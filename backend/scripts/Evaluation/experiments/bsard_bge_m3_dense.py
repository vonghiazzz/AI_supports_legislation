"""BSARD Phase 1D: frozen zero-shot BGE-M3 dense retrieval."""
from __future__ import annotations

import argparse, datetime as dt, gc, hashlib, json, math, os, platform, shutil, subprocess, time, uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ..benchmarking.bsard_adapter import load_bsard
from ..evaluation.bsard_metrics import paired_bootstrap_mean_delta
from .bsard_bm25_smoke import (
    DIAGNOSTIC_DEPTHS, _peak_rss_mb, _percentile, coverage_diagnostics,
    evaluate_records, ranking_fingerprint, sha256_file, sha256_json, validate_rankings,
)

PROTOCOL_ID="bsard_bge_m3_dense_v1"
CACHE_SCHEMA="bsard_bge_m3_embedding_cache_v1"
MODEL_ID="BAAI/bge-m3"
MODEL_REVISION="5617a9f61b028005a4858fdac845db406aefb181"
DIMENSION=1024
MAX_LENGTH=8192
BATCH_SIZE=1
SHARD_SIZE=100
TOP_K=500

class BSARDBGEDenseError(RuntimeError): pass

def _git_head(root:Path)->str:
    value=subprocess.run(["git","rev-parse","HEAD"],cwd=root,check=True,capture_output=True,text=True).stdout.strip()
    if len(value)!=40: raise BSARDBGEDenseError("valid git HEAD required")
    return value

def _array_fp(a:np.ndarray)->str:
    a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(json.dumps(list(a.shape),separators=(",",":")).encode()); h.update(a.tobytes()); return h.hexdigest()

def _write_json(path:Path,payload:Any)->None:
    with path.open("w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")

def resolve_snapshot(model:Mapping[str,Any])->Path:
    if model.get("id")!=MODEL_ID or model.get("revision")!=MODEL_REVISION or model.get("local_files_only") is not True: raise BSARDBGEDenseError("exact local BGE-M3 revision required")
    root=Path(os.environ.get("HF_HUB_CACHE",Path.home()/".cache/huggingface/hub"))
    path=root/"models--BAAI--bge-m3"/"snapshots"/MODEL_REVISION
    required=("config.json","modules.json","1_Pooling/config.json","pytorch_model.bin","tokenizer.json","sentencepiece.bpe.model")
    if path.name!=MODEL_REVISION or not path.is_dir() or any(not (path/x).is_file() for x in required): raise BSARDBGEDenseError("BGE_M3_EXACT_REVISION_CACHE_MISSING")
    return path

def model_provenance(snapshot:Path)->dict:
    files={}
    for name in ("config.json","modules.json","1_Pooling/config.json","sentence_bert_config.json","tokenizer_config.json","tokenizer.json","sentencepiece.bpe.model","pytorch_model.bin"):
        p=snapshot/name; resolved=p.resolve(strict=True)
        entry={"size":resolved.stat().st_size,"resolved_blob":resolved.name}
        if name!="pytorch_model.bin": entry["sha256"]=sha256_file(p)
        files[name]=entry
    return {"model_id":MODEL_ID,"revision":MODEL_REVISION,"snapshot_path":str(snapshot),"files":files}

def token_audit(tokenizer,texts:Sequence[str])->dict:
    lengths=[]
    for start in range(0,len(texts),64): lengths.extend(tokenizer(list(texts[start:start+64]),add_special_tokens=True,truncation=False,padding=False,return_length=True)["length"])
    a=np.asarray(lengths,dtype=np.int64)
    return {"count":len(a),"min":int(a.min()),"median":float(np.median(a)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":int(a.max()),"exceed_effective_max":int((a>MAX_LENGTH).sum()),"fraction_exceed_effective_max":float((a>MAX_LENGTH).mean())}

def representation_fp(ids:Sequence[str],texts:Sequence[str])->str:
    return sha256_json([{"id":i,"text_sha256":hashlib.sha256(t.encode()).hexdigest()} for i,t in zip(ids,texts)])

def cache_context(dataset,config_path:Path,dev_config_path:Path,runner:Path,commit:str)->dict:
    dids=[d["document_id"] for d in dataset.documents]; qids=[q["query_id"] for q in dataset.human_test]
    dtexts=[d["document_text"] for d in dataset.documents]; qtexts=[q["query_text"] for q in dataset.human_test]
    value={"cache_schema_version":CACHE_SCHEMA,"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"pooling":"cls","normalization":True,"query_instruction":None,"document_instruction":None,"effective_max_length":MAX_LENGTH,"embedding_dimension":DIMENSION,"dtype":"float32","batch_size":BATCH_SIZE,"corpus_fingerprint":dataset.fingerprints["corpus_ordered"],"test_split_fingerprint":dataset.fingerprints["test_split"],"document_representation_fingerprint":representation_fp(dids,dtexts),"query_representation_fingerprint":representation_fp(qids,qtexts),"ordered_document_ids_fingerprint":sha256_json(dids),"ordered_query_ids_fingerprint":sha256_json(qids),"config_fingerprint":sha256_file(config_path),"dev107_config_fingerprint":sha256_file(dev_config_path),"runner_source_fingerprint":sha256_file(runner),"git_commit":commit,"document_count":len(dids),"query_count":len(qids)}
    value["context_fingerprint"]=sha256_json(value); return value

def validate_context(actual:Mapping[str,Any],expected:Mapping[str,Any])->None:
    stored=actual.get("context_fingerprint"); body={k:v for k,v in actual.items() if k!="context_fingerprint"}
    if stored!=sha256_json(body) or dict(actual)!=dict(expected): raise BSARDBGEDenseError("cache provenance mismatch")

def _save_shard(path:Path,vectors:np.ndarray,metadata:dict)->None:
    if vectors.dtype!=np.float32 or not np.isfinite(vectors).all(): raise BSARDBGEDenseError("invalid shard embeddings")
    metadata={**metadata,"shape":list(vectors.shape),"dtype":"float32","matrix_fingerprint":_array_fp(vectors)}; metadata["metadata_fingerprint"]=sha256_json(metadata)
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f: np.savez(f,vectors=vectors,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)))
    os.replace(tmp,path)

def _load_shard(path:Path,expected:Mapping[str,Any],rows:int)->np.ndarray:
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={"vectors","metadata_json"}: raise BSARDBGEDenseError("incomplete shard")
        v=np.asarray(z["vectors"]); m=json.loads(str(z["metadata_json"].item()))
    stored=m.pop("metadata_fingerprint",None)
    if stored!=sha256_json(m) or any(m.get(k)!=x for k,x in expected.items()): raise BSARDBGEDenseError("shard provenance mismatch")
    if v.shape!=(rows,DIMENSION) or v.dtype!=np.float32 or not np.isfinite(v).all() or m.get("matrix_fingerprint")!=_array_fp(v): raise BSARDBGEDenseError("shard matrix validation failed")
    return v

def encode_cache(dataset,snapshot:Path,cache_root:Path,context:dict)->tuple[np.ndarray,np.ndarray,dict]:
    final=cache_root/CACHE_SCHEMA; partial=cache_root/(CACHE_SCHEMA+".partial")
    if final.exists(): return load_final_cache(final,context)
    partial.mkdir(parents=True,exist_ok=True); context_path=partial/"context.json"
    if context_path.exists(): validate_context(json.loads(context_path.read_text()),context)
    else: _write_json(context_path,context)
    from sentence_transformers import SentenceTransformer
    load_start=time.perf_counter(); model=SentenceTransformer(str(snapshot),device="cpu",local_files_only=True); model.max_seq_length=MAX_LENGTH; load_seconds=time.perf_counter()-load_start
    dtexts=[d["document_text"] for d in dataset.documents]; dids=[d["document_id"] for d in dataset.documents]
    encode_start=time.perf_counter(); shards=[]
    for start in range(0,len(dtexts),SHARD_SIZE):
        end=min(start+SHARD_SIZE,len(dtexts)); path=partial/f"documents_{start:06d}_{end:06d}.npz"; meta={"context_fingerprint":context["context_fingerprint"],"kind":"documents","start":start,"end":end,"ordered_ids_fingerprint":sha256_json(dids[start:end]),"source_text_fingerprint":representation_fp(dids[start:end],dtexts[start:end])}
        if path.exists(): v=_load_shard(path,meta,end-start)
        else:
            v=model.encode(dtexts[start:end],batch_size=BATCH_SIZE,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True).astype("float32"); _save_shard(path,v,meta)
        shards.append(v)
        if end%500==0 or end==len(dtexts): print(f"BGE-M3 documents {end}/{len(dtexts)}",flush=True)
    docs=np.concatenate(shards); document_seconds=time.perf_counter()-encode_start
    qtexts=[q["query_text"] for q in dataset.human_test]; qids=[q["query_id"] for q in dataset.human_test]; qpath=partial/"queries.npz"; qmeta={"context_fingerprint":context["context_fingerprint"],"kind":"queries","start":0,"end":len(qtexts),"ordered_ids_fingerprint":sha256_json(qids),"source_text_fingerprint":representation_fp(qids,qtexts)}
    qstart=time.perf_counter()
    if qpath.exists(): queries=_load_shard(qpath,qmeta,len(qtexts))
    else: queries=model.encode(qtexts,batch_size=BATCH_SIZE,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True).astype("float32"); _save_shard(qpath,queries,qmeta)
    query_seconds=time.perf_counter()-qstart
    del model; gc.collect()
    meta={**context,"document_shape":list(docs.shape),"query_shape":list(queries.shape),"document_matrix_fingerprint":_array_fp(docs),"query_matrix_fingerprint":_array_fp(queries),"model_load_seconds":load_seconds,"document_encoding_seconds":document_seconds,"query_encoding_seconds":query_seconds}; meta["metadata_fingerprint"]=sha256_json(meta)
    tmp=partial/"embeddings.tmp"; target=partial/"embeddings.npz"
    with tmp.open("wb") as f: np.savez(f,documents=docs,queries=queries,metadata_json=np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False)))
    os.replace(tmp,target); load_final_file(target,context); os.replace(partial,final); return docs,queries,meta

def load_final_file(path:Path,context:Mapping[str,Any])->tuple[np.ndarray,np.ndarray,dict]:
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={"documents","queries","metadata_json"}: raise BSARDBGEDenseError("incomplete final cache")
        d=np.asarray(z["documents"]); q=np.asarray(z["queries"]); m=json.loads(str(z["metadata_json"].item()))
    stored=m.pop("metadata_fingerprint",None)
    if stored!=sha256_json(m) or any(m.get(k)!=v for k,v in context.items()): raise BSARDBGEDenseError("final cache provenance mismatch")
    if d.shape!=(context["document_count"],DIMENSION) or q.shape!=(context["query_count"],DIMENSION) or d.dtype!=np.float32 or q.dtype!=np.float32 or not np.isfinite(d).all() or not np.isfinite(q).all() or m.get("document_matrix_fingerprint")!=_array_fp(d) or m.get("query_matrix_fingerprint")!=_array_fp(q): raise BSARDBGEDenseError("final cache matrix validation failed")
    m["metadata_fingerprint"]=stored; return d,q,m

def load_final_cache(final:Path,context:Mapping[str,Any]):
    if final.name.endswith(".partial") or not (final/"embeddings.npz").is_file(): raise BSARDBGEDenseError("partial cache cannot be final")
    return load_final_file(final/"embeddings.npz",context)

def dense_rankings(doc_vectors:np.ndarray,query_vectors:np.ndarray,documents,queries):
    import faiss
    start=time.perf_counter(); index=faiss.IndexFlatIP(DIMENSION); index.add(doc_vectors); build=time.perf_counter()-start
    records=[]; lat=[]; all_ids=[d["document_id"] for d in documents]
    for query,vector in zip(queries,query_vectors):
        t=time.perf_counter(); scores,indices=index.search(vector[None].astype("float32"),len(all_ids)); pairs=[(all_ids[int(i)],float(s)) for s,i in zip(scores[0],indices[0]) if i>=0]; pairs.sort(key=lambda x:(-x[1],int(x[0]))); lat.append(time.perf_counter()-t); records.append({"query_id":query["query_id"],"ranking":[{"rank":r,"standard_law_id":i,"score":s} for r,(i,s) in enumerate(pairs[:TOP_K],1)]})
    return records,{"index_build_seconds":build,"search_total_seconds":sum(lat),"search_mean_ms":float(np.mean(lat))*1000,"search_p50_ms":_percentile(lat,50)*1000,"search_p95_ms":_percentile(lat,95)*1000,"index_ntotal":int(index.ntotal),"index_dimension":int(index.d),"estimated_index_bytes":int(doc_vectors.nbytes)}

def validate_bm25(run:Path)->tuple[dict,dict]:
    if not run.is_dir(): raise BSARDBGEDenseError("BM25_CONTROL_ARTIFACT_UNAVAILABLE")
    m=json.loads((run/"run_manifest.json").read_text()); s=json.loads((run/"summary_metrics.json").read_text()); r=json.loads((run/"retrieval_rankings.json").read_text())
    if ranking_fingerprint(r["records"])!=m["retrieval_fingerprint"] or m["retrieval_fingerprint"]!="306b06754cecf8f564352bb3b9a817e28a2884c9301ecf16b67dea74a820ea5e": raise BSARDBGEDenseError("BM25 control retrieval mismatch")
    result=sha256_json({"retrieval_fingerprint":m["retrieval_fingerprint"],"official_compat":s["official_compat"],"safe_canonical":s["safe_canonical"],"coverage_diagnostics":s["coverage_diagnostics"]})
    if result!=m["result_fingerprint"] or result!="0a7109f7e48587d64690721447de59a93c43970300a70b2a0d9433f46779b150": raise BSARDBGEDenseError("BM25 control result mismatch")
    return s,r

def complementarity(bge,bm25,queries):
    br={r["query_id"]:[x["standard_law_id"] for x in r["ranking"]] for r in bge}; mr={r["query_id"]:[x["standard_law_id"] for x in r["ranking"]] for r in bm25}; out={}
    for n in DIAGNOSTIC_DEPTHS:
        qcount={"both_succeed":0,"bm25_only":0,"bge_only":0,"both_fail":0}; bf=set(); mf=set()
        for q in queries:
            qid=q["query_id"]; rel=set(q["relevant_document_ids"]); bh=rel&set(br[qid][:n]); mh=rel&set(mr[qid][:n]); bf|={(qid,x) for x in bh}; mf|={(qid,x) for x in mh}
            key="both_succeed" if bh and mh else "bge_only" if bh else "bm25_only" if mh else "both_fail"; qcount[key]+=1
        allrel={(q["query_id"],x) for q in queries for x in set(q["relevant_document_ids"])}; bden=qcount["bge_only"]+qcount["both_fail"]; mden=qcount["bm25_only"]+qcount["both_fail"]
        out[str(n)]={"queries":qcount,"references":{"found_by_both":len(bf&mf),"bm25_only":len(mf-bf),"bge_only":len(bf-mf),"missed_by_both":len(allrel-(bf|mf))},"bge_recovery_of_bm25_failures":{"count":qcount["bge_only"],"denominator":bden,"rate":qcount["bge_only"]/bden if bden else 0.0},"bm25_recovery_of_bge_failures":{"count":qcount["bm25_only"],"denominator":mden,"rate":qcount["bm25_only"]/mden if mden else 0.0}}
    return out

def paired(bge_safe,bm_safe):
    names=("recall@100","recall@200","recall@500","map@100","mrr@100","ndcg@10"); out={}
    for name in names:
        qids=sorted(bge_safe["per_query"],key=int); b=[bm_safe["per_query"][q]["metrics"][name] for q in qids]; t=[bge_safe["per_query"][q]["metrics"][name] for q in qids]; out[name]=paired_bootstrap_mean_delta(b,t,samples=20000,seed=42)
    return out

def research_decision(deltas:Mapping[str,float],comp:Mapping[str,Any])->str:
    rows=[comp[str(n)]["queries"] for n in DIAGNOSTIC_DEPTHS]
    if all(v>0 for v in deltas.values()) and all(r["bm25_only"]==0 for r in rows): return "DENSE_DOMINATES_BM25"
    if all(v<0 for v in deltas.values()) and all(r["bge_only"]==0 for r in rows): return "BM25_DOMINATES_DENSE"
    if sum(r["bge_only"]>0 and r["bm25_only"]>0 for r in rows)>=3: return "DENSE_COMPLEMENTS_BM25"
    if sum(r["bge_only"]+r["bm25_only"] for r in rows)<=26 and max(map(abs,deltas.values()))<0.01: return "LARGELY_REDUNDANT"
    return "MIXED_OR_INCONCLUSIVE"

def publish(run_dir:Path,config:Path,artifacts:Mapping[str,Any]):
    expected={"retrieval_rankings.json","summary_metrics.json","coverage_complementarity.json","run_manifest.json","token_length_audit.json"}; partial=run_dir.with_name(run_dir.name+".partial")
    if set(artifacts)!=expected or run_dir.exists() or partial.exists(): raise BSARDBGEDenseError("unsafe artifact publication")
    partial.mkdir(parents=True)
    try:
        for name,p in artifacts.items(): _write_json(partial/name,p)
        shutil.copy2(config,partial/"eval_config.snapshot.yaml")
        if {p.name for p in partial.iterdir()}!=expected|{"eval_config.snapshot.yaml"}: raise BSARDBGEDenseError("incomplete artifacts")
        os.replace(partial,run_dir)
    except Exception: shutil.rmtree(partial,ignore_errors=True); raise

def _read_json(path:Path):
    return json.loads(path.read_text(encoding="utf-8"),parse_constant=lambda value:(_ for _ in ()).throw(BSARDBGEDenseError(f"non-finite JSON constant: {value}")))

def reanalyze_run(source_run:Path,raw_root:Path,config_path:Path,output_root:Path)->Path:
    """Rebuild analysis from a validated frozen ranking/cache without model inference."""
    root=Path(__file__).resolve().parents[4]; runner=Path(__file__).resolve(); commit=_git_head(root)
    if source_run.name.endswith(".partial") or not source_run.is_dir(): raise BSARDBGEDenseError("validated source run required")
    manifest=_read_json(source_run/"run_manifest.json"); rankings=_read_json(source_run/"retrieval_rankings.json"); audit=_read_json(source_run/"token_length_audit.json")
    if manifest.get("protocol_id")!=PROTOCOL_ID or manifest.get("git_commit")!=commit or manifest.get("config_sha256")!=sha256_file(config_path): raise BSARDBGEDenseError("source run provenance mismatch")
    dataset=load_bsard(raw_root,strict_identity=True)
    expected={"raw_dataset_sha256":dict(dataset.provenance["raw_sha256"]),"corpus_fingerprint":dataset.fingerprints["corpus_ordered"],"test_split_fingerprint":dataset.fingerprints["test_split"],"test_unique_qrel_fingerprint":dataset.fingerprints["test_unique_qrels"],"document_count":len(dataset.documents),"query_count":len(dataset.human_test)}
    if any(manifest.get(k)!=v for k,v in expected.items()): raise BSARDBGEDenseError("source run dataset mismatch")
    from transformers import AutoTokenizer
    token_start=time.perf_counter(); tokenizer=AutoTokenizer.from_pretrained(resolve_snapshot({"id":MODEL_ID,"revision":MODEL_REVISION,"local_files_only":True}),local_files_only=True); replayed_audit={"tokenizer_class":tokenizer.__class__.__name__,"tokenizer_model_max_length":tokenizer.model_max_length,"effective_max_length":MAX_LENGTH,"truncation":"single-sequence tail truncation","documents":token_audit(tokenizer,[x["document_text"] for x in dataset.documents]),"queries":token_audit(tokenizer,[x["query_text"] for x in dataset.human_test])}; token_seconds=time.perf_counter()-token_start
    if replayed_audit!=audit: raise BSARDBGEDenseError("source token audit mismatch")
    records=rankings.get("records"); retrieval_fp=ranking_fingerprint(records)
    if retrieval_fp!=rankings.get("retrieval_fingerprint") or retrieval_fp!=manifest.get("retrieval_fingerprint"): raise BSARDBGEDenseError("source retrieval fingerprint mismatch")
    validate_rankings(records,dataset.human_test,[d["document_id"] for d in dataset.documents])
    cache=Path(manifest["cache_path"]); stored_context=_read_json(cache/"context.json"); docs,queries,cache_meta=load_final_file(cache/"embeddings.npz",stored_context)
    if cache_meta["metadata_fingerprint"]!=manifest["cache_metadata_fingerprint"] or cache_meta["document_matrix_fingerprint"]!=manifest["document_matrix_fingerprint"] or cache_meta["query_matrix_fingerprint"]!=manifest["query_matrix_fingerprint"]: raise BSARDBGEDenseError("source cache fingerprint mismatch")
    replay,replay_search=dense_rankings(docs,queries,dataset.documents,dataset.human_test)
    if ranking_fingerprint(replay)!=retrieval_fp: raise BSARDBGEDenseError("source cache replay mismatch")
    official,safe=evaluate_records(records,dataset.human_test,[d["document_id"] for d in dataset.documents]); coverage=coverage_diagnostics(records,dataset.human_test)
    bm_summary,bm_rank=validate_bm25(root/"backend/data/runs/bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"); comp=complementarity(records,bm_rank["records"],dataset.human_test); bootstrap=paired(safe,bm_summary["safe_canonical"])
    deltas={n:safe["metrics"][n]-bm_summary["safe_canonical"]["metrics"][n] for n in ("recall@100","recall@200","recall@500","map@100","mrr@100","ndcg@10")}; decision=research_decision(deltas,comp)
    old_summary=_read_json(source_run/"summary_metrics.json"); timing={**old_summary["timing"],"tokenization_audit_seconds":token_seconds,"corrected_analysis_cache_replay_faiss":replay_search}
    summary={"official_compat":official,"safe_canonical":safe,"coverage":coverage,"bm25_control":{"validated":True,"run_id":bm_summary.get("protocol_id","bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"),"metric_deltas":deltas,"paired_bootstrap":bootstrap},"complementarity":comp,"decision":decision,"timing":timing}
    result_fp=sha256_json({"retrieval_fingerprint":retrieval_fp,"official":official,"safe":safe,"coverage":coverage,"complementarity":comp,"bootstrap":bootstrap})
    run_id=f"{PROTOCOL_ID}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"; output_rankings={"run_id":run_id,"records":records,"retrieval_fingerprint":retrieval_fp}
    output_manifest={**manifest,"run_id":run_id,"runner_sha256":sha256_file(runner),"result_fingerprint":result_fp,"source_retrieval_run_id":manifest["run_id"],"source_retrieval_runner_sha256":manifest["runner_sha256"],"analysis_replayed_from_validated_cache":True}
    artifacts={"retrieval_rankings.json":output_rankings,"summary_metrics.json":summary,"coverage_complementarity.json":comp,"run_manifest.json":output_manifest,"token_length_audit.json":audit}; final=output_root/run_id; publish(final,config_path,artifacts); return final

def run(raw_root:Path,config_path:Path,output_root:Path)->Path:
    root=Path(__file__).resolve().parents[4]; runner=Path(__file__).resolve(); commit=_git_head(root); config=yaml.safe_load(config_path.read_text()); dev_path=root/"backend/scripts/Evaluation/configs/dev107_retrieval_model_screening.yaml"; dev=yaml.safe_load(dev_path.read_text()); model=dev["bge_m3"]
    if model!={"id":MODEL_ID,"revision":MODEL_REVISION,"snapshot_path":None,"architecture":"XLMRobertaModel","dimension":1024,"pooling":"cls","query_instruction":None,"normalize_embeddings":True,"max_seq_length":8192,"batch_size":1,"device":"cpu","dtype":"float32","local_files_only":True}: raise BSARDBGEDenseError("DEV107 BGE contract changed")
    snapshot=resolve_snapshot(model); dataset=load_bsard(raw_root,strict_identity=True)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True); token_start=time.perf_counter(); audit={"tokenizer_class":tok.__class__.__name__,"tokenizer_model_max_length":tok.model_max_length,"effective_max_length":MAX_LENGTH,"truncation":"single-sequence tail truncation","documents":token_audit(tok,[d["document_text"] for d in dataset.documents]),"queries":token_audit(tok,[q["query_text"] for q in dataset.human_test])}; token_seconds=time.perf_counter()-token_start
    if tok.model_max_length!=MAX_LENGTH or audit["queries"]["exceed_effective_max"]!=0: raise BSARDBGEDenseError("truncation safety gate failed")
    context=cache_context(dataset,config_path,dev_path,runner,commit); total=time.perf_counter(); docs,queries,cache_meta=encode_cache(dataset,snapshot,output_root,context); cache_seconds=time.perf_counter()-total
    records,search=dense_rankings(docs,queries,dataset.documents,dataset.human_test); validate_rankings(records,dataset.human_test,[d["document_id"] for d in dataset.documents]); retrieval_fp=ranking_fingerprint(records)
    replay,replay_search=dense_rankings(docs,queries,dataset.documents,dataset.human_test)
    if ranking_fingerprint(replay)!=retrieval_fp: raise BSARDBGEDenseError("cache replay mismatch")
    official,safe=evaluate_records(records,dataset.human_test,[d["document_id"] for d in dataset.documents]); coverage=coverage_diagnostics(records,dataset.human_test)
    bm_summary,bm_rank=validate_bm25(root/"backend/data/runs/bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"); comp=complementarity(records,bm_rank["records"],dataset.human_test); bootstrap=paired(safe,bm_summary["safe_canonical"])
    deltas={n:safe["metrics"][n]-bm_summary["safe_canonical"]["metrics"][n] for n in ("recall@100","recall@200","recall@500","map@100","mrr@100","ndcg@10")}
    decision=research_decision(deltas,comp)
    summary={"official_compat":official,"safe_canonical":safe,"coverage":coverage,"bm25_control":{"validated":True,"run_id":bm_summary["protocol_id"] if "protocol_id" in bm_summary else "bsard_bm25_smoke_v1_20261102_024146_040921_1e759195","metric_deltas":deltas,"paired_bootstrap":bootstrap},"complementarity":comp,"decision":decision,"timing":{"tokenization_audit_seconds":token_seconds,"cache_prepare_seconds":cache_seconds,**{k:cache_meta.get(k) for k in ("model_load_seconds","document_encoding_seconds","query_encoding_seconds")},"faiss":search,"replay_faiss":replay_search,"peak_rss_mb":_peak_rss_mb()}}
    result_fp=sha256_json({"retrieval_fingerprint":retrieval_fp,"official":official,"safe":safe,"coverage":coverage,"complementarity":comp,"bootstrap":bootstrap})
    run_id=f"{PROTOCOL_ID}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"; rankings={"run_id":run_id,"records":records,"retrieval_fingerprint":retrieval_fp}
    manifest={"run_id":run_id,"protocol_id":PROTOCOL_ID,"git_commit":commit,"runner_sha256":sha256_file(runner),"config_sha256":sha256_file(config_path),"raw_dataset_sha256":dict(dataset.provenance["raw_sha256"]),"corpus_fingerprint":dataset.fingerprints["corpus_ordered"],"test_split_fingerprint":dataset.fingerprints["test_split"],"test_unique_qrel_fingerprint":dataset.fingerprints["test_unique_qrels"],"model":model_provenance(snapshot),"cache_metadata_fingerprint":cache_meta["metadata_fingerprint"],"document_matrix_fingerprint":cache_meta["document_matrix_fingerprint"],"query_matrix_fingerprint":cache_meta["query_matrix_fingerprint"],"retrieval_fingerprint":retrieval_fp,"result_fingerprint":result_fp,"cache_path":str((output_root/CACHE_SCHEMA).resolve()),"document_count":len(dataset.documents),"query_count":len(dataset.human_test),"device":"cpu","dtype":"float32","batch_size":1,"hybrid_used":False,"reranker_used":False}
    artifacts={"retrieval_rankings.json":rankings,"summary_metrics.json":summary,"coverage_complementarity.json":comp,"run_manifest.json":manifest,"token_length_audit.json":audit}; final=output_root/run_id; publish(final,config_path,artifacts); return final

def main(argv=None):
    root=Path(__file__).resolve().parents[4]; p=argparse.ArgumentParser(); p.add_argument("--raw-root",type=Path,required=True); p.add_argument("--config",type=Path,default=root/"backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml"); p.add_argument("--output-root",type=Path,default=root/"backend/data/runs"); p.add_argument("--reanalyze-run",type=Path); a=p.parse_args(argv); result=reanalyze_run(a.reanalyze_run.resolve(),a.raw_root,a.config.resolve(),a.output_root.resolve()) if a.reanalyze_run else run(a.raw_root,a.config.resolve(),a.output_root.resolve()); print(result); return result
if __name__=="__main__": main()
