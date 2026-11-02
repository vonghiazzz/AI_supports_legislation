import copy,json,math
from pathlib import Path
import numpy as np
import pytest
from scripts.Evaluation.experiments import bsard_bge_m3_dense as dense
from scripts.Evaluation.experiments.bsard_bm25_smoke import validate_rankings

def context():
    value={"cache_schema_version":dense.CACHE_SCHEMA,"model_id":dense.MODEL_ID,"model_revision":dense.MODEL_REVISION,"pooling":"cls","normalization":True,"query_instruction":None,"document_instruction":None,"effective_max_length":8192,"embedding_dimension":1024,"dtype":"float32","batch_size":1,"corpus_fingerprint":"a"*64,"test_split_fingerprint":"b"*64,"document_representation_fingerprint":"c"*64,"query_representation_fingerprint":"d"*64,"ordered_document_ids_fingerprint":"e"*64,"ordered_query_ids_fingerprint":"f"*64,"config_fingerprint":"1"*64,"dev107_config_fingerprint":"2"*64,"runner_source_fingerprint":"3"*64,"git_commit":"4"*40,"document_count":2,"query_count":1}
    value["context_fingerprint"]=dense.sha256_json(value); return value

def final_cache(path,ctx,docs=None,queries=None,mutate=None):
    docs=np.zeros((2,1024),dtype=np.float32) if docs is None else docs; queries=np.zeros((1,1024),dtype=np.float32) if queries is None else queries
    meta={**ctx,"document_shape":list(docs.shape),"query_shape":list(queries.shape),"document_matrix_fingerprint":dense._array_fp(docs),"query_matrix_fingerprint":dense._array_fp(queries),"model_load_seconds":1.0,"document_encoding_seconds":2.0,"query_encoding_seconds":3.0}
    if mutate: mutate(meta)
    meta["metadata_fingerprint"]=dense.sha256_json(meta)
    with path.open("wb") as f: np.savez(f,documents=docs,queries=queries,metadata_json=np.asarray(json.dumps(meta,sort_keys=True)))

def test_frozen_contract_constants():
    assert dense.MODEL_ID=="BAAI/bge-m3" and dense.MODEL_REVISION=="5617a9f61b028005a4858fdac845db406aefb181"
    assert (dense.DIMENSION,dense.MAX_LENGTH,dense.BATCH_SIZE)==(1024,8192,1)

@pytest.mark.parametrize("field",["model_revision","corpus_fingerprint","query_representation_fingerprint","ordered_document_ids_fingerprint","ordered_query_ids_fingerprint"])
def test_context_rejects_provenance_mismatch(field):
    expected=context(); actual=copy.deepcopy(expected); actual[field]="0"*64; body={k:v for k,v in actual.items() if k!="context_fingerprint"}; actual["context_fingerprint"]=dense.sha256_json(body)
    with pytest.raises(dense.BSARDBGEDenseError,match="provenance"): dense.validate_context(actual,expected)

def test_context_rejects_dishonest_fingerprint():
    actual=context(); actual["corpus_fingerprint"]="0"*64
    with pytest.raises(dense.BSARDBGEDenseError,match="provenance"): dense.validate_context(actual,context())

@pytest.mark.parametrize("shape",[(3,1024),(2,100)])
def test_final_cache_rejects_wrong_shape(tmp_path,shape):
    p=tmp_path/"e.npz"; final_cache(p,context(),docs=np.zeros(shape,dtype=np.float32))
    with pytest.raises(dense.BSARDBGEDenseError,match="matrix"): dense.load_final_file(p,context())

def test_final_cache_rejects_wrong_dtype(tmp_path):
    p=tmp_path/"e.npz"; final_cache(p,context(),docs=np.zeros((2,1024),dtype=np.float64))
    with pytest.raises(dense.BSARDBGEDenseError,match="matrix"): dense.load_final_file(p,context())

@pytest.mark.parametrize("bad",[math.nan,math.inf,-math.inf])
def test_final_cache_rejects_nonfinite(tmp_path,bad):
    d=np.zeros((2,1024),dtype=np.float32); d[0,0]=bad; p=tmp_path/"e.npz"; final_cache(p,context(),docs=d)
    with pytest.raises(dense.BSARDBGEDenseError,match="matrix"): dense.load_final_file(p,context())

def test_final_cache_rejects_matrix_fingerprint_mismatch(tmp_path):
    p=tmp_path/"e.npz"; final_cache(p,context(),mutate=lambda m:m.update(document_matrix_fingerprint="0"*64))
    with pytest.raises(dense.BSARDBGEDenseError,match="matrix"): dense.load_final_file(p,context())

def test_valid_final_cache_loads(tmp_path):
    p=tmp_path/"e.npz"; final_cache(p,context()); d,q,m=dense.load_final_file(p,context()); assert d.shape==(2,1024) and q.shape==(1,1024) and m["metadata_fingerprint"]

def test_partial_cache_cannot_be_final(tmp_path):
    p=tmp_path/"cache.partial"; p.mkdir()
    with pytest.raises(dense.BSARDBGEDenseError,match="partial"): dense.load_final_cache(p,context())

def test_shard_rejects_provenance_and_fingerprint(tmp_path):
    p=tmp_path/"s.npz"; v=np.zeros((2,1024),dtype=np.float32); meta={"context_fingerprint":"a","kind":"documents","start":0,"end":2,"ordered_ids_fingerprint":"b","source_text_fingerprint":"c"}; dense._save_shard(p,v,meta); assert dense._load_shard(p,meta,2).shape==(2,1024)
    wrong={**meta,"ordered_ids_fingerprint":"x"}
    with pytest.raises(dense.BSARDBGEDenseError,match="provenance"): dense._load_shard(p,wrong,2)

def test_ranking_validation_rejects_duplicate_unknown_and_missing_qid():
    queries=[{"query_id":"1"}]; base=[{"query_id":"1","ranking":[{"rank":1,"standard_law_id":"1","score":1.0}]}]; validate_rankings(base,queries,{"1"})
    dup=copy.deepcopy(base); dup[0]["ranking"].append({"rank":2,"standard_law_id":"1","score":0.5})
    with pytest.raises(RuntimeError,match="duplicate"): validate_rankings(dup,queries,{"1"})
    with pytest.raises(RuntimeError,match="unknown"): validate_rankings([{"query_id":"1","ranking":[{"rank":1,"standard_law_id":"9","score":1.0}]}],queries,{"1"})
    with pytest.raises(RuntimeError,match="qid"): validate_rankings([],queries,{"1"})

def test_dense_tie_break_is_numeric_document_id(monkeypatch):
    faiss=pytest.importorskip("faiss"); docs=np.asarray([[1,0],[1,0],[1,0]],dtype=np.float32); qs=np.asarray([[1,0]],dtype=np.float32)
    monkeypatch.setattr(dense,"DIMENSION",2); monkeypatch.setattr(dense,"TOP_K",3)
    records,meta=dense.dense_rankings(docs,qs,[{"document_id":"10"},{"document_id":"2"},{"document_id":"1"}],[{"query_id":"7"}])
    assert [x["standard_law_id"] for x in records[0]["ranking"]]==["1","2","10"] and meta["index_ntotal"]==3

def test_token_audit_counts_overlength():
    class T:
        def __call__(self,texts,**kwargs): return {"length":[3,9000][:len(texts)]}
    result=dense.token_audit(T(),["a","b"]); assert result["exceed_effective_max"]==1 and result["max"]==9000

def test_research_decision_is_evidence_driven():
    comp={str(n):{"queries":{"both_succeed":1,"bm25_only":1,"bge_only":1,"both_fail":1}} for n in dense.DIAGNOSTIC_DEPTHS}
    assert dense.research_decision({"recall@100":.1},comp)=="DENSE_COMPLEMENTS_BM25"
    for row in comp.values(): row["queries"]["bm25_only"]=0
    assert dense.research_decision({"x":.1},comp)=="DENSE_DOMINATES_BM25"

def test_complementarity_counts_qid_qrel_pairs_not_global_document_ids(monkeypatch):
    monkeypatch.setattr(dense,"DIAGNOSTIC_DEPTHS",(1,))
    queries=[
        {"query_id":"1","relevant_document_ids":["7","8"]},
        {"query_id":"2","relevant_document_ids":["7","9"]},
    ]
    bge=[
        {"query_id":"1","ranking":[{"standard_law_id":"7"}]},
        {"query_id":"2","ranking":[{"standard_law_id":"7"}]},
    ]
    bm25=[
        {"query_id":"1","ranking":[{"standard_law_id":"7"}]},
        {"query_id":"2","ranking":[{"standard_law_id":"9"}]},
    ]
    result=dense.complementarity(bge,bm25,queries)["1"]
    assert result["references"]=={"found_by_both":1,"bm25_only":1,"bge_only":1,"missed_by_both":1}
    assert sum(result["references"].values())==4

def test_model_resolution_rejects_wrong_revision():
    with pytest.raises(dense.BSARDBGEDenseError,match="exact"): dense.resolve_snapshot({"id":dense.MODEL_ID,"revision":"bad","local_files_only":True})

def test_publish_strict_json_and_cleanup(tmp_path):
    config=tmp_path/"c.yaml"; config.write_text("x: 1\n"); names={"retrieval_rankings.json","summary_metrics.json","coverage_complementarity.json","run_manifest.json","token_length_audit.json"}; artifacts={n:{"ok":True} for n in names}; run=tmp_path/"run"; dense.publish(run,config,artifacts); assert {p.name for p in run.iterdir()}==names|{"eval_config.snapshot.yaml"}
    bad=tmp_path/"bad"; artifacts["summary_metrics.json"]={"x":math.nan}
    with pytest.raises(ValueError): dense.publish(bad,config,artifacts)
    assert not bad.exists() and not (tmp_path/"bad.partial").exists()

def test_bm25_control_artifact_validates():
    root=Path(__file__).resolve().parents[3]; summary,rankings=dense.validate_bm25(root/"backend/data/runs/bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"); assert summary["safe_canonical"]["query_count"]==222 and len(rankings["records"])==222
