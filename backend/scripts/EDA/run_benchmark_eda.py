import json
from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path("/Users/tannghiavo/Documents/AllProjects/AI/AI_supports_legislation/backend/data")
eval_file = BASE_DIR / "judgments_evaluation.json"
results_file = BASE_DIR / "pipeline_results.json"
expl_file = BASE_DIR / "retrieval_explainability.json"

# Load files
with open(eval_file, 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open(results_file, 'r', encoding='utf-8') as f:
    pipeline_results = json.load(f)

with open(expl_file, 'r', encoding='utf-8') as f:
    explainability_logs = json.load(f)

# Helper function to map law codes to family
def map_law_code_to_family(law_code):
    lc = str(law_code).lower().strip()
    if "bo_luat_dan_su" in lc or "luat_dan_su" in lc or "bds" in lc:
        return "Civil Code"
    elif "to_tung_dan_su" in lc or "ttds" in lc:
        return "Civil Procedure Code"
    elif "dat_dai" in lc or "dd" in lc or "nghi_dinh_43" in lc or "nghi_dinh_01_2017" in lc:
        return "Land Law"
    elif "hon_nhan" in lc or "gia_dinh" in lc:
        return "Marriage & Family Law"
    elif "nghi_dinh_163" in lc or "nghi_dinh_17" in lc:
        return "Civil Code"
    else:
        return "Others"

# 1. Dataset Distribution Audit
total_queries = len(queries)
lengths = [len(q.get("query", "").split()) for q in queries]
gt_sizes = []
procedural_count = 0
substantive_count = 0

court_levels = {}
procedure_stages = {}
years = {}
law_families = {}
article_frequencies = {}

for q in queries:
    meta = q.get("metadata", {})
    court_levels[meta.get("court_level", "unknown")] = court_levels.get(meta.get("court_level", "unknown"), 0) + 1
    procedure_stages[meta.get("procedure_stage", "unknown")] = procedure_stages.get(meta.get("procedure_stage", "unknown"), 0) + 1
    years[meta.get("year", "unknown")] = years.get(meta.get("year", "unknown"), 0) + 1
    
    gt = q.get("ground_truth", {})
    core = gt.get("core", [])
    supp = gt.get("supporting", [])
    
    if isinstance(core, list) and isinstance(supp, list):
        sub = [item for item in core + supp if item.get("domain") == "substantive"]
        proc = [item for item in core + supp if item.get("domain") == "procedural"]
    else:
        # Legacy fallback
        core_dict = core if isinstance(core, dict) else {}
        supp_dict = supp if isinstance(supp, dict) else {}
        sub = core_dict.get("substantive", []) + supp_dict.get("substantive", [])
        proc = core_dict.get("procedural", []) + supp_dict.get("procedural", [])
    
    substantive_count += len(sub)
    procedural_count += len(proc)
    gt_sizes.append(len(sub) + len(proc))
    
    for item in sub + proc:
        code = item.get("law_code", "unknown")
        fam = map_law_code_to_family(code)
        law_families[fam] = law_families.get(fam, 0) + 1
        
        art_id = f"{item.get('article', '')} {code}".strip().lower()
        article_frequencies[art_id] = article_frequencies.get(art_id, 0) + 1

print("--- 1. DATASET DISTRIBUTION ---")
print(f"Total Queries: {total_queries}")
print(f"Query Length: Min={min(lengths)}, Max={max(lengths)}, Mean={np.mean(lengths):.2f}, Median={np.median(lengths):.1f}")
print(f"Ground Truth Size: Min={min(gt_sizes)}, Max={max(gt_sizes)}, Mean={np.mean(gt_sizes):.2f}, Median={np.median(gt_sizes):.1f}")
print(f"Substantive Articles: {substantive_count} ({substantive_count/(substantive_count+procedural_count)*100:.2f}%)")
print(f"Procedural Articles: {procedural_count} ({procedural_count/(substantive_count+procedural_count)*100:.2f}%)")

print("\nCourt Level Distribution:")
for k, v in court_levels.items():
    print(f"  - {k}: {v} ({v/total_queries*100:.2f}%)")

print("\nProcedure Stage Distribution:")
for k, v in procedure_stages.items():
    print(f"  - {k}: {v} ({v/total_queries*100:.2f}%)")

print("\nYear Distribution:")
for k, v in sorted(years.items()):
    print(f"  - {k}: {v} ({v/total_queries*100:.2f}%)")

print("\nLaw Family Distribution in GT:")
total_fam = sum(law_families.values())
for k, v in sorted(law_families.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {k}: {v} ({v/total_fam*100:.2f}%)")

# Article frequencies and long-tail analysis
sorted_articles = sorted(article_frequencies.items(), key=lambda x: x[1], reverse=True)
print(f"\nTotal unique articles in GT: {len(sorted_articles)}")
print("Top 10 most frequent articles:")
for art, count in sorted_articles[:10]:
    print(f"  - {art}: {count} times ({count/total_queries*100:.2f}% of queries)")

single_occurrence_count = sum(1 for art, count in sorted_articles if count == 1)
print(f"Long-tail size (articles appearing exactly once): {single_occurrence_count} ({single_occurrence_count/len(sorted_articles)*100:.2f}%)")

# 2. Retrieval Failure Taxonomy from explainability logs
print("\n--- 2. RETRIEVAL FAILURE TAXONOMY ---")
for cfg in ["BM25", "FAISS", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"]:
    counts = {}
    for item in explainability_logs:
        err = item["configurations"].get(cfg, {}).get("error_type")
        if err:
            counts[err] = counts.get(err, 0) + 1
    print(f"\nConfiguration: {cfg}")
    total_failures = sum(counts.values())
    for k, v in counts.items():
        print(f"  - {k}: {v} ({v/total_failures*100:.2f}% of failures)")

# 3. Dynamic Hybrid failure modes
print("\n--- 3. DYNAMIC HYBRID FAILURE MODES ---")
failure_modes = {}
for item in explainability_logs:
    mode = item.get("diagnostics", {}).get("failure_mode", "neutral")
    failure_modes[mode] = failure_modes.get(mode, 0) + 1
for k, v in failure_modes.items():
    print(f"  - {k}: {v} ({v/total_queries*100:.2f}%)")

