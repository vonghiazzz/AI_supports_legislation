import json
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "Annotation"

    annotator_a_path = data_dir / "annotator_a.json"
    annotator_b_path = data_dir / "annotator_b.json"
    disagreements_path = data_dir / "resolved_disagreements.json"
    adjudicated_path = data_dir / "adjudicated_gt.json"

    print(f"[*] Loading Annotator A from: {annotator_a_path}")
    with open(annotator_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    print(f"[*] Loading Annotator B from: {annotator_b_path}")
    with open(annotator_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    print(f"[*] Loading reviewer resolutions from: {disagreements_path}")
    with open(disagreements_path, "r", encoding="utf-8") as f:
        resolutions = json.load(f)

    map_b = {item["id"]: item for item in data_b}

    adjudicated_dataset = []

    for item_a in data_a:
        q_id = item_a["id"]
        judgment_name = item_a["judgment_name"]
        query = item_a["query"]

        if q_id not in map_b:
            # If item only annotated by A, keep A's annotations
            print(f"[!] Warning: Query {q_id} only exists in Annotator A annotations. Keeping A's version.")
            adjudicated_dataset.append(item_a)
            continue

        item_b = map_b[q_id]

        # Gather annotations by A
        laws_a = {}
        for item in item_a["ground_truth"].get("core", []):
            laws_a[(item["law_code"], item["article"])] = {
                "label": "core",
                "source_type": item.get("source_type", "explicit"),
                "domain": item.get("domain", "substantive")
            }
        for item in item_a["ground_truth"].get("supporting", []):
            laws_a[(item["law_code"], item["article"])] = {
                "label": "supporting",
                "source_type": item.get("source_type", "explicit"),
                "domain": item.get("domain", "substantive")
            }

        # Gather annotations by B
        laws_b = {}
        for item in item_b["ground_truth"].get("core", []):
            laws_b[(item["law_code"], item["article"])] = {
                "label": "core",
                "source_type": item.get("source_type", "explicit"),
                "domain": item.get("domain", "substantive")
            }
        for item in item_b["ground_truth"].get("supporting", []):
            laws_b[(item["law_code"], item["article"])] = {
                "label": "supporting",
                "source_type": item.get("source_type", "explicit"),
                "domain": item.get("domain", "substantive")
            }

        all_laws = set(laws_a.keys()) | set(laws_b.keys())

        adjudicated_core = []
        adjudicated_supporting = []

        for law in all_laws:
            law_code, article = law
            meta_a = laws_a.get(law)
            meta_b = laws_b.get(law)

            rating_a = meta_a["label"] if meta_a else "none"
            rating_b = meta_b["label"] if meta_b else "none"

            final_label = None
            final_meta = meta_a if meta_a else meta_b

            if rating_a == rating_b:
                # Agreement
                final_label = rating_a
            else:
                # Disagreement - resolve via reviewer
                res_key = f"{q_id}:{law_code}:{article}"
                if res_key in resolutions:
                    final_label = resolutions[res_key]
                    print(f"[*] Adjudicating disagreement for {res_key}: A='{rating_a}', B='{rating_b}' -> RESOLVED TO '{final_label}'")
                else:
                    final_label = rating_a  # Default fallback
                    print(f"[!] Warning: Unresolved disagreement for {res_key}. Falling back to Annotator A label: '{final_label}'")

            if final_label == "core":
                adjudicated_core.append({
                    "law_code": law_code,
                    "article": article,
                    "reasoning": f"Adjudicated decisive legal basis ({law_code} - {article}) for resolving the dispute.",
                    "importance": "decisive",
                    "importance_score": 3,
                    "source_type": final_meta["source_type"],
                    "domain": final_meta["domain"]
                })
            elif final_label == "supporting":
                adjudicated_supporting.append({
                    "law_code": law_code,
                    "article": article,
                    "reasoning": f"Adjudicated supporting legal basis ({law_code} - {article}) providing secondary context.",
                    "importance": "supporting",
                    "importance_score": 1,
                    "source_type": final_meta["source_type"],
                    "domain": final_meta["domain"]
                })

        adjudicated_item = {
            "id": q_id,
            "judgment_name": judgment_name,
            "query": query,
            "ground_truth": {
                "core": adjudicated_core,
                "supporting": adjudicated_supporting
            }
        }
        adjudicated_dataset.append(adjudicated_item)

    # Save final version
    print(f"[*] Saving final adjudicated Ground Truth to: {adjudicated_path}")
    with open(adjudicated_path, "w", encoding="utf-8") as f:
        json.dump(adjudicated_dataset, f, ensure_ascii=False, indent=2)

    print("🎉 Adjudication pipeline successfully completed!")

if __name__ == "__main__":
    main()
