import json
from pathlib import Path

def interpret_kappa(kappa):
    if kappa < 0.20:
        return "Poor"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "Annotation"
    
    annotator_a_path = data_dir / "annotator_a.json"
    annotator_b_path = data_dir / "annotator_b.json"
    report_path = data_dir / "iaa_report.md"

    print(f"[*] Loading Annotator A from: {annotator_a_path}")
    with open(annotator_a_path, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    print(f"[*] Loading Annotator B from: {annotator_b_path}")
    with open(annotator_b_path, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    # Index by ID
    map_a = {item["id"]: item for item in data_a}
    map_b = {item["id"]: item for item in data_b}

    common_ids = sorted(list(set(map_a.keys()) & set(map_b.keys())))
    print(f"[*] Found {len(common_ids)} common sample queries.")

    paired_ratings = []
    disagreements = []
    
    for q_id in common_ids:
        q_a = map_a[q_id]
        q_b = map_b[q_id]
        judgment_name = q_a.get("judgment_name", "N/A")

        # Collect candidate laws from both
        laws_a = {}
        for item in q_a["ground_truth"].get("core", []):
            laws_a[(item["law_code"], item["article"])] = "core"
        for item in q_a["ground_truth"].get("supporting", []):
            laws_a[(item["law_code"], item["article"])] = "supporting"

        laws_b = {}
        for item in q_b["ground_truth"].get("core", []):
            laws_b[(item["law_code"], item["article"])] = "core"
        for item in q_b["ground_truth"].get("supporting", []):
            laws_b[(item["law_code"], item["article"])] = "supporting"

        all_laws = set(laws_a.keys()) | set(laws_b.keys())

        for law in all_laws:
            rating_a = laws_a.get(law, "none")
            rating_b = laws_b.get(law, "none")

            paired_ratings.append((rating_a, rating_b))

            if rating_a != rating_b:
                disagreements.append({
                    "query_id": q_id,
                    "judgment_name": judgment_name,
                    "law_code": law[0],
                    "article": law[1],
                    "annotator_a": rating_a,
                    "annotator_b": rating_b
                })

    total_labels = len(paired_ratings)
    if total_labels == 0:
        print("❌ No labels found to compare.")
        return

    agreement_count = sum(1 for a, b in paired_ratings if a == b)
    disagreement_count = total_labels - agreement_count
    agreement_rate = (agreement_count / total_labels) * 100

    # Calculate Kappa
    classes = ["none", "core", "supporting"]
    count_a = {c: 0 for c in classes}
    count_b = {c: 0 for c in classes}
    
    for a, b in paired_ratings:
        count_a[a] += 1
        count_b[b] += 1

    Po = agreement_count / total_labels
    Pe = sum((count_a[c] / total_labels) * (count_b[c] / total_labels) for c in classes)

    if Pe < 1.0:
        kappa = (Po - Pe) / (1.0 - Pe)
    else:
        kappa = 1.0

    interpretation = interpret_kappa(kappa)

    print("\n--- Inter-Annotator Agreement Statistics ---")
    print(f"Total Queries Evaluated: {len(common_ids)}")
    print(f"Total Labels Checked: {total_labels}")
    print(f"Agreement Count: {agreement_count}")
    print(f"Disagreement Count: {disagreement_count}")
    print(f"Agreement Rate: {agreement_rate:.2f}%")
    print(f"Cohen's Kappa: {kappa:.4f} ({interpretation})")

    # Generate Markdown Report
    report_content = f"""# Inter-Annotator Agreement (IAA) Report

## Agreement Metrics Summary

- **Total Queries Evaluated**: {len(common_ids)}
- **Total Unique Labels Evaluated**: {total_labels}
- **Observed Agreement**: {agreement_count} / {total_labels} ({agreement_rate:.2f}%)
- **Disagreement Count**: {disagreement_count}
- **Cohen's Kappa**: {kappa:.4f}
- **Interpretation**: **{interpretation}** (based on standard interval tables)

### Kappa Interpretation Scale
- `<0.20`: Poor
- `0.21-0.40`: Fair
- `0.41-0.60`: Moderate
- `0.61-0.80`: Substantial
- `0.81-1.00`: Almost Perfect

## Flagged Disagreements for Review

| Query ID | Judgment PDF | Law Code | Article | Annotator A | Annotator B |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for d in disagreements:
        report_content += f"| {d['query_id']} | {d['judgment_name']} | {d['law_code']} | {d['article']} | `{d['annotator_a']}` | `{d['annotator_b']}` |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n✅ Automatically generated IAA report at: {report_path}")

if __name__ == "__main__":
    main()
