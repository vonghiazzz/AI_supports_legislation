import json
import shutil
from pathlib import Path
import jsonschema

def migrate_record(record):
    """
    Migrates a single record from Schema v1 to Schema v2.
    """
    old_gt = record.get("ground_truth", {})
    new_core = []
    new_supporting = []

    # Map core items
    old_core = old_gt.get("core", {})
    for domain_key in ["substantive", "procedural"]:
        for item in old_core.get(domain_key, []):
            law_code = item.get("law_code")
            article = item.get("article")
            source_type = item.get("source_type", "explicit")
            
            # Construct reasoning
            reasoning = f"Decisive {domain_key} legal basis ({law_code} - {article}) for resolving the dispute in the case query."
            
            new_item = {
                "law_code": law_code,
                "article": article,
                "reasoning": reasoning,
                "importance": "decisive",
                "importance_score": 3,
                "source_type": source_type,
                "domain": domain_key
            }
            new_core.append(new_item)

    # Map supporting items
    old_supporting = old_gt.get("supporting", {})
    for domain_key in ["substantive", "procedural"]:
        for item in old_supporting.get(domain_key, []):
            law_code = item.get("law_code")
            article = item.get("article")
            source_type = item.get("source_type", "explicit")
            
            # Construct reasoning
            reasoning = f"Supporting {domain_key} legal basis ({law_code} - {article}) providing secondary context or procedural guidance."
            
            new_item = {
                "law_code": law_code,
                "article": article,
                "reasoning": reasoning,
                "importance": "supporting",
                "importance_score": 1,
                "source_type": source_type,
                "domain": domain_key
            }
            new_supporting.append(new_item)

    new_record = record.copy()
    new_record["ground_truth"] = {
        "core": new_core,
        "supporting": new_supporting
    }
    return new_record

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    dataset_path = data_dir / "judgments_evaluation.json"
    schema_path = data_dir / "gt_schema_v2.json"
    report_path = base_dir / "migration_report.md"

    print(f"[*] Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_records = len(dataset)
    print(f"[*] Found {total_records} records in dataset.")

    migrated_dataset = []
    for record in dataset:
        migrated_record = migrate_record(record)
        migrated_dataset.append(migrated_record)

    # Load JSON Schema
    print(f"[*] Loading schema from: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validate dataset against schema
    print("[*] Validating migrated dataset against Schema v2...")
    validation_success = True
    validation_error_msg = ""
    try:
        jsonschema.validate(instance=migrated_dataset, schema=schema)
        print("✅ Validation successful: Migrated dataset matches Schema v2!")
    except jsonschema.exceptions.ValidationError as e:
        validation_success = False
        validation_error_msg = str(e)
        print(f"❌ Schema validation failed: {e}")

    # Create backup of legacy file
    backup_path = dataset_path.with_suffix(".json.bak")
    print(f"[*] Backing up legacy dataset to: {backup_path}")
    shutil.copy2(dataset_path, backup_path)

    # Write migrated dataset to file
    print(f"[*] Writing migrated dataset to: {dataset_path}")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(migrated_dataset, f, ensure_ascii=False, indent=2)

    # Generate migration report
    print(f"[*] Generating migration report at: {report_path}")
    coverage_pct = (len(migrated_dataset) / total_records) * 100 if total_records > 0 else 0.0
    
    report_content = f"""# Dataset Migration Report: Ground Truth Schema v2

## Migration Statistics

- **Total Records in Dataset**: {total_records}
- **Migrated Records**: {len(migrated_dataset)}
- **Migration Coverage**: {coverage_pct:.2f}%
- **Schema Validation Result**: {"SUCCESS" if validation_success else "FAILURE"}

## Validation Details

"""
    if validation_success:
        report_content += "✅ All migrated records successfully validated against the `gt_schema_v2.json` schema.\n"
    else:
        report_content += f"❌ Schema validation failed with the following error:\n```\n{validation_error_msg}\n```\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("🎉 Migration completed successfully!")

if __name__ == "__main__":
    main()
