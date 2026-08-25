import json
from pathlib import Path
from typing import Any, Dict, List

class ValidationError(Exception):
    """Custom exception raised when benchmark file fails schema validation."""
    pass

def validate_benchmark_file(file_path: Path) -> bool:
    """
    Validates the structure and content of the benchmark evaluation JSON file.
    Raises ValidationError if any schema check fails.
    """
    if not file_path.exists():
        raise ValidationError(f"Benchmark file not found at: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON format: {e}")

    if not isinstance(data, list):
        raise ValidationError("Root of benchmark dataset must be a JSON array/list.")

    if len(data) == 0:
        raise ValidationError("Benchmark dataset is empty.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Item at index {idx} must be a JSON object.")

        # Check required root keys
        for key in ["id", "query", "ground_truth"]:
            if key not in item:
                raise ValidationError(f"Item at index {idx} (ID: {item.get('id', 'N/A')}) is missing required key '{key}'")

        # Validate types
        if not isinstance(item["id"], int):
            raise ValidationError(f"Item at index {idx}: 'id' must be an integer, got {type(item['id']).__name__}")
        
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValidationError(f"Item at index {idx} (ID: {item['id']}): 'query' must be a non-empty string.")

        # Validate metadata if present
        if "metadata" in item:
            metadata = item["metadata"]
            if not isinstance(metadata, dict):
                raise ValidationError(f"Item ID {item['id']}: 'metadata' must be a JSON object.")
            # Check year can be parsed to integer or is null
            year = metadata.get("year")
            if year is not None:
                try:
                    int(year)
                except (ValueError, TypeError):
                    raise ValidationError(f"Item ID {item['id']}: 'metadata.year' must be an integer or null, got '{year}'.")

        # Validate ground_truth
        gt = item["ground_truth"]
        if not isinstance(gt, dict):
            raise ValidationError(f"Item ID {item['id']}: 'ground_truth' must be a JSON object.")

    # Load JSON Schema v2
    schema_path = Path(__file__).parent.parent.parent / "data" / "gt_schema_v2.json"
    if schema_path.exists():
        try:
            import jsonschema
            with open(schema_path, "r", encoding="utf-8") as sf:
                schema = json.load(sf)
            jsonschema.validate(instance=data, schema=schema)
        except ImportError:
            # Fallback manual validation if jsonschema is not available
            _manual_validate(data)
        except jsonschema.exceptions.ValidationError as e:
            raise ValidationError(f"JSON Schema Validation Error: {e.message} at path {list(e.absolute_path)}")
    else:
        # Fallback manual validation if schema file is not found
        _manual_validate(data)

    print(f"✅ Schema Validation Success: {file_path.name} is fully valid.")
    return True

def _manual_validate(data: List[Any]):
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(f"Item at index {idx} must be a JSON object.")

        for key in ["id", "query", "ground_truth"]:
            if key not in item:
                raise ValidationError(f"Item at index {idx} (ID: {item.get('id', 'N/A')}) is missing required key '{key}'")

        if not isinstance(item["id"], int):
            raise ValidationError(f"Item at index {idx}: 'id' must be an integer, got {type(item['id']).__name__}")
        
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValidationError(f"Item at index {idx} (ID: {item['id']}): 'query' must be a non-empty string.")

        gt = item["ground_truth"]
        if not isinstance(gt, dict):
            raise ValidationError(f"Item ID {item['id']}: 'ground_truth' must be a JSON object.")

        for partition_key in ["core", "supporting"]:
            if partition_key not in gt:
                raise ValidationError(f"Item ID {item['id']}: 'ground_truth' is missing required partition '{partition_key}'")
            
            partition_list = gt[partition_key]
            if not isinstance(partition_list, list):
                raise ValidationError(f"Item ID {item['id']}: 'ground_truth.{partition_key}' must be a JSON array.")

            for g_idx, g_item in enumerate(partition_list):
                if not isinstance(g_item, dict):
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}] must be an object.")
                
                # Required keys in v2
                required_keys = ["law_code", "article", "reasoning", "importance", "importance_score", "source_type", "domain"]
                for req_g_key in required_keys:
                    if req_g_key not in g_item:
                        raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}] is missing key '{req_g_key}'")
                    
                if not isinstance(g_item["law_code"], str) or not g_item["law_code"].strip():
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}].'law_code' must be a non-empty string.")

                if not isinstance(g_item["article"], str):
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}].'article' must be a string.")

                if not isinstance(g_item["reasoning"], str) or not g_item["reasoning"].strip():
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}].'reasoning' must be a non-empty string.")

                if partition_key == "core":
                    if g_item["importance"] != "decisive":
                        raise ValidationError(f"Item ID {item['id']}, ground_truth.core[{g_idx}].'importance' must be 'decisive', got '{g_item['importance']}'")
                    if g_item["importance_score"] != 3:
                        raise ValidationError(f"Item ID {item['id']}, ground_truth.core[{g_idx}].'importance_score' must be 3, got {g_item['importance_score']}")
                elif partition_key == "supporting":
                    if g_item["importance"] != "supporting":
                        raise ValidationError(f"Item ID {item['id']}, ground_truth.supporting[{g_idx}].'importance' must be 'supporting', got '{g_item['importance']}'")
                    if g_item["importance_score"] != 1:
                        raise ValidationError(f"Item ID {item['id']}, ground_truth.supporting[{g_idx}].'importance_score' must be 1, got {g_item['importance_score']}")

                source_type = g_item["source_type"]
                if source_type not in ["explicit", "inferred"]:
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}].'source_type' must be 'explicit' or 'inferred', got '{source_type}'")

                domain = g_item["domain"]
                if domain not in ["substantive", "procedural"]:
                    raise ValidationError(f"Item ID {item['id']}, ground_truth.{partition_key}[{g_idx}].'domain' must be 'substantive' or 'procedural', got '{domain}'")

if __name__ == "__main__":
    # Test on the actual dataset
    dataset_path = Path(__file__).parent.parent.parent / "data" / "judgments_evaluation.json"
    try:
        validate_benchmark_file(dataset_path)
    except ValidationError as e:
        print(f"❌ Validation Error: {e}")
        exit(1)
