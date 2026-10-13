import os
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))

import json
import pytest
from backend.scripts.Evaluation.validate_benchmark import validate_benchmark_file, ValidationError

# Base directory setup
BASE_DIR = Path(__file__).parent.parent.parent

def test_benchmark_exists_and_valid():
    """Assert that the benchmark evaluation file exists and is valid according to the schema."""
    benchmark_path = BASE_DIR / "data" / "judgments_evaluation.json"
    assert benchmark_path.exists(), "judgments_evaluation.json does not exist."
    
    # Must validate successfully
    assert validate_benchmark_file(benchmark_path) is True

def test_benchmark_schema_validation_failure():
    """Assert that validate_benchmark_file correctly catches invalid schema patterns."""
    # 1. Missing ID
    invalid_item_1 = [{"query": "test query", "ground_truth": {"core": [], "supporting": []}}]
    # 2. Missing supporting ground truth
    invalid_item_2 = [{"id": 1, "query": "test query", "ground_truth": {"core": []}}]
    # 3. Invalid importance in core
    invalid_item_3 = [{
        "id": 1, 
        "query": "test query", 
        "ground_truth": {
            "core": [{"law_code": "code", "article": "art", "reasoning": "reason", "importance": "invalid", "importance_score": 3, "source_type": "explicit", "domain": "substantive"}],
            "supporting": []
        }
    }]
    # 4. Invalid importance_score in core
    invalid_item_4 = [{
        "id": 1, 
        "query": "test query", 
        "ground_truth": {
            "core": [{"law_code": "code", "article": "art", "reasoning": "reason", "importance": "decisive", "importance_score": 1, "source_type": "explicit", "domain": "substantive"}],
            "supporting": []
        }
    }]
    # 5. Invalid source_type
    invalid_item_5 = [{
        "id": 1, 
        "query": "test query", 
        "ground_truth": {
            "core": [{"law_code": "code", "article": "art", "reasoning": "reason", "importance": "decisive", "importance_score": 3, "source_type": "invalid_type", "domain": "substantive"}],
            "supporting": []
        }
    }]
    # 6. Invalid domain
    invalid_item_6 = [{
        "id": 1, 
        "query": "test query", 
        "ground_truth": {
            "core": [{"law_code": "code", "article": "art", "reasoning": "reason", "importance": "decisive", "importance_score": 3, "source_type": "explicit", "domain": "invalid_domain"}],
            "supporting": []
        }
    }]
    
    tmp_dir = BASE_DIR / "scratch"
    tmp_dir.mkdir(exist_ok=True)
    
    invalid_cases = [invalid_item_1, invalid_item_2, invalid_item_3, invalid_item_4, invalid_item_5, invalid_item_6]
    for idx, invalid_data in enumerate(invalid_cases):
        tmp_file = tmp_dir / f"invalid_test_{idx}.json"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(invalid_data, f)
            
        with pytest.raises(ValidationError):
            validate_benchmark_file(tmp_file)
            
        # Clean up
        if tmp_file.exists():
            tmp_file.unlink()

def test_laws_config_ranges():
    """Assert that the laws_config.json file exists and has correct start/end ranges."""
    config_path = BASE_DIR / "dataLaw" / "laws_config.json"
    assert config_path.exists(), "laws_config.json does not exist."
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    assert isinstance(config, list)
    for entry in config:
        assert "key" in entry
        assert "law_name" in entry
        assert "active_start_year" in entry
        assert "active_end_year" in entry
        
        start = entry["active_start_year"]
        end = entry["active_end_year"]
        assert start <= end, f"Active ranges mismatch for {entry['key']}: {start} > {end}"

def test_manifest_metadata():
    """Assert dataset versioning metadata matches expected definitions."""
    manifest_path = BASE_DIR / "data" / "dataset_manifest.json"
    assert manifest_path.exists(), "dataset_manifest.json does not exist."
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest.get("version") == "1.0.0"
    assert manifest.get("schema_version") == "2.0.0"
    assert manifest.get("total_queries") == 107
