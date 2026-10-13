import os
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))

import pytest
from backend.scripts.Evaluation import run_rag_evaluation

def test_configurable_weighted_recall():
    """
    Test that get_weighted_recall calculates the recall correctly using configurable weights
    and that changing the weights dynamically changes the output metric values.
    """
    # Mock evaluation results
    mock_results = [
        {
            "core_gt_filtered": ["điều 1 luat_a", "điều 2 luat_a"],
            "supporting_gt_filtered": ["điều 3 luat_a"],
            "metrics": {
                "test_cfg": {
                    "top_k": [
                        {"standard_law_id": "điều 1 luat_a"}, # core hit
                        {"standard_law_id": "điều 4 luat_a"}  # miss
                    ]
                }
            }
        }
    ]

    # Test Case 1: alpha_core = 3, alpha_support = 1 (default)
    run_rag_evaluation.ALPHA_CORE = 3
    run_rag_evaluation.ALPHA_SUPPORT = 1
    recall_1 = run_rag_evaluation.get_weighted_recall(mock_results, "test_cfg", k=5, is_filtered=True)
    # Expected: (3 * 1 + 1 * 0) / (3 * 2 + 1 * 1) = 3 / 7 approx 0.42857
    assert abs(recall_1 - (3.0 / 7.0)) < 1e-5

    # Test Case 2: alpha_core = 5, alpha_support = 2
    run_rag_evaluation.ALPHA_CORE = 5
    run_rag_evaluation.ALPHA_SUPPORT = 2
    recall_2 = run_rag_evaluation.get_weighted_recall(mock_results, "test_cfg", k=5, is_filtered=True)
    # Expected: (5 * 1 + 2 * 0) / (5 * 2 + 2 * 1) = 5 / 12 approx 0.41667
    assert abs(recall_2 - (5.0 / 12.0)) < 1e-5

    # Test Case 3: alpha_core = 1, alpha_support = 1 (equal weights)
    run_rag_evaluation.ALPHA_CORE = 1
    run_rag_evaluation.ALPHA_SUPPORT = 1
    recall_3 = run_rag_evaluation.get_weighted_recall(mock_results, "test_cfg", k=5, is_filtered=True)
    # Expected: (1 * 1 + 1 * 0) / (1 * 2 + 1 * 1) = 1 / 3 approx 0.33333
    assert abs(recall_3 - (1.0 / 3.0)) < 1e-5

    # Assert that changing weights actually results in different recall values
    assert recall_1 != recall_2
    assert recall_2 != recall_3
    print("✅ Unit tests for configurable weighted recall passed successfully!")
