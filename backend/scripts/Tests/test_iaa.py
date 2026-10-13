import os
import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.resolve()))

import pytest

def compute_kappa_raw(paired_ratings, classes):
    total_labels = len(paired_ratings)
    if total_labels == 0:
        return 1.0

    agreement_count = sum(1 for a, b in paired_ratings if a == b)
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
    return kappa

def test_cohens_kappa_math():
    """
    Assert that the custom Cohen's Kappa formula produces exact values against a manually computed matrix.
    """
    classes = ["none", "core", "supporting"]
    
    # 6 items:
    # 1. ("none", "none") - Agree
    # 2. ("none", "core") - Disagree
    # 3. ("core", "core") - Agree
    # 4. ("core", "none") - Disagree
    # 5. ("supporting", "supporting") - Agree
    # 6. ("supporting", "core") - Disagree
    paired_ratings = [
        ("none", "none"),
        ("none", "core"),
        ("core", "core"),
        ("core", "none"),
        ("supporting", "supporting"),
        ("supporting", "core")
    ]
    
    kappa = compute_kappa_raw(paired_ratings, classes)
    
    # Expected Po = 3/6 = 0.5
    # Expected Pe = (2/6 * 2/6) + (2/6 * 3/6) + (2/6 * 1/6) = 1/9 + 1/6 + 1/18 = 6/18 = 1/3
    # Expected Kappa = (0.5 - 1/3) / (1 - 1/3) = (1/6) / (2/3) = 1/4 = 0.25
    assert abs(kappa - 0.25) < 1e-5
    print("✅ Cohen's Kappa math verification test passed!")
