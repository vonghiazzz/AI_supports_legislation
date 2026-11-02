import ast
import json
from pathlib import Path

import pytest

from scripts.Evaluation.reporting import cross_benchmark_evidence_freeze as freeze


def per_query(control=0.0, treatment=1.0):
    left = {str(index): {"10": {metric: control for metric in ("recall", "weighted_recall", "mrr", "ndcg")}} for index in range(1, 5)}
    right = {str(index): {"10": {metric: treatment for metric in ("recall", "weighted_recall", "mrr", "ndcg")}} for index in range(1, 5)}
    return left, right


def test_missing_source_artifact_fails_closed(tmp_path):
    with pytest.raises(freeze.EvidenceFreezeError, match="missing final source run"):
        freeze._load_run(tmp_path, "missing", ("run_manifest",))


def test_wrong_source_fingerprint_fails_closed():
    with pytest.raises(freeze.EvidenceFreezeError, match="fingerprint mismatch"):
        freeze.validate_source_identity({"fingerprint": "bad"}, {"fingerprint": "good"}, "SOURCE")


def test_benchmark_roles_are_frozen():
    roles = freeze.benchmark_role_freeze()
    assert roles["DEV107"]["status"] == "DEVELOPMENT_EXPLORATORY"
    assert roles["DEV107"]["confirmatory"] is False
    assert roles["BSARD"]["status"] == "EXTERNAL_EXPLORATORY_FROZEN"
    assert roles["INDEPENDENT_VIETNAMESE"]["status"] == "FUTURE_CONFIRMATORY_TARGET"
    assert roles["INDEPENDENT_VIETNAMESE"]["current_state"] == "DRAFT"


def test_bsard_freeze_declaration_and_terminal_commit():
    row = freeze.benchmark_role_freeze()["BSARD"]
    assert row["declaration"] == "BSARD_EXPLORATORY_TRACK_FROZEN_AFTER_PHASE_1F"
    assert row["terminal_commit"] == "22ee5a048349c74ebe3eaa2c5025374870b61b4c"
    assert "new_method_tuning" in row["prohibited_without_new_exploratory_protocol"]


def test_no_cross_benchmark_metric_pooling_contract():
    matrix = {"raw_metric_pooling": False, "common_estimand_declared": False}
    assert matrix["raw_metric_pooling"] is False
    assert matrix["common_estimand_declared"] is False
    assert "pooled_metrics" not in matrix


def test_claim_status_vocabulary_accepts_only_frozen_values():
    for status in freeze.CLAIM_STATUSES:
        claim = freeze._claim("X", "text", {}, {}, status, status, status, [], "allowed")
        assert claim["cross_benchmark_status"] == status
    with pytest.raises(freeze.EvidenceFreezeError, match="invalid claim status"):
        freeze._claim("X", "text", {}, {}, "PROVEN", "PROVEN", "PROVEN", [], "bad")


def test_failure_components_sum_and_fractions_sum_to_one():
    row = freeze.failure_ratio(7, 2, 1, 10)
    assert row["candidate_absence_dominates_ranking_failure"] is True
    assert sum(row["fractions"].values()) == pytest.approx(1.0)


def test_failure_sum_mismatch_fails_closed():
    with pytest.raises(freeze.EvidenceFreezeError, match="do not sum"):
        freeze.failure_ratio(7, 2, 2, 10)


def test_candidate_dominance_is_derived_not_claim_text():
    assert freeze.failure_ratio(2, 7, 1, 10)["candidate_absence_dominates_ranking_failure"] is False
    assert freeze.failure_ratio(7, 2, 1, 10)["candidate_absence_dominates_ranking_failure"] is True


def test_paired_bootstrap_is_deterministic_and_positive():
    control, treatment = per_query()
    first = freeze.paired_bootstrap(control, treatment)
    second = freeze.paired_bootstrap(control, treatment)
    assert first == second
    assert all(row["ci_lower"] > 0 for row in first.values())


def test_paired_bootstrap_rejects_qid_mismatch():
    control, treatment = per_query(); treatment.pop("4")
    with pytest.raises(freeze.EvidenceFreezeError, match="identical"):
        freeze.paired_bootstrap(control, treatment)


def test_research_hypothesis_is_not_reported_as_success():
    payload = freeze.hypotheses()
    assert payload["policy"]["do_not_design_algorithm_in_this_phase"] is True
    assert payload["future_independent_baselines"]["status"] == "PROPOSED_PRE_REGISTRATION_SET"
    assert any("not" in value.lower() or "whether" in value.lower() for value in payload["NOT_YET_TESTED"])


def test_future_method_firewall_freezes_required_inputs():
    firewall = freeze.benchmark_role_freeze()["firewall"]
    assert firewall["new_method_development_inputs"] == ["DEV107", "synthetic_or_unit_level_analysis"]
    assert firewall["BSARD_may_motivate_but_not_optimize"] is True
    assert "implementation_commit" in firewall["freeze_before_confirmatory_evaluation"]
    assert "candidate_budget" in firewall["freeze_before_confirmatory_evaluation"]


def test_no_model_or_retrieval_library_imports():
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
    prohibited = {"transformers", "sentence_transformers", "faiss", "rank_bm25", "google", "torch"}
    assert imports.isdisjoint(prohibited)


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(freeze.EvidenceFreezeError, match="non-finite"):
        freeze.strict_load(path)


def test_strict_writer_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        freeze.strict_write(tmp_path / "bad.json", {"value": float("inf")})


def test_json_fingerprint_is_key_order_deterministic():
    assert freeze.sha256_json({"a": 1, "b": 2}) == freeze.sha256_json({"b": 2, "a": 1})


def test_markdown_is_generated_from_structured_claims():
    claim = freeze._claim("E1", "claim text", {}, {}, "SUPPORTED_SINGLE_BENCHMARK", "SUPPORTED_SINGLE_BENCHMARK", "REPLICATED_PATTERN", [], "allowed conclusion")
    matrix = {"claims": [claim]}; failures = {"systems": {"S": freeze.failure_ratio(7, 2, 1, 10)}}
    text = freeze.render_markdown(freeze.benchmark_role_freeze(), matrix, failures, freeze.hypotheses())
    assert "E1" in text and "claim text" in text and "allowed conclusion" in text
    assert "BSARD_EXPLORATORY_TRACK_FROZEN_AFTER_PHASE_1F" in text


def test_publish_is_atomic_and_records_fingerprints(tmp_path):
    structured = {
        "evidence_matrix.json": {"claims": []},
        "failure_pattern_summary.json": {"systems": {}},
        "benchmark_role_freeze.json": freeze.benchmark_role_freeze(),
        "research_hypotheses.json": freeze.hypotheses(),
        "source_artifact_inventory.json": {"artifacts": []},
    }
    result = freeze.publish(tmp_path, structured, "# report\n", {"git_commit": freeze.TERMINAL_COMMIT})
    assert result.is_dir() and not result.with_name(result.name + ".partial").exists()
    manifest = freeze.strict_load(result / "run_manifest.json")
    assert manifest["models_executed"] == [] and manifest["retrievers_executed"] == []
    assert manifest["raw_metric_pooling"] is False


def test_deterministic_structured_report_fingerprint():
    structured = {
        "roles": freeze.benchmark_role_freeze(),
        "hypotheses": freeze.hypotheses(),
        "failure": freeze.failure_ratio(7, 2, 1, 10),
    }
    assert freeze.sha256_json(structured) == freeze.sha256_json(structured)
