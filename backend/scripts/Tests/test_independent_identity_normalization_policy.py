import ast
import copy
import hashlib
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import (
    identity_normalization_policy as policy,
)
from backend.scripts.Evaluation.benchmarking.source_intake import (
    validate_source_identity_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / policy.DEFAULT_POLICY_RELATIVE
MANIFEST_PATH = (
    PROJECT_ROOT / "backend/dataLaw/independent_intake/acquisition_manifest.json"
)
SOURCE_INTAKE_PATH = (
    PROJECT_ROOT / "backend/scripts/Evaluation/benchmarking/source_intake.py"
)


def _loaded():
    return policy.load_policy(POLICY_PATH)


def _refingerprint(value):
    changed = copy.deepcopy(value)
    changed["policy_fingerprint"] = policy.policy_fingerprint(changed)
    return changed


def test_exact_source039_rule_is_accepted():
    checked = policy.load_and_validate_policy(PROJECT_ROOT)
    assert checked["corroboration"]["result"] == "PASS"
    assert checked["corroboration"]["independently_represented_identity"] == (
        "25/2026/QĐST-DS"
    )
    assert (
        policy.normalize_allowed_identity_value(
            checked,
            "batch2dc-20261105-039",
            "judgment_number",
            "25 2026/QĐST-DS",
        )
        == "25/2026/QĐST-DS"
    )


def test_canonical_result_satisfies_current_identity_validator():
    canonical = policy.normalize_allowed_identity_value(
        _loaded(),
        policy.EXPECTED_ACQUISITION_ID,
        policy.EXPECTED_FIELD,
        policy.EXPECTED_RAW_VALUE,
    )
    checked = validate_source_identity_payload(
        {
            "identity_version": "independent_source_identity_v1",
            "verified_metadata": {
                "judgment_number": canonical,
                "case_id": "26/2025/TLST-DS",
                "court": "Tòa án nhân dân Khu vực 2 - Nghệ An",
                "decision_date": "2026-07-31",
                "procedure_stage": "so_tham",
            },
            "verified_fields": {
                "judgment_number": True,
                "case_id": True,
                "court": True,
                "decision_date": True,
                "procedure_stage": True,
            },
            "related_decision_ids": [],
            "underlying_dispute_id": None,
        }
    )
    assert checked["verified_metadata"]["judgment_number"] == canonical


@pytest.mark.parametrize(
    ("acquisition_id", "field", "raw_value"),
    [
        ("batch2dc-20261105-038", "judgment_number", "25 2026/QĐST-DS"),
        ("batch2dc-20261105-*", "judgment_number", "25 2026/QĐST-DS"),
        ("batch2dc-20261105-039", "case_id", "25 2026/QĐST-DS"),
        ("batch2dc-20261105-039", "related_decision_ids", "25 2026/QĐST-DS"),
        ("batch2dc-20261105-039", "judgment_number", "25  2026/QĐST-DS"),
        ("batch2dc-20261105-039", "judgment_number", "26 2026/QĐST-DS"),
        ("batch2dc-20261105-039", "judgment_number", "25 2025/QĐST-DS"),
        ("batch2dc-20261105-039", "judgment_number", "25 2026/DS-ST"),
        ("batch2dc-20261105-039", "judgment_number", "25 2026//QĐST-DS"),
        ("batch2dc-20261105-039", "judgment_number", "25 2026 QĐST-DS"),
    ],
)
def test_nonexact_scope_or_edits_are_rejected(acquisition_id, field, raw_value):
    with pytest.raises(policy.IdentityNormalizationPolicyError):
        policy.normalize_allowed_identity_value(
            _loaded(), acquisition_id, field, raw_value
        )


@pytest.mark.parametrize("result", [None, "NEUTRAL", "CONFLICT"])
def test_missing_neutral_or_conflicting_corroboration_is_rejected(result):
    changed = _loaded()
    if result is None:
        changed["corroboration"].pop("result")
    else:
        changed["corroboration"]["result"] = result
    changed = _refingerprint(changed)
    with pytest.raises(
        policy.IdentityNormalizationPolicyError,
        match="corroboration|PASS",
    ):
        policy.validate_policy(changed, PROJECT_ROOT)


def test_wildcard_or_general_rule_is_rejected():
    changed = _loaded()
    changed["scope"]["acquisition_id"] = "batch2dc-20261105-*"
    changed = _refingerprint(changed)
    with pytest.raises(
        policy.IdentityNormalizationPolicyError, match="scope differs"
    ):
        policy.validate_policy(changed, PROJECT_ROOT)


def test_more_than_one_separator_replacement_is_rejected():
    changed = _loaded()
    changed["transformation"]["changed_boundary_count"] = 2
    changed = _refingerprint(changed)
    with pytest.raises(
        policy.IdentityNormalizationPolicyError, match="transformation differs"
    ):
        policy.validate_policy(changed, PROJECT_ROOT)


def test_policy_binds_unchanged_source_intake_contract():
    assert (
        policy.sha256_file(SOURCE_INTAKE_PATH)
        == policy.EXPECTED_SOURCE_INTAKE_SHA256
    )
    assert _loaded()["bindings"]["source_intake_sha256"] == (
        policy.EXPECTED_SOURCE_INTAKE_SHA256
    )


def test_validation_and_normalization_do_not_mutate_manifest():
    before = MANIFEST_PATH.read_bytes()
    checked = policy.load_and_validate_policy(PROJECT_ROOT)
    policy.normalize_allowed_identity_value(
        checked,
        policy.EXPECTED_ACQUISITION_ID,
        policy.EXPECTED_FIELD,
        policy.EXPECTED_RAW_VALUE,
    )
    assert MANIFEST_PATH.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == policy.EXPECTED_MANIFEST_SHA256


def test_module_has_no_network_import_or_identity_mutation_call():
    tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint(
        {"requests", "urllib", "http", "httpx", "socket", "aiohttp"}
    )
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "record_source_identity" not in called_names


def test_policy_fingerprint_and_single_source_scope_are_stable():
    loaded = _loaded()
    assert loaded["policy_fingerprint"] == policy.policy_fingerprint(loaded)
    assert loaded["scope"] == policy.EXPECTED_SCOPE
