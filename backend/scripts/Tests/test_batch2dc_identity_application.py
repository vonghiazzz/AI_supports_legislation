import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import (
    batch2dc_identity_application as application,
)
from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking.identity_normalization_policy import (
    EXPECTED_CANONICAL_VALUE,
    EXPECTED_FIELD,
    EXPECTED_RAW_VALUE,
    IdentityNormalizationPolicyError,
    load_policy,
    normalize_allowed_identity_value,
)
from backend.scripts.Tests.test_independent_source_intake import (
    PROJECT_ROOT,
    _manifest,
    _materialize,
    _minimal_pdf_bytes,
    _source,
    _structured,
    _write_manifest,
)


SOURCE_A = "batch2dc-20261105-001"
SOURCE_B = application.SOURCE039_ID
SOURCE_OTHER = "unrelated-source-001"


def _payload(
    number: str,
    *,
    case_id: str,
    court: str = "Tòa án nhân dân tỉnh Test",
    date: str = "2026-01-02",
) -> dict:
    return {
        "identity_version": source_intake.IDENTITY_PAYLOAD_VERSION,
        "verified_metadata": {
            "judgment_number": number,
            "case_id": case_id,
            "court": court,
            "decision_date": date,
            "procedure_stage": "so_tham",
        },
        "verified_fields": {
            field: True for field in source_intake.REVIEW_METADATA_FIELDS
        },
        "related_decision_ids": [],
        "underlying_dispute_id": None,
    }


def _make_extracted_source(root: Path, acquisition_id: str, filename: str) -> dict:
    source = _source(acquisition_id, filename, status="extracted")
    for field in source_intake.REVIEW_METADATA_FIELDS:
        source[field] = None
        source["metadata_verification"][field] = False
    for field in source_intake.REVIEW_COMPLETION_FIELDS:
        source["metadata_verification"][field] = False
    source["related_decision_ids"] = []
    source["underlying_dispute_id"] = None
    source["rejection_reasons"] = []
    source["notes"] = f"preserve:{acquisition_id}"
    _materialize(
        root,
        source,
        pdf_bytes=_minimal_pdf_bytes(acquisition_id),
        structured_payload=_structured(filename, f"Nội dung {acquisition_id}"),
    )
    return source


def _atomic_fixture(tmp_path: Path, monkeypatch, *, include_unrelated=True):
    left = _make_extracted_source(tmp_path, SOURCE_A, "left.pdf")
    right = _make_extracted_source(tmp_path, SOURCE_B, "right.pdf")
    sources = [left, right]
    if include_unrelated:
        other = _make_extracted_source(tmp_path, SOURCE_OTHER, "other.pdf")
        sources.append(other)
    manifest = _manifest(sources)
    _write_manifest(tmp_path, manifest)
    development = tmp_path / "backend/data/judgments_evaluation.json"
    development.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE
    pre_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setattr(application, "EXPECTED_SOURCE_IDS", (SOURCE_A, SOURCE_B))
    monkeypatch.setattr(application, "EXPECTED_PRE_MANIFEST_SHA256", pre_sha)
    payloads = {
        SOURCE_A: _payload("01/2026/DS-ST", case_id="01/2026/TLST-DS"),
        SOURCE_B: _payload(
            EXPECTED_CANONICAL_VALUE,
            case_id="26/2025/TLST-DS",
            court="Tòa án nhân dân Khu vực 2 - Nghệ An",
            date="2026-07-31",
        ),
    }
    plan = {
        "protocol_id": application.PROTOCOL_ID,
        "baseline_head": application.BASELINE_HEAD,
        "baseline_parent": application.BASELINE_PARENT,
        "blocked_run_id": application.EXPECTED_BLOCKED_RUN_ID,
        "blocked_protocol_fingerprint": (
            application.EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT
        ),
        "blocked_run_fingerprint": application.EXPECTED_BLOCKED_RUN_FINGERPRINT,
        "policy_fingerprint": application.FROZEN_POLICY_FINGERPRINT,
        "provisional_45_payload_fingerprint": (
            application.EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT
        ),
        "source039_normalization": {
            "acquisition_id": SOURCE_B,
            "field": EXPECTED_FIELD,
            "raw_value": EXPECTED_RAW_VALUE,
            "canonical_value": EXPECTED_CANONICAL_VALUE,
            "transformation": (
                "replace exactly one ASCII-space boundary with one slash"
            ),
        },
        "source_ids": [SOURCE_A, SOURCE_B],
        "payloads": payloads,
        "evidence_bindings": {
            source["acquisition_id"]: {
                "source_sha256": source["sha256"],
                "structured_sha256": source["structured_sha256"],
            }
            for source in (left, right)
        },
        "pre_manifest_sha256": pre_sha,
        "post_manifest_sha256": "0" * 64,
        "plan_fingerprint": "0" * 64,
    }
    checked = source_intake.load_acquisition_manifest(tmp_path)
    updated = application._apply_plan_in_memory(tmp_path, checked, plan)
    plan["post_manifest_sha256"] = hashlib.sha256(
        source_intake._canonical_json_bytes(updated)
    ).hexdigest()
    plan["plan_fingerprint"] = application._plan_fingerprint(plan)
    return manifest_path, plan


@pytest.fixture(scope="module")
def frozen_inputs():
    policy = load_policy(PROJECT_ROOT / application.POLICY_RELATIVE)
    ledger = json.loads((PROJECT_ROOT / application.LEDGER_RELATIVE).read_text())
    return policy, ledger


def test_production_plan_reconstructs_exact_46_without_mutation():
    path = PROJECT_ROOT / source_intake.ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    plan = application.build_batch2dc_application_plan(PROJECT_ROOT)
    assert len(plan["source_ids"]) == 46
    assert plan["source_ids"] == sorted(application.EXPECTED_SOURCE_IDS)
    assert plan["provisional_45_payload_fingerprint"] == (
        application.EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT
    )
    assert plan["payloads"][SOURCE_B]["verified_metadata"][EXPECTED_FIELD] == (
        EXPECTED_CANONICAL_VALUE
    )
    assert path.read_bytes() == before


def test_original_45_payload_fingerprint_is_reproducible(frozen_inputs):
    _, ledger = frozen_inputs
    payloads = {
        record["acquisition_id"]: application._payload_from_record(record)
        for record in ledger["records"]
        if record["status"] == "RESOLVED_NOT_APPLIED"
    }
    assert len(payloads) == 45
    assert application.canonical_fingerprint(payloads) == (
        application.EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT
    )


def test_source039_uses_only_frozen_normalizer(frozen_inputs):
    policy, _ = frozen_inputs
    assert normalize_allowed_identity_value(
        policy, SOURCE_B, EXPECTED_FIELD, EXPECTED_RAW_VALUE
    ) == EXPECTED_CANONICAL_VALUE


@pytest.mark.parametrize(
    "ids, message",
    [
        (list(application.EXPECTED_SOURCE_IDS[:-1]), "missing"),
        (list(application.EXPECTED_SOURCE_IDS) + ["extra"], "extra"),
        (
            list(application.EXPECTED_SOURCE_IDS[:-1])
            + [application.EXPECTED_SOURCE_IDS[0]],
            "Duplicate",
        ),
    ],
)
def test_exact_source_set_rejects_missing_extra_and_duplicate(ids, message):
    with pytest.raises(application.Batch2DCIdentityApplicationError, match=message):
        application._validate_exact_source_ids(ids)


@pytest.mark.parametrize(
    "acquisition_id, field, raw",
    [
        (SOURCE_B, EXPECTED_FIELD, "25  2026/QĐST-DS"),
        (SOURCE_A, EXPECTED_FIELD, EXPECTED_RAW_VALUE),
        (SOURCE_B, "case_id", EXPECTED_RAW_VALUE),
    ],
)
def test_invalid_or_outside_allowlist_normalization_is_rejected(
    frozen_inputs, acquisition_id, field, raw
):
    policy, _ = frozen_inputs
    with pytest.raises(IdentityNormalizationPolicyError):
        normalize_allowed_identity_value(policy, acquisition_id, field, raw)


def test_modified_provisional_payload_is_rejected(frozen_inputs):
    policy, ledger = frozen_inputs
    changed = copy.deepcopy(ledger)
    changed["records"][0]["court"] += " changed"
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="Provisional 45-payload fingerprint mismatch",
    ):
        application.reconstruct_frozen_payloads(changed, policy)


def test_wrong_expected_provisional_fingerprint_is_rejected(
    frozen_inputs, monkeypatch
):
    policy, ledger = frozen_inputs
    monkeypatch.setattr(application, "EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT", "0" * 64)
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="Provisional 45-payload fingerprint mismatch",
    ):
        application.reconstruct_frozen_payloads(ledger, policy)


def test_modified_policy_is_rejected(frozen_inputs):
    policy, ledger = frozen_inputs
    changed = copy.deepcopy(policy)
    changed["scope"]["raw_observed_value"] = "changed"
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="Frozen policy fingerprint mismatch",
    ):
        application.reconstruct_frozen_payloads(ledger, changed)


def test_invalid_source046_causes_zero_production_mutations():
    path = PROJECT_ROOT / source_intake.ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    plan = application.build_batch2dc_application_plan(PROJECT_ROOT)
    plan["payloads"]["batch2dc-20261105-046"]["verified_metadata"][
        "judgment_number"
    ] = "invalid"
    plan["plan_fingerprint"] = application._plan_fingerprint(plan)
    with pytest.raises(source_intake.SourceIntakeError):
        application.apply_identity_plan_atomically(PROJECT_ROOT, plan)
    assert path.read_bytes() == before


def test_invalid_identity_field_is_rejected_before_manifest_open(
    tmp_path, monkeypatch
):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    before = manifest_path.read_bytes()
    plan["payloads"][SOURCE_A]["verified_metadata"]["decision_date"] = "bad"
    plan["plan_fingerprint"] = application._plan_fingerprint(plan)
    with pytest.raises(source_intake.SourceIntakeError):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert manifest_path.read_bytes() == before


def test_manifest_sha_mismatch_rejected(tmp_path, monkeypatch):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][-1]["notes"] = "concurrent-before-call"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    changed = manifest_path.read_bytes()
    with pytest.raises(
        application.Batch2DCIdentityApplicationError, match="Manifest SHA mismatch"
    ):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert manifest_path.read_bytes() == changed


def test_stale_concurrent_modification_is_preserved(tmp_path, monkeypatch):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)

    def mutate(checkpoint, acquisition_id=None):
        if checkpoint == "before_manifest_compare":
            manifest = json.loads(manifest_path.read_text())
            manifest["sources"][-1]["notes"] = "concurrent-during-call"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )

    monkeypatch.setattr(application, "_atomic_application_checkpoint", mutate)
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="unexpected manifest state",
    ):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert json.loads(manifest_path.read_text())["sources"][-1]["notes"] == (
        "concurrent-during-call"
    )


def test_failure_during_in_memory_application_is_byte_stable(tmp_path, monkeypatch):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    before = manifest_path.read_bytes()
    calls = 0

    def fail(checkpoint, acquisition_id=None):
        nonlocal calls
        if checkpoint == "after_source_application":
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated application failure")

    monkeypatch.setattr(application, "_atomic_application_checkpoint", fail)
    with pytest.raises(RuntimeError, match="simulated application failure"):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert manifest_path.read_bytes() == before


def test_persistence_failure_preserves_original_bytes(tmp_path, monkeypatch):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    before = manifest_path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(source_intake, "_atomic_write_bytes_at", fail)
    with pytest.raises(OSError, match="simulated persistence failure"):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert manifest_path.read_bytes() == before


def test_post_replace_failure_rolls_back_and_retry_is_safe(tmp_path, monkeypatch):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    before = manifest_path.read_bytes()

    def fail(checkpoint, acquisition_id=None):
        if checkpoint == "after_manifest_replacement":
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(application, "_atomic_application_checkpoint", fail)
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="previous manifest restored",
    ):
        application.apply_identity_plan_atomically(tmp_path, plan)
    assert manifest_path.read_bytes() == before
    monkeypatch.setattr(
        application,
        "_atomic_application_checkpoint",
        lambda *args, **kwargs: None,
    )
    first = application.apply_identity_plan_atomically(tmp_path, plan)
    published = manifest_path.read_bytes()
    second = application.apply_identity_plan_atomically(tmp_path, plan)
    assert first["applied_count"] == 2
    assert first["idempotent_noop"] is False
    assert second["applied_count"] == 0
    assert second["idempotent_noop"] is True
    assert manifest_path.read_bytes() == published


def test_atomic_success_changes_only_targets_and_never_promotes(
    tmp_path, monkeypatch
):
    manifest_path, plan = _atomic_fixture(tmp_path, monkeypatch)
    before = source_intake.load_acquisition_manifest(tmp_path)
    other_before = next(
        source for source in before["sources"] if source["acquisition_id"] == SOURCE_OTHER
    )
    result = application.apply_identity_plan_atomically(tmp_path, plan)
    after = source_intake.load_acquisition_manifest(tmp_path)
    other_after = next(
        source for source in after["sources"] if source["acquisition_id"] == SOURCE_OTHER
    )
    targets = [
        source
        for source in after["sources"]
        if source["acquisition_id"] in (SOURCE_A, SOURCE_B)
    ]
    assert result["applied_count"] == 2
    assert other_after == other_before
    assert all(source["acquisition_status"] == "extracted" for source in targets)
    assert all(
        not source["metadata_verification"][field]
        for source in targets
        for field in source_intake.REVIEW_COMPLETION_FIELDS
    )
    assert manifest_path.exists()


def test_atomic_application_does_not_touch_benchmark_or_sampling(
    tmp_path, monkeypatch
):
    _, plan = _atomic_fixture(tmp_path, monkeypatch)
    benchmark = tmp_path / application.BENCHMARK_RELATIVE
    sampling = tmp_path / application.SAMPLING_RELATIVE
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_bytes(b'{"state":"DRAFT"}\n')
    sampling.write_bytes(
        b'{"state":"DRAFT","selection_status":"NOT_STARTED",'
        b'"sampling_seed":null}\n'
    )
    before = (benchmark.read_bytes(), sampling.read_bytes())
    application.apply_identity_plan_atomically(tmp_path, plan)
    assert (benchmark.read_bytes(), sampling.read_bytes()) == before


def test_application_module_has_no_network_or_promotion_paths():
    path = Path(application.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert imported_roots.isdisjoint(
        {"requests", "urllib", "httpx", "aiohttp", "socket"}
    )
    assert "record_source_review" not in called_names
    assert "promote_source_to_eligible" not in called_names


def test_protected_transition_requires_exact_draft_state():
    state = {
        "global_total": 363,
        "global_eligible": 257,
        "global_extracted": 105,
        "global_hash_verified": 1,
        "batch2dc_count": 46,
        "batch2dc_extracted": 46,
        "batch2dc_eligible": 0,
        "batch2dc_identity_ready": 0,
        "benchmark_state": "DRAFT",
        "sampling_state": "DRAFT",
        "selection_status": "NOT_STARTED",
        "sampling_seed": None,
        "selected_count": 0,
        "query_count": 0,
        "ground_truth_count": 0,
        "benchmark_sha256": "a" * 64,
        "sampling_sha256": "b" * 64,
    }
    after = copy.deepcopy(state)
    after["batch2dc_identity_ready"] = 46
    application._assert_protected_transition(state, after)
    changed = copy.deepcopy(after)
    changed["sampling_seed"] = 42
    with pytest.raises(application.Batch2DCIdentityApplicationError):
        application._assert_protected_transition(state, changed)


def test_regression_evidence_requires_at_least_one_real_pass(tmp_path):
    path = tmp_path / "tests.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "batch2dc_r2_test_evidence_v1",
                "results": [
                    {
                        "name": "focused",
                        "command": "pytest focused",
                        "outcome": "PASS",
                        "passed": 23,
                        "failed": 0,
                        "duration_seconds": 5.4,
                        "notes": "executed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert application._load_test_evidence(path)["results"][0]["passed"] == 23
    value = json.loads(path.read_text())
    value["results"][0]["outcome"] = "INTERRUPTED_COST_BOUND"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(
        application.Batch2DCIdentityApplicationError,
        match="no passing suite",
    ):
        application._load_test_evidence(path)
