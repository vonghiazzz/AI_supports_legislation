import base64
import copy
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from backend.scripts.Evaluation.benchmarking import cutover_authorization_lifecycle as lifecycle
from backend.scripts.Evaluation.benchmarking import cutover_operational_closure as r12
from backend.scripts.Evaluation.benchmarking import dedup_performance_v1
from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking import trusted_review_protocol as trusted
from backend.scripts.Tests.test_independent_source_intake import _extracted_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _public(private_key):
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _sign(payload, private_key):
    unsigned = {
        key: value for key, value in payload.items() if key != "signature_base64"
    }
    payload["signature_base64"] = base64.b64encode(
        private_key.sign(trusted.canonical_bytes(unsigned))
    ).decode("ascii")
    return payload


def _registry(*, status="ACTIVE"):
    private_key = Ed25519PrivateKey.generate()
    issuer = {
        "issuer_id": "r12-test-issuer",
        "issuer_version": trusted.ISSUER_VERSION,
        "ed25519_public_key_base64": _public(private_key),
        "status": status,
        "approved_at": "2026-11-13T00:00:00+00:00",
        "approval_provenance": "R12_TEMPORARY_TEST_FIXTURE_ONLY",
        "revoked_at": (
            None if status == "ACTIVE" else "2026-11-13T00:30:00+00:00"
        ),
    }
    return (
        trusted.freeze_registry(
            activation_state="ACTIVE",
            issuers=[issuer],
            producers=[],
            approval_provenance="R12_TEMPORARY_TEST_FIXTURE_ONLY",
        ),
        private_key,
    )


def _receipt():
    cutover = {
        "authorization_id": "r12-cutover-once",
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": "c" * 64,
    }
    return lifecycle.build_activation_receipt(
        cutover,
        pre_manifest_sha256="1" * 64,
        post_manifest_sha256="2" * 64,
        published_at="2026-11-13T01:00:00+00:00",
    )


def _operational(private_key, receipt, *, now=None, issuer="r12-test-issuer"):
    now = now or datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)
    payload = {
        "authorization_version": lifecycle.OPERATIONAL_AUTHORIZATION_VERSION,
        "authorization_id": "r12-operational-fixture",
        "issuer_id": issuer,
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "activation_receipt_fingerprint": receipt["receipt_fingerprint"],
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "operational_code_bundle_fingerprint": lifecycle.operational_code_bundle_fingerprint(
            PROJECT_ROOT
        ),
        "issued_at": (now - timedelta(minutes=2)).isoformat(),
        "not_before": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        "nonce": "r12-operational-nonce",
        "signature_base64": "PENDING",
    }
    return _sign(payload, private_key), now


def _recovery(private_key, receipt, *, now=None):
    now = now or datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)
    payload = {
        "authorization_version": lifecycle.RECOVERY_AUTHORIZATION_VERSION,
        "authorization_id": "r12-recovery-fixture",
        "issuer_id": "r12-test-issuer",
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "activation_receipt_fingerprint": receipt["receipt_fingerprint"],
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "purpose": "READ_ONLY_EVIDENCE_RECOVERY",
        "issued_at": (now - timedelta(minutes=2)).isoformat(),
        "not_before": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        "nonce": "r12-recovery-nonce",
        "signature_base64": "PENDING",
    }
    return _sign(payload, private_key), now


def _cutover(private_key, pre_sha256, *, now=None):
    now = now or datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)
    payload = {
        "authorization_version": trusted.ACTIVATION_AUTHORIZATION_VERSION,
        "authorization_id": "r12-authorized-clone-cutover",
        "issuer_id": "r12-test-issuer",
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "target_manifest_pre_sha256": pre_sha256,
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": "c" * 64,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        "nonce": "r12-authorized-clone-cutover-nonce",
        "signature_base64": "PENDING",
    }
    return _sign(payload, private_key), now


def test_baseline_and_r11_provenance_are_exact():
    baseline = r12._baseline(PROJECT_ROOT)
    assert baseline["manifest_sha256"] == r12.SOURCE_MANIFEST_SHA256
    assert baseline["policy_fingerprint"] == r12.POLICY_FINGERPRINT
    assert baseline["r11_run_fingerprint"] == r12.R11_RUN_FINGERPRINT


def test_r11_cutover_expiry_lockout_is_reproduced():
    registry, key = _registry()
    now = datetime(2026, 11, 13, 1, 0, tzinfo=timezone.utc)
    payload = {
        "authorization_version": trusted.ACTIVATION_AUTHORIZATION_VERSION,
        "authorization_id": "r11-expiry-reproduction",
        "issuer_id": "r12-test-issuer",
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "target_manifest_pre_sha256": "1" * 64,
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": "2" * 64,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=1)).isoformat(),
        "nonce": "r11-expiry-reproduction-nonce",
        "signature_base64": "PENDING",
    }
    _sign(payload, key)
    assert trusted.verify_activation_authorization(
        payload,
        registry,
        expected_manifest_sha256="1" * 64,
        expected_code_bundle_fingerprint="2" * 64,
        now=now,
    )["authorized"]
    with pytest.raises(trusted.TrustedReviewProtocolError, match="not currently valid"):
        trusted.verify_activation_authorization(
            payload,
            registry,
            expected_manifest_sha256="1" * 64,
            expected_code_bundle_fingerprint="2" * 64,
            now=now + timedelta(minutes=2),
        )


def test_operational_authority_is_independent_and_restart_stable():
    registry, key = _registry()
    receipt = _receipt()
    authorization, now = _operational(key, receipt)
    first = lifecycle.verify_operational_authorization(
        authorization, receipt, registry, project_root=PROJECT_ROOT, now=now
    )
    second = lifecycle.verify_operational_authorization(
        copy.deepcopy(authorization),
        copy.deepcopy(receipt),
        copy.deepcopy(registry),
        project_root=PROJECT_ROOT,
        now=now,
    )
    assert first == second
    assert first["authorized"]


def test_operational_expiry_and_revocation_fail_closed():
    registry, key = _registry()
    receipt = _receipt()
    authorization, now = _operational(key, receipt)
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="expired"):
        lifecycle.verify_operational_authorization(
            authorization,
            receipt,
            registry,
            project_root=PROJECT_ROOT,
            now=now + timedelta(minutes=6),
        )
    revoked, _ = _registry(status="REVOKED")
    revoked["issuers"][0]["ed25519_public_key_base64"] = registry["issuers"][0][
        "ed25519_public_key_base64"
    ]
    revoked["registry_fingerprint"] = trusted.canonical_fingerprint(
        revoked, excluded_key="registry_fingerprint"
    )
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="revoked"):
        lifecycle.verify_operational_authorization(
            authorization,
            receipt,
            revoked,
            project_root=PROJECT_ROOT,
            now=now,
        )


def test_unknown_issuer_forgery_and_code_drift_fail_closed(monkeypatch):
    registry, key = _registry()
    receipt = _receipt()
    unknown, now = _operational(key, receipt, issuer="unknown")
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="unknown"):
        lifecycle.verify_operational_authorization(
            unknown, receipt, registry, project_root=PROJECT_ROOT, now=now
        )
    valid, now = _operational(key, receipt)
    monkeypatch.setattr(
        lifecycle, "operational_code_bundle_fingerprint", lambda _root: "0" * 64
    )
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="binding mismatch"):
        lifecycle.verify_operational_authorization(
            valid, receipt, registry, project_root=PROJECT_ROOT, now=now
        )


def test_inactive_or_unavailable_registry_fails_closed():
    registry, key = _registry()
    receipt = _receipt()
    authorization, now = _operational(key, receipt)
    registry["activation_state"] = "INACTIVE"
    registry["registry_fingerprint"] = trusted.canonical_fingerprint(
        registry, excluded_key="registry_fingerprint"
    )
    with pytest.raises(
        lifecycle.AuthorizationLifecycleError,
        match="unavailable or invalid|inactive",
    ):
        lifecycle.verify_operational_authorization(
            authorization,
            receipt,
            registry,
            project_root=PROJECT_ROOT,
            now=now,
        )


def test_recovery_authority_is_read_only_and_distinct():
    registry, key = _registry()
    receipt = _receipt()
    authorization, now = _recovery(key, receipt)
    result = lifecycle.verify_recovery_authorization(
        authorization, receipt, registry, now=now
    )
    assert result["read_only"] is True
    assert "operational_authorization_fingerprint" not in result
    operational, _ = _operational(key, receipt)
    with pytest.raises(lifecycle.AuthorizationLifecycleError):
        lifecycle.verify_recovery_authorization(
            operational, receipt, registry, now=now
        )


def test_activation_receipt_is_create_once_and_replay_safe(tmp_path):
    receipt = _receipt()
    assert lifecycle.record_activation_receipt_once(tmp_path, receipt) == receipt
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="replay"):
        lifecycle.record_activation_receipt_once(tmp_path, receipt)


def test_signed_clone_cutover_persists_receipt_and_rejects_reuse(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    clone.mkdir()
    _extracted_fixture(clone)
    manifest_path = clone / source_intake.ACQUISITION_MANIFEST_RELATIVE
    pre_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    registry, key = _registry()
    authorization, now = _cutover(key, pre_sha256)
    monkeypatch.setattr(
        trusted, "cutover_code_bundle_fingerprint", lambda _root: "c" * 64
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    result = lifecycle.execute_authorized_clone_cutover(
        clone, authorization, registry, ledger, now=now
    )
    assert result["migration"]["result"] == "MIGRATED"
    assert result["receipt_persisted"] is True
    assert len(list(ledger.glob("*.json"))) == 1
    with pytest.raises(lifecycle.AuthorizationLifecycleError):
        lifecycle.execute_authorized_clone_cutover(
            clone, authorization, registry, ledger, now=now
        )


def test_receipt_failure_rolls_clone_back_to_exact_v1(tmp_path, monkeypatch):
    clone = tmp_path / "clone"
    clone.mkdir()
    _extracted_fixture(clone)
    manifest_path = clone / source_intake.ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    pre_sha256 = hashlib.sha256(before).hexdigest()
    registry, key = _registry()
    authorization, now = _cutover(key, pre_sha256)
    monkeypatch.setattr(
        trusted, "cutover_code_bundle_fingerprint", lambda _root: "c" * 64
    )
    monkeypatch.setattr(
        lifecycle,
        "record_activation_receipt_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            lifecycle.AuthorizationLifecycleError("injected receipt failure")
        ),
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    with pytest.raises(lifecycle.AuthorizationLifecycleError, match="receipt failure"):
        lifecycle.execute_authorized_clone_cutover(
            clone, authorization, registry, ledger, now=now
        )
    assert manifest_path.read_bytes() == before
    assert not (
        clone / source_intake.INACTIVE_V21_STAGING_MARKER_RELATIVE
    ).exists()


def test_authority_spec_has_no_implicit_review_or_promotion():
    spec = lifecycle.authority_separation_specification()
    operational = spec["authorities"]["OPERATIONAL_AUTHORIZATION"]
    recovery = spec["authorities"]["RECOVERY_AUTHORIZATION"]
    assert operational["implies_review_authority"] is False
    assert operational["implies_promotion_authority"] is False
    assert recovery["normal_writes_allowed"] is False


def test_live_v21_gate_uses_operational_bundle_not_cutover_token(tmp_path, monkeypatch):
    simulated_source = tmp_path / "backend/scripts/Evaluation/benchmarking/source_intake.py"
    simulated_source.parent.mkdir(parents=True)
    simulated_source.write_text("fixture")
    monkeypatch.setattr(source_intake, "__file__", str(simulated_source))
    observed = []

    def verify(root, access):
        observed.append((root, access))
        return {"authorized": True}

    monkeypatch.setattr(lifecycle, "verify_live_operational_access", verify)
    bundle = {"test": "operational"}
    source_intake._authorize_v21_access(
        tmp_path,
        None,
        allow_inactive_v21=False,
        v21_activation_authorization=None,
        v21_operational_access=bundle,
    )
    assert observed == [(tmp_path.resolve(), bundle)]
    with pytest.raises(source_intake.SourceIntakeError, match="One-time cutover"):
        source_intake._authorize_v21_access(
            tmp_path,
            None,
            allow_inactive_v21=False,
            v21_activation_authorization={"test": "cutover"},
            v21_operational_access=None,
        )


def test_optimized_dedup_is_exactly_equivalent_on_control_fixture():
    inventory = [
        {
            "source_id": f"independent:s{index}",
            "source_name": f"source-{index}.pdf",
            "source_sha256": str(index) * 64,
            "structured_sha256": str(index + 1) * 64,
            "case_id": f"{index}/2026/DS-ST",
            "judgment_number": f"{index}/2026/DS-ST",
            "related_decision_ids": [],
            "underlying_dispute_id": None,
            "_dedup_text": text,
        }
        for index, text in enumerate(
            ("alpha beta gamma delta epsilon", "alpha beta gamma delta epsilon", "unique zeta eta theta")
        )
    ]
    reference_stats = {}
    optimized_stats = {}
    reference = dedup_performance_v1.deduplicate_reference_profiled(
        inventory,
        [],
        performance_stats=reference_stats,
    )
    optimized = dedup_performance_v1.deduplicate_memoized(
        inventory,
        [],
        performance_stats=optimized_stats,
    )
    assert reference == optimized
    assert optimized_stats["shingle_computations"] < reference_stats["shingle_computations"]


@pytest.fixture(scope="module")
def full_population_profile():
    return r12.profile_full_population_v1(PROJECT_ROOT)


def test_full_population_v1_profile_is_exact(full_population_profile):
    profile = full_population_profile
    assert profile["population_record_count"] == 469
    assert profile["exact_eligible_equivalence"] is True
    assert profile["exact_report_equivalence"] is True
    assert profile["scientific_semantics_changed"] is False


@pytest.fixture(scope="module")
def full_clone_result():
    return r12.perform_full_clone_v1_review_promotion(PROJECT_ROOT)


def test_real_full_clone_v1_review_and_promotion_complete(full_clone_result):
    assert full_clone_result["original_source_count"] == 363
    assert full_clone_result["review_failure_rollback_verified"] is True
    assert full_clone_result["promotion_status"] == "eligible"
    assert full_clone_result["historical_sources_unchanged"] is True
    assert full_clone_result["dedup_mocked"] is False
    assert full_clone_result["live_mutation_executed"] is False


def test_output_decision_remains_not_activated(full_population_profile, full_clone_result):
    outputs = r12.build_outputs(
        PROJECT_ROOT,
        "f" * 40,
        r12.PROTOCOL_ID + "_fixture",
        full_population_profile,
        full_clone_result,
    )
    decision = outputs["migration_go_no_go.json"]
    assert decision["MIGRATION_ENGINEERING_GO"] is True
    assert decision["LIVE_MIGRATION_AUTHORIZED"] is False
    assert outputs["validation_summary.json"]["live_manifest_mutated"] is False
