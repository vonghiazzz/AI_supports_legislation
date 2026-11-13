import base64
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as policy_v2
from backend.scripts.Evaluation.benchmarking import dual_version_manifest_safety as v21
from backend.scripts.Evaluation.benchmarking import live_cutover_readiness as r11
from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking import trusted_review_protocol as trusted
from backend.scripts.Evaluation.benchmarking.source_intake import SourceIntakeError
from backend.scripts.Tests.test_independent_source_intake import (
    FIXED_TIME,
    _extracted_fixture,
    _manifest,
    _materialize,
    _minimal_pdf_bytes,
    _source,
    _write_manifest,
    _write_valid_extractor_output_fd,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / source_intake.ACQUISITION_MANIFEST_RELATIVE


def _public_key_base64(private_key):
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw).decode("ascii")


def _sign(payload, private_key):
    unsigned = {key: value for key, value in payload.items() if key != "signature_base64"}
    payload["signature_base64"] = base64.b64encode(
        private_key.sign(trusted.canonical_bytes(unsigned))
    ).decode("ascii")
    return payload


def _active_registry(*, producer_status="ACTIVE"):
    issuer_key = Ed25519PrivateKey.generate()
    producer_key = Ed25519PrivateKey.generate()
    issuer = {
        "issuer_id": "research-governance",
        "issuer_version": trusted.ISSUER_VERSION,
        "ed25519_public_key_base64": _public_key_base64(issuer_key),
        "status": "ACTIVE",
        "approved_at": "2026-11-12T00:00:00+00:00",
        "approval_provenance": "approved-r11-fixture",
        "revoked_at": None,
    }
    producer = {
        "producer_id": "dedup-review-producer",
        "producer_version": "1.0.0",
        "ed25519_public_key_base64": _public_key_base64(producer_key),
        "approved_code_sha256": "a" * 64,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "review_contract_version": trusted.REVIEW_CONTRACT_VERSION,
        "evidence_amendment_id": trusted.EVIDENCE_AMENDMENT_ID,
        "trusted_execution_identity": "isolated-r11-fixture",
        "issuer_id": issuer["issuer_id"],
        "approval_id": "approval-r11-fixture",
        "approved_at": "2026-11-12T00:00:00+00:00",
        "approval_provenance": "approved-r11-fixture",
        "status": producer_status,
        "revoked_at": (
            None if producer_status == "ACTIVE" else "2026-11-12T00:01:00+00:00"
        ),
    }
    registry = trusted.freeze_registry(
        activation_state="ACTIVE",
        issuers=[issuer],
        producers=[producer],
        approval_provenance="r11-cryptographic-test-fixture-only",
    )
    return registry, issuer_key, producer_key


def _activation(issuer_key, *, issuer_id="research-governance"):
    now = datetime(2026, 11, 12, 1, 0, tzinfo=timezone.utc)
    payload = {
        "authorization_version": trusted.ACTIVATION_AUTHORIZATION_VERSION,
        "authorization_id": "activation-r11-fixture",
        "issuer_id": issuer_id,
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "target_manifest_pre_sha256": "b" * 64,
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": "c" * 64,
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        "nonce": "activation-nonce-r11",
        "signature_base64": "PENDING",
    }
    return _sign(payload, issuer_key), now


def _attestation_context():
    return {
        "acquisition_id": "batch2dc-20261105-025",
        "source_id": "independent:batch2dc-20261105-025",
        "source_local_evidence_fingerprint": "1" * 64,
        "current_inventory_fingerprint": "2" * 64,
        "candidate_set_fingerprint": "3" * 64,
        "recomputed_findings_fingerprint": "4" * 64,
        "adjudication_evidence_bindings": ["5" * 64, "6" * 64],
        "review_decision_fingerprint": "7" * 64,
    }


def _attestation(producer_key, context=None):
    context = context or _attestation_context()
    payload = {
        "attestation_version": trusted.ATTESTATION_VERSION,
        "attestation_id": "attestation-r11-fixture",
        "producer_id": "dedup-review-producer",
        "producer_version": "1.0.0",
        "issuer_id": "research-governance",
        **copy.deepcopy(context),
        "producer_code_sha256": "a" * 64,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "review_contract_version": trusted.REVIEW_CONTRACT_VERSION,
        "evidence_amendment_id": trusted.EVIDENCE_AMENDMENT_ID,
        "trusted_execution_identity": "isolated-r11-fixture",
        "issuance_provenance": "r11-test-fixture-no-live-authority",
        "issued_at": "2026-11-12T01:00:00+00:00",
        "nonce": "attestation-nonce-r11",
        "signature_base64": "PENDING",
    }
    return _sign(payload, producer_key)


def _write_marker(root):
    path = root / source_intake.INACTIVE_V21_STAGING_MARKER_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source_intake._canonical_json_bytes(source_intake.INACTIVE_V21_STAGING_MARKER))


def _migrate_fixture(root):
    path = root / source_intake.ACQUISITION_MANIFEST_RELATIVE
    before = json.loads(path.read_text())
    migrated = v21.migrate_v1_to_v21_copy(before)
    path.write_bytes(source_intake._canonical_json_bytes(migrated))
    _write_marker(root)
    return migrated


def _install_policy_assets(root):
    for relative in (
        policy_v2.DEFAULT_POLICY_RELATIVE,
        policy_v2.EVIDENCE_FIXTURE_RELATIVE,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())
    (root / "backend/dataLaw/ban_an").mkdir(parents=True, exist_ok=True)
    (root / "backend/data/structured_judgments").mkdir(parents=True, exist_ok=True)


def test_baseline_and_r10_provenance_are_exact():
    assert policy_v2.sha256_file(MANIFEST_PATH) == r11.SOURCE_MANIFEST_SHA256
    run = json.loads((PROJECT_ROOT / r11.R10_RUN_RELATIVE / "run_manifest.json").read_text())
    assert run["run_fingerprint"] == r11.R10_RUN_FINGERPRINT


def test_live_registry_is_exactly_inactive_and_empty():
    checked = trusted.validate_registry(trusted.LIVE_TRUST_REGISTRY)
    assert checked["activation_state"] == "INACTIVE"
    assert checked["issuers"] == []
    assert checked["producers"] == []


def test_live_authorization_cannot_be_created_by_caller_json():
    with pytest.raises(trusted.TrustedReviewProtocolError, match="inactive"):
        trusted.verify_live_activation_authorization(
            PROJECT_ROOT,
            {"caller": "AUTHORIZED"},
        )


def test_signed_activation_fixture_verifies():
    registry, issuer_key, _ = _active_registry()
    authorization, now = _activation(issuer_key)
    result = trusted.verify_activation_authorization(
        authorization,
        registry,
        expected_manifest_sha256="b" * 64,
        expected_code_bundle_fingerprint="c" * 64,
        now=now,
    )
    assert result["authorized"] is True


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("target_manifest_pre_sha256", "0" * 64, "manifest"),
        ("cutover_code_bundle_fingerprint", "0" * 64, "code"),
        ("policy_fingerprint", "0" * 64, "policy"),
    ],
)
def test_activation_binding_mismatch_fails_closed(field, value, match):
    registry, issuer_key, _ = _active_registry()
    authorization, now = _activation(issuer_key)
    authorization[field] = value
    with pytest.raises(trusted.TrustedReviewProtocolError, match=match):
        trusted.verify_activation_authorization(
            authorization,
            registry,
            expected_manifest_sha256="b" * 64,
            expected_code_bundle_fingerprint="c" * 64,
            now=now,
        )


def test_unknown_activation_issuer_fails_closed():
    registry, issuer_key, _ = _active_registry()
    authorization, now = _activation(issuer_key, issuer_id="unknown-issuer")
    with pytest.raises(trusted.TrustedReviewProtocolError, match="unknown"):
        trusted.verify_activation_authorization(
            authorization,
            registry,
            expected_manifest_sha256="b" * 64,
            expected_code_bundle_fingerprint="c" * 64,
            now=now,
        )


def test_expired_activation_fails_closed():
    registry, issuer_key, _ = _active_registry()
    authorization, now = _activation(issuer_key)
    with pytest.raises(trusted.TrustedReviewProtocolError, match="currently valid"):
        trusted.verify_activation_authorization(
            authorization,
            registry,
            expected_manifest_sha256="b" * 64,
            expected_code_bundle_fingerprint="c" * 64,
            now=now + timedelta(days=1),
        )


def test_signed_attestation_fixture_verifies():
    registry, _, producer_key = _active_registry()
    context = _attestation_context()
    result = trusted.verify_review_attestation(
        _attestation(producer_key, context),
        registry,
        authoritative_context=context,
        consumed_attestation_ids=set(),
        consumed_nonces=set(),
    )
    assert result["verified"] is True
    assert result["authoritative_only_with_create_once_replay_ledger"] is True


def test_verified_attestation_create_once_ledger_rejects_replay(tmp_path):
    registry, _, producer_key = _active_registry()
    context = _attestation_context()
    verified = trusted.verify_review_attestation(
        _attestation(producer_key, context),
        registry,
        authoritative_context=context,
        consumed_attestation_ids=set(),
        consumed_nonces=set(),
    )
    ledger = tmp_path / "trusted-ledger"
    ledger.mkdir()
    first = trusted.record_attestation_once(ledger, verified)
    assert first["attestation_id"] == verified["attestation_id"]
    with pytest.raises(trusted.TrustedReviewProtocolError, match="replay"):
        trusted.record_attestation_once(ledger, verified)


def test_symlinked_replay_ledger_fails_closed(tmp_path):
    target = tmp_path / "target-ledger"
    target.mkdir()
    link = tmp_path / "ledger-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(trusted.TrustedReviewProtocolError, match="regular directory"):
        trusted.record_attestation_once(
            link,
            {
                "attestation_id": "fixture-id",
                "nonce": "fixture-nonce",
                "producer_id": "fixture-producer",
                "attestation_fingerprint": "a" * 64,
                "verified": True,
                "authoritative_only_with_create_once_replay_ledger": True,
            },
        )


def test_attestation_payload_tampering_breaks_signature():
    registry, _, producer_key = _active_registry()
    context = _attestation_context()
    attestation = _attestation(producer_key, context)
    attestation["issuance_provenance"] = "tampered-after-signing"
    with pytest.raises(trusted.TrustedReviewProtocolError, match="signature"):
        trusted.verify_review_attestation(
            attestation,
            registry,
            authoritative_context=context,
            consumed_attestation_ids=set(),
            consumed_nonces=set(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("producer_code_sha256", "0" * 64),
        ("policy_fingerprint", "0" * 64),
        ("trusted_execution_identity", "caller-controlled"),
    ],
)
def test_attestation_producer_binding_mismatch_fails_closed(field, value):
    registry, _, producer_key = _active_registry()
    context = _attestation_context()
    attestation = _attestation(producer_key, context)
    attestation[field] = value
    attestation = _sign(attestation, producer_key)
    with pytest.raises(trusted.TrustedReviewProtocolError, match="binding mismatch"):
        trusted.verify_review_attestation(
            attestation,
            registry,
            authoritative_context=context,
            consumed_attestation_ids=set(),
            consumed_nonces=set(),
        )


def test_stale_inventory_attestation_fails_closed():
    registry, _, producer_key = _active_registry()
    context = _attestation_context()
    stale = copy.deepcopy(context)
    stale["current_inventory_fingerprint"] = "0" * 64
    with pytest.raises(trusted.TrustedReviewProtocolError, match="stale"):
        trusted.verify_review_attestation(
            _attestation(producer_key, stale),
            registry,
            authoritative_context=context,
            consumed_attestation_ids=set(),
            consumed_nonces=set(),
        )


def test_revoked_producer_fails_closed():
    registry, _, producer_key = _active_registry(producer_status="REVOKED")
    with pytest.raises(trusted.TrustedReviewProtocolError, match="revoked"):
        trusted.verify_review_attestation(
            _attestation(producer_key),
            registry,
            authoritative_context=_attestation_context(),
            consumed_attestation_ids=set(),
            consumed_nonces=set(),
        )


@pytest.mark.parametrize(
    "seen_ids,seen_nonces",
    [({"attestation-r11-fixture"}, set()), (set(), {"attestation-nonce-r11"})],
)
def test_attestation_replay_fails_closed(seen_ids, seen_nonces):
    registry, _, producer_key = _active_registry()
    with pytest.raises(trusted.TrustedReviewProtocolError, match="replay"):
        trusted.verify_review_attestation(
            _attestation(producer_key),
            registry,
            authoritative_context=_attestation_context(),
            consumed_attestation_ids=seen_ids,
            consumed_nonces=seen_nonces,
        )


def test_unknown_producer_fails_closed():
    registry, _, producer_key = _active_registry()
    attestation = _attestation(producer_key)
    attestation["producer_id"] = "unknown-producer"
    attestation = _sign(attestation, producer_key)
    with pytest.raises(trusted.TrustedReviewProtocolError, match="unknown"):
        trusted.verify_review_attestation(
            attestation,
            registry,
            authoritative_context=_attestation_context(),
            consumed_attestation_ids=set(),
            consumed_nonces=set(),
        )


def test_historical_v1_promotion_operates_on_staged_v21(tmp_path):
    source, _ = _extracted_fixture(tmp_path)
    _install_policy_assets(tmp_path)
    migrated = _migrate_fixture(tmp_path)
    dynamic_before = r11.load_dynamic_inventory(
        tmp_path, allow_inactive_v21=True
    )
    manifest = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    findings = source_intake._eligibility_review_findings(
        tmp_path, manifest, source["acquisition_id"]
    )
    reviewed = source_intake.record_source_review(
        tmp_path,
        source["acquisition_id"],
        {
            "review_version": source_intake.REVIEW_PAYLOAD_VERSION,
            "reviews": {
                field: {"completed": True, "findings": findings[field]}
                for field in source_intake.REVIEW_COMPLETION_FIELDS
            },
        },
        allow_inactive_v21=True,
    )
    assert reviewed["acquisition_status"] == "extracted"
    result = source_intake.promote_source_to_eligible(
        tmp_path, source["acquisition_id"], allow_inactive_v21=True
    )
    assert result["acquisition_status"] == "eligible"
    loaded = source_intake.load_acquisition_manifest(tmp_path, allow_inactive_v21=True)
    assert loaded["review_records"] == migrated["review_records"] == []
    dynamic_after = r11.load_dynamic_inventory(
        tmp_path, allow_inactive_v21=True
    )
    assert (
        dynamic_before["complete_inventory_fingerprint"]
        != dynamic_after["complete_inventory_fingerprint"]
    )


def test_v21_review_record_never_authorizes_promotion(tmp_path):
    source, manifest = _extracted_fixture(tmp_path)
    migrated = v21.migrate_v1_to_v21_copy(manifest)
    migrated["review_records"] = [
        v21.build_v21_evidence_record(
            migrated,
            source["source_id"],
            complete_inventory_fingerprint="1" * 64,
            candidate_set_fingerprint="2" * 64,
            evidence_fingerprint="3" * 64,
        )
    ]
    (tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE).write_bytes(
        source_intake._canonical_json_bytes(migrated)
    )
    _write_marker(tmp_path)
    with pytest.raises(SourceIntakeError, match="V2.1 review-record promotion"):
        source_intake.promote_source_to_eligible(
            tmp_path, source["acquisition_id"], allow_inactive_v21=True
        )


def test_live_staging_flag_cannot_bypass_activation_guard():
    with pytest.raises(SourceIntakeError, match="live repository"):
        source_intake.load_acquisition_manifest(
            PROJECT_ROOT, allow_inactive_v21=True
        )


def test_r12_rejects_r11_activation_as_operational_credential(tmp_path, monkeypatch):
    _extracted_fixture(tmp_path)
    _migrate_fixture(tmp_path)
    code_relatives = (
        "backend/scripts/Evaluation/benchmarking/source_intake.py",
        "backend/scripts/Evaluation/benchmarking/dual_version_manifest_safety.py",
        "backend/scripts/Evaluation/benchmarking/trusted_review_protocol.py",
        "backend/scripts/Evaluation/benchmarking/live_cutover_readiness.py",
        "backend/scripts/Evaluation/benchmarking/cutover_authorization_lifecycle.py",
        "backend/scripts/Evaluation/benchmarking/dedup_performance_v1.py",
    )
    for relative in code_relatives:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())
    simulated_source_file = tmp_path / code_relatives[0]
    monkeypatch.setattr(source_intake, "__file__", str(simulated_source_file))
    registry, issuer_key, _ = _active_registry()
    monkeypatch.setattr(trusted, "LIVE_TRUST_REGISTRY", registry)
    now = datetime.now(timezone.utc)
    authorization = {
        "authorization_version": trusted.ACTIVATION_AUTHORIZATION_VERSION,
        "authorization_id": "isolated-live-simulation",
        "issuer_id": "research-governance",
        "issuer_version": trusted.ISSUER_VERSION,
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "target_manifest_pre_sha256": trusted.LIVE_LINEAGE_ROOT_SHA256,
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": trusted.cutover_code_bundle_fingerprint(
            tmp_path
        ),
        "issued_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(minutes=5)).isoformat(),
        "nonce": "isolated-live-simulation-nonce",
        "signature_base64": "PENDING",
    }
    _sign(authorization, issuer_key)
    with pytest.raises(SourceIntakeError, match="One-time cutover authorization"):
        source_intake.load_acquisition_manifest(
            tmp_path,
            v21_activation_authorization=authorization,
        )
    with pytest.raises(SourceIntakeError, match="live repository"):
        source_intake.load_acquisition_manifest(
            tmp_path,
            allow_inactive_v21=True,
            v21_activation_authorization=authorization,
        )


def test_clone_migration_rolls_back_and_retries(tmp_path):
    _extracted_fixture(tmp_path)
    pre = policy_v2.sha256_file(tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE)

    def fail(name):
        if name == "after_atomic_publication":
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        r11.migrate_clone_manifest(tmp_path, pre, checkpoint=fail)
    assert policy_v2.sha256_file(tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE) == pre
    first = r11.migrate_clone_manifest(tmp_path, pre)
    second = r11.migrate_clone_manifest(tmp_path, pre)
    assert first["result"] == "MIGRATED"
    assert second["result"] == "IDEMPOTENT_ALREADY_MIGRATED"


def test_clone_hard_stop_forward_recovers(tmp_path):
    _extracted_fixture(tmp_path)
    pre = policy_v2.sha256_file(tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE)

    def stop(name):
        if name == "after_atomic_publication":
            raise r11.SimulatedHardStop(name)

    with pytest.raises(r11.SimulatedHardStop):
        r11.migrate_clone_manifest(tmp_path, pre, checkpoint=stop)
    loaded = source_intake.load_acquisition_manifest(tmp_path, allow_inactive_v21=True)
    assert loaded["manifest_version"] == v21.V21_MANIFEST_VERSION
    assert r11.migrate_clone_manifest(tmp_path, pre)["result"] == "IDEMPOTENT_ALREADY_MIGRATED"


def test_dynamic_inventory_has_no_historical_count_invariant(tmp_path):
    source, _ = _extracted_fixture(tmp_path, text="nguồn động thứ nhất độc lập")
    _install_policy_assets(tmp_path)
    _migrate_fixture(tmp_path)
    result = r11.recompute_dynamic_candidates(
        tmp_path, {source["source_id"]}, allow_inactive_v21=True
    )
    assert len(result["records"]) == 1
    assert result["historical_count_invariants_used"] is False


def test_dynamic_inventory_changes_after_registration_without_fixed_count(tmp_path):
    source, _ = _extracted_fixture(tmp_path, text="nguồn động thứ nhất độc lập")
    _install_policy_assets(tmp_path)
    _migrate_fixture(tmp_path)
    before = r11.load_dynamic_inventory(tmp_path, allow_inactive_v21=True)
    input_pdf = tmp_path / "registered.pdf"
    input_pdf.write_bytes(_minimal_pdf_bytes("r11-registration"))
    source_intake.register_source(
        tmp_path,
        "r11-registered",
        input_pdf,
        "https://example.test/registered.pdf",
        "example.test",
        FIXED_TIME,
        "registered.pdf",
        allow_inactive_v21=True,
    )
    after = r11.load_dynamic_inventory(tmp_path, allow_inactive_v21=True)
    assert len(before["records"]) == len(after["records"]) == 1
    assert before["complete_inventory_fingerprint"] != after["complete_inventory_fingerprint"]
    assert source["source_id"] in {item["source_id"] for item in after["records"]}


def test_dynamic_inventory_tracks_newly_extracted_incomplete_identity(
    tmp_path, monkeypatch
):
    source = _source("dynamic-new", "dynamic-new.pdf", status="hash_verified")
    for field in source_intake.REVIEW_METADATA_FIELDS:
        source[field] = None
        source["metadata_verification"][field] = False
    pdf = _minimal_pdf_bytes("dynamic-new")
    path = tmp_path / source["stored_pdf_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf)
    source["sha256"] = policy_v2.sha256_file(path)
    _write_manifest(tmp_path, _manifest([source]))
    _install_policy_assets(tmp_path)
    _migrate_fixture(tmp_path)

    def extractor(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", extractor)
    source_intake.extract_intake_source(
        tmp_path, source["acquisition_id"], allow_inactive_v21=True
    )
    dynamic = r11.load_dynamic_inventory(tmp_path, allow_inactive_v21=True)
    assert dynamic["independent_candidate_count"] == 0
    assert dynamic["incomplete_candidate_sources"][0]["source_id"] == source["source_id"]
    with pytest.raises(r11.LiveCutoverReadinessError, match="identity"):
        r11.recompute_dynamic_candidates(
            tmp_path, {source["source_id"]}, allow_inactive_v21=True
        )


def test_relation_specific_evidence_amendment_is_frozen_not_activated():
    spec = trusted.evidence_amendment_specification()
    assert spec["exact_duplicate"]["document_type_required"] is False
    assert spec["exact_duplicate"]["weak_metadata_can_cancel_strong_positive"] is False
    assert spec["weak_distinct_control"]["generic_override_allowed"] is False
    assert spec["related_decision_or_dispute"]["generic_override_allowed"] is False
    assert spec["targeted_adjudications_preserved"] == ["027", "029"]
    assert spec["unresolved_sources_preserved"] == ["008", "014", "022", "024", "034"]
    assert spec["document_type_coverage_claim"] == "LIMITED_65_SOURCE_AUDIT_NOT_ALL_469"
    assert spec["frozen_policy_v2_modified"] is False


@pytest.fixture(scope="module")
def full_clone_rehearsal():
    return r11.perform_full_clone_rehearsal(PROJECT_ROOT)


def test_real_population_full_clone_rehearsal(full_clone_rehearsal):
    result = full_clone_rehearsal
    assert result["clone_type"] == "REAL_REPOSITORY_BACKED_FULL_INTAKE_POPULATION"
    assert result["original_source_count"] == 363
    assert result["failure_injection_rolled_back"] is True
    assert result["migration_result"] == "MIGRATED"
    assert result["idempotent_retry_result"] == "IDEMPOTENT_ALREADY_MIGRATED"
    assert result["historical_v1_promotion_status"] == (
        "EXERCISED_ON_ISOLATED_V21_EXACT_GATE_FIXTURE"
    )
    assert result["full_clone_review_and_promotion_exercised"] is False
    assert result["v2_review_record_promotion_exercised"] is False
    assert result["existing_sources_unchanged"] is True
    assert result["original_provenance_unchanged"] is True
    assert result["live_mutation_executed"] is False


def test_r11_outputs_are_not_activated(full_clone_rehearsal):
    outputs = r11.build_outputs(
        PROJECT_ROOT,
        "a" * 40,
        r11.PROTOCOL_ID + "_test",
        full_clone_rehearsal,
    )
    assert set(outputs) == {
        "cutover_dependency_audit.json",
        "live_capability_safety_matrix.json",
        "v1_operational_continuity.json",
        "dynamic_inventory_validation.json",
        "trusted_producer_registry_spec.json",
        "attestation_protocol_spec.json",
        "evidence_amendment_validation.json",
        "full_clone_migration_rehearsal.json",
        "activation_go_no_go.json",
        "validation_summary.json",
    }
    decision = outputs["activation_go_no_go.json"]
    assert decision["migration_engineering_ready"] is True
    assert decision["trusted_producer_protocol_ready"] is True
    assert decision["MIGRATION_ENGINEERING_READY"] is True
    assert decision["TRUSTED_PRODUCER_READY"] is True
    assert decision["LIVE_MIGRATION_AUTHORIZED"] is False
    assert decision["trusted_producer_live_registry_active"] is False
    assert decision["live_migration_authorized"] is False
    assert decision["live_review_authorized"] is False
    assert decision["live_promotion_authorized"] is False
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == policy_v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )


def test_protected_state_and_live_manifest_unchanged(full_clone_rehearsal):
    del full_clone_rehearsal
    assert policy_v2.sha256_file(MANIFEST_PATH) == r11.SOURCE_MANIFEST_SHA256
    state = r11._protected_state(PROJECT_ROOT)
    assert state["global"] == {
        "total": 363,
        "eligible": 257,
        "extracted": 105,
        "hash_verified": 1,
    }
    assert state["batch2dc"] == {
        "total": 46,
        "extracted": 46,
        "identity_ready": 46,
        "eligible": 0,
    }
    assert state["benchmark_state"] == "DRAFT"
    assert state["sampling_state"] == "DRAFT"
    assert state["selection_status"] == "NOT_STARTED"
    assert state["sampling_seed"] is None
    assert state["selected_cases"] == state["query_records"] == state["ground_truth_records"] == 0
