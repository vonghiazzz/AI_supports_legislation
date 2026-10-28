import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    canonical_json_fingerprint,
    inventory_judgment_sources,
)
from backend.scripts.Evaluation.benchmarking.source_intake import (
    ACQUISITION_MANIFEST_RELATIVE,
    IDENTITY_PAYLOAD_VERSION,
    INTAKE_ROOT_RELATIVE,
    MANIFEST_VERSION,
    PROVENANCE_ROOT_RELATIVE,
    REVIEW_COMPLETION_FIELDS,
    REVIEW_METADATA_FIELDS,
    REVIEW_PAYLOAD_VERSION,
    SourceIntakeError,
    extract_intake_source,
    inventory_independent_sources,
    load_acquisition_manifest,
    promote_source_to_eligible,
    record_source_identity,
    record_source_review,
    register_source,
)
from backend.scripts.Tests.test_independent_source_intake import (
    BATCH_ONE_BLOCKED_IDS,
    BATCH_ONE_ELIGIBLE_IDS,
    BATCH_ONE_SOURCE_IDS,
    FIXED_TIME,
    PROJECT_ROOT,
    _eligible_fixture,
    _extracted_fixture,
    _hash_verified_fixture,
    _manifest,
    _materialize,
    _minimal_pdf_bytes,
    _source,
    _structured,
    _write_manifest,
    _write_valid_extractor_output_fd,
)


BATCH_ONE_SOURCE_RECORDS_FINGERPRINT = (
    "a48eb95ce9117637bcd18cf0e270e9704055dd0ef64f36173b53177301349fe8"
)
DEVELOPMENT_FINGERPRINT = (
    "fd6a6f05e4b71a2c499584fc0296cf25cedf5eac958aea6d079bf449c864f754"
)
BENCHMARK_DIR = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _payload(**metadata_overrides) -> dict:
    metadata = {
        "judgment_number": "09/2026/DS-PT",
        "case_id": "09/2026/DS-PT",
        "court": "Tòa án nhân dân thành phố Đà Nẵng",
        "decision_date": "2026-01-09",
        "procedure_stage": "phuc_tham",
    }
    metadata.update(metadata_overrides)
    return {
        "review_version": REVIEW_PAYLOAD_VERSION,
        "verified_metadata": metadata,
        "verified_fields": {field: True for field in REVIEW_METADATA_FIELDS},
        "related_decision_ids": [
            "59/2025/DS-ST",
            "141/2025/TLPT-DS",
            "392/2025/QĐPT-DS",
            "483/2025/QĐ-PT",
        ],
        "underlying_dispute_id": None,
        "reviews": {
            field: {"completed": True, "findings": []}
            for field in REVIEW_COMPLETION_FIELDS
        },
    }


def _identity_payload(**metadata_overrides) -> dict:
    payload = _payload(**metadata_overrides)
    return {
        "identity_version": IDENTITY_PAYLOAD_VERSION,
        "verified_metadata": payload["verified_metadata"],
        "verified_fields": payload["verified_fields"],
        "related_decision_ids": payload["related_decision_ids"],
        "underlying_dispute_id": payload["underlying_dispute_id"],
    }


def _review_only_payload() -> dict:
    return {
        "review_version": REVIEW_PAYLOAD_VERSION,
        "reviews": {
            field: {"completed": True, "findings": []}
            for field in REVIEW_COMPLETION_FIELDS
        },
    }


def _review_fixture(
    root: Path,
    *,
    acquisition_id: str = "src-review-001",
    structured_payload: dict | None = None,
) -> tuple[dict, dict]:
    source, manifest = _extracted_fixture(
        root,
        acquisition_id,
        "review.pdf",
        structured_payload=structured_payload,
    )
    for field in REVIEW_METADATA_FIELDS:
        source[field] = None
        source["metadata_verification"][field] = False
    for field in REVIEW_COMPLETION_FIELDS:
        source["metadata_verification"][field] = False
    source["related_decision_ids"] = []
    source["underlying_dispute_id"] = None
    source["rejection_reasons"] = []
    _write_manifest(root, manifest)
    return source, manifest


def _manifest_source(root: Path, acquisition_id: str = "src-review-001") -> dict:
    return next(
        source
        for source in load_acquisition_manifest(root)["sources"]
        if source["acquisition_id"] == acquisition_id
    )


def _payload_with_authoritative_findings(
    root: Path,
    manifest: dict,
    acquisition_id: str,
    payload: dict,
) -> dict:
    candidate = copy.deepcopy(manifest)
    source = next(
        item
        for item in candidate["sources"]
        if item["acquisition_id"] == acquisition_id
    )
    if "verified_metadata" in payload:
        for field, value in payload["verified_metadata"].items():
            source[field] = value
            source["metadata_verification"][field] = True
        source["related_decision_ids"] = list(payload["related_decision_ids"])
        source["underlying_dispute_id"] = payload["underlying_dispute_id"]
    for field in REVIEW_COMPLETION_FIELDS:
        source["metadata_verification"][field] = payload["reviews"][field][
            "completed"
        ]
    findings = source_intake._eligibility_review_findings(
        root, candidate, acquisition_id
    )
    for field in REVIEW_COMPLETION_FIELDS:
        payload["reviews"][field]["findings"] = findings[field]
    return payload


def test_valid_review_persists_only_existing_contract_and_does_not_promote(tmp_path):
    source, _ = _review_fixture(tmp_path)
    history = copy.deepcopy(source["status_history"])
    snapshot_paths = tuple(
        BENCHMARK_DIR / name
        for name in (
            "benchmark_manifest.json",
            "sampling_manifest.json",
            "source_dedup_report.json",
        )
    )
    snapshot_bytes = {path: path.read_bytes() for path in snapshot_paths}

    result = record_source_review(tmp_path, source["acquisition_id"], _payload())

    reviewed = _manifest_source(tmp_path)
    assert result["acquisition_status"] == "extracted"
    assert result["gate_failures"] == []
    assert result["idempotent_noop"] is False
    assert reviewed["acquisition_status"] == "extracted"
    assert reviewed["status_history"] == history
    assert reviewed["judgment_number"] == "09/2026/DS-PT"
    assert reviewed["related_decision_ids"] == _payload()["related_decision_ids"]
    assert reviewed["underlying_dispute_id"] is None
    assert all(reviewed["metadata_verification"].values())
    inventory, report, _ = inventory_independent_sources(tmp_path)
    assert inventory == []
    assert report["eligible_source_count"] == 0
    assert {path: path.read_bytes() for path in snapshot_paths} == snapshot_bytes


def test_review_missing_source_is_rejected_without_manifest_change(tmp_path):
    _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    with pytest.raises(SourceIntakeError, match="Unknown acquisition_id"):
        record_source_review(tmp_path, "missing-source", _payload())
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["hash_verified", "eligible", "rejected", "quarantined"])
def test_review_rejects_every_non_extracted_status(tmp_path, status):
    if status == "hash_verified":
        source, _, _, _ = _hash_verified_fixture(tmp_path)
    elif status == "eligible":
        source, _ = _eligible_fixture(tmp_path)
    else:
        source, manifest = _extracted_fixture(tmp_path)
        source["acquisition_status"] = status
        source["status_history"].append(
            {"status": status, "timestamp": FIXED_TIME, "reasons": ["terminal"]}
        )
        _write_manifest(tmp_path, manifest)
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    with pytest.raises(SourceIntakeError, match="requires extracted status"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("judgment_number", "not canonical", "canonical Vietnamese decision"),
        ("case_id", "", "non-empty string"),
        ("decision_date", "2026-02-30", "valid YYYY-MM-DD"),
        ("procedure_stage", "appeal", "allowed value"),
        ("court", "", "non-empty string"),
    ],
)
def test_review_rejects_invalid_verified_identity(tmp_path, field, value, message):
    source, _ = _review_fixture(tmp_path)
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    with pytest.raises(SourceIntakeError, match=message):
        record_source_review(
            tmp_path, source["acquisition_id"], _payload(**{field: value})
        )
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


def test_review_rejects_missing_case_id_even_if_verified_true(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    payload["verified_metadata"].pop("case_id")
    with pytest.raises(SourceIntakeError, match="verified_metadata fields"):
        record_source_review(tmp_path, source["acquisition_id"], payload)


def test_review_rejects_metadata_value_without_explicit_verification(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    payload["verified_fields"]["court"] = False
    with pytest.raises(SourceIntakeError, match="verified=true"):
        record_source_review(tmp_path, source["acquisition_id"], payload)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "unknown",
        "status",
        "status_history",
        "source_id",
        "source_url",
        "stored_pdf_path",
        "sha256",
        "structured_path",
        "structured_sha256",
    ],
)
def test_review_payload_rejects_unknown_and_forbidden_manifest_fields(
    tmp_path, forbidden_field
):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    payload[forbidden_field] = "forbidden"
    with pytest.raises(SourceIntakeError, match="invalid root fields"):
        record_source_review(tmp_path, source["acquisition_id"], payload)


def test_incomplete_overlap_review_remains_an_eligibility_gate(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    payload["reviews"]["development_overlap_check"]["completed"] = False
    result = record_source_review(tmp_path, source["acquisition_id"], payload)
    assert "incomplete_development_overlap_check" in result["gate_failures"]
    with pytest.raises(SourceIntakeError, match="incomplete_development_overlap_check"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


def test_development_overlap_cannot_be_falsely_marked_clean(tmp_path):
    source, _ = _review_fixture(tmp_path)
    development = tmp_path / "backend/data/judgments_evaluation.json"
    development.write_text(
        json.dumps([{"id": 1, "judgment_name": source["original_filename"]}]),
        encoding="utf-8",
    )
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    with pytest.raises(SourceIntakeError, match="findings mismatch"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


def _add_peer_source(
    root: Path,
    manifest: dict,
    *,
    same_pdf: bool,
    same_text: bool,
) -> dict:
    peer = _source("src-review-002", "peer.pdf", status="extracted")
    peer["case_id"] = "22/2026/DS-ST"
    peer["judgment_number"] = "22/2026/DS-ST"
    target = manifest["sources"][0]
    pdf = (
        (root / target["stored_pdf_path"]).read_bytes()
        if same_pdf
        else _minimal_pdf_bytes("peer-source")
    )
    _materialize(
        root,
        peer,
        pdf_bytes=pdf,
        structured_payload=_structured(
            "peer.pdf",
            "Nội dung bản án hợp lệ" if same_text else "Văn bản hoàn toàn khác",
        ),
    )
    manifest["sources"].append(peer)
    manifest["immutable_provenance"].append(
        source_intake.immutable_provenance_snapshot(peer)
    )
    _write_manifest(root, manifest)
    return peer


def _cohort_identity(index: int, *, underlying_dispute_id=None) -> dict:
    payload = _identity_payload(
        judgment_number=f"{40 + index}/2026/DS-ST",
        case_id=f"{40 + index}/2026/DS-ST",
        court=f"Tòa án nhân dân tỉnh Cohort {index}",
        decision_date=f"2026-02-{index:02d}",
        procedure_stage="so_tham",
    )
    payload["related_decision_ids"] = []
    payload["underlying_dispute_id"] = underlying_dispute_id
    return payload


def _multi_source_cohort(
    root: Path,
    *,
    exact_duplicate_ab: bool = False,
    near_duplicate_ab: bool = False,
) -> tuple[dict, list[str]]:
    existing = _source("existing-eligible", "existing.pdf", status="eligible")
    _materialize(
        root,
        existing,
        pdf_bytes=_minimal_pdf_bytes("existing-eligible"),
        structured_payload=_structured(
            "existing.pdf",
            "Quan hệ lao động tiền lương bảo hiểm và kỷ luật sa thải.",
        ),
    )
    texts = {
        "cohort-a": "Tranh chấp hợp đồng vay tài sản, lãi suất và nghĩa vụ hoàn trả.",
        "cohort-b": "Yêu cầu bồi thường thiệt hại sức khỏe do tai nạn giao thông.",
        "cohort-c": "Phân chia di sản thừa kế và quyền sử dụng đất của các đồng thừa kế.",
    }
    sources = [existing]
    acquisition_ids = ["cohort-a", "cohort-b", "cohort-c"]
    a_pdf = _minimal_pdf_bytes("cohort-a")
    for acquisition_id in acquisition_ids:
        source = _source(
            acquisition_id, f"{acquisition_id}.pdf", status="extracted"
        )
        for field in REVIEW_METADATA_FIELDS:
            source[field] = None
            source["metadata_verification"][field] = False
        for field in REVIEW_COMPLETION_FIELDS:
            source["metadata_verification"][field] = False
        source["related_decision_ids"] = []
        source["underlying_dispute_id"] = None
        source["rejection_reasons"] = []
        is_b = acquisition_id == "cohort-b"
        _materialize(
            root,
            source,
            pdf_bytes=(
                a_pdf
                if acquisition_id == "cohort-a" or (is_b and exact_duplicate_ab)
                else _minimal_pdf_bytes(acquisition_id)
            ),
            structured_payload=_structured(
                source["original_filename"],
                texts["cohort-a"] if is_b and near_duplicate_ab else texts[acquisition_id],
            ),
        )
        sources.append(source)
    manifest = _manifest(sources)
    _write_manifest(root, manifest)
    development = root / "backend/data/judgments_evaluation.json"
    development.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("[]", encoding="utf-8")
    return manifest, acquisition_ids


def _identify_cohort(root: Path, acquisition_ids: list[str], *, shared_dispute=False):
    for index, acquisition_id in enumerate(acquisition_ids, start=1):
        record_source_identity(
            root,
            acquisition_id,
            _cohort_identity(
                index,
                underlying_dispute_id=(
                    "DISPUTE-SHARED" if shared_dispute and index <= 2 else None
                ),
            ),
        )


def _review_semantics(root: Path, acquisition_ids: list[str]) -> dict:
    manifest = load_acquisition_manifest(root)
    return {
        acquisition_id: {
            "metadata_verification": copy.deepcopy(
                next(
                    source
                    for source in manifest["sources"]
                    if source["acquisition_id"] == acquisition_id
                )["metadata_verification"]
            ),
            "rejection_reasons": list(
                next(
                    source
                    for source in manifest["sources"]
                    if source["acquisition_id"] == acquisition_id
                )["rejection_reasons"]
            ),
        }
        for acquisition_id in acquisition_ids
    }


def test_exact_duplicate_finding_cannot_be_overwritten_as_clean(tmp_path):
    source, manifest = _review_fixture(tmp_path)
    _add_peer_source(tmp_path, manifest, same_pdf=True, same_text=False)
    with pytest.raises(SourceIntakeError, match="findings mismatch"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())


def test_near_duplicate_unresolved_review_remains_blocking(tmp_path):
    source, manifest = _review_fixture(tmp_path)
    _add_peer_source(tmp_path, manifest, same_pdf=False, same_text=True)
    live = load_acquisition_manifest(tmp_path)
    payload = _payload()
    payload["reviews"]["internal_dedup_check"]["completed"] = False
    payload = _payload_with_authoritative_findings(
        tmp_path, live, source["acquisition_id"], payload
    )
    result = record_source_review(tmp_path, source["acquisition_id"], payload)
    assert "incomplete_internal_dedup_check" in result["gate_failures"]
    assert "unresolved_rejection_reasons" in result["gate_failures"]
    assert any("new_pool_near_duplicate_source_text" in value for value in result["gate_failures"])


def test_underlying_dispute_and_related_decisions_use_existing_fields(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    payload["underlying_dispute_id"] = "DISPUTE-2026-001"
    reviewed_payload = record_source_review(
        tmp_path, source["acquisition_id"], payload
    )
    reviewed = _manifest_source(tmp_path)
    assert reviewed_payload["gate_failures"] == []
    assert reviewed["underlying_dispute_id"] == "DISPUTE-2026-001"
    assert reviewed["related_decision_ids"] == payload["related_decision_ids"]


def test_multi_source_two_phase_lifecycle_has_no_review_deadlock(tmp_path):
    _, acquisition_ids = _multi_source_cohort(tmp_path)
    original_histories = {
        acquisition_id: copy.deepcopy(
            _manifest_source(tmp_path, acquisition_id)["status_history"]
        )
        for acquisition_id in acquisition_ids
    }

    _identify_cohort(tmp_path, acquisition_ids)

    for acquisition_id in acquisition_ids:
        source = _manifest_source(tmp_path, acquisition_id)
        assert source["acquisition_status"] == "extracted"
        assert source["status_history"] == original_histories[acquisition_id]
        assert all(
            source["metadata_verification"][field]
            for field in REVIEW_METADATA_FIELDS
        )
        assert all(
            not source["metadata_verification"][field]
            for field in REVIEW_COMPLETION_FIELDS
        )

    for acquisition_id in acquisition_ids:
        result = record_source_review(
            tmp_path, acquisition_id, _review_only_payload()
        )
        assert result["gate_failures"] == []
        assert _manifest_source(tmp_path, acquisition_id)["acquisition_status"] == "extracted"

    for acquisition_id in acquisition_ids:
        promoted = promote_source_to_eligible(tmp_path, acquisition_id)
        assert promoted["acquisition_status"] == "eligible"


def test_review_fails_closed_until_every_extracted_peer_is_identity_ready(tmp_path):
    _, acquisition_ids = _multi_source_cohort(tmp_path)
    for index, acquisition_id in enumerate(acquisition_ids[:2], start=1):
        record_source_identity(
            tmp_path, acquisition_id, _cohort_identity(index)
        )
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    with pytest.raises(
        SourceIntakeError,
        match=r"Peer identity not ready.*cohort-c",
    ):
        record_source_review(
            tmp_path, acquisition_ids[0], _review_only_payload()
        )

    assert path.read_bytes() == before
    assert all(
        not _manifest_source(tmp_path, acquisition_ids[0])["metadata_verification"][field]
        for field in REVIEW_COMPLETION_FIELDS
    )
    record_source_identity(tmp_path, acquisition_ids[2], _cohort_identity(3))
    assert record_source_review(
        tmp_path, acquisition_ids[0], _review_only_payload()
    )["gate_failures"] == []


def test_identity_ready_peers_need_not_be_reviewed_for_cross_source_review(tmp_path):
    _, acquisition_ids = _multi_source_cohort(tmp_path)
    _identify_cohort(tmp_path, acquisition_ids)

    result = record_source_review(
        tmp_path, acquisition_ids[0], _review_only_payload()
    )

    assert result["gate_failures"] == []
    for acquisition_id in acquisition_ids[1:]:
        peer = _manifest_source(tmp_path, acquisition_id)
        assert all(
            not peer["metadata_verification"][field]
            for field in REVIEW_COMPLETION_FIELDS
        )


@pytest.mark.parametrize(
    ("fixture_option", "expected_reason"),
    [
        ("exact", "new_pool_duplicate_sha256"),
        ("near", "new_pool_near_duplicate_source_text"),
    ],
)
def test_identity_ready_unreviewed_duplicate_peer_cannot_be_marked_clean(
    tmp_path, fixture_option, expected_reason
):
    _, acquisition_ids = _multi_source_cohort(
        tmp_path,
        exact_duplicate_ab=fixture_option == "exact",
        near_duplicate_ab=fixture_option == "near",
    )
    _identify_cohort(tmp_path, acquisition_ids)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    with pytest.raises(SourceIntakeError, match="findings mismatch"):
        record_source_review(
            tmp_path, acquisition_ids[0], _review_only_payload()
        )

    assert path.read_bytes() == before
    manifest = load_acquisition_manifest(tmp_path)
    findings = source_intake._eligibility_review_findings(
        tmp_path, manifest, acquisition_ids[0]
    )
    assert any(
        expected_reason in reason
        for reasons in findings.values()
        for reason in reasons
    )


def test_same_underlying_dispute_is_detected_before_either_peer_is_reviewed(tmp_path):
    _, acquisition_ids = _multi_source_cohort(tmp_path)
    _identify_cohort(tmp_path, acquisition_ids, shared_dispute=True)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    with pytest.raises(SourceIntakeError, match="findings mismatch"):
        record_source_review(
            tmp_path, acquisition_ids[0], _review_only_payload()
        )

    assert path.read_bytes() == before
    findings = source_intake._eligibility_review_findings(
        tmp_path, load_acquisition_manifest(tmp_path), acquisition_ids[0]
    )
    assert any(
        "same_underlying_dispute" in reason
        for reason in findings["underlying_dispute_review"]
    )


def test_review_order_is_semantically_independent(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _, acquisition_ids = _multi_source_cohort(first_root)
    _multi_source_cohort(second_root)
    _identify_cohort(first_root, acquisition_ids)
    _identify_cohort(second_root, acquisition_ids)

    for acquisition_id in acquisition_ids:
        record_source_review(first_root, acquisition_id, _review_only_payload())
    for acquisition_id in reversed(acquisition_ids):
        record_source_review(second_root, acquisition_id, _review_only_payload())

    assert _review_semantics(first_root, acquisition_ids) == _review_semantics(
        second_root, acquisition_ids
    )


def test_blank_page_review_never_rewrites_structured_artifact(tmp_path):
    structured = {
        "judgment_name": "review.pdf",
        "page_count": 2,
        "pages": [
            {"page": 1, "text": "Nội dung hoàn chỉnh", "extraction_status": "native_text"},
            {"page": 2, "text": "", "extraction_status": "empty_after_native_and_ocr"},
        ],
        "sections": {
            "header": "Nội dung hoàn chỉnh",
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }
    source, _ = _review_fixture(tmp_path, structured_payload=structured)
    path = tmp_path / source["structured_path"]
    before = path.read_bytes()
    record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert path.read_bytes() == before


def test_identical_second_review_is_byte_stable_noop(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _payload()
    record_source_review(tmp_path, source["acquisition_id"], payload)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    result = record_source_review(tmp_path, source["acquisition_id"], payload)
    assert result["idempotent_noop"] is True
    assert manifest_path.read_bytes() == before


def test_conflicting_second_review_fails_closed(tmp_path):
    source, _ = _review_fixture(tmp_path)
    record_source_review(tmp_path, source["acquisition_id"], _payload())
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    conflict = _payload(court="Tòa án nhân dân tỉnh Khác")
    with pytest.raises(SourceIntakeError, match="conflicts with existing court"):
        record_source_review(tmp_path, source["acquisition_id"], conflict)
    assert path.read_bytes() == before


def test_manifest_race_fails_closed_without_lost_update(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE

    def concurrent_update(checkpoint):
        if checkpoint == "before_manifest_compare":
            concurrent = json.loads(path.read_text(encoding="utf-8"))
            concurrent["sources"][0]["notes"] = "retained concurrent update"
            path.write_text(json.dumps(concurrent, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(source_intake, "_review_publication_checkpoint", concurrent_update)
    with pytest.raises(SourceIntakeError, match="unexpected manifest state"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert _manifest_source(tmp_path)["notes"] == "retained concurrent update"


def test_failure_before_atomic_replace_preserves_old_manifest(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("before replacement")

    monkeypatch.setattr(source_intake, "_atomic_write_bytes_at", fail)
    with pytest.raises(OSError, match="before replacement"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert path.read_bytes() == before


def test_failure_after_atomic_replace_rolls_back(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    def fail_after_replace(checkpoint):
        if checkpoint == "after_manifest_replacement":
            raise OSError("after replacement")

    monkeypatch.setattr(source_intake, "_review_publication_checkpoint", fail_after_replace)
    with pytest.raises(SourceIntakeError, match="previous manifest restored"):
        record_source_review(tmp_path, source["acquisition_id"], _payload())
    assert path.read_bytes() == before
    assert _manifest_source(tmp_path)["judgment_number"] is None


def test_review_preserves_unrelated_source_and_all_protected_artifacts(tmp_path):
    source, manifest = _review_fixture(tmp_path)
    peer = _source("unrelated-002", "unrelated.pdf", status="extracted")
    peer["case_id"] = "77/2026/DS-ST"
    peer["judgment_number"] = "77/2026/DS-ST"
    _materialize(
        tmp_path,
        peer,
        pdf_bytes=_minimal_pdf_bytes("unrelated"),
        structured_payload=_structured("unrelated.pdf", "qwerty zxcv asdf ghjkl"),
    )
    manifest["sources"].append(peer)
    manifest["immutable_provenance"].append(source_intake.immutable_provenance_snapshot(peer))
    _write_manifest(tmp_path, manifest)
    benchmark = tmp_path / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_bytes(b'{"state":"DRAFT"}\n')
    protected_paths = [
        tmp_path / source["stored_pdf_path"],
        tmp_path / source["structured_path"],
        tmp_path / PROVENANCE_ROOT_RELATIVE / f"{source['acquisition_id']}.json",
        benchmark,
    ]
    protected = {path: path.read_bytes() for path in protected_paths}
    peer_before = copy.deepcopy(peer)

    record_source_review(tmp_path, source["acquisition_id"], _payload())

    assert _manifest_source(tmp_path, peer["acquisition_id"]) == peer_before
    assert {path: path.read_bytes() for path in protected_paths} == protected


def test_review_cli_delegates_strict_payload_and_never_promotes(tmp_path, monkeypatch, capsys):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    calls = []

    def fake_review(root, acquisition_id, payload):
        calls.append((Path(root), acquisition_id, payload))
        return {"acquisition_id": acquisition_id, "acquisition_status": "extracted"}

    def forbidden_promote(*args, **kwargs):
        raise AssertionError("review CLI must not promote")

    monkeypatch.setattr(source_intake, "record_source_review", fake_review)
    monkeypatch.setattr(source_intake, "promote_source_to_eligible", forbidden_promote)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "review",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-review-001",
            "--review-file",
            str(review_path),
        ],
    )
    source_intake.main()
    assert calls == [(tmp_path, "src-review-001", _payload())]
    assert json.loads(capsys.readouterr().out)["acquisition_status"] == "extracted"


@pytest.mark.parametrize("argument", ["--force-ocr", "--source-url", "--pdf"])
def test_review_cli_rejects_unrelated_mutation_arguments(tmp_path, monkeypatch, argument):
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(_payload()), encoding="utf-8")
    argv = [
        "source_intake",
        "review",
        "--project-root",
        str(tmp_path),
        "--acquisition-id",
        "src-review-001",
        "--review-file",
        str(review_path),
        argument,
    ]
    if argument == "--source-url":
        argv.append("https://example.test/rewrite")
    elif argument == "--pdf":
        argv.append(str(tmp_path / "rewrite.pdf"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        source_intake.main()


def test_review_cli_rejects_symlinked_review_file(tmp_path, monkeypatch):
    actual = tmp_path / "actual-review.json"
    actual.write_text(json.dumps(_payload()), encoding="utf-8")
    link = tmp_path / "review-link.json"
    link.symlink_to(actual)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "review",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-review-001",
            "--review-file",
            str(link),
        ],
    )
    with pytest.raises(SystemExit):
        source_intake.main()


def test_temporary_end_to_end_register_extract_review_then_explicit_promote(
    tmp_path, monkeypatch
):
    development = tmp_path / "backend/data/judgments_evaluation.json"
    development.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("[]", encoding="utf-8")
    for relative in (
        INTAKE_ROOT_RELATIVE,
        INTAKE_ROOT_RELATIVE / "raw",
        INTAKE_ROOT_RELATIVE / "structured",
        PROVENANCE_ROOT_RELATIVE,
        source_intake.TRANSACTION_ROOT_RELATIVE,
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    empty_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": "2026-10-26T09:00:00+07:00",
        "sources": [],
        "immutable_provenance": [],
        "known_duplicate_links": [],
    }
    (tmp_path / ACQUISITION_MANIFEST_RELATIVE).write_bytes(
        source_intake._canonical_json_bytes(empty_manifest)
    )
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(_minimal_pdf_bytes("end-to-end"))
    acquisition_id = "e2e-review-001"
    registered = register_source(
        tmp_path,
        acquisition_id,
        input_pdf,
        "https://example.test/judgments/e2e.pdf",
        "example.test",
        "2026-10-26T09:00:00+07:00",
        "e2e.pdf",
        None,
    )
    assert registered["acquisition_status"] == "hash_verified"
    def fake_extractor(_raw_fd, output_fd, filename, _force_ocr):
        _write_valid_extractor_output_fd(output_fd, filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", fake_extractor)
    assert extract_intake_source(tmp_path, acquisition_id)["acquisition_status"] == "extracted"
    reviewed = record_source_review(tmp_path, acquisition_id, _payload())
    assert reviewed["gate_failures"] == []
    assert _manifest_source(tmp_path, acquisition_id)["acquisition_status"] == "extracted"
    assert inventory_independent_sources(tmp_path)[0] == []
    promoted = promote_source_to_eligible(tmp_path, acquisition_id)
    assert promoted["acquisition_status"] == "eligible"
    assert [item["source_id"] for item in inventory_independent_sources(tmp_path)[0]] == [
        f"independent:{acquisition_id}"
    ]


def test_temporary_three_source_batch_register_extract_identify_review_promote(
    tmp_path, monkeypatch
):
    development = tmp_path / "backend/data/judgments_evaluation.json"
    development.parent.mkdir(parents=True, exist_ok=True)
    development.write_text("[]", encoding="utf-8")
    for relative in (
        INTAKE_ROOT_RELATIVE,
        INTAKE_ROOT_RELATIVE / "raw",
        INTAKE_ROOT_RELATIVE / "structured",
        PROVENANCE_ROOT_RELATIVE,
        source_intake.TRANSACTION_ROOT_RELATIVE,
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    empty_manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": "2026-10-26T09:00:00+07:00",
        "sources": [],
        "immutable_provenance": [],
        "known_duplicate_links": [],
    }
    (tmp_path / ACQUISITION_MANIFEST_RELATIVE).write_bytes(
        source_intake._canonical_json_bytes(empty_manifest)
    )
    acquisition_ids = ["batch-e2e-a", "batch-e2e-b", "batch-e2e-c"]
    unique_text = {
        "batch-e2e-a.pdf": "Nghĩa vụ thanh toán hợp đồng mua bán hàng hóa và tiền lãi.",
        "batch-e2e-b.pdf": "Tranh chấp quyền nuôi con và nghĩa vụ cấp dưỡng sau ly hôn.",
        "batch-e2e-c.pdf": "Yêu cầu hủy quyết định hành chính về quản lý đất đai.",
    }
    for acquisition_id in acquisition_ids:
        input_pdf = tmp_path / f"input-{acquisition_id}.pdf"
        input_pdf.write_bytes(_minimal_pdf_bytes(acquisition_id))
        registered = register_source(
            tmp_path,
            acquisition_id,
            input_pdf,
            f"https://example.test/judgments/{acquisition_id}.pdf",
            "example.test",
            "2026-10-26T09:00:00+07:00",
            f"{acquisition_id}.pdf",
            None,
        )
        assert registered["acquisition_status"] == "hash_verified"

    def fake_extractor(_raw_fd, output_fd, filename, _force_ocr):
        payload = json.dumps(
            _structured(filename, unique_text[filename]),
            ensure_ascii=False,
        ).encode("utf-8")
        output_name = f"{Path(filename).stem}_structured.json"
        artifact_fd = os.open(
            output_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=output_fd,
        )
        try:
            os.write(artifact_fd, payload)
        finally:
            os.close(artifact_fd)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", fake_extractor)
    for acquisition_id in acquisition_ids:
        assert extract_intake_source(
            tmp_path, acquisition_id
        )["acquisition_status"] == "extracted"
    _identify_cohort(tmp_path, acquisition_ids)
    for acquisition_id in acquisition_ids:
        assert record_source_review(
            tmp_path, acquisition_id, _review_only_payload()
        )["gate_failures"] == []
    for acquisition_id in acquisition_ids:
        assert promote_source_to_eligible(
            tmp_path, acquisition_id
        )["acquisition_status"] == "eligible"
    assert {
        record["source_id"]
        for record in inventory_independent_sources(tmp_path)[0]
    } == {f"independent:{acquisition_id}" for acquisition_id in acquisition_ids}


def test_real_batch_one_production_state_is_preserved():
    manifest = load_acquisition_manifest(PROJECT_ROOT)
    sources_by_id = {
        source["acquisition_id"]: source for source in manifest["sources"]
    }
    assert BATCH_ONE_SOURCE_IDS <= set(sources_by_id)
    batch_one_sources = [
        source
        for source in manifest["sources"]
        if source["acquisition_id"] in BATCH_ONE_SOURCE_IDS
    ]
    assert len(batch_one_sources) == len(BATCH_ONE_SOURCE_IDS)
    assert canonical_json_fingerprint(batch_one_sources) == (
        BATCH_ONE_SOURCE_RECORDS_FINGERPRINT
    )
    scale = next(source for source in manifest["sources"] if source["acquisition_id"] == "scale-20261026-001")
    assert scale["acquisition_status"] == "eligible"
    assert {field: scale[field] for field in REVIEW_METADATA_FIELDS} == {
        "judgment_number": "09/2026/DS-PT",
        "case_id": "09/2026/DS-PT",
        "court": "Tòa án nhân dân thành phố Đà Nẵng",
        "decision_date": "2026-01-09",
        "procedure_stage": "phuc_tham",
    }
    assert all(scale["metadata_verification"].values())
    batch = [
        source
        for source in manifest["sources"]
        if source["acquisition_id"].startswith("scale-20261026-")
        and source["acquisition_id"] != "scale-20261026-001"
    ]
    assert len(batch) == 49
    blocked_ids = {
        "scale-20261026-003",
        "scale-20261026-008",
        "scale-20261026-009",
        "scale-20261026-012",
        "scale-20261026-020",
        "scale-20261026-022",
        "scale-20261026-023",
        "scale-20261026-042",
        "scale-20261026-046",
        "scale-20261026-047",
        "scale-20261026-050",
    }
    assert blocked_ids == BATCH_ONE_BLOCKED_IDS
    blocked = [source for source in batch if source["acquisition_id"] in blocked_ids]
    promoted = [source for source in batch if source["acquisition_id"] not in blocked_ids]
    assert {source["acquisition_id"] for source in promoted} == (
        BATCH_ONE_ELIGIBLE_IDS
        - {f"pilot-20261025-{index:03d}" for index in range(1, 25)}
        - {"scale-20261026-001"}
    )
    assert len(blocked) == 11
    assert len(promoted) == 38
    assert all(source["acquisition_status"] == "extracted" for source in blocked)
    assert all(source["acquisition_status"] == "eligible" for source in promoted)
    assert all(
        not any(entry["status"] == "eligible" for entry in source["status_history"])
        for source in blocked
    )
    assert all(
        [entry["status"] for entry in source["status_history"]]
        == ["downloaded", "hash_verified", "extracted", "eligible"]
        for source in promoted
    )
    assert all(
        all(source[field] is not None for field in REVIEW_METADATA_FIELDS)
        and all(source["metadata_verification"].values())
        for source in batch
    )
    assert all(
        source["rejection_reasons"]
        == ["dedup_or_development_overlap:new_pool_related_decision_ids"]
        for source in blocked
    )
    assert all(source["rejection_reasons"] == [] for source in promoted)


def test_historical_and_stale_draft_snapshot_remain_monotonically_safe():
    inventory = inventory_judgment_sources(PROJECT_ROOT)
    assert len(inventory) == 107
    assert canonical_json_fingerprint(inventory) == DEVELOPMENT_FINGERPRINT
    benchmark = json.loads((BENCHMARK_DIR / "benchmark_manifest.json").read_text(encoding="utf-8"))
    sampling = json.loads((BENCHMARK_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    independent, _, _ = inventory_independent_sources(PROJECT_ROOT)
    independent_ids = {record["source_id"] for record in independent}
    independent_by_id = {record["source_id"]: record for record in independent}
    snapshot_eligible = sampling["eligible_sources"]
    snapshot_eligible_ids = {
        record["source_id"] for record in snapshot_eligible
    }
    assert benchmark["state"] == "DRAFT"
    assert sampling["state"] == "DRAFT"
    assert sampling["selection_status"] == "NOT_STARTED"
    assert sampling["sampling_seed"] is None
    assert snapshot_eligible_ids <= independent_ids
    assert len(snapshot_eligible_ids) == len(snapshot_eligible)
    assert {
        source_id: independent_by_id[source_id]
        for source_id in snapshot_eligible_ids
    } == {record["source_id"]: record for record in snapshot_eligible}
    assert len(sampling["available_source_inventory"]) == (
        len(inventory) + len(snapshot_eligible)
    )
    assert benchmark["current_counts"]["eligible_sources"] == len(
        snapshot_eligible
    )
    assert benchmark["current_counts"]["available_sources"] == (
        len(inventory) + len(snapshot_eligible)
    )
    assert not (BENCHMARK_DIR / "selected_sources.json").exists()
    for name in ("queries.json", "ground_truth.json", "selected_sources.json"):
        candidate = BENCHMARK_DIR / name
        assert not candidate.exists() or json.loads(candidate.read_text(encoding="utf-8")) == []
