import copy
import hashlib
import json
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
    record_source_review,
    register_source,
)
from backend.scripts.Tests.test_independent_source_intake import (
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


PRODUCTION_MANIFEST_SHA256 = (
    "9a5aa079b231fe2b4fed63cd5911ed800ccd47ebfa884e0c48ccf6b59f948b90"
)
DEVELOPMENT_FINGERPRINT = (
    "fd6a6f05e4b71a2c499584fc0296cf25cedf5eac958aea6d079bf449c864f754"
)


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


def test_real_scale_source_completed_production_lifecycle():
    path = PROJECT_ROOT / ACQUISITION_MANIFEST_RELATIVE
    payload = path.read_bytes()
    assert _sha(payload) == PRODUCTION_MANIFEST_SHA256
    manifest = json.loads(payload.decode("utf-8"))
    assert len(manifest["sources"]) == 25
    assert sum(source["acquisition_status"] == "eligible" for source in manifest["sources"]) == 25
    assert sum(source["acquisition_status"] == "extracted" for source in manifest["sources"]) == 0
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


def test_historical_and_benchmark_state_remain_sealed():
    inventory = inventory_judgment_sources(PROJECT_ROOT)
    assert len(inventory) == 107
    assert canonical_json_fingerprint(inventory) == DEVELOPMENT_FINGERPRINT
    benchmark_dir = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"
    benchmark = json.loads((benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    sampling = json.loads((benchmark_dir / "sampling_manifest.json").read_text(encoding="utf-8"))
    assert benchmark["state"] == "DRAFT"
    assert sampling["state"] == "DRAFT"
    assert sampling["selection_status"] == "NOT_STARTED"
    assert len(sampling["eligible_sources"]) == 25
    assert len(sampling["available_source_inventory"]) == 132
    assert not (benchmark_dir / "selected_sources.json").exists()
    for name in ("queries.json", "ground_truth.json", "selected_sources.json"):
        candidate = benchmark_dir / name
        assert not candidate.exists() or json.loads(candidate.read_text(encoding="utf-8")) == []
