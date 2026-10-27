import copy
import hashlib
import json
import os
import sys
import types
import unicodedata
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import independent_benchmark, source_intake
from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    canonical_json_fingerprint,
    deduplicate_against_development,
    inventory_judgment_sources,
    phase1_payloads,
)
from backend.scripts.Evaluation.benchmarking.source_intake import (
    ACQUISITION_MANIFEST_RELATIVE,
    ELIGIBLE_STATUS_HISTORY,
    INTAKE_ROOT_RELATIVE,
    MANIFEST_VERSION,
    PROVENANCE_ROOT_RELATIVE,
    REVIEW_METADATA_FIELDS,
    SourceIntakeError,
    TRANSACTION_ROOT_RELATIVE,
    create_provenance_anchor,
    extract_intake_source,
    immutable_provenance_snapshot,
    inventory_independent_sources,
    load_acquisition_manifest,
    promote_source_to_eligible,
    stable_source_id,
    validate_acquisition_manifest,
    validate_pdf_source,
    validate_structured_judgment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"
FIXED_TIME = "2026-10-25T00:00:00+07:00"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _minimal_pdf_bytes(label: str = "fixture") -> bytes:
    """Build a deterministic, parseable one-page PDF without test dependencies."""

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    payload = bytearray(f"%PDF-1.4\n% {label}\n".encode("ascii"))
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def _structured(filename: str, text: str = "Nội dung bản án hợp lệ") -> dict:
    text = unicodedata.normalize("NFC", text)
    return {
        "judgment_name": filename,
        "page_count": 1,
        "pages": [{"page": 1, "text": text}],
        "sections": {
            "header": text,
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }


def _status_history(status: str = "eligible") -> list[dict]:
    if status in ELIGIBLE_STATUS_HISTORY:
        statuses = ELIGIBLE_STATUS_HISTORY[: ELIGIBLE_STATUS_HISTORY.index(status) + 1]
    else:
        statuses = (status,)
    return [
        {"status": value, "timestamp": FIXED_TIME, "reasons": []}
        for value in statuses
    ]


def _source(
    acquisition_id: str = "src-001",
    filename: str = "decision.pdf",
    *,
    status: str = "eligible",
) -> dict:
    base = INTAKE_ROOT_RELATIVE.as_posix()
    return {
        "acquisition_id": acquisition_id,
        "source_id": f"independent:{acquisition_id}",
        "original_filename": filename,
        "stored_pdf_path": f"{base}/raw/{acquisition_id}/{filename}",
        "source_url": "https://example.test/judgments/decision.pdf",
        "offline_provenance": None,
        "source_domain": "example.test",
        "download_timestamp": FIXED_TIME,
        "sha256": "a" * 64,
        "case_id": "CASE-001",
        "judgment_number": "01/2026/DS-ST",
        "court": "TAND tỉnh Test",
        "decision_date": "2026-01-02",
        "procedure_stage": "so_tham",
        "underlying_dispute_id": None,
        "related_decision_ids": [],
        "structured_path": (
            f"{base}/structured/{acquisition_id}/{Path(filename).stem}_structured.json"
        ),
        "structured_sha256": (
            "b" * 64 if status in ELIGIBLE_STATUS_HISTORY[2:] else None
        ),
        "metadata_verification": {
            "judgment_number": True,
            "case_id": True,
            "court": True,
            "decision_date": True,
            "procedure_stage": True,
            "development_overlap_check": True,
            "internal_dedup_check": True,
            "underlying_dispute_review": True,
        },
        "acquisition_status": status,
        "status_history": _status_history(status),
        "rejection_reasons": [],
        "notes": None,
    }


def _manifest(sources: list[dict], links=None) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": FIXED_TIME,
        "sources": sources,
        "immutable_provenance": [
            immutable_provenance_snapshot(source) for source in sources
        ],
        "known_duplicate_links": links or [],
    }


def _materialize(
    root: Path,
    source: dict,
    *,
    pdf_bytes: bytes | None = None,
    structured_payload: dict | None = None,
) -> None:
    pdf_bytes = pdf_bytes or _minimal_pdf_bytes()
    structured_payload = structured_payload or _structured(source["original_filename"])
    pdf_path = root / source["stored_pdf_path"]
    structured_path = root / source["structured_path"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    structured_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    structured_bytes = (
        json.dumps(structured_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    structured_path.write_bytes(structured_bytes)
    source["sha256"] = _sha(pdf_bytes)
    source["structured_sha256"] = _sha(structured_bytes)


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / PROVENANCE_ROOT_RELATIVE).mkdir(parents=True, exist_ok=True)
    (root / TRANSACTION_ROOT_RELATIVE).mkdir(parents=True, exist_ok=True)
    for source in manifest["sources"]:
        anchor = root / PROVENANCE_ROOT_RELATIVE / f"{source['acquisition_id']}.json"
        if not os.path.lexists(anchor):
            create_provenance_anchor(root, source)
    path = root / ACQUISITION_MANIFEST_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def _eligible_fixture(
    root: Path,
    acquisition_id: str = "src-001",
    filename: str = "decision.pdf",
    *,
    pdf_bytes: bytes | None = None,
    text: str = "Nội dung bản án hợp lệ",
) -> tuple[dict, dict]:
    source = _source(acquisition_id, filename)
    source["case_id"] = f"CASE-{acquisition_id}"
    source["judgment_number"] = f"JUDGMENT-{acquisition_id}"
    _materialize(
        root,
        source,
        pdf_bytes=pdf_bytes,
        structured_payload=_structured(filename, text),
    )
    manifest = _manifest([source])
    _write_manifest(root, manifest)
    return source, manifest


def _extracted_fixture(
    root: Path,
    acquisition_id: str = "src-001",
    filename: str = "decision.pdf",
    *,
    text: str = "Nội dung bản án hợp lệ",
    structured_payload: dict | None = None,
) -> tuple[dict, dict]:
    source = _source(acquisition_id, filename, status="extracted")
    source["case_id"] = f"CASE-{acquisition_id}"
    source["judgment_number"] = f"JUDGMENT-{acquisition_id}"
    _materialize(
        root,
        source,
        pdf_bytes=_minimal_pdf_bytes(acquisition_id),
        structured_payload=(
            structured_payload or _structured(filename, text)
        ),
    )
    manifest = _manifest([source])
    _write_manifest(root, manifest)
    development_path = root / "backend/data/judgments_evaluation.json"
    development_path.parent.mkdir(parents=True, exist_ok=True)
    development_path.write_text("[]", encoding="utf-8")
    return source, manifest


def _hash_verified_fixture(
    root: Path,
    acquisition_id: str = "src-001",
    filename: str = "decision.pdf",
) -> tuple[dict, Path, Path, bytes]:
    source = _source(acquisition_id, filename, status="hash_verified")
    pdf_bytes = _minimal_pdf_bytes(acquisition_id)
    pdf_path = root / source["stored_pdf_path"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    source["sha256"] = _sha(pdf_bytes)
    _write_manifest(root, _manifest([source]))
    manifest_path = root / ACQUISITION_MANIFEST_RELATIVE
    return source, pdf_path, root / source["structured_path"], manifest_path.read_bytes()


def _write_valid_extractor_output(output_dir: Path, filename: str = "decision.pdf") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{Path(filename).stem}_structured.json").write_text(
        json.dumps(_structured(filename), ensure_ascii=False),
        encoding="utf-8",
    )


def _write_valid_extractor_output_fd(
    output_dir_fd: int, filename: str = "decision.pdf"
) -> None:
    payload = json.dumps(_structured(filename), ensure_ascii=False).encode("utf-8")
    descriptor = os.open(
        f"{Path(filename).stem}_structured.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=output_dir_fd,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _assert_failed_extraction_is_clean(
    root: Path, output_path: Path, original_manifest: bytes
) -> None:
    assert not output_path.exists()
    assert not output_path.is_symlink()
    assert (root / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest
    assert not list(
        (root / ACQUISITION_MANIFEST_RELATIVE).parent.glob(
            ".acquisition-manifest-*.partial"
        )
    )
    if output_path.parent.exists() and not output_path.parent.is_symlink():
        assert not list(output_path.parent.glob(".extract-partial-*"))


def test_valid_acquisition_manifest_and_current_production_intake_state():
    checked = load_acquisition_manifest(PROJECT_ROOT)
    pilot_ids = {
        f"pilot-20261025-{index:03d}" for index in range(1, 25)
    }
    scale_id = "scale-20261026-001"
    batch_ids = {
        f"scale-20261026-{index:03d}" for index in range(2, 51)
    }
    blocked_ids = {
        f"scale-20261026-{index:03d}"
        for index in (3, 8, 9, 12, 20, 22, 23, 42, 46, 47, 50)
    }
    eligible_ids = pilot_ids | {scale_id} | (batch_ids - blocked_ids)
    expected_ids = eligible_ids | batch_ids
    assert {source["acquisition_id"] for source in checked["sources"]} == expected_ids
    assert {
        source["acquisition_id"]
        for source in checked["sources"]
        if source["acquisition_status"] == "extracted"
    } == blocked_ids
    assert {
        source["acquisition_id"]
        for source in checked["sources"]
        if source["acquisition_status"] == "hash_verified"
    } == set()
    assert {
        source["acquisition_id"]
        for source in checked["sources"]
        if source["acquisition_status"] == "eligible"
    } == eligible_ids
    assert all(source["structured_sha256"] is not None for source in checked["sources"])
    assert {
        path.parent.name
        for path in (PROJECT_ROOT / INTAKE_ROOT_RELATIVE / "structured").rglob("*.json")
    } == expected_ids
    pilot_023 = next(
        source
        for source in checked["sources"]
        if source["acquisition_id"] == "pilot-20261025-023"
    )
    assert pilot_023["acquisition_status"] == "eligible"
    assert pilot_023["structured_sha256"] is not None
    scale = next(
        source
        for source in checked["sources"]
        if source["acquisition_id"] == scale_id
    )
    assert scale["acquisition_status"] == "eligible"
    assert {field: scale[field] for field in REVIEW_METADATA_FIELDS} == {
        "judgment_number": "09/2026/DS-PT",
        "case_id": "09/2026/DS-PT",
        "court": "Tòa án nhân dân thành phố Đà Nẵng",
        "decision_date": "2026-01-09",
        "procedure_stage": "phuc_tham",
    }
    assert all(scale["metadata_verification"].values())
    assert not list(
        (PROJECT_ROOT / TRANSACTION_ROOT_RELATIVE).glob("*.json")
    )


def test_duplicate_acquisition_id_is_rejected():
    left = _source()
    right = copy.deepcopy(left)
    with pytest.raises(SourceIntakeError, match="Duplicate acquisition_id"):
        validate_acquisition_manifest(_manifest([left, right]))


def test_malformed_pdf_sha_is_rejected():
    source = _source(status="hash_verified")
    source["sha256"] = "ABC"
    with pytest.raises(SourceIntakeError, match="sha256"):
        validate_acquisition_manifest(_manifest([source]))


def test_non_offset_timestamp_is_rejected():
    source = _source()
    source["download_timestamp"] = "2026-10-25T00:00:00"
    with pytest.raises(SourceIntakeError, match="offset-aware"):
        validate_acquisition_manifest(_manifest([source]))


def test_path_escape_outside_intake_root_is_rejected(tmp_path):
    source, _ = _eligible_fixture(tmp_path)
    source["stored_pdf_path"] = "../escaped.pdf"
    manifest = _manifest([source])
    _write_manifest(tmp_path, manifest)
    with pytest.raises(
        SourceIntakeError,
        match="repository-relative|acquisition-scoped|provenance differs from anchor",
    ):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing_pdf", "regular non-symlink"),
        ("invalid_pdf", "PDF signature is invalid"),
        ("pdf_hash", "PDF SHA256 mismatch"),
        ("missing_structured", "Structured artifact invalid"),
        ("malformed_structured", "valid structured JSON"),
        ("page_count", "page_count"),
        ("empty_page", "usable text"),
        ("structured_hash", "SHA256 mismatch"),
    ],
)
def test_file_and_structured_gates_fail_closed(tmp_path, mutation, expected):
    source, manifest = _eligible_fixture(tmp_path)
    pdf_path = tmp_path / source["stored_pdf_path"]
    structured_path = tmp_path / source["structured_path"]
    if mutation == "missing_pdf":
        pdf_path.unlink()
    elif mutation == "invalid_pdf":
        pdf_path.write_bytes(b"not a PDF")
    elif mutation == "pdf_hash":
        pdf_path.write_bytes(_minimal_pdf_bytes("tampered"))
    elif mutation == "missing_structured":
        structured_path.unlink()
    elif mutation == "malformed_structured":
        structured_path.write_text("{", encoding="utf-8")
    elif mutation == "page_count":
        payload = _structured(source["original_filename"])
        payload["page_count"] = 2
        structured_path.write_text(json.dumps(payload), encoding="utf-8")
        source["structured_sha256"] = hashlib.sha256(structured_path.read_bytes()).hexdigest()
    elif mutation == "empty_page":
        payload = _structured(source["original_filename"])
        payload["pages"][0]["text"] = "   "
        structured_path.write_text(json.dumps(payload), encoding="utf-8")
        source["structured_sha256"] = hashlib.sha256(structured_path.read_bytes()).hexdigest()
    elif mutation == "structured_hash":
        source["structured_sha256"] = "b" * 64
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match=expected):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("case_id", "unverified_case_id"),
        ("judgment_number", "unverified_judgment_number"),
        ("court", "unverified_court"),
        ("decision_date", "unverified_decision_date"),
        ("procedure_stage", "unverified_procedure_stage"),
    ],
)
def test_unverified_metadata_blocks_eligibility(tmp_path, field, expected):
    source, manifest = _eligible_fixture(tmp_path)
    source["metadata_verification"][field] = False
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match=expected):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize("field", ["court", "decision_date", "procedure_stage"])
def test_missing_critical_metadata_blocks_eligibility(tmp_path, field):
    source, manifest = _eligible_fixture(tmp_path)
    source[field] = None
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match=f"missing_{field}"):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize(
    "check",
    [
        "development_overlap_check",
        "internal_dedup_check",
        "underlying_dispute_review",
    ],
)
def test_unfinished_review_checks_block_eligibility(tmp_path, check):
    source, manifest = _eligible_fixture(tmp_path)
    source["metadata_verification"][check] = False
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match=f"incomplete_{check}"):
        inventory_independent_sources(tmp_path)


def test_unresolved_rejection_reason_blocks_eligibility(tmp_path):
    source, manifest = _eligible_fixture(tmp_path)
    source["rejection_reasons"] = ["manual review unresolved"]
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match="unresolved_rejection_reasons"):
        inventory_independent_sources(tmp_path)


def test_development_overlap_is_excluded(tmp_path):
    source, _ = _eligible_fixture(tmp_path)
    inventory, _, links = inventory_independent_sources(tmp_path, include_private_text=True)
    eligible, report = deduplicate_against_development(
        inventory,
        [{"id": 1, "judgment_name": source["original_filename"]}],
        known_duplicate_links=links,
    )
    assert eligible == []
    assert report["reason_counts"]["development_benchmark_exact_source"] == 1


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        ("pdf_sha", "new_pool_duplicate_sha256"),
        ("case_id", "new_pool_duplicate_case_id"),
        ("structured_sha", "new_pool_duplicate_structured_sha256"),
        ("near_text", "new_pool_near_duplicate_source_text"),
        ("underlying", "new_pool_same_underlying_dispute"),
        ("normalized", "new_pool_duplicate_normalized_identity"),
        ("related", "new_pool_related_decision_ids"),
    ],
)
def test_internal_intake_dedup_is_deterministic(tmp_path, kind, expected_reason):
    long_text = "Tranh chấp hợp đồng quyền sử dụng đất và nghĩa vụ dân sự. " * 30
    left, _ = _eligible_fixture(
        tmp_path,
        "src-001",
        "same.pdf" if kind == "structured_sha" else "left.pdf",
        pdf_bytes=_minimal_pdf_bytes("left"),
        text=long_text,
    )
    right = _source(
        "src-002", "same.pdf" if kind == "structured_sha" else "right.pdf"
    )
    right["case_id"] = "CASE-src-002"
    right["judgment_number"] = "JUDGMENT-src-002"
    right_text = long_text + ("Khác." if kind == "near_text" else " Hoàn toàn khác.")
    right_pdf = (
        _minimal_pdf_bytes("left")
        if kind == "pdf_sha"
        else _minimal_pdf_bytes("right")
    )
    _materialize(
        tmp_path,
        right,
        pdf_bytes=right_pdf,
        structured_payload=_structured(right["original_filename"], right_text),
    )
    if kind == "case_id":
        right["case_id"] = left["case_id"]
    if kind == "structured_sha":
        # Same filename and exact payload yield the same structured hash.
        left_path = tmp_path / left["structured_path"]
        right_path = tmp_path / right["structured_path"]
        right_path.write_bytes(left_path.read_bytes())
        right["structured_sha256"] = _sha(right_path.read_bytes())
    if kind == "underlying":
        left["underlying_dispute_id"] = "DISPUTE-1"
        right["underlying_dispute_id"] = "DISPUTE-1"
    if kind == "normalized":
        right["judgment_number"] = left["judgment_number"]
    if kind == "related":
        right["related_decision_ids"] = [left["judgment_number"]]
    manifest = _manifest([right, left])
    _write_manifest(tmp_path, manifest)
    inventory, _, links = inventory_independent_sources(tmp_path, include_private_text=True)
    eligible, report = deduplicate_against_development(
        inventory, [], known_duplicate_links=links
    )
    assert [record["source_id"] for record in eligible] == ["independent:src-001"]
    reasons = [
        reason
        for exclusion in report["exclusions"]
        for reason in exclusion["reasons"]
    ]
    assert any(reason.startswith(expected_reason) for reason in reasons)


def test_known_duplicate_links_are_loaded_by_phase1_flow(tmp_path, monkeypatch):
    left, _ = _eligible_fixture(tmp_path, "src-001", "left.pdf")
    right = _source("src-002", "right.pdf")
    right["case_id"] = "CASE-src-002"
    right["judgment_number"] = "JUDGMENT-src-002"
    _materialize(tmp_path, right, pdf_bytes=_minimal_pdf_bytes("right"))
    links = [{
        "acquisition_id": "src-001",
        "related_acquisition_id": "src-002",
        "reason": "manual duplicate review",
    }]
    _write_manifest(tmp_path, _manifest([right, left], links))
    development_path = tmp_path / "backend/data/judgments_evaluation.json"
    development_path.parent.mkdir(parents=True, exist_ok=True)
    development_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(independent_benchmark, "inventory_judgment_sources", lambda *a, **k: [])
    monkeypatch.setattr(independent_benchmark, "git_commit", lambda root: "a" * 40)
    sampling, report = phase1_payloads(tmp_path, generation_timestamp=FIXED_TIME)
    assert [item["source_id"] for item in sampling["eligible_sources"]] == [
        "independent:src-001"
    ]
    assert report["known_duplicate_links"] == [
        ["independent:src-001", "independent:src-002"]
    ]
    assert report["reason_counts"]["new_pool_known_duplicate_link"] == 1


def test_retrieval_fields_are_forbidden_recursively():
    source = _source(status="downloaded")
    source["offline_provenance"] = {"description": "offline", "nested": {"rank": 1}}
    source["source_url"] = None
    with pytest.raises(SourceIntakeError, match="forbidden"):
        validate_acquisition_manifest(_manifest([source]))


def test_source_id_is_stable_and_filename_independent():
    assert stable_source_id("source-123") == "independent:source-123"
    first = _source("source-123", "first.pdf")
    second = _source("source-123", "renamed.pdf")
    assert stable_source_id(first["acquisition_id"]) == stable_source_id(
        second["acquisition_id"]
    )
    assert first["source_id"] == second["source_id"] == "independent:source-123"


def test_persisted_source_id_must_match_acquisition_id():
    source = _source("src-001", status="downloaded")
    source["source_id"] = "independent:someone-else"
    with pytest.raises(SourceIntakeError, match="source_id must equal"):
        validate_acquisition_manifest(_manifest([source]))


def test_input_order_does_not_affect_eligible_output(tmp_path):
    left, _ = _eligible_fixture(tmp_path, "src-001", "left.pdf")
    right = _source("src-002", "right.pdf")
    right["case_id"] = "CASE-src-002"
    right["judgment_number"] = "JUDGMENT-src-002"
    _materialize(tmp_path, right, pdf_bytes=_minimal_pdf_bytes("right"))
    _write_manifest(tmp_path, _manifest([right, left]))
    forward, _, _ = inventory_independent_sources(tmp_path)
    _write_manifest(tmp_path, _manifest([left, right]))
    reverse, _, _ = inventory_independent_sources(tmp_path)
    assert forward == reverse


def test_acquisition_id_is_immutable_against_prior_manifest():
    prior = _manifest([_source("src-001", status="downloaded")])
    current = _manifest([_source("src-002", status="downloaded")])
    with pytest.raises(SourceIntakeError, match="[Ii]mmutable"):
        validate_acquisition_manifest(current, prior_manifest=prior)


def test_extraction_is_acquisition_scoped_and_records_structured_hash(
    tmp_path, monkeypatch
):
    source = _source("src-001", "decision.pdf", status="hash_verified")
    pdf_bytes = _minimal_pdf_bytes()
    pdf_path = tmp_path / source["stored_pdf_path"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    source["sha256"] = _sha(pdf_bytes)
    _write_manifest(tmp_path, _manifest([source]))
    calls = []

    def fake_extract(raw_fd, output_dir_fd, original_filename, force_ocr):
        calls.append((raw_fd, output_dir_fd, original_filename, force_ocr))
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(
        "backend.scripts.Evaluation.benchmarking.source_intake._run_local_extractor",
        fake_extract,
    )
    result = extract_intake_source(tmp_path, "src-001")
    updated = json.loads(
        (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )["sources"][0]
    assert len(calls) == 1
    assert result["structured_path"] == source["structured_path"]
    assert result["structured_sha256"] == updated["structured_sha256"]
    assert updated["acquisition_status"] == "extracted"
    assert updated["status_history"][-1]["status"] == "extracted"


def test_extraction_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    source = _source("src-001", "decision.pdf", status="hash_verified")
    pdf_bytes = _minimal_pdf_bytes()
    pdf_path = tmp_path / source["stored_pdf_path"]
    output_path = tmp_path / source["structured_path"]
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    output_path.write_text("do not overwrite", encoding="utf-8")
    source["sha256"] = _sha(pdf_bytes)
    _write_manifest(tmp_path, _manifest([source]))
    called = False

    def fake_extract(*args):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(
        "backend.scripts.Evaluation.benchmarking.source_intake._run_local_extractor",
        fake_extract,
    )
    with pytest.raises(SourceIntakeError, match="Orphan structured artifact"):
        extract_intake_source(tmp_path, "src-001")
    assert called is False
    assert output_path.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize("object_kind", ["symlink", "directory"])
def test_extraction_rejects_any_existing_non_file_output_object(
    tmp_path, monkeypatch, object_kind
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if object_kind == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text("outside", encoding="utf-8")
        output_path.symlink_to(outside)
    else:
        output_path.mkdir()
    monkeypatch.setattr(
        source_intake,
        "_run_local_extractor",
        lambda *args: pytest.fail("extractor must not run"),
    )
    with pytest.raises(SourceIntakeError, match="Orphan structured artifact"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


@pytest.mark.parametrize("symlink_level", ["structured_root", "acquisition"])
def test_structured_output_symlink_escape_is_rejected(
    tmp_path, monkeypatch, symlink_level
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    structured_root = (
        tmp_path / INTAKE_ROOT_RELATIVE / "structured"
    )
    if symlink_level == "structured_root":
        structured_root.parent.mkdir(parents=True, exist_ok=True)
        structured_root.symlink_to(outside, target_is_directory=True)
    else:
        structured_root.mkdir(parents=True, exist_ok=True)
        output_path.parent.symlink_to(outside, target_is_directory=True)

    called = False

    def fake_extract(*args):
        nonlocal called
        called = True
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", fake_extract)
    with pytest.raises(SourceIntakeError, match="[Uu]nsafe|[Ss]ymlink"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert called is False
    assert not (outside / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_cross_acquisition_structured_symlink_alias_is_rejected(tmp_path, monkeypatch):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    other = tmp_path / INTAKE_ROOT_RELATIVE / "structured" / "src-002"
    other.mkdir(parents=True)
    output_path.parent.symlink_to(other, target_is_directory=True)
    monkeypatch.setattr(
        source_intake,
        "_run_local_extractor",
        lambda *args: pytest.fail("extractor must not run"),
    )
    with pytest.raises(SourceIntakeError, match="[Uu]nsafe|[Ss]ymlink"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not list(other.iterdir())
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


@pytest.mark.parametrize("target_kind", ["outside", "cross_acquisition"])
def test_raw_input_symlink_alias_is_rejected(tmp_path, monkeypatch, target_kind):
    source = _source("src-001", status="hash_verified")
    pdf_bytes = _minimal_pdf_bytes(target_kind)
    if target_kind == "outside":
        target = tmp_path / "outside.pdf"
    else:
        target = (
            tmp_path
            / INTAKE_ROOT_RELATIVE
            / "raw"
            / "src-002"
            / "decision.pdf"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(pdf_bytes)
    raw_path = tmp_path / source["stored_pdf_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.symlink_to(target)
    source["sha256"] = _sha(pdf_bytes)
    _write_manifest(tmp_path, _manifest([source]))
    original_manifest = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    monkeypatch.setattr(
        source_intake,
        "_run_local_extractor",
        lambda *args: pytest.fail("extractor must not run"),
    )
    with pytest.raises(SourceIntakeError, match="[Ss]ymlink"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    output_path = tmp_path / source["structured_path"]
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


def test_authoritative_pdf_validator_accepts_parseable_minimal_pdf(tmp_path):
    payload = _minimal_pdf_bytes("valid")
    path = tmp_path / "valid.pdf"
    path.write_bytes(payload)
    assert validate_pdf_source(path, _sha(payload)) == _sha(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"%PDF-1.7\ngarbage",
        _minimal_pdf_bytes("truncated")[:-80],
        b"",
        b"not a PDF",
    ],
    ids=["pdf-prefix-garbage", "truncated", "zero-byte", "non-pdf"],
)
def test_authoritative_pdf_validator_rejects_invalid_documents(tmp_path, payload):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(payload)
    with pytest.raises(SourceIntakeError, match="signature|parser rejected|page"):
        validate_pdf_source(path, _sha(payload))


def test_pdf_parser_failure_blocks_eligibility(tmp_path, monkeypatch):
    _eligible_fixture(tmp_path)
    monkeypatch.setattr(
        "pdfplumber.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("parser failed")),
    )
    with pytest.raises(SourceIntakeError, match="PDF parser rejected"):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("original_filename", "renamed.pdf"),
        (
            "stored_pdf_path",
            f"{INTAKE_ROOT_RELATIVE.as_posix()}/raw/src-001/renamed.pdf",
        ),
        ("sha256", "c" * 64),
        ("source_url", "https://example.test/judgments/changed.pdf"),
        ("download_timestamp", "2026-10-25T01:00:00+07:00"),
        ("source_id", "independent:changed"),
        ("acquisition_id", "src-002"),
    ],
)
def test_immutable_acquisition_field_drift_is_rejected(field, replacement):
    source = _source(status="downloaded")
    manifest = _manifest([source])
    source[field] = replacement
    with pytest.raises(SourceIntakeError):
        validate_acquisition_manifest(manifest)


def test_offline_provenance_identity_drift_is_rejected():
    source = _source(status="downloaded")
    source["source_url"] = None
    source["offline_provenance"] = {"description": "court archive A"}
    manifest = _manifest([source])
    source["offline_provenance"] = {"description": "court archive B"}
    with pytest.raises(SourceIntakeError, match="immutable provenance"):
        validate_acquisition_manifest(manifest)


def test_internally_consistent_filename_path_and_sha_rewrite_is_detected():
    source = _source(status="downloaded")
    manifest = _manifest([source])
    source["original_filename"] = "renamed.pdf"
    source["stored_pdf_path"] = (
        f"{INTAKE_ROOT_RELATIVE.as_posix()}/raw/src-001/renamed.pdf"
    )
    source["structured_path"] = (
        f"{INTAKE_ROOT_RELATIVE.as_posix()}/structured/src-001/renamed_structured.json"
    )
    source["sha256"] = "c" * 64
    with pytest.raises(SourceIntakeError, match="immutable provenance"):
        validate_acquisition_manifest(manifest)


def test_live_manifest_loader_enforces_immutable_snapshot(tmp_path):
    source = _source(status="downloaded")
    manifest = _manifest([source])
    source["source_url"] = "https://example.test/judgments/rewritten.pdf"
    _write_manifest(tmp_path, manifest)
    with pytest.raises(SourceIntakeError, match="immutable provenance"):
        load_acquisition_manifest(tmp_path)


def test_mutable_legal_metadata_can_progress_without_provenance_drift():
    source = _source(status="downloaded")
    for field in (
        "judgment_number",
        "case_id",
        "court",
        "decision_date",
        "procedure_stage",
        "underlying_dispute_id",
    ):
        source[field] = None
    source["metadata_verification"] = {
        field: False for field in source["metadata_verification"]
    }
    prior = _manifest([source])
    current = copy.deepcopy(prior)
    updated = current["sources"][0]
    updated.update(
        {
            "judgment_number": "01/2026/DS-ST",
            "case_id": "CASE-001",
            "court": "TAND tỉnh Test",
            "decision_date": "2026-01-02",
            "procedure_stage": "so_tham",
            "underlying_dispute_id": "DISPUTE-001",
            "structured_sha256": "b" * 64,
            "acquisition_status": "metadata_verified",
            "status_history": _status_history("metadata_verified"),
        }
    )
    updated["metadata_verification"] = {
        field: True for field in updated["metadata_verification"]
    }
    validate_acquisition_manifest(current, prior_manifest=prior)


def test_extractor_failure_is_transactional_and_retryable(tmp_path, monkeypatch):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    monkeypatch.setattr(source_intake, "_run_local_extractor", lambda *args: False)
    with pytest.raises(SourceIntakeError, match="Local extraction failed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    result = extract_intake_source(tmp_path, source["acquisition_id"])
    assert result["acquisition_status"] == "extracted"
    assert output_path.is_file()


def test_structured_validation_failure_rolls_back(tmp_path, monkeypatch):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)

    def writes_invalid(raw_fd, output_dir_fd, original_filename, force_ocr):
        descriptor = os.open(
            output_path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=output_dir_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("{}")
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", writes_invalid)
    with pytest.raises(SourceIntakeError, match="Structured judgment"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


def test_published_structured_hash_failure_rolls_back(tmp_path, monkeypatch):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    original_hash = source_intake._hash_regular_file_at

    def wrong_published_hash(directory_fd, name):
        if name == output_path.name:
            return "0" * 64
        return original_hash(directory_fd, name)

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    monkeypatch.setattr(source_intake, "_hash_regular_file_at", wrong_published_hash)
    with pytest.raises(SourceIntakeError, match="Published structured SHA256 mismatch"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


@pytest.mark.parametrize("failure_stage", ["validation", "manifest_publication"])
def test_manifest_failure_never_leaves_inconsistent_final_state(
    tmp_path, monkeypatch, failure_stage
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    if failure_stage == "validation":
        original_validate = source_intake.validate_acquisition_manifest

        def fail_updated(manifest, prior_manifest=None):
            if prior_manifest is not None:
                raise SourceIntakeError("injected manifest validation failure")
            return original_validate(manifest)

        monkeypatch.setattr(source_intake, "validate_acquisition_manifest", fail_updated)
        expected = "validation failure"
    else:
        original_atomic_write = source_intake._atomic_write_bytes_at

        def fail_manifest_publication(directory_fd, name, payload):
            if name == ACQUISITION_MANIFEST_RELATIVE.name:
                raise OSError("injected publication failure")
            return original_atomic_write(directory_fd, name, payload)

        monkeypatch.setattr(
            source_intake, "_atomic_write_bytes_at", fail_manifest_publication
        )
        expected = "publication failure"
    with pytest.raises((SourceIntakeError, OSError), match=expected):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


def test_successful_extraction_publishes_one_final_with_matching_hash(
    tmp_path, monkeypatch
):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    result = extract_intake_source(tmp_path, source["acquisition_id"])
    manifest = load_acquisition_manifest(tmp_path)
    updated = manifest["sources"][0]
    assert output_path.is_file() and not output_path.is_symlink()
    assert [path.name for path in output_path.parent.iterdir()] == [output_path.name]
    assert _sha(output_path.read_bytes()) == updated["structured_sha256"]
    assert result["structured_sha256"] == updated["structured_sha256"]
    assert updated["acquisition_status"] == "extracted"


def test_pdf_tamper_before_extraction_is_rejected(tmp_path, monkeypatch):
    source, pdf_path, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    pdf_path.write_bytes(_minimal_pdf_bytes("changed-before"))
    monkeypatch.setattr(
        source_intake,
        "_run_local_extractor",
        lambda *args: pytest.fail("extractor must not run"),
    )
    with pytest.raises(SourceIntakeError, match="PDF SHA256 mismatch"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


def test_pdf_tamper_during_extraction_is_rejected(tmp_path, monkeypatch):
    source, pdf_path, output_path, original_manifest = _hash_verified_fixture(tmp_path)

    def tampers_input(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        pdf_path.write_bytes(_minimal_pdf_bytes("changed-during"))
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", tampers_input)
    with pytest.raises(SourceIntakeError, match="PDF SHA256 mismatch"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    _assert_failed_extraction_is_clean(tmp_path, output_path, original_manifest)


def test_manifest_provenance_mutation_during_extraction_aborts_publication(
    tmp_path, monkeypatch
):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE

    def mutates_manifest(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sources"][0]["source_url"] = (
            "https://example.test/judgments/concurrent-change.pdf"
        )
        _write_manifest(tmp_path, manifest)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", mutates_manifest)
    with pytest.raises(SourceIntakeError, match="manifest changed during extraction"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not output_path.exists()
    assert not list(output_path.parent.glob(".extract-partial-*"))
    with pytest.raises(SourceIntakeError, match="immutable provenance"):
        load_acquisition_manifest(tmp_path)


@pytest.mark.parametrize("replacement", ["outside_symlink", "cross_acquisition"])
def test_parent_replacement_immediately_before_publication_cannot_escape_fd(
    tmp_path, monkeypatch, replacement
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    outside = tmp_path / "outside"
    if replacement == "cross_acquisition":
        outside = tmp_path / INTAKE_ROOT_RELATIVE / "structured" / "src-002"
    outside.mkdir(parents=True)
    displaced = tmp_path / "displaced-src-001"

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    def swap_at_checkpoint(name):
        if name == "before_artifact_publication":
            output_path.parent.rename(displaced)
            output_path.parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_at_checkpoint)
    with pytest.raises(
        SourceIntakeError, match="directory identity changed|not a directory"
    ):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (outside / output_path.name).exists()
    assert not (displaced / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_acquisition_directory_swap_after_fd_open_never_publishes_manifest(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    displaced = tmp_path / "detached-acquisition"

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    def swap_after_artifact(name):
        if name == "after_artifact_published":
            output_path.parent.rename(displaced)
            output_path.parent.mkdir()

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_after_artifact)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not output_path.exists()
    assert not (displaced / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_parent_swap_inside_artifact_link_still_targets_held_fd(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    displaced = tmp_path / "fd-held-acquisition"
    original_link = source_intake.os.link

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    def swap_during_link(source_name, destination_name, **kwargs):
        if (
            source_name == output_path.name
            and destination_name == output_path.name
            and kwargs.get("src_dir_fd") != kwargs.get("dst_dir_fd")
        ):
            output_path.parent.rename(displaced)
            output_path.parent.symlink_to(outside, target_is_directory=True)
        return original_link(source_name, destination_name, **kwargs)

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    monkeypatch.setattr(source_intake.os, "link", swap_during_link)
    with pytest.raises(
        SourceIntakeError, match="directory identity changed|not a directory"
    ):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (outside / output_path.name).exists()
    assert not (displaced / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_final_symlink_substitution_immediately_before_publication_is_rejected(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")

    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    def substitute_final(name):
        if name == "before_artifact_publication":
            output_path.symlink_to(outside)

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", substitute_final)
    with pytest.raises(SourceIntakeError, match="Refusing existing structured output"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert outside.read_text(encoding="utf-8") == "unchanged"
    assert output_path.is_symlink()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_structured_ancestor_moved_outside_and_symlinked_back_is_rejected(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    structured_root = output_path.parent.parent
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = outside / "structured"

    def swap_ancestor(name):
        if name == "before_artifact_publication":
            structured_root.rename(moved)
            structured_root.symlink_to(moved, target_is_directory=True)

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_ancestor)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (moved / source["acquisition_id"] / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


@pytest.mark.parametrize("replacement", ["real_directory", "recreated_path"])
def test_structured_ancestor_real_directory_replacement_is_rejected(
    tmp_path, monkeypatch, replacement
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    structured_root = output_path.parent.parent
    moved = tmp_path / f"moved-structured-{replacement}"
    original_leaf_identity = []

    def swap_ancestor(name):
        if name == "before_artifact_publication":
            metadata = output_path.parent.stat()
            original_leaf_identity.append((metadata.st_dev, metadata.st_ino))
            structured_root.rename(moved)
            structured_root.mkdir()

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_ancestor)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    moved_leaf = moved / source["acquisition_id"]
    assert (moved_leaf.stat().st_dev, moved_leaf.stat().st_ino) == original_leaf_identity[0]
    assert not (moved_leaf / output_path.name).exists()
    assert not (structured_root / source["acquisition_id"] / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_independent_intake_root_namespace_replacement_is_rejected(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    intake_root = tmp_path / INTAKE_ROOT_RELATIVE
    moved_intake = tmp_path / "moved-independent-intake"

    def swap_intake_root(name):
        if name == "before_artifact_publication":
            intake_root.rename(moved_intake)
            intake_root.mkdir()

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_intake_root)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    moved_output = moved_intake / "structured" / source["acquisition_id"] / output_path.name
    assert not moved_output.exists()
    assert (
        moved_intake.joinpath("acquisition_manifest.json").read_bytes()
        == original_manifest
    )


def test_structured_and_acquisition_subtree_move_together_is_rejected(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    structured_root = output_path.parent.parent
    moved = tmp_path / "moved-whole-structured-subtree"

    def move_whole_subtree(name):
        if name == "before_artifact_publication":
            structured_root.rename(moved)
            structured_root.symlink_to(moved, target_is_directory=True)

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", move_whole_subtree)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (moved / source["acquisition_id"] / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_cross_acquisition_structured_subtree_substitution_is_rejected(
    tmp_path, monkeypatch
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    structured_root = output_path.parent.parent
    moved = tmp_path / "original-structured-subtree"
    replacement = tmp_path / "cross-acquisition-structured-subtree"
    replacement_leaf = replacement / source["acquisition_id"]
    replacement_leaf.mkdir(parents=True)

    def substitute_subtree(name):
        if name == "before_artifact_publication":
            structured_root.rename(moved)
            replacement.rename(structured_root)

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", substitute_subtree)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (replacement_leaf / output_path.name).exists()
    assert not (moved / source["acquisition_id"] / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


@pytest.mark.parametrize(
    "checkpoint",
    ["after_prepared", "after_artifact_published", "after_manifest_published"],
)
def test_structured_ancestor_swap_rolls_back_transaction_and_allows_retry(
    tmp_path, monkeypatch, checkpoint
):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    structured_root = output_path.parent.parent
    moved = tmp_path / f"moved-at-{checkpoint}"

    def swap_ancestor(name):
        if name == checkpoint:
            structured_root.rename(moved)
            structured_root.symlink_to(moved, target_is_directory=True)

    _install_successful_extractor(monkeypatch)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", swap_ancestor)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not (moved / source["acquisition_id"] / output_path.name).exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest
    assert list((tmp_path / TRANSACTION_ROOT_RELATIVE).glob("*.json")) == []

    structured_root.unlink()
    moved.rename(structured_root)
    monkeypatch.setattr(source_intake, "_publication_checkpoint", lambda name: None)
    result = extract_intake_source(tmp_path, source["acquisition_id"])
    assert result["acquisition_status"] == "extracted"


def test_raw_ancestor_move_and_symlink_back_is_rejected(tmp_path, monkeypatch):
    source, _, output_path, original_manifest = _hash_verified_fixture(tmp_path)
    raw_path = tmp_path / source["stored_pdf_path"]
    raw_root = raw_path.parent.parent
    moved = tmp_path / "moved-raw-subtree"

    def succeeds_after_raw_swap(raw_fd, output_dir_fd, original_filename, force_ocr):
        raw_root.rename(moved)
        raw_root.symlink_to(moved, target_is_directory=True)
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds_after_raw_swap)
    with pytest.raises(SourceIntakeError, match="directory identity changed"):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert not output_path.exists()
    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == original_manifest


def test_create_once_external_anchor_and_overwrite_rejection(tmp_path):
    source = _source(status="downloaded")
    (tmp_path / INTAKE_ROOT_RELATIVE).mkdir(parents=True)
    anchor_path = create_provenance_anchor(tmp_path, source)
    original = anchor_path.read_bytes()
    assert anchor_path.is_file() and not anchor_path.is_symlink()
    assert len(list(anchor_path.parent.glob("src-001.json"))) == 1
    with pytest.raises(SourceIntakeError, match="create-once"):
        create_provenance_anchor(tmp_path, source)
    assert anchor_path.read_bytes() == original


@pytest.mark.parametrize(
    "mutations",
    [
        {
            "original_filename": "renamed.pdf",
            "stored_pdf_path": (
                f"{INTAKE_ROOT_RELATIVE.as_posix()}/raw/src-001/renamed.pdf"
            ),
            "structured_path": (
                f"{INTAKE_ROOT_RELATIVE.as_posix()}/structured/src-001/renamed_structured.json"
            ),
        },
        {
            "stored_pdf_path": (
                f"{INTAKE_ROOT_RELATIVE.as_posix()}/raw/src-001/other.pdf"
            )
        },
        {"sha256": "c" * 64},
        {
            "source_url": "https://changed.test/judgment.pdf",
            "source_domain": "changed.test",
            "download_timestamp": "2026-10-25T01:00:00+07:00",
        },
    ],
    ids=["filename", "stored-path", "pdf-sha", "url-domain-timestamp"],
)
def test_coordinated_manifest_snapshot_rewrite_is_rejected_by_anchor(
    tmp_path, mutations
):
    source = _source(status="downloaded")
    _write_manifest(tmp_path, _manifest([source]))
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0].update(mutations)
    if "download_timestamp" in mutations:
        manifest["sources"][0]["status_history"][0]["timestamp"] = mutations[
            "download_timestamp"
        ]
    manifest["immutable_provenance"] = [
        immutable_provenance_snapshot(manifest["sources"][0])
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SourceIntakeError, match="differs from anchor"):
        load_acquisition_manifest(tmp_path)


def test_missing_or_symlinked_anchor_fails_authoritative_load(tmp_path):
    source = _source(status="downloaded")
    _write_manifest(tmp_path, _manifest([source]))
    anchor = tmp_path / PROVENANCE_ROOT_RELATIVE / "src-001.json"
    payload = anchor.read_bytes()
    anchor.unlink()
    with pytest.raises(SourceIntakeError, match="exactly one anchor"):
        load_acquisition_manifest(tmp_path)
    outside = tmp_path / "outside-anchor.json"
    outside.write_bytes(payload)
    anchor.symlink_to(outside)
    with pytest.raises(SourceIntakeError, match="regular intake file"):
        load_acquisition_manifest(tmp_path)


def test_live_legal_metadata_progress_does_not_modify_anchor(tmp_path):
    source = _source(status="downloaded")
    source["judgment_number"] = None
    source["case_id"] = None
    source["court"] = None
    source["decision_date"] = None
    source["procedure_stage"] = None
    source["underlying_dispute_id"] = None
    _write_manifest(tmp_path, _manifest([source]))
    anchor = tmp_path / PROVENANCE_ROOT_RELATIVE / "src-001.json"
    original_anchor = anchor.read_bytes()
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0].update(
        {
            "judgment_number": "01/2026/DS-ST",
            "case_id": "CASE-001",
            "court": "TAND tỉnh Test",
            "decision_date": "2026-01-02",
            "procedure_stage": "so_tham",
            "underlying_dispute_id": "DISPUTE-001",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_acquisition_manifest(tmp_path)
    assert loaded["sources"][0]["court"] == "TAND tỉnh Test"
    assert anchor.read_bytes() == original_anchor


class _SimulatedCrash(BaseException):
    pass


def _install_successful_extractor(monkeypatch):
    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)


def _crash_at(monkeypatch, expected_checkpoint):
    def checkpoint(name):
        if name == expected_checkpoint:
            raise _SimulatedCrash(name)

    monkeypatch.setattr(source_intake, "_publication_checkpoint", checkpoint)


def test_prepared_crash_is_cleaned_deterministically_on_next_load(
    tmp_path, monkeypatch
):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    _crash_at(monkeypatch, "after_prepared")
    with pytest.raises(_SimulatedCrash):
        extract_intake_source(tmp_path, source["acquisition_id"])
    journal = tmp_path / TRANSACTION_ROOT_RELATIVE / "src-001.json"
    assert journal.is_file() and not output_path.exists()
    loaded = load_acquisition_manifest(tmp_path)
    assert loaded["sources"][0]["acquisition_status"] == "hash_verified"
    assert not journal.exists()


def test_artifact_published_crash_rolls_back_and_allows_retry(tmp_path, monkeypatch):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    _crash_at(monkeypatch, "after_artifact_published")
    with pytest.raises(_SimulatedCrash):
        extract_intake_source(tmp_path, source["acquisition_id"])
    assert output_path.is_file()
    loaded = load_acquisition_manifest(tmp_path)
    assert loaded["sources"][0]["acquisition_status"] == "hash_verified"
    assert not output_path.exists()
    monkeypatch.setattr(source_intake, "_publication_checkpoint", lambda name: None)
    result = extract_intake_source(tmp_path, source["acquisition_id"])
    assert result["acquisition_status"] == "extracted"


def test_manifest_published_crash_is_recognized_as_consistent_completion(
    tmp_path, monkeypatch
):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    _crash_at(monkeypatch, "after_manifest_published")
    with pytest.raises(_SimulatedCrash):
        extract_intake_source(tmp_path, source["acquisition_id"])
    journal = tmp_path / TRANSACTION_ROOT_RELATIVE / "src-001.json"
    assert output_path.is_file() and journal.is_file()
    loaded = load_acquisition_manifest(tmp_path)
    assert loaded["sources"][0]["acquisition_status"] == "extracted"
    assert not journal.exists()


def test_normal_success_leaves_no_pending_journal(tmp_path, monkeypatch):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    result = extract_intake_source(tmp_path, source["acquisition_id"])
    assert output_path.is_file()
    assert result["structured_sha256"] == _sha(output_path.read_bytes())
    assert list((tmp_path / TRANSACTION_ROOT_RELATIVE).glob("*.json")) == []


@pytest.mark.parametrize("tamper", ["missing", "changed"])
def test_extracted_manifest_with_missing_or_changed_artifact_fails_load(
    tmp_path, monkeypatch, tamper
):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    extract_intake_source(tmp_path, source["acquisition_id"])
    if tamper == "missing":
        output_path.unlink()
    else:
        output_path.write_bytes(output_path.read_bytes() + b"changed")
    with pytest.raises(
        SourceIntakeError, match="missing|invalid|structured JSON|SHA256"
    ):
        load_acquisition_manifest(tmp_path)


def test_hash_verified_orphan_without_journal_fails_closed(tmp_path):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_structured(source["original_filename"])), encoding="utf-8"
    )
    with pytest.raises(SourceIntakeError, match="Orphan structured artifact"):
        load_acquisition_manifest(tmp_path)


@pytest.mark.parametrize(
    "tamper",
    [
        "corrupt",
        "wrong_acquisition",
        "structured_hash",
        "previous_manifest_hash",
        "unsafe_temp_path",
    ],
)
def test_corrupt_or_inconsistent_journal_fails_closed(tmp_path, monkeypatch, tamper):
    source, _, output_path, _ = _hash_verified_fixture(tmp_path)
    _install_successful_extractor(monkeypatch)
    _crash_at(monkeypatch, "after_artifact_published")
    with pytest.raises(_SimulatedCrash):
        extract_intake_source(tmp_path, source["acquisition_id"])
    journal_path = tmp_path / TRANSACTION_ROOT_RELATIVE / "src-001.json"
    if tamper == "corrupt":
        journal_path.write_text("{", encoding="utf-8")
    else:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if tamper == "wrong_acquisition":
            journal["acquisition_id"] = "src-002"
        elif tamper == "structured_hash":
            journal["structured_sha256"] = "0" * 64
        elif tamper == "unsafe_temp_path":
            journal["temp_directory_name"] = "../src-002"
        else:
            journal["previous_manifest_sha256"] = "0" * 64
        journal = source_intake._journal_payload(journal)
        journal_path.write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(SourceIntakeError):
        load_acquisition_manifest(tmp_path)
    assert output_path.exists()


@pytest.mark.parametrize("artifact", ["pdf", "structured"])
def test_eligible_artifact_tamper_fails_next_authoritative_inventory(tmp_path, artifact):
    source, _ = _eligible_fixture(tmp_path)
    path = tmp_path / source[
        "stored_pdf_path" if artifact == "pdf" else "structured_path"
    ]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(SourceIntakeError, match="SHA256|sha256|structured JSON"):
        inventory_independent_sources(tmp_path)


@pytest.mark.parametrize(
    "history,current_status",
    [
        (["downloaded", "extracted"], "extracted"),
        (["downloaded", "rejected", "eligible"], "eligible"),
        (["downloaded", "quarantined", "eligible"], "eligible"),
    ],
)
def test_invalid_status_order_and_terminal_recovery_are_rejected(history, current_status):
    source = _source(status="downloaded")
    source["acquisition_status"] = current_status
    source["structured_sha256"] = "b" * 64
    source["status_history"] = [
        {"status": status, "timestamp": FIXED_TIME, "reasons": []}
        for status in history
    ]
    with pytest.raises(SourceIntakeError, match="ordered status history"):
        validate_acquisition_manifest(_manifest([source]))


def test_decreasing_status_timestamps_are_rejected():
    source = _source(status="hash_verified")
    source["status_history"][0]["timestamp"] = "2026-10-25T02:00:00+07:00"
    source["status_history"][1]["timestamp"] = "2026-10-25T01:00:00+07:00"
    source["download_timestamp"] = "2026-10-25T00:00:00+07:00"
    with pytest.raises(SourceIntakeError, match="chronological"):
        validate_acquisition_manifest(_manifest([source]))


def test_forged_eligible_manifest_without_artifacts_is_rejected(tmp_path):
    _write_manifest(tmp_path, _manifest([_source(status="eligible")]))
    with pytest.raises(SourceIntakeError, match="Structured artifact missing"):
        inventory_independent_sources(tmp_path)


def test_intake_ocr_adapter_is_explicit_scoped_and_non_overwriting(
    tmp_path, monkeypatch
):
    calls = []

    def fake_process(
        input_path,
        output_dir,
        *,
        force_ocr,
        overwrite,
        output_dir_fd,
        original_filename,
    ):
        calls.append(
            (
                input_path,
                output_dir,
                force_ocr,
                overwrite,
                output_dir_fd,
                original_filename,
            )
        )
        return True

    fake_module = types.ModuleType("backend.scripts.ocrFullBanAn")
    fake_module.process_judgment_file = fake_process
    monkeypatch.setitem(sys.modules, "backend.scripts.ocrFullBanAn", fake_module)
    monkeypatch.setattr(
        sys.modules["backend.scripts"],
        "ocrFullBanAn",
        fake_module,
        raising=False,
    )
    raw_path = tmp_path / "decision.pdf"
    output_dir = tmp_path / "structured"
    raw_path.write_bytes(b"test")
    output_dir.mkdir()
    raw_fd = os.open(raw_path, os.O_RDONLY)
    output_fd = os.open(output_dir, os.O_RDONLY)
    try:
        assert (
            source_intake._run_local_extractor(
                raw_fd, output_fd, "decision.pdf", False
            )
            is True
        )
        assert calls == [
            (
                Path(f"/dev/fd/{raw_fd}"),
                Path("."),
                False,
                False,
                output_fd,
                "decision.pdf",
            )
        ]
    finally:
        os.close(output_fd)
        os.close(raw_fd)


def test_structured_json_requires_nfc():
    payload = _structured("decision.pdf")
    payload["pages"][0]["text"] = unicodedata.normalize("NFD", "Tòa án")
    with pytest.raises(SourceIntakeError, match="NFC"):
        validate_structured_judgment(payload, "decision.pdf")


def test_historical_inventory_and_fingerprint_remain_unchanged():
    inventory = inventory_judgment_sources(PROJECT_ROOT)
    assert len(inventory) == 107
    assert canonical_json_fingerprint(inventory) == (
        "fd6a6f05e4b71a2c499584fc0296cf25cedf5eac958aea6d079bf449c864f754"
    )


def test_current_independent_pool_has_63_eligible_sources_and_benchmark_is_draft():
    inventory, report, links = inventory_independent_sources(PROJECT_ROOT)
    benchmark_manifest = json.loads(
        (BENCHMARK_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    blocked_numbers = {3, 8, 9, 12, 20, 22, 23, 42, 46, 47, 50}
    assert [record["source_id"] for record in inventory] == [
        f"independent:pilot-20261025-{index:03d}" for index in range(1, 25)
    ] + [
        f"independent:scale-20261026-{index:03d}"
        for index in range(1, 51)
        if index not in blocked_numbers
    ]
    assert report["eligible_source_count"] == 63
    assert links == []
    assert benchmark_manifest["state"] == "DRAFT"
    assert benchmark_manifest["current_counts"]["selected_cases"] == 0
    assert benchmark_manifest["current_counts"]["query_records"] == 0
    assert benchmark_manifest["current_counts"]["ground_truth_records"] == 0


def test_explicit_blank_page_is_visible_without_automatic_eligibility(tmp_path):
    source = _source("src-001", "decision.pdf", status="extracted")
    payload = {
        "judgment_name": "decision.pdf",
        "page_count": 2,
        "pages": [
            {
                "page": 1,
                "text": "Nội dung bản án hợp lệ",
                "extraction_status": "native_text",
            },
            {
                "page": 2,
                "text": "",
                "extraction_status": "empty_after_native_and_ocr",
            },
        ],
        "sections": {
            "header": "Nội dung bản án hợp lệ",
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }
    _materialize(tmp_path, source, structured_payload=payload)
    _write_manifest(tmp_path, _manifest([source]))

    inventory, report, _ = inventory_independent_sources(tmp_path)

    assert inventory == []
    assert report["eligible_source_count"] == 0
    assert report["sources"] == [
        {
            "acquisition_id": "src-001",
            "source_id": "independent:src-001",
            "claimed_status": "extracted",
            "eligible": False,
            "gate_failures": [],
            "blank_page_numbers": [2],
        }
    ]


def test_promote_extracted_source_to_eligible_is_atomic_and_artifact_blind(
    tmp_path,
):
    source, _ = _extracted_fixture(tmp_path)
    second = _source("src-002", "other.pdf", status="hash_verified")
    second_pdf = _minimal_pdf_bytes("src-002")
    second_pdf_path = tmp_path / second["stored_pdf_path"]
    second_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    second_pdf_path.write_bytes(second_pdf)
    second["sha256"] = _sha(second_pdf)
    manifest = _manifest([source, second])
    _write_manifest(tmp_path, manifest)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    raw_path = tmp_path / source["stored_pdf_path"]
    structured_path = tmp_path / source["structured_path"]
    anchor_path = (
        tmp_path / PROVENANCE_ROOT_RELATIVE / f"{source['acquisition_id']}.json"
    )
    before_history = copy.deepcopy(source["status_history"])
    protected = {
        "raw": raw_path.read_bytes(),
        "structured": structured_path.read_bytes(),
        "anchor": anchor_path.read_bytes(),
    }
    before_inventory, before_report, _ = inventory_independent_sources(tmp_path)
    assert source["source_id"] not in {
        item["source_id"] for item in before_inventory
    }
    assert not next(
        item
        for item in before_report["sources"]
        if item["acquisition_id"] == source["acquisition_id"]
    )["eligible"]

    result = promote_source_to_eligible(tmp_path, source["acquisition_id"])

    checked = load_acquisition_manifest(tmp_path)
    promoted = next(
        item for item in checked["sources"] if item["acquisition_id"] == "src-001"
    )
    untouched = next(
        item for item in checked["sources"] if item["acquisition_id"] == "src-002"
    )
    assert result["acquisition_status"] == "eligible"
    assert promoted["acquisition_status"] == "eligible"
    assert promoted["status_history"][:-1] == before_history
    assert promoted["status_history"][-1]["status"] == "eligible"
    assert promoted["status_history"][-1]["reasons"] == []
    assert len(promoted["status_history"]) == len(before_history) + 1
    assert untouched == second
    assert raw_path.read_bytes() == protected["raw"]
    assert structured_path.read_bytes() == protected["structured"]
    assert anchor_path.read_bytes() == protected["anchor"]
    assert manifest_path.is_file() and not manifest_path.is_symlink()
    inventory, report, _ = inventory_independent_sources(tmp_path)
    assert [item["source_id"] for item in inventory] == [source["source_id"]]
    assert next(
        item for item in report["sources"] if item["acquisition_id"] == "src-001"
    )["eligible"]


@pytest.mark.parametrize(
    "status",
    ["downloaded", "hash_verified", "rejected", "quarantined", "eligible"],
)
def test_promote_rejects_every_non_extracted_status(tmp_path, status):
    if status == "eligible":
        source, _ = _eligible_fixture(tmp_path)
    elif status == "hash_verified":
        source, _, _, _ = _hash_verified_fixture(tmp_path)
    elif status == "downloaded":
        source = _source(status="downloaded")
        pdf = _minimal_pdf_bytes("downloaded")
        path = tmp_path / source["stored_pdf_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf)
        source["sha256"] = _sha(pdf)
        _write_manifest(tmp_path, _manifest([source]))
    else:
        source, _ = _extracted_fixture(tmp_path)
        source["acquisition_status"] = status
        source["status_history"].append({
            "status": status,
            "timestamp": FIXED_TIME,
            "reasons": ["terminal review"],
        })
        _write_manifest(tmp_path, _manifest([source]))
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()

    with pytest.raises(
        SourceIntakeError, match="requires extracted status"
    ):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


def test_promote_rejects_missing_source(tmp_path):
    _extracted_fixture(tmp_path)
    with pytest.raises(SourceIntakeError, match="Unknown acquisition_id"):
        promote_source_to_eligible(tmp_path, "src-missing")


def test_promote_rejects_development_overlap(tmp_path):
    source, _ = _extracted_fixture(tmp_path)
    development_path = tmp_path / "backend/data/judgments_evaluation.json"
    development_path.write_text(
        json.dumps([{"id": 1, "judgment_name": source["original_filename"]}]),
        encoding="utf-8",
    )
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()

    with pytest.raises(
        SourceIntakeError, match="development_benchmark_exact_source"
    ):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


def test_promote_rejects_noncanonical_exact_duplicate(tmp_path):
    left, _ = _extracted_fixture(tmp_path, "src-001", "left.pdf")
    right = _source("src-002", "right.pdf", status="extracted")
    right["case_id"] = "CASE-src-002"
    right["judgment_number"] = "JUDGMENT-src-002"
    _materialize(
        tmp_path,
        right,
        pdf_bytes=(tmp_path / left["stored_pdf_path"]).read_bytes(),
        structured_payload=_structured("right.pdf", "Nội dung khác hợp lệ"),
    )
    _write_manifest(tmp_path, _manifest([left, right]))
    before = (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes()

    with pytest.raises(SourceIntakeError, match="new_pool_duplicate_sha256"):
        promote_source_to_eligible(tmp_path, right["acquisition_id"])

    assert (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_bytes() == before


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("metadata", "unverified_court"),
        ("development_review", "incomplete_development_overlap_check"),
        ("dedup_review", "incomplete_internal_dedup_check"),
        ("underlying_review", "incomplete_underlying_dispute_review"),
        ("rejection_reason", "unresolved_rejection_reasons"),
    ],
)
def test_promote_rejects_incomplete_eligibility_contract(
    tmp_path, mutation, expected
):
    source, manifest = _extracted_fixture(tmp_path)
    if mutation == "metadata":
        source["metadata_verification"]["court"] = False
    elif mutation == "rejection_reason":
        source["rejection_reasons"] = ["manual review unresolved"]
    else:
        field = {
            "development_review": "development_overlap_check",
            "dedup_review": "internal_dedup_check",
            "underlying_review": "underlying_dispute_review",
        }[mutation]
        source["metadata_verification"][field] = False
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SourceIntakeError, match=expected):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


def test_promote_rejects_coordinated_provenance_drift(tmp_path):
    source, manifest = _extracted_fixture(tmp_path)
    source["source_url"] = "https://example.test/judgments/rewritten.pdf"
    manifest["immutable_provenance"] = [immutable_provenance_snapshot(source)]
    (tmp_path / ACQUISITION_MANIFEST_RELATIVE).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(SourceIntakeError, match="differs from anchor"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


def test_promote_rejects_raw_pdf_drift(tmp_path):
    source, _ = _extracted_fixture(tmp_path)
    raw_path = tmp_path / source["stored_pdf_path"]
    raw_path.write_bytes(_minimal_pdf_bytes("tampered"))

    with pytest.raises(SourceIntakeError, match="PDF SHA256 mismatch"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("drift", "valid structured JSON|Structured artifact"),
        ("missing", "Structured artifact"),
    ],
)
def test_promote_rejects_structured_artifact_failure(
    tmp_path, mutation, expected
):
    source, _ = _extracted_fixture(tmp_path)
    structured_path = tmp_path / source["structured_path"]
    if mutation == "drift":
        structured_path.write_bytes(structured_path.read_bytes() + b"tampered")
    else:
        structured_path.unlink()

    with pytest.raises(SourceIntakeError, match=expected):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


def test_promote_rejects_invalid_blank_page_artifact(tmp_path):
    payload = {
        "judgment_name": "decision.pdf",
        "page_count": 2,
        "pages": [
            {"page": 1, "text": "Nội dung hợp lệ", "extraction_status": "native_text"},
            {
                "page": 2,
                "text": "Không được có nội dung",
                "extraction_status": "empty_after_native_and_ocr",
            },
        ],
        "sections": {
            "header": "Nội dung hợp lệ",
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }
    source, _ = _extracted_fixture(
        tmp_path, structured_payload=_structured("decision.pdf")
    )
    structured_path = tmp_path / source["structured_path"]
    structured_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = json.loads(
        (tmp_path / ACQUISITION_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )
    manifest["sources"][0]["structured_sha256"] = _sha(structured_path.read_bytes())
    _write_manifest(tmp_path, manifest)

    with pytest.raises(SourceIntakeError, match="empty extraction status"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])


def test_promote_accepts_valid_explicit_blank_page_without_rewriting_it(tmp_path):
    payload = {
        "judgment_name": "decision.pdf",
        "page_count": 2,
        "pages": [
            {"page": 1, "text": "Nội dung hợp lệ", "extraction_status": "native_text"},
            {
                "page": 2,
                "text": "",
                "extraction_status": "empty_after_native_and_ocr",
            },
        ],
        "sections": {
            "header": "Nội dung hợp lệ",
            "facts": "",
            "reasoning": "",
            "decision": "",
        },
    }
    source, _ = _extracted_fixture(tmp_path, structured_payload=payload)
    structured_path = tmp_path / source["structured_path"]
    before = structured_path.read_bytes()

    promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert structured_path.read_bytes() == before
    _, report, _ = inventory_independent_sources(tmp_path)
    assert report["sources"][0]["blank_page_numbers"] == [2]


def test_promote_publication_failure_preserves_old_manifest(tmp_path, monkeypatch):
    source, _ = _extracted_fixture(tmp_path)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()

    def fail_publication(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(source_intake, "_atomic_write_bytes_at", fail_publication)
    with pytest.raises(OSError, match="simulated publication failure"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert manifest_path.read_bytes() == before
    assert load_acquisition_manifest(tmp_path)["sources"][0][
        "acquisition_status"
    ] == "extracted"


def test_promote_failure_after_atomic_replace_rolls_back_manifest(
    tmp_path, monkeypatch
):
    source, _ = _extracted_fixture(tmp_path)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    real_atomic_write = source_intake._atomic_write_bytes_at
    calls = 0

    def replace_then_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        real_atomic_write(*args, **kwargs)
        if calls == 1:
            raise OSError("simulated post-replace failure")

    monkeypatch.setattr(
        source_intake, "_atomic_write_bytes_at", replace_then_fail
    )
    with pytest.raises(SourceIntakeError, match="previous manifest restored"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert calls == 2
    assert manifest_path.read_bytes() == before
    assert load_acquisition_manifest(tmp_path)["sources"][0][
        "acquisition_status"
    ] == "extracted"


def test_promote_manifest_race_fails_without_lost_update(tmp_path, monkeypatch):
    source, _ = _extracted_fixture(tmp_path)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE

    def concurrent_update(checkpoint):
        if checkpoint != "before_manifest_compare":
            return
        concurrent = json.loads(manifest_path.read_text(encoding="utf-8"))
        concurrent["sources"][0]["notes"] = "concurrent update retained"
        manifest_path.write_text(
            json.dumps(concurrent, ensure_ascii=False), encoding="utf-8"
        )

    monkeypatch.setattr(
        source_intake, "_eligibility_publication_checkpoint", concurrent_update
    )
    with pytest.raises(SourceIntakeError, match="changed during eligibility"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    checked = load_acquisition_manifest(tmp_path)["sources"][0]
    assert checked["acquisition_status"] == "extracted"
    assert checked["notes"] == "concurrent update retained"


def test_promote_postpublication_failure_rolls_back_manifest(tmp_path, monkeypatch):
    source, _ = _extracted_fixture(tmp_path)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()

    def fail_postpublication(*args, **kwargs):
        raise SourceIntakeError("simulated post-publication verification failure")

    monkeypatch.setattr(
        source_intake, "inventory_independent_sources", fail_postpublication
    )
    with pytest.raises(SourceIntakeError, match="previous manifest restored"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])

    assert manifest_path.read_bytes() == before
    assert load_acquisition_manifest(tmp_path)["sources"][0][
        "acquisition_status"
    ] == "extracted"


def test_promote_cli_delegates_without_force_or_bypass(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_promote(root, acquisition_id):
        calls.append((Path(root), acquisition_id))
        return {
            "acquisition_id": acquisition_id,
            "source_id": f"independent:{acquisition_id}",
            "acquisition_status": "eligible",
            "status_history_length": 4,
        }

    monkeypatch.setattr(source_intake, "promote_source_to_eligible", fake_promote)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "promote-eligible",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-001",
        ],
    )
    source_intake.main()
    assert calls == [(tmp_path, "src-001")]
    assert json.loads(capsys.readouterr().out)["acquisition_status"] == "eligible"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "promote-eligible",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-001",
            "--force-ocr",
        ],
    )
    with pytest.raises(SystemExit):
        source_intake.main()
    assert calls == [(tmp_path, "src-001")]
