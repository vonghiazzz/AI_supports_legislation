import hashlib
import json
import sys
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking.source_intake import (
    ACQUISITION_MANIFEST_RELATIVE,
    INTAKE_ROOT_RELATIVE,
    MANIFEST_VERSION,
    PROVENANCE_ROOT_RELATIVE,
    TRANSACTION_ROOT_RELATIVE,
    SourceIntakeError,
    inventory_independent_sources,
    load_acquisition_manifest,
    register_source,
)


DOWNLOAD_TIME = "2026-01-02T03:04:05+07:00"
ACQUISITION_ID = "scale-20261026-001"
SOURCE_URL = "https://congbobanan.toaan.gov.vn/example/decision.pdf"
SOURCE_DOMAIN = "congbobanan.toaan.gov.vn"
ORIGINAL_FILENAME = "decision.pdf"


class _SimulatedCrash(BaseException):
    pass


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _minimal_pdf_bytes(label: str = "registration") -> bytes:
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


def _canonical_bytes(value: dict) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _empty_manifest() -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": DOWNLOAD_TIME,
        "sources": [],
        "immutable_provenance": [],
        "known_duplicate_links": [],
    }


@pytest.fixture
def registration_root(tmp_path):
    for relative in (
        INTAKE_ROOT_RELATIVE,
        INTAKE_ROOT_RELATIVE / "raw",
        INTAKE_ROOT_RELATIVE / "structured",
        PROVENANCE_ROOT_RELATIVE,
        TRANSACTION_ROOT_RELATIVE,
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    manifest_path.write_bytes(_canonical_bytes(_empty_manifest()))
    pdf_path = tmp_path / "candidate.pdf"
    pdf_path.write_bytes(_minimal_pdf_bytes())
    benchmark_path = tmp_path / "benchmark-sentinel.json"
    benchmark_path.write_text('{"state":"DRAFT"}\n', encoding="utf-8")
    return tmp_path, pdf_path, manifest_path, benchmark_path


def _register(root: Path, pdf_path: Path, **overrides):
    values = {
        "project_root": root,
        "acquisition_id": ACQUISITION_ID,
        "input_pdf": pdf_path,
        "source_url": SOURCE_URL,
        "source_domain": SOURCE_DOMAIN,
        "download_timestamp": DOWNLOAD_TIME,
        "original_filename": ORIGINAL_FILENAME,
    }
    values.update(overrides)
    return register_source(**values)


def _paths(root: Path):
    raw = root / INTAKE_ROOT_RELATIVE / "raw" / ACQUISITION_ID / ORIGINAL_FILENAME
    anchor = root / PROVENANCE_ROOT_RELATIVE / f"{ACQUISITION_ID}.json"
    journal = root / TRANSACTION_ROOT_RELATIVE / f"{ACQUISITION_ID}.register.json"
    structured = root / INTAKE_ROOT_RELATIVE / "structured" / ACQUISITION_ID
    return raw, anchor, journal, structured


def test_valid_pdf_registers_and_authoritative_reload_accepts_complete_source(
    registration_root,
):
    root, pdf_path, _, benchmark_path = registration_root
    benchmark_before = benchmark_path.read_bytes()

    result = _register(root, pdf_path)
    manifest = load_acquisition_manifest(root)
    source = manifest["sources"][0]
    raw, anchor, journal, structured = _paths(root)

    assert result["acquisition_status"] == "hash_verified"
    assert source["source_id"] == f"independent:{ACQUISITION_ID}"
    assert source["stored_pdf_path"] == raw.relative_to(root).as_posix()
    assert source["structured_path"].endswith(
        f"/{ACQUISITION_ID}/decision_structured.json"
    )
    assert source["sha256"] == _sha(pdf_path.read_bytes()) == _sha(raw.read_bytes())
    assert source["structured_sha256"] is None
    assert source["acquisition_status"] == "hash_verified"
    assert [item["status"] for item in source["status_history"]] == [
        "downloaded",
        "hash_verified",
    ]
    assert source["status_history"][0]["timestamp"] == DOWNLOAD_TIME
    assert source["case_id"] is None and source["judgment_number"] is None
    assert not any(source["metadata_verification"].values())
    assert anchor.is_file() and not journal.exists() and not structured.exists()
    anchor_payload = json.loads(anchor.read_text(encoding="utf-8"))
    assert anchor_payload["pdf_sha256"] == source["sha256"]
    assert anchor_payload["source_id"] == source["source_id"]
    assert manifest["immutable_provenance"][0]["sha256"] == source["sha256"]
    inventory, report, _ = inventory_independent_sources(root)
    assert inventory == [] and report["eligible_source_count"] == 0
    assert benchmark_path.read_bytes() == benchmark_before


def test_provenance_anchor_is_create_once(registration_root):
    root, pdf_path, _, _ = registration_root
    _register(root, pdf_path)
    _, anchor, _, _ = _paths(root)
    original = anchor.read_bytes()
    with pytest.raises(SourceIntakeError, match="Duplicate acquisition_id"):
        _register(root, pdf_path)
    assert anchor.read_bytes() == original


def test_invalid_pdf_is_rejected_before_authoritative_mutation(registration_root):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    pdf_path.write_bytes(b"%PDF-not-a-parseable-document")
    with pytest.raises(SourceIntakeError, match="PDF parser rejected"):
        _register(root, pdf_path)
    raw, anchor, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == before
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()


def test_symlink_input_pdf_is_rejected_before_authoritative_mutation(
    registration_root,
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    link = root / "candidate-link.pdf"
    link.symlink_to(pdf_path)
    with pytest.raises(SourceIntakeError, match="non-symlink"):
        _register(root, link)
    raw, anchor, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == before
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()


def test_duplicate_acquisition_and_stable_source_identity_are_rejected(
    registration_root,
):
    root, pdf_path, _, _ = registration_root
    first = _register(root, pdf_path)
    with pytest.raises(SourceIntakeError, match="Duplicate acquisition_id"):
        _register(root, pdf_path)
    assert first["source_id"] == f"independent:{ACQUISITION_ID}"
    assert len(load_acquisition_manifest(root)["sources"]) == 1


def test_existing_raw_destination_is_rejected(registration_root):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    raw, _, journal, _ = _paths(root)
    raw.parent.mkdir()
    raw.write_bytes(pdf_path.read_bytes())
    with pytest.raises(
        SourceIntakeError, match="Orphan raw|Raw acquisition destination"
    ):
        _register(root, pdf_path)
    assert manifest_path.read_bytes() == before and not journal.exists()


def test_existing_provenance_anchor_is_rejected(registration_root):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    _, anchor, journal, _ = _paths(root)
    anchor.write_text("{}", encoding="utf-8")
    with pytest.raises(SourceIntakeError, match="exactly one anchor"):
        _register(root, pdf_path)
    assert manifest_path.read_bytes() == before and not journal.exists()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"acquisition_id": "../escape"}, "acquisition_id"),
        ({"acquisition_id": "/absolute"}, "acquisition_id"),
        ({"acquisition_id": "x"}, "acquisition_id"),
        ({"original_filename": "../escape.pdf"}, "PDF basename"),
        ({"original_filename": "/tmp/escape.pdf"}, "PDF basename"),
    ],
)
def test_invalid_identity_or_path_traversal_is_rejected_before_mutation(
    registration_root, overrides, match
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    with pytest.raises(SourceIntakeError, match=match):
        _register(root, pdf_path, **overrides)
    assert manifest_path.read_bytes() == before


def test_symlink_raw_directory_attack_is_rejected(registration_root):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    raw_root = root / INTAKE_ROOT_RELATIVE / "raw"
    raw_root.rmdir()
    outside = root / "outside-raw"
    outside.mkdir()
    raw_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceIntakeError, match="Unsafe intake directory"):
        _register(root, pdf_path)
    assert manifest_path.read_bytes() == before and list(outside.iterdir()) == []


def test_symlink_provenance_directory_attack_is_rejected(registration_root):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    provenance = root / PROVENANCE_ROOT_RELATIVE
    provenance.rmdir()
    outside = root / "outside-provenance"
    outside.mkdir()
    provenance.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceIntakeError, match="Unsafe intake directory"):
        _register(root, pdf_path)
    assert manifest_path.read_bytes() == before and list(outside.iterdir()) == []


@pytest.mark.parametrize("checkpoint", ["after_prepared", "after_raw_published"])
def test_raw_ancestor_replacement_is_detected_without_redirected_write(
    registration_root, monkeypatch, checkpoint
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    raw_root = root / INTAKE_ROOT_RELATIVE / "raw"
    moved = root / "moved-raw"

    def replace_raw(name):
        if name == checkpoint:
            raw_root.rename(moved)
            raw_root.mkdir()

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", replace_raw
    )
    with pytest.raises(SourceIntakeError, match="identity changed|rollback failed"):
        _register(root, pdf_path)
    assert manifest_path.read_bytes() == before
    assert list(moved.iterdir()) == [] and list(raw_root.iterdir()) == []
    assert list((root / TRANSACTION_ROOT_RELATIVE).iterdir()) == []


def test_provenance_ancestor_replacement_cleans_through_trusted_fd(
    registration_root, monkeypatch
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()
    provenance = root / PROVENANCE_ROOT_RELATIVE
    moved = root / "moved-provenance"

    def replace_provenance(name):
        if name == "after_anchor_published":
            provenance.rename(moved)
            provenance.mkdir()

    monkeypatch.setattr(
        source_intake,
        "_registration_publication_checkpoint",
        replace_provenance,
    )
    with pytest.raises(SourceIntakeError, match="identity changed|rollback failed"):
        _register(root, pdf_path)
    raw, _, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == before
    assert not raw.parent.exists() and list(moved.iterdir()) == []
    assert list(provenance.iterdir()) == [] and not journal.exists()


def test_concurrent_manifest_modification_is_preserved_and_fails_closed(
    registration_root, monkeypatch
):
    root, pdf_path, manifest_path, _ = registration_root
    concurrent = _empty_manifest()
    concurrent["created_at"] = "2026-01-03T03:04:05+07:00"
    concurrent_bytes = _canonical_bytes(concurrent)

    def mutate_manifest(name):
        if name == "after_anchor_published":
            manifest_path.write_bytes(concurrent_bytes)

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", mutate_manifest
    )
    with pytest.raises(SourceIntakeError, match="changed during registration"):
        _register(root, pdf_path)
    raw, anchor, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == concurrent_bytes
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()


@pytest.mark.parametrize(
    "checkpoint",
    ["after_prepared", "after_raw_published", "after_anchor_published", "before_manifest_publication"],
)
def test_normal_failure_before_manifest_publication_rolls_back_everything(
    registration_root, monkeypatch, checkpoint
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()

    def fail(name):
        if name == checkpoint:
            raise RuntimeError(checkpoint)

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", fail
    )
    with pytest.raises(RuntimeError, match=checkpoint):
        _register(root, pdf_path)
    raw, anchor, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == before
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()
    assert load_acquisition_manifest(root)["sources"] == []


@pytest.mark.parametrize(
    "checkpoint",
    ["after_prepared", "after_raw_published", "after_anchor_published"],
)
def test_hard_stop_before_manifest_is_recovered_to_old_state(
    registration_root, monkeypatch, checkpoint
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()

    def crash(name):
        if name == checkpoint:
            raise _SimulatedCrash(checkpoint)

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", crash
    )
    with pytest.raises(_SimulatedCrash):
        _register(root, pdf_path)
    loaded = load_acquisition_manifest(root)
    raw, anchor, journal, _ = _paths(root)
    assert loaded["sources"] == [] and manifest_path.read_bytes() == before
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()
    assert load_acquisition_manifest(root) == loaded


def test_hard_stop_after_manifest_is_recovered_as_consistent_completion(
    registration_root, monkeypatch
):
    root, pdf_path, _, _ = registration_root

    def crash(name):
        if name == "after_manifest_published":
            raise _SimulatedCrash(name)

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", crash
    )
    with pytest.raises(_SimulatedCrash):
        _register(root, pdf_path)
    loaded = load_acquisition_manifest(root)
    raw, anchor, journal, _ = _paths(root)
    assert loaded["sources"][0]["acquisition_status"] == "hash_verified"
    assert raw.is_file() and anchor.is_file() and not journal.exists()
    assert load_acquisition_manifest(root) == loaded


def test_normal_failure_after_manifest_publication_rolls_back_registration(
    registration_root, monkeypatch
):
    root, pdf_path, manifest_path, _ = registration_root
    before = manifest_path.read_bytes()

    def fail(name):
        if name == "after_manifest_published":
            raise RuntimeError(name)

    monkeypatch.setattr(
        source_intake, "_registration_publication_checkpoint", fail
    )
    with pytest.raises(RuntimeError, match="after_manifest_published"):
        _register(root, pdf_path)
    raw, anchor, journal, _ = _paths(root)
    assert manifest_path.read_bytes() == before
    assert not raw.parent.exists() and not anchor.exists() and not journal.exists()
    assert load_acquisition_manifest(root)["sources"] == []


def test_existing_unrelated_source_remains_byte_for_byte_untouched(
    registration_root,
):
    root, pdf_path, _, _ = registration_root
    _register(root, pdf_path)
    raw, anchor, _, _ = _paths(root)
    raw_before = raw.read_bytes()
    anchor_before = anchor.read_bytes()
    manifest_before = load_acquisition_manifest(root)

    second_pdf = root / "candidate-2.pdf"
    second_pdf.write_bytes(_minimal_pdf_bytes("second"))
    _register(
        root,
        second_pdf,
        acquisition_id="scale-20261026-002",
        original_filename="decision-2.pdf",
        source_url="https://congbobanan.toaan.gov.vn/example/decision-2.pdf",
    )
    loaded = load_acquisition_manifest(root)
    original_after = next(
        item for item in loaded["sources"] if item["acquisition_id"] == ACQUISITION_ID
    )
    assert original_after == manifest_before["sources"][0]
    assert raw.read_bytes() == raw_before and anchor.read_bytes() == anchor_before


def test_cli_register_delegates_to_authoritative_function(
    registration_root, monkeypatch, capsys
):
    root, pdf_path, _, _ = registration_root
    calls = []

    def fake_register(*args):
        calls.append(args)
        return {"acquisition_id": ACQUISITION_ID, "acquisition_status": "hash_verified"}

    monkeypatch.setattr(source_intake, "register_source", fake_register)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake.py",
            "register",
            "--project-root",
            str(root),
            "--acquisition-id",
            ACQUISITION_ID,
            "--pdf",
            str(pdf_path),
            "--source-url",
            SOURCE_URL,
            "--source-domain",
            SOURCE_DOMAIN,
            "--download-timestamp",
            DOWNLOAD_TIME,
            "--original-filename",
            ORIGINAL_FILENAME,
        ],
    )
    source_intake.main()
    assert len(calls) == 1
    assert calls[0][:7] == (
        root,
        ACQUISITION_ID,
        pdf_path,
        SOURCE_URL,
        SOURCE_DOMAIN,
        DOWNLOAD_TIME,
        ORIGINAL_FILENAME,
    )
    assert json.loads(capsys.readouterr().out)["acquisition_status"] == "hash_verified"


@pytest.mark.parametrize("forbidden", ["--force", "--overwrite", "--skip-validation"])
def test_cli_register_has_no_force_overwrite_or_validation_bypass(
    registration_root, monkeypatch, forbidden
):
    root, pdf_path, _, _ = registration_root
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake.py",
            "register",
            "--project-root",
            str(root),
            "--acquisition-id",
            ACQUISITION_ID,
            "--pdf",
            str(pdf_path),
            "--source-url",
            SOURCE_URL,
            "--source-domain",
            SOURCE_DOMAIN,
            "--download-timestamp",
            DOWNLOAD_TIME,
            "--original-filename",
            ORIGINAL_FILENAME,
            forbidden,
        ],
    )
    with pytest.raises(SystemExit):
        source_intake.main()
