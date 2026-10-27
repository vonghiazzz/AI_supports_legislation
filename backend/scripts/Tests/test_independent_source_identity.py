import copy
import json
import sys
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking.source_intake import (
    ACQUISITION_MANIFEST_RELATIVE,
    IDENTITY_PAYLOAD_VERSION,
    PROVENANCE_ROOT_RELATIVE,
    REVIEW_COMPLETION_FIELDS,
    REVIEW_METADATA_FIELDS,
    SourceIntakeError,
    load_acquisition_manifest,
    promote_source_to_eligible,
    record_source_identity,
)
from backend.scripts.Tests.test_independent_source_intake import (
    _eligible_fixture,
    _hash_verified_fixture,
)
from backend.scripts.Tests.test_independent_source_review import (
    _identity_payload,
    _review_fixture,
)


def _manifest_source(root: Path, acquisition_id: str = "src-review-001") -> dict:
    return next(
        source
        for source in load_acquisition_manifest(root)["sources"]
        if source["acquisition_id"] == acquisition_id
    )


def test_valid_identity_persists_only_intrinsic_fields_and_keeps_lifecycle(tmp_path):
    source, _ = _review_fixture(tmp_path)
    history = copy.deepcopy(source["status_history"])
    manifest_path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    raw_path = tmp_path / source["stored_pdf_path"]
    structured_path = tmp_path / source["structured_path"]
    anchor_path = (
        tmp_path / PROVENANCE_ROOT_RELATIVE / f"{source['acquisition_id']}.json"
    )
    protected = {
        path: path.read_bytes()
        for path in (raw_path, structured_path, anchor_path)
    }
    benchmark = (
        tmp_path
        / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
    )
    benchmark.parent.mkdir(parents=True, exist_ok=True)
    benchmark.write_bytes(b'{"state":"DRAFT"}\n')
    benchmark_before = benchmark.read_bytes()

    result = record_source_identity(
        tmp_path, source["acquisition_id"], _identity_payload()
    )

    identified = _manifest_source(tmp_path)
    assert result["acquisition_status"] == "extracted"
    assert result["idempotent_noop"] is False
    assert identified["acquisition_status"] == "extracted"
    assert identified["status_history"] == history
    assert {field: identified[field] for field in REVIEW_METADATA_FIELDS} == (
        _identity_payload()["verified_metadata"]
    )
    assert all(
        identified["metadata_verification"][field]
        for field in REVIEW_METADATA_FIELDS
    )
    assert all(
        not identified["metadata_verification"][field]
        for field in REVIEW_COMPLETION_FIELDS
    )
    assert identified["rejection_reasons"] == []
    assert all(
        f"incomplete_{field}" in result["gate_failures"]
        for field in REVIEW_COMPLETION_FIELDS
    )
    assert {path: path.read_bytes() for path in protected} == protected
    assert benchmark.read_bytes() == benchmark_before
    assert manifest_path.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("judgment_number", "invalid", "canonical Vietnamese decision"),
        ("case_id", "", "non-empty string"),
        ("court", "", "non-empty string"),
        ("decision_date", "2026-02-30", "valid YYYY-MM-DD"),
        ("procedure_stage", "unknown", "allowed value"),
    ],
)
def test_identity_rejects_invalid_metadata_without_mutation(
    tmp_path, field, value, message
):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    with pytest.raises(SourceIntakeError, match=message):
        record_source_identity(
            tmp_path,
            source["acquisition_id"],
            _identity_payload(**{field: value}),
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "source_id",
        "status",
        "status_history",
        "source_url",
        "stored_pdf_path",
        "sha256",
        "structured_path",
        "reviews",
        "benchmark_state",
        "unknown",
    ],
)
def test_identity_rejects_unknown_or_forbidden_fields(
    tmp_path, forbidden_field
):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    payload = _identity_payload()
    payload[forbidden_field] = "forbidden"
    with pytest.raises(SourceIntakeError, match="invalid root fields"):
        record_source_identity(tmp_path, source["acquisition_id"], payload)
    assert path.read_bytes() == before


def test_identity_requires_every_verification_flag_true(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _identity_payload()
    payload["verified_fields"]["court"] = False
    with pytest.raises(SourceIntakeError, match="verified=true"):
        record_source_identity(tmp_path, source["acquisition_id"], payload)


@pytest.mark.parametrize("fixture", ["hash_verified", "eligible"])
def test_identity_rejects_non_extracted_source(tmp_path, fixture):
    if fixture == "hash_verified":
        source, _, _, _ = _hash_verified_fixture(tmp_path)
    else:
        source, _ = _eligible_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    with pytest.raises(SourceIntakeError, match="requires extracted status"):
        record_source_identity(tmp_path, source["acquisition_id"], _identity_payload())
    assert path.read_bytes() == before


def test_identical_identity_is_a_byte_stable_noop(tmp_path):
    source, _ = _review_fixture(tmp_path)
    payload = _identity_payload()
    record_source_identity(tmp_path, source["acquisition_id"], payload)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    result = record_source_identity(tmp_path, source["acquisition_id"], payload)
    assert result["idempotent_noop"] is True
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("court", "Tòa án nhân dân tỉnh Khác"),
        ("judgment_number", "10/2026/DS-PT"),
        ("decision_date", "2026-01-10"),
    ],
)
def test_conflicting_verified_identity_fails_closed(tmp_path, field, value):
    source, _ = _review_fixture(tmp_path)
    record_source_identity(
        tmp_path, source["acquisition_id"], _identity_payload()
    )
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    with pytest.raises(SourceIntakeError, match=f"conflicts with existing {field}"):
        record_source_identity(
            tmp_path,
            source["acquisition_id"],
            _identity_payload(**{field: value}),
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["related_decision_ids", "underlying_dispute_id"])
def test_conflicting_verified_relationship_identity_fails_closed(tmp_path, field):
    source, _ = _review_fixture(tmp_path)
    initial = _identity_payload()
    initial["related_decision_ids"] = []
    initial["underlying_dispute_id"] = None
    record_source_identity(tmp_path, source["acquisition_id"], initial)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    conflict = copy.deepcopy(initial)
    conflict[field] = (
        ["55/2026/DS-ST"]
        if field == "related_decision_ids"
        else "DISPUTE-CONFLICT"
    )
    with pytest.raises(SourceIntakeError, match=f"conflicts with existing {field}"):
        record_source_identity(tmp_path, source["acquisition_id"], conflict)
    assert path.read_bytes() == before


def test_identity_manifest_race_fails_without_lost_update(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE

    def concurrent_update(checkpoint):
        if checkpoint == "before_manifest_compare":
            concurrent = json.loads(path.read_text(encoding="utf-8"))
            concurrent["sources"][0]["notes"] = "retained concurrent update"
            path.write_text(json.dumps(concurrent, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(
        source_intake, "_identity_publication_checkpoint", concurrent_update
    )
    with pytest.raises(SourceIntakeError, match="unexpected manifest state"):
        record_source_identity(
            tmp_path, source["acquisition_id"], _identity_payload()
        )
    assert _manifest_source(tmp_path)["notes"] == "retained concurrent update"


def test_identity_failure_before_replace_preserves_manifest(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("before replacement")

    monkeypatch.setattr(source_intake, "_atomic_write_bytes_at", fail)
    with pytest.raises(OSError, match="before replacement"):
        record_source_identity(
            tmp_path, source["acquisition_id"], _identity_payload()
        )
    assert path.read_bytes() == before


def test_identity_failure_after_replace_rolls_back(tmp_path, monkeypatch):
    source, _ = _review_fixture(tmp_path)
    path = tmp_path / ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()

    def fail_after_replace(checkpoint):
        if checkpoint == "after_manifest_replacement":
            raise OSError("after replacement")

    monkeypatch.setattr(
        source_intake, "_identity_publication_checkpoint", fail_after_replace
    )
    with pytest.raises(SourceIntakeError, match="previous manifest restored"):
        record_source_identity(
            tmp_path, source["acquisition_id"], _identity_payload()
        )
    assert path.read_bytes() == before


def test_identity_alone_never_allows_promotion(tmp_path):
    source, _ = _review_fixture(tmp_path)
    record_source_identity(
        tmp_path, source["acquisition_id"], _identity_payload()
    )
    with pytest.raises(SourceIntakeError, match="incomplete_development_overlap_check"):
        promote_source_to_eligible(tmp_path, source["acquisition_id"])
    assert _manifest_source(tmp_path)["acquisition_status"] == "extracted"


def test_identity_cli_delegates_strict_payload_and_never_reviews_or_promotes(
    tmp_path, monkeypatch, capsys
):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(
        json.dumps(_identity_payload(), ensure_ascii=False), encoding="utf-8"
    )
    calls = []

    def fake_identity(root, acquisition_id, payload):
        calls.append((Path(root), acquisition_id, payload))
        return {"acquisition_id": acquisition_id, "acquisition_status": "extracted"}

    monkeypatch.setattr(source_intake, "record_source_identity", fake_identity)
    monkeypatch.setattr(
        source_intake,
        "record_source_review",
        lambda *args, **kwargs: pytest.fail("identify CLI must not review"),
    )
    monkeypatch.setattr(
        source_intake,
        "promote_source_to_eligible",
        lambda *args, **kwargs: pytest.fail("identify CLI must not promote"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "identify",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-review-001",
            "--identity-file",
            str(identity_path),
        ],
    )
    source_intake.main()
    assert calls == [(tmp_path, "src-review-001", _identity_payload())]
    assert json.loads(capsys.readouterr().out)["acquisition_status"] == "extracted"


@pytest.mark.parametrize("argument", ["--force-ocr", "--source-url", "--pdf", "--review-file"])
def test_identity_cli_rejects_unrelated_arguments(tmp_path, monkeypatch, argument):
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(_identity_payload()), encoding="utf-8")
    argv = [
        "source_intake",
        "identify",
        "--project-root",
        str(tmp_path),
        "--acquisition-id",
        "src-review-001",
        "--identity-file",
        str(identity_path),
        argument,
    ]
    if argument == "--source-url":
        argv.append("https://example.test/rewrite")
    elif argument in {"--pdf", "--review-file"}:
        argv.append(str(tmp_path / "forbidden"))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        source_intake.main()


def test_identity_cli_rejects_symlinked_identity_file(tmp_path, monkeypatch):
    actual = tmp_path / "actual-identity.json"
    actual.write_text(json.dumps(_identity_payload()), encoding="utf-8")
    link = tmp_path / "identity-link.json"
    link.symlink_to(actual)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "source_intake",
            "identify",
            "--project-root",
            str(tmp_path),
            "--acquisition-id",
            "src-review-001",
            "--identity-file",
            str(link),
        ],
    )
    with pytest.raises(SystemExit):
        source_intake.main()
