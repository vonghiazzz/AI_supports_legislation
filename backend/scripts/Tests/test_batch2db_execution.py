import ast
import copy
import hashlib
import json
import ssl
import urllib.error
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import batch2db_execution as runner


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COHORT_PATH = PROJECT_ROOT / runner.COHORT_RELATIVE
INTAKE_PATH = PROJECT_ROOT / runner.MANIFEST_RELATIVE


@pytest.fixture(scope="module")
def cohort():
    return runner.strict_json_load(COHORT_PATH)


@pytest.fixture(scope="module")
def intake():
    return runner.strict_json_load(INTAKE_PATH)


class FakeResponse:
    def __init__(self, body=b"%PDF-test", status=200, url=None):
        self.body = body
        self.status = status
        self.url = url or "https://congbobanan.toaan.gov.vn/frozen.pdf"
        self.headers = {}
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def getcode(self):
        return self.status

    def read(self, size=-1):
        if self.offset:
            return b""
        self.offset = len(self.body)
        return self.body


def _entry(index=1):
    return {
        "acquisition_id": f"batch2db-20261103-{index:03d}",
        "source_url": f"https://congbobanan.toaan.gov.vn/frozen-{index}.pdf",
        "source_domain": runner.OFFICIAL_DOMAIN,
        "official_document_title_or_filename": f"document-{index}.pdf",
    }


def _download_record(entry, disposition="DOWNLOADED_VALID_PDF"):
    return {
        "acquisition_id": entry["acquisition_id"],
        "source_url": entry["source_url"],
        "attempt_count": 1,
        "http_terminal_status": 200,
        "final_host": runner.OFFICIAL_DOMAIN,
        "download_timestamp": "2026-11-03T12:00:00+07:00",
        "staged_byte_count": 9,
        "staged_sha256": hashlib.sha256(b"%PDF-test").hexdigest(),
        "terminal_disposition": disposition,
        "technical_failure": None,
    }


def _registration_record(entry, disposition="REGISTERED_HASH_VERIFIED"):
    return {
        "acquisition_id": entry["acquisition_id"],
        "source_url": entry["source_url"],
        "terminal_disposition": disposition,
        "technical_failure": None,
        "raw_sha256": "a" * 64 if disposition == "REGISTERED_HASH_VERIFIED" else None,
    }


def test_frozen_constants_are_exact():
    assert runner.COHORT_CHECKPOINT_COMMIT == "c34825798754de0e97943abfbd372f439256b7e0"
    assert runner.COHORT_FINGERPRINT == "7f98015c8127bbf282098215d5db3bcb6037803ca1c7fc5f8c587bd4a26d4802"
    assert runner.COHORT_FILE_SHA256 == "2372285a1cce328943a86b769609157bfd0db2024efc451007fb3ab11a43c85c"
    assert runner.EXPECTED_IDS == [f"batch2db-20261103-{index:03d}" for index in range(1, 51)]


def test_real_frozen_cohort_validates(cohort, intake):
    validated = runner.validate_frozen_cohort(cohort, intake)
    assert validated["cohort_fingerprint"] == runner.COHORT_FINGERPRINT


def test_real_execution_inputs_and_checkpoint_ancestry_are_valid():
    loaded, manifest, head = runner.load_execution_inputs(PROJECT_ROOT)
    assert loaded["cohort_fingerprint"] == runner.COHORT_FINGERPRINT
    assert len(manifest["sources"]) == 271
    assert len(head) == 40
    assert int(head, 16) >= 0


def test_replacement_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = "https://congbobanan.toaan.gov.vn/replacement.pdf"
    with pytest.raises(Exception):
        runner.validate_frozen_cohort(changed, intake)


def test_id_051_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][-1]["acquisition_id"] = "batch2db-20261103-051"
    with pytest.raises(Exception):
        runner.validate_frozen_cohort(changed, intake)


@pytest.mark.parametrize(
    "filename",
    ["../escape.pdf", "/absolute.pdf", "nested/file.pdf", "file.txt", "bad\n.pdf", ""],
)
def test_original_filename_validation_rejects_unsafe_values(filename):
    with pytest.raises(runner.Batch2DBExecutionError):
        runner.validate_original_filename(filename)


def test_original_filename_validation_accepts_frozen_basename():
    assert runner.validate_original_filename("446.pdf") == "446.pdf"


def test_redirect_off_official_domain_is_rejected(tmp_path):
    calls = []

    def opener(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(url="https://example.com/redirected.pdf")

    result = runner.download_entry(_entry(), tmp_path, opener=opener, sleeper=lambda _: None)
    assert result["terminal_disposition"] == "DOWNLOAD_REDIRECT_DOMAIN_REJECTED"
    assert result["attempt_count"] == 1
    assert len(calls) == 1


def test_permanent_404_is_not_retried(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    result = runner.download_entry(_entry(), tmp_path, opener=opener, sleeper=lambda _: None)
    assert result["terminal_disposition"] == "DOWNLOAD_HTTP_FAILURE"
    assert result["attempt_count"] == 1
    assert calls == [_entry()["source_url"]]


def test_retryable_503_uses_exact_frozen_attempt_count_and_backoff(tmp_path):
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, "busy", {}, None)

    result = runner.download_entry(_entry(), tmp_path, opener=opener, sleeper=sleeps.append)
    assert result["terminal_disposition"] == "DOWNLOAD_HTTP_FAILURE"
    assert result["attempt_count"] == 3
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_html_200_is_rejected_as_non_pdf(tmp_path, monkeypatch):
    def reject(path, sha):
        raise runner.source_intake.SourceIntakeError("PDF signature is invalid")

    monkeypatch.setattr(runner.source_intake, "validate_pdf_source", reject)
    result = runner.download_entry(
        _entry(),
        tmp_path,
        opener=lambda *args, **kwargs: FakeResponse(body=b"<html>error</html>"),
        sleeper=lambda _: None,
    )
    assert result["terminal_disposition"] == "DOWNLOAD_INVALID_PDF"
    assert result["staged_byte_count"] == len(b"<html>error</html>")
    assert not (tmp_path / "batch2db-20261103-001.pdf").exists()


def test_successful_download_records_raw_sha_and_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.source_intake, "validate_pdf_source", lambda path, sha: sha)
    result = runner.download_entry(
        _entry(),
        tmp_path,
        opener=lambda *args, **kwargs: FakeResponse(),
        sleeper=lambda _: None,
    )
    assert result["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
    assert result["staged_sha256"] == hashlib.sha256(b"%PDF-test").hexdigest()
    assert result["download_timestamp"].endswith(("+00:00", "+07:00"))


def test_download_ledger_covers_exactly_50_and_uses_only_frozen_urls(tmp_path):
    entries = [_entry(index) for index in range(1, 51)]
    events = []

    def downloader(entry, directory):
        events.append(("download", entry["acquisition_id"]))
        return _download_record(entry)

    records = runner.run_download_stage(entries, tmp_path, downloader=downloader)
    assert len(records) == 50
    assert [item["acquisition_id"] for item in records] == runner.EXPECTED_IDS
    assert [item["source_url"] for item in records] == [item["source_url"] for item in entries]
    assert len(events) == 50


def test_registration_requires_complete_download_barrier(tmp_path):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry) for entry in entries[:-1]]
    called = []
    with pytest.raises(runner.Batch2DBExecutionError, match="exact frozen cohort"):
        runner.run_registration_stage(
            tmp_path,
            entries,
            downloads,
            tmp_path,
            registrar=lambda **kwargs: called.append(kwargs),
        )
    assert called == []


def test_extraction_requires_complete_registration_barrier(tmp_path):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry) for entry in entries]
    registrations = [_registration_record(entry) for entry in entries[:-1]]
    called = []
    with pytest.raises(runner.Batch2DBExecutionError, match="exact frozen cohort"):
        runner.run_extraction_stage(
            tmp_path,
            entries,
            downloads,
            registrations,
            tmp_path,
            extractor=lambda *args: called.append(args),
        )
    assert called == []


def test_failure_ledgers_cover_all_50_without_registration_or_extraction(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [
        _download_record(entry, "DOWNLOAD_HTTP_FAILURE") for entry in entries
    ]
    monkeypatch.setattr(runner, "_manifest_by_id", lambda root: ({"sources": []}, {}))
    monkeypatch.setattr(runner.source_intake, "inventory_independent_sources", lambda root: ([], {}, []))
    registrations = runner.run_registration_stage(
        tmp_path,
        entries,
        downloads,
        tmp_path,
        registrar=lambda **kwargs: pytest.fail("registration crossed download barrier"),
    )
    extractions = runner.run_extraction_stage(
        tmp_path,
        entries,
        downloads,
        registrations,
        tmp_path,
        extractor=lambda *args: pytest.fail("extraction crossed registration barrier"),
    )
    assert len(registrations) == len(extractions) == 50
    assert {item["terminal_disposition"] for item in registrations} == {
        "NOT_REGISTERED_DOWNLOAD_FAILURE"
    }
    assert {item["terminal_disposition"] for item in extractions} == {
        "NOT_EXTRACTED_DOWNLOAD_FAILURE"
    }


def test_resume_ledger_url_mismatch_fails_closed(tmp_path):
    entries = [_entry(index) for index in range(1, 51)]
    bad = _download_record(entries[0])
    bad["source_url"] = "https://congbobanan.toaan.gov.vn/replacement.pdf"
    runner.strict_json_write(
        tmp_path / "download_ledger.json",
        runner._ledger("batch2db_download_ledger_v1", [bad]),
    )
    with pytest.raises(runner.Batch2DBExecutionError, match="resume URL mismatch"):
        runner.run_download_stage(entries, tmp_path, downloader=lambda *args: {})


def test_resume_protocol_fingerprint_mismatch_fails_closed(tmp_path):
    payload = {
        "protocol_id": runner.PROTOCOL_ID,
        "run_id": tmp_path.name,
        "parent_cohort_checkpoint_commit": runner.COHORT_CHECKPOINT_COMMIT,
        "parent_cohort_fingerprint": "0" * 64,
        "parent_cohort_file_sha256": runner.COHORT_FILE_SHA256,
        "execution_git_commit": runner.COHORT_CHECKPOINT_COMMIT,
        "runner_sha256": runner._runner_sha256(),
        "baseline_manifest_sha256": runner.BASELINE_MANIFEST_SHA256,
        "baseline_source_count": runner.BASELINE_SOURCE_COUNT,
    }
    runner.strict_json_write(tmp_path / "execution_protocol.json", payload)
    with pytest.raises(runner.Batch2DBExecutionError, match="parent_cohort_fingerprint"):
        runner._load_resume_protocol(tmp_path, runner.COHORT_CHECKPOINT_COMMIT)


def test_manifest_count_reconciliation_rejects_wrong_delta(tmp_path, monkeypatch):
    (tmp_path / runner.MANIFEST_RELATIVE).parent.mkdir(parents=True)
    (tmp_path / runner.MANIFEST_RELATIVE).write_text("{}\n", encoding="utf-8")
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry, "DOWNLOAD_HTTP_FAILURE") for entry in entries]
    registrations = [
        _registration_record(entry, "NOT_REGISTERED_DOWNLOAD_FAILURE")
        for entry in entries
    ]
    extractions = [
        {
            "acquisition_id": entry["acquisition_id"],
            "source_url": entry["source_url"],
            "terminal_disposition": "NOT_EXTRACTED_DOWNLOAD_FAILURE",
            "technical_failure": "DOWNLOAD_HTTP_FAILURE",
            "structured_path": None,
            "structured_sha256": None,
        }
        for entry in entries
    ]
    monkeypatch.setattr(
        runner.source_intake,
        "load_acquisition_manifest",
        lambda root: {"sources": [{"acquisition_id": f"old-{i}"} for i in range(270)]},
    )
    with pytest.raises(runner.Batch2DBExecutionError, match="does not reconcile"):
        runner._finalize(
            tmp_path,
            tmp_path,
            {"execution_git_commit": runner.COHORT_CHECKPOINT_COMMIT, "runner_sha256": "a" * 64, "baseline_manifest_sha256": runner.BASELINE_MANIFEST_SHA256},
            entries,
            downloads,
            registrations,
            extractions,
        )


def test_runner_exposes_no_identity_review_promotion_or_scientific_actions():
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "record_source_identity",
        "record_source_review",
        "promote_source_to_eligible",
        "build_preliminary_stratification_for_frame",
        "generate_query",
        "generate_gt",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden)


def test_authoritative_mutations_delegate_to_source_intake():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "source_intake.register_source" in source
    assert "source_intake.extract_intake_source" in source
    assert "force_ocr=False" in source


def test_retry_policy_is_frozen():
    assert runner.MAX_ATTEMPTS == 3
    assert runner.RETRY_HTTP_STATUSES == {408, 429, 500, 502, 503, 504}
    assert runner.BACKOFF_SECONDS == (1.0, 2.0)


def test_tls_certificate_verification_failure_is_recorded_fail_closed(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        verification_error = ssl.SSLCertVerificationError(
            1, "unable to get local issuer certificate"
        )
        raise urllib.error.URLError(verification_error)

    result = runner.download_entry(
        _entry(), tmp_path, opener=opener, sleeper=lambda _: None
    )
    assert result["terminal_disposition"] == "DOWNLOAD_NETWORK_FAILURE"
    assert result["attempt_count"] == runner.MAX_ATTEMPTS
    assert "unable to get local issuer certificate" in result["technical_failure"]
    assert calls == [_entry()["source_url"]] * runner.MAX_ATTEMPTS
    assert not list(tmp_path.iterdir())


def test_runner_contains_no_tls_bypass_or_ca_ignore_escape_hatch():
    source = Path(runner.__file__).read_text(encoding="utf-8").casefold()
    forbidden_fragments = {
        "verify=false",
        "verify = false",
        "_create_unverified_context",
        "check_hostname=false",
        "check_hostname = false",
        "cert_none",
        "--insecure",
        "--no-verify",
        "--no_verify",
        "--ca-ignore",
        "--ca_ignore",
        "http://",
    }
    assert all(fragment not in source for fragment in forbidden_fragments)
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "ssl" not in imported_roots


def test_frozen_cohort_validation_rejects_https_to_http_downgrade(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = changed["entries"][0][
        "source_url"
    ].replace("https://", "http://", 1)
    with pytest.raises(Exception):
        runner.validate_frozen_cohort(changed, intake)


@pytest.mark.parametrize("unsafe_flag", ["--insecure", "--no-verify", "--ca-ignore"])
def test_cli_rejects_arbitrary_tls_bypass_flags(unsafe_flag, capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--project-root", str(PROJECT_ROOT), unsafe_flag])
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_no_main_stress_seed_query_gt_or_model_fields_in_protocol(tmp_path, cohort):
    protocol = runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
    )
    assert "main" not in protocol
    assert "stress" not in protocol
    assert "sampling_seed" not in protocol
    assert "query_records" not in protocol
    assert "ground_truth_records" not in protocol
    assert protocol["selection_seed_query_gt_model_retrieval_forbidden"] is True


def test_initial_baseline_gate_matches_repository(intake):
    runner.validate_initial_baseline(PROJECT_ROOT, intake)
