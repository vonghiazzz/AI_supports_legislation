import ast
import copy
import hashlib
import http.client
import json
import ssl
import urllib.error
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import batch2dc_execution as runner


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


class IncompleteReadResponse(FakeResponse):
    def __init__(self, partial=b"partial-body", expected=17, **kwargs):
        super().__init__(**kwargs)
        self.partial = partial
        self.expected = expected

    def read(self, size=-1):
        raise http.client.IncompleteRead(self.partial, self.expected)


class ChunkThenIncompleteReadResponse(FakeResponse):
    def __init__(self, first_chunk=b"stale-first-attempt", **kwargs):
        super().__init__(**kwargs)
        self.first_chunk = first_chunk
        self.read_count = 0

    def read(self, size=-1):
        self.read_count += 1
        if self.read_count == 1:
            return self.first_chunk
        raise http.client.IncompleteRead(b"truncated-tail", 23)


def _entry(index=1):
    return {
        "acquisition_id": f"batch2dc-20261105-{index:03d}",
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


def _tls_trust():
    return {
        "schema_version": runner.TLS_PROTOCOL_SCHEMA_VERSION,
        "mode": runner.TLS_TRUST_MODE,
        "base_certifi_sha256": runner.EXPECTED_BASE_CERTIFI_SHA256,
        "supplemental_intermediate_der_sha256": runner.EXPECTED_INTERMEDIATE_DER_SHA256,
        "expected_root_r3_sha256": runner.EXPECTED_ROOT_R3_SHA256,
        "combined_ca_bundle_sha256": runner.EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256,
        "verify_mode": "CERT_REQUIRED",
        "check_hostname": True,
        "partial_chain_enabled": False,
        "base_trust": "certifi",
        "intermediate_role": "CHAIN_COMPLETION_ONLY",
    }


def _valid_test_bundle(tmp_path, monkeypatch):
    path = tmp_path / "verified-supplemental.pem"
    path.write_bytes(b"test-only-supplemental-bundle\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256", digest)
    return path


class FakeSSLContext:
    def __init__(
        self,
        *,
        verify_mode=ssl.CERT_REQUIRED,
        check_hostname=True,
        verify_flags=0,
    ):
        self.verify_mode = verify_mode
        self.check_hostname = check_hostname
        self.verify_flags = verify_flags


def test_frozen_constants_are_exact():
    assert runner.COHORT_CHECKPOINT_COMMIT == "1d6c2dc69e9251bf78db31ae9ee6dce2125ea638"
    assert runner.COHORT_FINGERPRINT == "e4490f6f5cba6aecfee563edf9b5035944ff964160e8703e43533be8a8f9b8e4"
    assert runner.COHORT_FILE_SHA256 == "3afc710119577b5a154b1ae09e41a3c225ea6122130fa5e7217840a2901fbaf1"
    assert runner.BASELINE_MANIFEST_SHA256 == "4716e61118ce77dee60597940f68247423be685a27b6dde1c3c479393bbc5094"
    assert runner.BASELINE_SOURCE_COUNT == 317
    assert runner.BASELINE_STATUS_COUNTS == {
        "eligible": 257,
        "extracted": 59,
        "hash_verified": 1,
    }
    assert runner.EXPECTED_IDS == [
        f"batch2dc-20261105-{index:03d}" for index in range(1, 51)
    ]
    assert runner.TLS_TRUST_MODE == "CERTIFI_PLUS_VERIFIED_GLOBALSIGN_INTERMEDIATE"
    assert runner.EXPECTED_BASE_CERTIFI_SHA256 == "fc9165a12403263e7ebfbdad7be7a3eac0fa5d325d3c70465f28d3690072ca28"
    assert runner.EXPECTED_INTERMEDIATE_DER_SHA256 == "b676ffa3179e8812093a1b5eafee876ae7a6aaf231078dad1bfb21cd2893764a"
    assert runner.EXPECTED_ROOT_R3_SHA256 == "cbb522d7b7f127ad6a0113865bdf1cd4102e7d0759af635a7cf4720dc963c53b"
    assert runner.EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256 == "f08632660e469b1953e83ecc06d02ffb9a6dd4ad3de44502979679197d6f3fcc"


def test_real_frozen_cohort_validates(cohort, intake):
    validated = runner.validate_frozen_cohort(cohort, intake)
    assert validated["cohort_fingerprint"] == runner.COHORT_FINGERPRINT


def test_real_execution_inputs_and_checkpoint_ancestry_are_valid(monkeypatch):
    monkeypatch.setattr(
        runner, "validate_git_gate", lambda root: runner.COHORT_CHECKPOINT_COMMIT
    )
    loaded, manifest, head = runner.load_execution_inputs(PROJECT_ROOT)
    assert loaded["cohort_fingerprint"] == runner.COHORT_FINGERPRINT
    assert len(manifest["sources"]) == 317
    assert len(head) == 40
    assert int(head, 16) >= 0


def test_replacement_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = "https://congbobanan.toaan.gov.vn/replacement.pdf"
    with pytest.raises(Exception):
        runner.validate_frozen_cohort(changed, intake)


def test_id_051_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][-1]["acquisition_id"] = "batch2dc-20261105-051"
    with pytest.raises(Exception):
        runner.validate_frozen_cohort(changed, intake)


@pytest.mark.parametrize(
    "filename",
    ["../escape.pdf", "/absolute.pdf", "nested/file.pdf", "file.txt", "bad\n.pdf", ""],
)
def test_original_filename_validation_rejects_unsafe_values(filename):
    with pytest.raises(runner.Batch2DCExecutionError):
        runner.validate_original_filename(filename)


def test_original_filename_validation_accepts_frozen_basename():
    assert runner.validate_original_filename("446.pdf") == "446.pdf"


def test_registration_filename_uses_frozen_url_basename_when_title_is_not_safe():
    entry = _entry()
    entry["official_document_title_or_filename"] = "37/2026/DSPT.pdf"
    entry["source_url"] = (
        "https://congbobanan.toaan.gov.vn/5ta1t1cvn/official%20source.pdf"
    )
    assert runner.registration_filename(entry) == "official source.pdf"


def test_registration_filename_fails_closed_when_both_sources_are_unsafe():
    entry = _entry()
    entry["official_document_title_or_filename"] = "37/2026/DSPT.pdf"
    entry["source_url"] = "https://congbobanan.toaan.gov.vn/5ta1t1cvn/bad%2Fname.pdf"
    with pytest.raises(runner.Batch2DCExecutionError, match="no safe deterministic"):
        runner.registration_filename(entry)


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
    assert not list(tmp_path.iterdir())


def test_incomplete_read_retries_same_url_then_succeeds(tmp_path, monkeypatch):
    complete_pdf = b"%PDF-complete-second-response"
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            return IncompleteReadResponse(url=request.full_url)
        return FakeResponse(body=complete_pdf, url=request.full_url)

    monkeypatch.setattr(
        runner.source_intake, "validate_pdf_source", lambda path, sha: sha
    )
    result = runner.download_entry(
        _entry(), tmp_path, opener=opener, sleeper=sleeps.append
    )

    assert result["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
    assert result["attempt_count"] == 2
    assert calls == [_entry()["source_url"], _entry()["source_url"]]
    assert sleeps == [1.0]
    assert result["staged_byte_count"] == len(complete_pdf)
    assert result["staged_sha256"] == hashlib.sha256(complete_pdf).hexdigest()
    assert (tmp_path / "batch2dc-20261105-001.pdf").read_bytes() == complete_pdf
    assert not (tmp_path / ".batch2dc-20261105-001.part").exists()


def test_incomplete_read_exhaustion_terminalizes_without_partial_file(tmp_path):
    calls, sleeps = [], []

    def opener(request, timeout):
        calls.append(request.full_url)
        return IncompleteReadResponse(
            partial=b"never-persist-this-body", expected=41, url=request.full_url
        )

    result = runner.download_entry(
        _entry(), tmp_path, opener=opener, sleeper=sleeps.append
    )

    assert result["terminal_disposition"] == "DOWNLOAD_NETWORK_FAILURE"
    assert result["attempt_count"] == runner.MAX_ATTEMPTS
    assert result["staged_byte_count"] is None
    assert result["staged_sha256"] is None
    assert result["technical_failure"] == (
        "IncompleteRead: 23 bytes read, 41 more expected"
    )
    assert calls == [_entry()["source_url"]] * runner.MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0]
    assert not list(tmp_path.iterdir())


def test_incomplete_read_retry_discards_all_cross_attempt_bytes(
    tmp_path, monkeypatch
):
    stale = b"%PDF-stale-first-attempt"
    complete_pdf = b"%PDF-only-the-second-response-counts"
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            return ChunkThenIncompleteReadResponse(
                first_chunk=stale, url=request.full_url
            )
        return FakeResponse(body=complete_pdf, url=request.full_url)

    monkeypatch.setattr(
        runner.source_intake, "validate_pdf_source", lambda path, sha: sha
    )
    result = runner.download_entry(
        _entry(), tmp_path, opener=opener, sleeper=lambda _: None
    )

    assert result["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
    assert result["attempt_count"] == 2
    assert result["staged_byte_count"] == len(complete_pdf)
    assert result["staged_sha256"] == hashlib.sha256(complete_pdf).hexdigest()
    assert (tmp_path / "batch2dc-20261105-001.pdf").read_bytes() == complete_pdf
    assert stale not in complete_pdf
    assert not (tmp_path / ".batch2dc-20261105-001.part").exists()


def test_incomplete_read_terminal_record_allows_full_ledger_continuation(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    calls = []

    def downloader(entry, downloads_dir):
        def opener(request, timeout):
            calls.append(entry["acquisition_id"])
            if entry["acquisition_id"] == entries[0]["acquisition_id"]:
                return IncompleteReadResponse(url=request.full_url)
            return FakeResponse(url=request.full_url)

        return runner.download_entry(
            entry, downloads_dir, opener=opener, sleeper=lambda _: None
        )

    monkeypatch.setattr(
        runner.source_intake, "validate_pdf_source", lambda path, sha: sha
    )
    records = runner.run_download_stage(entries, tmp_path, downloader=downloader)

    assert len(records) == 50
    assert records[0]["terminal_disposition"] == "DOWNLOAD_NETWORK_FAILURE"
    assert records[0]["attempt_count"] == runner.MAX_ATTEMPTS
    assert records[1]["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
    assert records[-1]["acquisition_id"] == "batch2dc-20261105-050"
    assert calls.count("batch2dc-20261105-001") == runner.MAX_ATTEMPTS
    assert calls[-1] == "batch2dc-20261105-050"
    ledger = runner.strict_json_load(tmp_path / "download_ledger.json")
    assert len(ledger["records"]) == 50
    assert not list((tmp_path / "downloads").glob(".*.part"))


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
    assert not (tmp_path / "batch2dc-20261105-001.pdf").exists()


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
    with pytest.raises(runner.Batch2DCExecutionError, match="exact frozen cohort"):
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
    with pytest.raises(runner.Batch2DCExecutionError, match="exact frozen cohort"):
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
        runner._ledger("batch2dc_download_ledger_v1", [bad]),
    )
    with pytest.raises(runner.Batch2DCExecutionError, match="resume URL mismatch"):
        runner.run_download_stage(entries, tmp_path, downloader=lambda *args: {})


def test_resume_protocol_fingerprint_mismatch_fails_closed(tmp_path, cohort):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    path = tmp_path / "execution_protocol.json"
    payload = runner.strict_json_load(path)
    payload["parent_cohort_fingerprint"] = "0" * 64
    runner.strict_json_write(path, payload)
    with pytest.raises(runner.Batch2DCExecutionError, match="parent_cohort_fingerprint"):
        runner._load_resume_protocol(
            tmp_path,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            cohort["entries"],
        )


def test_exact_ca_bundle_path_is_accepted(tmp_path, monkeypatch):
    bundle = _valid_test_bundle(tmp_path, monkeypatch)
    assert runner.validate_ca_bundle(bundle) == bundle.resolve()


def test_wrong_ca_bundle_sha_is_rejected(tmp_path):
    bundle = tmp_path / "wrong.pem"
    bundle.write_bytes(b"wrong bundle\n")
    with pytest.raises(runner.Batch2DCExecutionError, match="SHA256 mismatch"):
        runner.validate_ca_bundle(bundle)


def test_missing_ca_bundle_is_rejected(tmp_path):
    with pytest.raises(runner.Batch2DCExecutionError, match="does not exist"):
        runner.validate_ca_bundle(tmp_path / "missing.pem")


def test_symlink_ca_bundle_is_rejected(tmp_path, monkeypatch):
    bundle = _valid_test_bundle(tmp_path, monkeypatch)
    link = tmp_path / "bundle-link.pem"
    link.symlink_to(bundle)
    with pytest.raises(runner.Batch2DCExecutionError, match="must not be a symlink"):
        runner.validate_ca_bundle(link)


def test_directory_ca_bundle_is_rejected(tmp_path):
    with pytest.raises(runner.Batch2DCExecutionError, match="regular file"):
        runner.validate_ca_bundle(tmp_path)


def test_unreadable_ca_bundle_is_rejected(tmp_path, monkeypatch):
    bundle = tmp_path / "unreadable.pem"
    bundle.write_bytes(b"bundle\n")
    monkeypatch.setattr(runner.os, "access", lambda path, mode: False)
    with pytest.raises(runner.Batch2DCExecutionError, match="not readable"):
        runner.validate_ca_bundle(bundle)


def test_verified_ssl_context_and_provenance_are_exact(tmp_path, monkeypatch):
    bundle = _valid_test_bundle(tmp_path, monkeypatch)
    context = FakeSSLContext()
    monkeypatch.setattr(
        runner.ssl, "create_default_context", lambda *, cafile: context
    )
    actual_context, provenance = runner.build_verified_ssl_context(bundle)
    assert actual_context is context
    assert provenance == _tls_trust()
    assert provenance["base_trust"] == "certifi"
    assert provenance["intermediate_role"] == "CHAIN_COMPLETION_ONLY"


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (FakeSSLContext(verify_mode=ssl.CERT_NONE), "certificate verification"),
        (FakeSSLContext(check_hostname=False), "hostname verification"),
        (
            FakeSSLContext(
                verify_flags=int(getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0))
            ),
            "partial-chain",
        ),
    ],
)
def test_unsafe_ssl_context_is_rejected(tmp_path, monkeypatch, context, message):
    if message == "partial-chain" and not getattr(
        ssl, "VERIFY_X509_PARTIAL_CHAIN", 0
    ):
        pytest.skip("Runtime has no partial-chain flag")
    bundle = _valid_test_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner.ssl, "create_default_context", lambda *, cafile: context
    )
    with pytest.raises(runner.Batch2DCExecutionError, match=message):
        runner.build_verified_ssl_context(bundle)


def test_tls_downloader_injects_verified_context_and_preserves_stage_barrier(
    tmp_path, monkeypatch
):
    context = FakeSSLContext()
    calls = []

    def fake_urlopen(request, timeout, context=None):
        calls.append((request.full_url, timeout, context))
        return FakeResponse(url=request.full_url)

    monkeypatch.setattr(
        runner.source_intake, "validate_pdf_source", lambda path, sha: sha
    )
    entries = [_entry(index) for index in range(1, 51)]
    downloader = runner.make_tls_downloader(context, urlopen=fake_urlopen)
    records = runner.run_download_stage(entries, tmp_path, downloader=downloader)
    assert len(records) == 50
    assert len(calls) == 50
    assert all(call[2] is context for call in calls)
    assert [record["acquisition_id"] for record in records] == runner.EXPECTED_IDS


def test_tls_protocol_records_exact_provenance(tmp_path, cohort):
    protocol = runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    assert protocol["tls_protocol_schema_version"] == runner.TLS_PROTOCOL_SCHEMA_VERSION
    assert protocol["tls_trust"] == _tls_trust()


def test_protocol_freezes_exact_50_cohort_urls_in_order(tmp_path, cohort):
    protocol = runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    expected = [entry["source_url"] for entry in cohort["entries"]]
    assert len(expected) == len(set(expected)) == 50
    assert protocol["allowed_source_urls"] == expected
    assert protocol["frozen_source_count"] == 50


def test_resume_with_same_tls_provenance_is_accepted(tmp_path, cohort):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    loaded = runner._load_resume_protocol(
        tmp_path, runner.COHORT_CHECKPOINT_COMMIT, _tls_trust(), cohort["entries"]
    )
    assert loaded["tls_trust"] == _tls_trust()


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("combined_ca_bundle_sha256", "0" * 64),
        ("base_certifi_sha256", "1" * 64),
        ("supplemental_intermediate_der_sha256", "2" * 64),
        ("partial_chain_enabled", True),
        ("check_hostname", False),
    ],
)
def test_resume_with_changed_tls_provenance_is_rejected(
    tmp_path, cohort, field, changed
):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    changed_trust = _tls_trust()
    changed_trust[field] = changed
    with pytest.raises(runner.Batch2DCExecutionError, match="tls_trust"):
        runner._load_resume_protocol(
            tmp_path,
            runner.COHORT_CHECKPOINT_COMMIT,
            changed_trust,
            cohort["entries"],
        )


def test_old_protocol_without_tls_section_is_rejected(tmp_path):
    old = {
        "protocol_id": runner.PROTOCOL_ID,
        "run_id": tmp_path.name,
        "parent_cohort_checkpoint_commit": runner.COHORT_CHECKPOINT_COMMIT,
        "parent_cohort_fingerprint": runner.COHORT_FINGERPRINT,
        "parent_cohort_file_sha256": runner.COHORT_FILE_SHA256,
        "execution_git_commit": runner.COHORT_CHECKPOINT_COMMIT,
        "runner_sha256": runner._runner_sha256(),
        "baseline_manifest_sha256": runner.BASELINE_MANIFEST_SHA256,
        "baseline_source_count": runner.BASELINE_SOURCE_COUNT,
    }
    runner.strict_json_write(tmp_path / "execution_protocol.json", old)
    with pytest.raises(
        runner.Batch2DCExecutionError, match="protocol schema"
    ):
        runner._load_resume_protocol(
            tmp_path,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            [_entry(index) for index in range(1, 51)],
        )


def test_foreign_batch2db_partial_run_is_rejected_before_network(cohort):
    historical = (
        PROJECT_ROOT
        / runner.RUNS_RELATIVE
        / "independent_batch2db_execution_v1_20261103_075028_894809_8bdd74ba"
    )
    assert historical.is_dir()
    with pytest.raises(runner.Batch2DCExecutionError):
        runner._load_resume_protocol(
            historical,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            cohort["entries"],
        )


def test_foreign_batch2db_blocked_run_is_rejected_before_network(cohort):
    blocked_r5 = (
        PROJECT_ROOT
        / runner.RUNS_RELATIVE
        / "independent_batch2db_execution_v1_20261103_153757_174475_8e8a10d9"
    )
    assert blocked_r5.is_dir()
    with pytest.raises(runner.Batch2DCExecutionError):
        runner._load_resume_protocol(
            blocked_r5,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            cohort["entries"],
        )


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
        lambda root: {
            "sources": [
                {"acquisition_id": f"old-{i}"}
                for i in range(runner.BASELINE_SOURCE_COUNT - 1)
            ]
        },
    )
    with pytest.raises(runner.Batch2DCExecutionError, match="does not reconcile"):
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
    assert "verify_flags |= " not in source
    assert "verify_flags = verify_flags |" not in source


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
        runner.main(
            [
                "--project-root",
                str(PROJECT_ROOT),
                "--ca-bundle",
                "unused.pem",
                unsafe_flag,
            ]
        )
    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_cli_requires_explicit_ca_bundle(capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--project-root", str(PROJECT_ROOT)])
    assert exc.value.code == 2
    assert "--ca-bundle" in capsys.readouterr().err


def test_no_main_stress_seed_query_gt_or_model_fields_in_protocol(tmp_path, cohort):
    protocol = runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    assert "main" not in protocol
    assert "stress" not in protocol
    assert "sampling_seed" not in protocol
    assert "query_records" not in protocol
    assert "ground_truth_records" not in protocol
    assert protocol["selection_seed_query_gt_model_retrieval_forbidden"] is True


def test_initial_baseline_gate_matches_repository(intake):
    runner.validate_initial_baseline(PROJECT_ROOT, intake)


def test_wrong_baseline_manifest_sha_is_rejected(intake, monkeypatch):
    monkeypatch.setattr(runner, "BASELINE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(runner.Batch2DCExecutionError, match="manifest SHA256"):
        runner.validate_initial_baseline(PROJECT_ROOT, intake)


def test_existing_batch2dc_source_is_rejected_for_new_run(intake, monkeypatch):
    changed = copy.deepcopy(intake)
    changed["sources"][0]["acquisition_id"] = runner.EXPECTED_IDS[0]
    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path: runner.BASELINE_MANIFEST_SHA256,
    )
    with pytest.raises(runner.Batch2DCExecutionError, match="already registered"):
        runner.validate_initial_baseline(PROJECT_ROOT, changed)


def test_download_outside_official_domain_is_rejected(tmp_path):
    entry = _entry()
    entry["source_url"] = "https://example.com/outside.pdf"
    with pytest.raises(runner.Batch2DCExecutionError, match="outside official"):
        runner.download_entry(entry, tmp_path, opener=lambda *args: pytest.fail())


def test_registration_failure_propagates_for_all_frozen_members(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry) for entry in entries]
    calls = []

    def fail_registration(**kwargs):
        calls.append(kwargs["acquisition_id"])
        raise runner.source_intake.SourceIntakeError("synthetic registration failure")

    monkeypatch.setattr(runner, "_manifest_by_id", lambda root: ({"sources": []}, {}))
    monkeypatch.setattr(
        runner.source_intake,
        "inventory_independent_sources",
        lambda root: ([], {}, []),
    )
    records = runner.run_registration_stage(
        tmp_path, entries, downloads, tmp_path, registrar=fail_registration
    )
    assert calls == runner.EXPECTED_IDS
    assert {record["terminal_disposition"] for record in records} == {
        "REGISTRATION_FAILED"
    }
    assert all(record["raw_sha256"] is None for record in records)


def test_extraction_failure_propagates_for_all_registered_members(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry) for entry in entries]
    registrations = [_registration_record(entry) for entry in entries]
    sources = {
        entry["acquisition_id"]: {
            "acquisition_id": entry["acquisition_id"],
            "acquisition_status": "hash_verified",
        }
        for entry in entries
    }
    calls = []

    def fail_extraction(root, acquisition_id, force_ocr):
        calls.append((acquisition_id, force_ocr))
        raise runner.source_intake.SourceIntakeError("synthetic extraction failure")

    monkeypatch.setattr(
        runner, "_manifest_by_id", lambda root: ({"sources": list(sources.values())}, sources)
    )
    monkeypatch.setattr(
        runner.source_intake,
        "inventory_independent_sources",
        lambda root: ([], {}, []),
    )
    records = runner.run_extraction_stage(
        tmp_path,
        entries,
        downloads,
        registrations,
        tmp_path,
        extractor=fail_extraction,
    )
    assert [item[0] for item in calls] == runner.EXPECTED_IDS
    assert all(item[1] is False for item in calls)
    assert {record["terminal_disposition"] for record in records} == {
        "EXTRACTION_FAILED"
    }


def _registered_source(entry, download, status="hash_verified"):
    return {
        "acquisition_id": entry["acquisition_id"],
        "source_url": entry["source_url"],
        "source_domain": runner.OFFICIAL_DOMAIN,
        "original_filename": runner.registration_filename(entry),
        "sha256": download["staged_sha256"],
        "download_timestamp": download["download_timestamp"],
        "acquisition_status": status,
        "case_id": None,
        "judgment_number": None,
        "court": None,
        "decision_date": None,
        "procedure_stage": None,
        "metadata_verification": {
            "case_id": False,
            "court": False,
            "decision_date": False,
            "development_overlap_check": False,
            "internal_dedup_check": False,
            "judgment_number": False,
            "procedure_stage": False,
            "underlying_dispute_review": False,
        },
        "rejection_reasons": [],
        "structured_path": None,
        "structured_sha256": None,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("case_id", "01/2026/TLST-DS"), "Identity metadata"),
        (("judgment_number", "01/2026/DS-ST"), "Identity metadata"),
        (("court", "TAND synthetic"), "Identity metadata"),
        (("decision_date", "2026-11-06"), "Identity metadata"),
        (("procedure_stage", "Sơ thẩm"), "Identity metadata"),
        (("metadata_verification", {"case_id": True}), "Review metadata"),
        (("rejection_reasons", ["reviewed"]), "review rejection"),
        (("acquisition_status", "eligible"), "forbidden status"),
    ],
)
def test_identity_review_and_promotion_fields_fail_closed(mutation, message):
    entry = _entry()
    download = _download_record(entry)
    source = _registered_source(entry, download)
    source[mutation[0]] = mutation[1]
    with pytest.raises(runner.Batch2DCExecutionError, match=message):
        runner._verify_registered_source(source, entry, download)


def test_valid_download_resume_prefix_is_accepted(tmp_path, monkeypatch):
    entries = [_entry(index) for index in range(1, 51)]
    first = _download_record(entries[0])
    runner.strict_json_write(
        tmp_path / "download_ledger.json",
        runner._ledger("batch2dc_download_ledger_v1", [first]),
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / f"{entries[0]['acquisition_id']}.pdf").write_bytes(b"%PDF-test")
    monkeypatch.setattr(runner, "_manifest_by_id", lambda root: ({"sources": []}, {}))
    runner.validate_resume_state(tmp_path, tmp_path, entries)


def test_resume_registration_cannot_cross_incomplete_download_barrier(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    runner.strict_json_write(
        tmp_path / "download_ledger.json",
        runner._ledger("batch2dc_download_ledger_v1", [_download_record(entries[0])]),
    )
    runner.strict_json_write(
        tmp_path / "registration_ledger.json",
        runner._ledger(
            "batch2dc_registration_ledger_v1", [_registration_record(entries[0])]
        ),
    )
    monkeypatch.setattr(runner, "_manifest_by_id", lambda root: ({"sources": []}, {}))
    with pytest.raises(runner.Batch2DCExecutionError, match="download barrier"):
        runner.validate_resume_state(tmp_path, tmp_path, entries)


def test_resume_extraction_cannot_cross_incomplete_registration_barrier(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [_download_record(entry) for entry in entries]
    runner.strict_json_write(
        tmp_path / "download_ledger.json",
        runner._ledger("batch2dc_download_ledger_v1", downloads),
    )
    runner.strict_json_write(
        tmp_path / "registration_ledger.json",
        runner._ledger(
            "batch2dc_registration_ledger_v1", [_registration_record(entries[0])]
        ),
    )
    runner.strict_json_write(
        tmp_path / "extraction_ledger.json",
        runner._ledger(
            "batch2dc_extraction_ledger_v1",
            [
                {
                    "acquisition_id": entries[0]["acquisition_id"],
                    "source_url": entries[0]["source_url"],
                    "terminal_disposition": "EXTRACTION_FAILED",
                }
            ],
        ),
    )
    monkeypatch.setattr(runner, "_manifest_by_id", lambda root: ({"sources": []}, {}))
    with pytest.raises(runner.Batch2DCExecutionError, match="registration barrier"):
        runner.validate_resume_state(tmp_path, tmp_path, entries)


def test_resume_manifest_cannot_run_ahead_of_registration_ledger(
    tmp_path, monkeypatch
):
    entries = [_entry(index) for index in range(1, 51)]
    source = {"acquisition_id": entries[0]["acquisition_id"]}
    monkeypatch.setattr(
        runner, "_manifest_by_id", lambda root: ({"sources": [source]}, {source["acquisition_id"]: source})
    )
    (tmp_path / "downloads").mkdir()
    with pytest.raises(runner.Batch2DCExecutionError, match="ahead of registration"):
        runner.validate_resume_state(tmp_path, tmp_path, entries)


def test_changed_execution_head_is_rejected_on_resume(tmp_path, cohort):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    with pytest.raises(runner.Batch2DCExecutionError, match="execution HEAD"):
        runner._load_resume_protocol(
            tmp_path, "f" * 40, _tls_trust(), cohort["entries"]
        )


def test_changed_runner_sha_is_rejected_on_resume(tmp_path, cohort):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    protocol_path = tmp_path / "execution_protocol.json"
    protocol = runner.strict_json_load(protocol_path)
    protocol["runner_sha256"] = "0" * 64
    runner.strict_json_write(protocol_path, protocol)
    with pytest.raises(runner.Batch2DCExecutionError, match="runner_sha256"):
        runner._load_resume_protocol(
            tmp_path,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            cohort["entries"],
        )


def test_changed_frozen_url_scope_is_rejected_on_resume(tmp_path, cohort):
    runner._create_protocol(
        PROJECT_ROOT,
        tmp_path,
        cohort,
        runner.COHORT_CHECKPOINT_COMMIT,
        runner.BASELINE_MANIFEST_SHA256,
        _tls_trust(),
    )
    protocol_path = tmp_path / "execution_protocol.json"
    protocol = runner.strict_json_load(protocol_path)
    protocol["allowed_source_urls"][0] = (
        "https://congbobanan.toaan.gov.vn/replacement.pdf"
    )
    runner.strict_json_write(protocol_path, protocol)
    with pytest.raises(runner.Batch2DCExecutionError, match="allowed_source_urls"):
        runner._load_resume_protocol(
            tmp_path,
            runner.COHORT_CHECKPOINT_COMMIT,
            _tls_trust(),
            cohort["entries"],
        )


def test_finalize_rejects_eligible_promotion(tmp_path, monkeypatch):
    (tmp_path / runner.MANIFEST_RELATIVE).parent.mkdir(parents=True)
    (tmp_path / runner.MANIFEST_RELATIVE).write_text("{}\n", encoding="utf-8")
    entry = _entry()
    download = _download_record(entry)
    registration = _registration_record(entry)
    extraction = {
        "acquisition_id": entry["acquisition_id"],
        "source_url": entry["source_url"],
        "terminal_disposition": "EXTRACTION_FAILED",
        "technical_failure": "synthetic",
        "structured_path": None,
        "structured_sha256": None,
    }
    sources = [
        {"acquisition_id": f"old-{index}", "acquisition_status": "eligible"}
        for index in range(runner.BASELINE_SOURCE_COUNT)
    ] + [{"acquisition_id": entry["acquisition_id"], "acquisition_status": "eligible"}]
    monkeypatch.setattr(
        runner.source_intake, "load_acquisition_manifest", lambda root: {"sources": sources}
    )
    with pytest.raises(runner.Batch2DCExecutionError, match="statuses do not reconcile"):
        runner._finalize(
            tmp_path,
            tmp_path,
            {
                "execution_git_commit": runner.COHORT_CHECKPOINT_COMMIT,
                "runner_sha256": "a" * 64,
                "baseline_manifest_sha256": runner.BASELINE_MANIFEST_SHA256,
                "tls_trust": _tls_trust(),
            },
            [entry],
            [download],
            [registration],
            [extraction],
        )


def test_partial_finalize_reconciles_without_replacement(tmp_path, monkeypatch):
    (tmp_path / runner.MANIFEST_RELATIVE).parent.mkdir(parents=True)
    (tmp_path / runner.MANIFEST_RELATIVE).write_text("{}\n", encoding="utf-8")
    entries = [_entry(index) for index in range(1, 51)]
    downloads = [
        _download_record(entry, "DOWNLOAD_HTTP_FAILURE") for entry in entries
    ]
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
    for name, version, records in (
        ("download_ledger.json", "batch2dc_download_ledger_v1", downloads),
        (
            "registration_ledger.json",
            "batch2dc_registration_ledger_v1",
            registrations,
        ),
        ("extraction_ledger.json", "batch2dc_extraction_ledger_v1", extractions),
    ):
        runner.strict_json_write(tmp_path / name, runner._ledger(version, records))
    runner.strict_json_write(tmp_path / "execution_protocol.json", {"frozen": True})
    monkeypatch.setattr(
        runner.source_intake,
        "load_acquisition_manifest",
        lambda root: {
            "sources": [
                {"acquisition_id": f"old-{index}", "acquisition_status": "eligible"}
                for index in range(runner.BASELINE_SOURCE_COUNT)
            ]
        },
    )
    monkeypatch.setattr(
        runner.source_intake,
        "inventory_independent_sources",
        lambda root: ([], {}, []),
    )
    manifest = runner._finalize(
        tmp_path,
        tmp_path,
        {
            "execution_git_commit": runner.COHORT_CHECKPOINT_COMMIT,
            "runner_sha256": "a" * 64,
            "baseline_manifest_sha256": runner.BASELINE_MANIFEST_SHA256,
            "tls_trust": _tls_trust(),
        },
        entries,
        downloads,
        registrations,
        extractions,
    )
    summary = runner.strict_json_load(tmp_path / "stage_summary.json")
    assert manifest["status"] == "TECHNICAL_ACQUISITION_PARTIAL"
    assert summary["replacement_or_topup_count"] == 0
    assert summary["registered_count"] == summary["extracted_count"] == 0
