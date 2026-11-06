"""Execute frozen Batch 2D-C technical acquisition stages.

The runner is intentionally limited to download, registration, and normal
extraction.  Registration and extraction delegate every authoritative
repository mutation to :mod:`source_intake`.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import http.client
import json
import os
import re
import ssl
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from . import source_intake, topup_acquisition_cohort
from .sampling_policy import sha256_file, strict_json_load, strict_json_write


PROTOCOL_ID = "independent_batch2dc_execution_v1"
TLS_PROTOCOL_SCHEMA_VERSION = "batch2dc_verified_tls_trust_v1"
TLS_TRUST_MODE = "CERTIFI_PLUS_VERIFIED_GLOBALSIGN_INTERMEDIATE"
EXPECTED_BASE_CERTIFI_SHA256 = (
    "fc9165a12403263e7ebfbdad7be7a3eac0fa5d325d3c70465f28d3690072ca28"
)
EXPECTED_INTERMEDIATE_DER_SHA256 = (
    "b676ffa3179e8812093a1b5eafee876ae7a6aaf231078dad1bfb21cd2893764a"
)
EXPECTED_ROOT_R3_SHA256 = (
    "cbb522d7b7f127ad6a0113865bdf1cd4102e7d0759af635a7cf4720dc963c53b"
)
EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256 = (
    "f08632660e469b1953e83ecc06d02ffb9a6dd4ad3de44502979679197d6f3fcc"
)
COHORT_ID = "batch2dc-20261105"
COHORT_CHECKPOINT_COMMIT = "1d6c2dc69e9251bf78db31ae9ee6dce2125ea638"
COHORT_FINGERPRINT = "e4490f6f5cba6aecfee563edf9b5035944ff964160e8703e43533be8a8f9b8e4"
COHORT_FILE_SHA256 = "3afc710119577b5a154b1ae09e41a3c225ea6122130fa5e7217840a2901fbaf1"
BASELINE_MANIFEST_SHA256 = "4716e61118ce77dee60597940f68247423be685a27b6dde1c3c479393bbc5094"
BASELINE_SOURCE_COUNT = 317
BASELINE_STATUS_COUNTS = {"eligible": 257, "extracted": 59, "hash_verified": 1}
OFFICIAL_DOMAIN = "congbobanan.toaan.gov.vn"
EXPECTED_IDS = [f"batch2dc-20261105-{index:03d}" for index in range(1, 51)]
MAX_ATTEMPTS = 3
RETRY_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
BACKOFF_SECONDS = (1.0, 2.0)
COHORT_RELATIVE = Path(
    "backend/dataLaw/independent_intake/cohorts/batch2dc-20261105.json"
)
MANIFEST_RELATIVE = source_intake.ACQUISITION_MANIFEST_RELATIVE
RUNS_RELATIVE = Path("backend/data/runs")

DOWNLOAD_TERMINAL = {
    "DOWNLOADED_VALID_PDF",
    "DOWNLOAD_HTTP_FAILURE",
    "DOWNLOAD_NETWORK_FAILURE",
    "DOWNLOAD_REDIRECT_DOMAIN_REJECTED",
    "DOWNLOAD_INVALID_PDF",
}
REGISTRATION_TERMINAL = {
    "REGISTERED_HASH_VERIFIED",
    "NOT_REGISTERED_DOWNLOAD_FAILURE",
    "REGISTRATION_FAILED",
}
EXTRACTION_TERMINAL = {
    "EXTRACTED",
    "NOT_EXTRACTED_DOWNLOAD_FAILURE",
    "NOT_EXTRACTED_REGISTRATION_FAILURE",
    "EXTRACTION_FAILED",
}


class Batch2DCExecutionError(RuntimeError):
    """Raised when the frozen execution protocol cannot proceed safely."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _runner_sha256() -> str:
    return sha256_file(Path(__file__))


def _git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Batch2DCExecutionError(f"Git command failed: git {' '.join(args)}")
    return result.stdout.strip()


def validate_git_gate(project_root: Path) -> str:
    """Require a committed HEAD descended from the frozen cohort checkpoint."""

    head = _git_output(project_root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise Batch2DCExecutionError("Execution requires a valid Git HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", COHORT_CHECKPOINT_COMMIT, head],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise Batch2DCExecutionError(
            "Frozen cohort checkpoint is not an ancestor of execution HEAD"
        )
    if subprocess.run(
        ["git", "diff", "--quiet", "--"], cwd=project_root, check=False
    ).returncode != 0:
        raise Batch2DCExecutionError("Tracked execution inputs are dirty")
    if subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=project_root,
        check=False,
    ).returncode != 0:
        raise Batch2DCExecutionError("Staged execution inputs are forbidden")
    return head


def validate_original_filename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise Batch2DCExecutionError("Frozen original filename must be non-empty")
    if value != unicodedata.normalize("NFC", value):
        raise Batch2DCExecutionError("Frozen original filename must use Unicode NFC")
    if Path(value).name != value or value in {".", ".."}:
        raise Batch2DCExecutionError("Frozen original filename must be a basename")
    if Path(value).suffix.casefold() != ".pdf":
        raise Batch2DCExecutionError("Frozen original filename must end in .pdf")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise Batch2DCExecutionError("Frozen original filename contains controls")
    return value


def registration_filename(entry: Mapping[str, Any]) -> str:
    """Resolve the safe original basename from immutable cohort metadata."""

    documentary_name = entry.get("official_document_title_or_filename")
    try:
        return validate_original_filename(documentary_name)
    except Batch2DCExecutionError:
        source_url = entry.get("source_url")
        if not isinstance(source_url, str):
            raise Batch2DCExecutionError("Frozen source URL must be a string")
        url_basename = unicodedata.normalize(
            "NFC", unquote(urlsplit(source_url).path.rsplit("/", 1)[-1])
        )
        try:
            return validate_original_filename(url_basename)
        except Batch2DCExecutionError as exc:
            raise Batch2DCExecutionError(
                "Frozen metadata provides no safe deterministic PDF basename"
            ) from exc


def validate_frozen_cohort(cohort: dict, intake: dict) -> dict[str, Any]:
    """Validate exact cohort identity while ignoring only its own resume records."""

    filtered_intake = copy.deepcopy(intake)
    filtered_intake["sources"] = [
        source
        for source in filtered_intake["sources"]
        if source.get("acquisition_id") not in EXPECTED_IDS
    ]
    filtered_intake["immutable_provenance"] = [
        item
        for item in filtered_intake["immutable_provenance"]
        if item.get("acquisition_id") not in EXPECTED_IDS
    ]
    development = strict_json_load(
        Path(__file__).resolve().parents[4]
        / topup_acquisition_cohort.DEFAULT_DEVELOPMENT_DEDUP_RELATIVE
    )
    validated = topup_acquisition_cohort.validate_cohort(
        cohort, filtered_intake, development
    )
    entries = cohort["entries"]
    if cohort.get("cohort_fingerprint") != COHORT_FINGERPRINT:
        raise Batch2DCExecutionError("Frozen cohort fingerprint mismatch")
    if [entry["acquisition_id"] for entry in entries] != EXPECTED_IDS:
        raise Batch2DCExecutionError("Frozen cohort IDs must be exactly 001..050")
    for entry in entries:
        if entry["source_domain"] != OFFICIAL_DOMAIN:
            raise Batch2DCExecutionError("Frozen source domain mismatch")
        if urlsplit(entry["source_url"]).hostname != OFFICIAL_DOMAIN:
            raise Batch2DCExecutionError("Frozen source URL leaves official domain")
        registration_filename(entry)
    return validated


def load_execution_inputs(project_root: Path) -> tuple[dict, dict, str]:
    root = Path(project_root).resolve()
    cohort_path = root / COHORT_RELATIVE
    manifest_path = root / MANIFEST_RELATIVE
    if sha256_file(cohort_path) != COHORT_FILE_SHA256:
        raise Batch2DCExecutionError("Committed frozen cohort file SHA256 mismatch")
    cohort = strict_json_load(cohort_path)
    intake = source_intake.load_acquisition_manifest(root)
    validate_frozen_cohort(cohort, intake)
    head = validate_git_gate(root)
    return cohort, intake, head


def validate_initial_baseline(project_root: Path, manifest: dict) -> None:
    manifest_path = Path(project_root).resolve() / MANIFEST_RELATIVE
    if sha256_file(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise Batch2DCExecutionError("Baseline acquisition manifest SHA256 mismatch")
    if len(manifest["sources"]) != BASELINE_SOURCE_COUNT:
        raise Batch2DCExecutionError("Baseline acquisition source count mismatch")
    statuses = {
        status: sum(source["acquisition_status"] == status for source in manifest["sources"])
        for status in ("eligible", "extracted", "hash_verified")
    }
    if statuses != BASELINE_STATUS_COUNTS:
        raise Batch2DCExecutionError("Baseline acquisition statuses mismatch")
    if any(source["acquisition_id"] in EXPECTED_IDS for source in manifest["sources"]):
        raise Batch2DCExecutionError("Batch 2D-C source already registered")
    for relative in (
        source_intake.INTAKE_ROOT_RELATIVE / "raw",
        source_intake.INTAKE_ROOT_RELATIVE / "structured",
        source_intake.PROVENANCE_ROOT_RELATIVE,
        source_intake.TRANSACTION_ROOT_RELATIVE,
    ):
        path = Path(project_root).resolve() / relative
        if any(path.glob("batch2dc-20261105-*")):
            raise Batch2DCExecutionError(f"Unexpected Batch 2D-C artifact under {relative}")


def validate_ca_bundle(ca_bundle: Path) -> Path:
    """Validate the exact, previously audited supplemental CA bundle."""

    candidate = Path(ca_bundle)
    if not candidate.exists():
        raise Batch2DCExecutionError("Explicit CA bundle does not exist")
    if candidate.is_symlink():
        raise Batch2DCExecutionError("Explicit CA bundle must not be a symlink")
    if not candidate.is_file():
        raise Batch2DCExecutionError("Explicit CA bundle must be a regular file")
    if not os.access(candidate, os.R_OK):
        raise Batch2DCExecutionError("Explicit CA bundle is not readable")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Batch2DCExecutionError("Explicit CA bundle cannot be resolved") from exc
    if not resolved.is_file():
        raise Batch2DCExecutionError("Resolved CA bundle must be a regular file")
    if sha256_file(resolved) != EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256:
        raise Batch2DCExecutionError("Explicit CA bundle SHA256 mismatch")
    return resolved


def build_verified_ssl_context(ca_bundle: Path) -> tuple[ssl.SSLContext, dict[str, Any]]:
    """Build fail-closed TLS trust from the fingerprint-pinned bundle."""

    resolved = validate_ca_bundle(ca_bundle)
    context = ssl.create_default_context(cafile=str(resolved))
    if context.verify_mode != ssl.CERT_REQUIRED:
        raise Batch2DCExecutionError("TLS context must require certificate verification")
    if context.check_hostname is not True:
        raise Batch2DCExecutionError("TLS context must enforce hostname verification")
    partial_chain_flag = getattr(ssl, "VERIFY_X509_PARTIAL_CHAIN", 0)
    partial_chain_enabled = bool(
        partial_chain_flag and context.verify_flags & partial_chain_flag
    )
    if partial_chain_enabled:
        raise Batch2DCExecutionError("TLS partial-chain trust is forbidden")
    provenance = {
        "schema_version": TLS_PROTOCOL_SCHEMA_VERSION,
        "mode": TLS_TRUST_MODE,
        "base_certifi_sha256": EXPECTED_BASE_CERTIFI_SHA256,
        "supplemental_intermediate_der_sha256": EXPECTED_INTERMEDIATE_DER_SHA256,
        "expected_root_r3_sha256": EXPECTED_ROOT_R3_SHA256,
        "combined_ca_bundle_sha256": EXPECTED_SUPPLEMENTAL_BUNDLE_SHA256,
        "verify_mode": "CERT_REQUIRED",
        "check_hostname": True,
        "partial_chain_enabled": False,
        "base_trust": "certifi",
        "intermediate_role": "CHAIN_COMPLETION_ONLY",
    }
    return context, provenance


def make_tls_downloader(
    context: ssl.SSLContext,
    *,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> Callable[[dict, Path], dict]:
    """Bind the verified context to the existing download implementation."""

    def verified_opener(request: urllib.request.Request, timeout: int) -> Any:
        return urlopen(request, timeout=timeout, context=context)

    def downloader(entry: dict, downloads_dir: Path) -> dict:
        return download_entry(entry, downloads_dir, opener=verified_opener)

    return downloader


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    strict_json_write(path, dict(value))


def _ledger(version: str, records: list[dict]) -> dict[str, Any]:
    return {"ledger_version": version, "cohort_id": COHORT_ID, "records": records}


def _validate_record_coverage(
    records: list[dict], entries: list[dict], allowed_statuses: set[str], label: str
) -> None:
    expected = [entry["acquisition_id"] for entry in entries]
    actual = [record.get("acquisition_id") for record in records]
    if actual != expected:
        raise Batch2DCExecutionError(f"{label} ledger does not cover exact frozen cohort")
    for record, entry in zip(records, entries):
        if record.get("source_url") != entry["source_url"]:
            raise Batch2DCExecutionError(f"{label} ledger source URL mismatch")
        if record.get("terminal_disposition") not in allowed_statuses:
            raise Batch2DCExecutionError(f"{label} ledger has non-terminal disposition")


def _response_status(response: Any) -> int:
    value = getattr(response, "status", None)
    if value is None and hasattr(response, "getcode"):
        value = response.getcode()
    return int(value)


def download_entry(
    entry: dict,
    downloads_dir: Path,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Download one exact frozen URL using the fixed retry policy."""

    acquisition_id = entry["acquisition_id"]
    if acquisition_id not in EXPECTED_IDS:
        raise Batch2DCExecutionError("Download ID is outside frozen cohort")
    source_url = entry["source_url"]
    if urlsplit(source_url).hostname != OFFICIAL_DOMAIN:
        raise Batch2DCExecutionError("Download URL is outside official domain")
    destination = downloads_dir / f"{acquisition_id}.pdf"
    partial = downloads_dir / f".{acquisition_id}.part"
    attempts = 0
    last_http_status: int | None = None
    last_error: str | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt
        try:
            request = urllib.request.Request(
                source_url,
                method="GET",
                headers={"User-Agent": "AI-supports-legislation-research/1.0"},
            )
            with opener(request, timeout=45) as response:
                last_http_status = _response_status(response)
                final_url = response.geturl()
                final_host = urlsplit(final_url).hostname
                if final_host != OFFICIAL_DOMAIN:
                    return {
                        "acquisition_id": acquisition_id,
                        "source_url": source_url,
                        "attempt_count": attempts,
                        "http_terminal_status": last_http_status,
                        "final_host": final_host,
                        "download_timestamp": None,
                        "staged_byte_count": None,
                        "staged_sha256": None,
                        "terminal_disposition": "DOWNLOAD_REDIRECT_DOMAIN_REJECTED",
                        "technical_failure": "Final redirect host left official domain",
                    }
                if last_http_status != 200:
                    raise urllib.error.HTTPError(
                        final_url, last_http_status, "Non-200 response", response.headers, None
                    )
                digest = hashlib.sha256()
                byte_count = 0
                with partial.open("xb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                completed_at = dt.datetime.now().astimezone().isoformat()
                os.replace(partial, destination)
                if destination.is_symlink() or not destination.is_file():
                    raise Batch2DCExecutionError("Staged download is not a regular file")
                sha256 = digest.hexdigest()
                try:
                    source_intake.validate_pdf_source(destination, sha256)
                except source_intake.SourceIntakeError as exc:
                    destination.unlink(missing_ok=True)
                    return {
                        "acquisition_id": acquisition_id,
                        "source_url": source_url,
                        "attempt_count": attempts,
                        "http_terminal_status": last_http_status,
                        "final_host": final_host,
                        "download_timestamp": completed_at,
                        "staged_byte_count": byte_count,
                        "staged_sha256": sha256,
                        "terminal_disposition": "DOWNLOAD_INVALID_PDF",
                        "technical_failure": str(exc),
                    }
                return {
                    "acquisition_id": acquisition_id,
                    "source_url": source_url,
                    "attempt_count": attempts,
                    "http_terminal_status": last_http_status,
                    "final_host": final_host,
                    "download_timestamp": completed_at,
                    "staged_byte_count": byte_count,
                    "staged_sha256": sha256,
                    "terminal_disposition": "DOWNLOADED_VALID_PDF",
                    "technical_failure": None,
                }
        except http.client.IncompleteRead as exc:
            partial.unlink(missing_ok=True)
            partial_byte_count = len(exc.partial)
            last_error = (
                "IncompleteRead: "
                f"{partial_byte_count} bytes read, {exc.expected} more expected"
            )
            if attempt < MAX_ATTEMPTS:
                sleeper(BACKOFF_SECONDS[attempt - 1])
                continue
            return {
                "acquisition_id": acquisition_id,
                "source_url": source_url,
                "attempt_count": attempts,
                "http_terminal_status": last_http_status,
                "final_host": None,
                "download_timestamp": None,
                "staged_byte_count": None,
                "staged_sha256": None,
                "terminal_disposition": "DOWNLOAD_NETWORK_FAILURE",
                "technical_failure": last_error,
            }
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            last_http_status = int(exc.code)
            last_error = f"HTTP {exc.code}"
            if exc.code in RETRY_HTTP_STATUSES and attempt < MAX_ATTEMPTS:
                sleeper(BACKOFF_SECONDS[attempt - 1])
                continue
            return {
                "acquisition_id": acquisition_id,
                "source_url": source_url,
                "attempt_count": attempts,
                "http_terminal_status": last_http_status,
                "final_host": None,
                "download_timestamp": None,
                "staged_byte_count": None,
                "staged_sha256": None,
                "terminal_disposition": "DOWNLOAD_HTTP_FAILURE",
                "technical_failure": last_error,
            }
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError) as exc:
            partial.unlink(missing_ok=True)
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                sleeper(BACKOFF_SECONDS[attempt - 1])
                continue
            return {
                "acquisition_id": acquisition_id,
                "source_url": source_url,
                "attempt_count": attempts,
                "http_terminal_status": last_http_status,
                "final_host": None,
                "download_timestamp": None,
                "staged_byte_count": None,
                "staged_sha256": None,
                "terminal_disposition": "DOWNLOAD_NETWORK_FAILURE",
                "technical_failure": last_error,
            }
        finally:
            partial.unlink(missing_ok=True)
    raise Batch2DCExecutionError("Unreachable retry state")


def _load_records(path: Path, expected_version: str) -> list[dict]:
    if not path.exists():
        return []
    payload = strict_json_load(path)
    if payload.get("ledger_version") != expected_version or payload.get("cohort_id") != COHORT_ID:
        raise Batch2DCExecutionError("Execution ledger identity mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise Batch2DCExecutionError("Execution ledger records must be a list")
    return records


def _validate_prefix(records: list[dict], entries: list[dict], allowed: set[str], label: str) -> None:
    if len(records) > len(entries):
        raise Batch2DCExecutionError(f"{label} ledger is longer than frozen cohort")
    for record, entry in zip(records, entries):
        if record.get("acquisition_id") != entry["acquisition_id"]:
            raise Batch2DCExecutionError(f"{label} resume ID mismatch")
        if record.get("source_url") != entry["source_url"]:
            raise Batch2DCExecutionError(f"{label} resume URL mismatch")
        if record.get("terminal_disposition") not in allowed:
            raise Batch2DCExecutionError(f"{label} resume disposition mismatch")


def run_download_stage(
    entries: list[dict],
    run_dir: Path,
    *,
    downloader: Callable[[dict, Path], dict] | None = None,
) -> list[dict]:
    downloads_dir = run_dir / "downloads"
    downloads_dir.mkdir(exist_ok=True)
    path = run_dir / "download_ledger.json"
    records = _load_records(path, "batch2dc_download_ledger_v1")
    _validate_prefix(records, entries, DOWNLOAD_TERMINAL, "Download")
    download = downloader or download_entry
    for entry in entries[len(records) :]:
        record = download(entry, downloads_dir)
        records.append(record)
        _atomic_json(path, _ledger("batch2dc_download_ledger_v1", records))
    _validate_record_coverage(records, entries, DOWNLOAD_TERMINAL, "Download")
    return records


def _manifest_by_id(project_root: Path) -> tuple[dict, dict[str, dict]]:
    manifest = source_intake.load_acquisition_manifest(project_root)
    return manifest, {source["acquisition_id"]: source for source in manifest["sources"]}


def _verify_registered_source(source: dict, entry: dict, download: dict) -> None:
    if source.get("source_url") != entry["source_url"]:
        raise Batch2DCExecutionError("Registered source URL differs from frozen cohort")
    if source.get("source_domain") != OFFICIAL_DOMAIN:
        raise Batch2DCExecutionError("Registered source domain mismatch")
    if source.get("original_filename") != registration_filename(entry):
        raise Batch2DCExecutionError("Registered original filename mismatch")
    if source.get("sha256") != download.get("staged_sha256"):
        raise Batch2DCExecutionError("Registered PDF SHA differs from download ledger")
    if source.get("download_timestamp") != download.get("download_timestamp"):
        raise Batch2DCExecutionError("Registered download timestamp mismatch")
    if source.get("acquisition_status") not in {"hash_verified", "extracted"}:
        raise Batch2DCExecutionError("Registered source has forbidden status")
    if any(
        source.get(field) is not None
        for field in (
            "case_id",
            "judgment_number",
            "court",
            "decision_date",
            "procedure_stage",
        )
    ):
        raise Batch2DCExecutionError("Identity metadata was populated during Phase 2D-B3.3B")
    verification = source.get("metadata_verification")
    if not isinstance(verification, dict) or any(verification.values()):
        raise Batch2DCExecutionError("Review metadata was populated during technical acquisition")
    if source.get("rejection_reasons") != []:
        raise Batch2DCExecutionError("Technical acquisition populated review rejection reasons")


def run_registration_stage(
    project_root: Path,
    entries: list[dict],
    downloads: list[dict],
    run_dir: Path,
    *,
    registrar: Callable[..., dict] | None = None,
) -> list[dict]:
    _validate_record_coverage(downloads, entries, DOWNLOAD_TERMINAL, "Download")
    path = run_dir / "registration_ledger.json"
    records = _load_records(path, "batch2dc_registration_ledger_v1")
    _validate_prefix(records, entries, REGISTRATION_TERMINAL, "Registration")
    register = registrar or source_intake.register_source
    _, by_id = _manifest_by_id(project_root)
    for index, entry in enumerate(entries[len(records) :], start=len(records)):
        download = downloads[index]
        acquisition_id = entry["acquisition_id"]
        if download["terminal_disposition"] != "DOWNLOADED_VALID_PDF":
            record = {
                "acquisition_id": acquisition_id,
                "source_url": entry["source_url"],
                "terminal_disposition": "NOT_REGISTERED_DOWNLOAD_FAILURE",
                "technical_failure": download["terminal_disposition"],
                "raw_sha256": None,
            }
        else:
            if acquisition_id in by_id:
                raise Batch2DCExecutionError(
                    "Manifest contains a source absent from registration ledger"
                )
            try:
                result = register(
                    project_root=project_root,
                    acquisition_id=acquisition_id,
                    input_pdf=run_dir / "downloads" / f"{acquisition_id}.pdf",
                    source_url=entry["source_url"],
                    source_domain=OFFICIAL_DOMAIN,
                    download_timestamp=download["download_timestamp"],
                    original_filename=registration_filename(entry),
                )
                record = {
                    "acquisition_id": acquisition_id,
                    "source_url": entry["source_url"],
                    "terminal_disposition": "REGISTERED_HASH_VERIFIED",
                    "technical_failure": None,
                    "raw_sha256": download["staged_sha256"],
                    "source_id": result.get("source_id"),
                }
                _, by_id = _manifest_by_id(project_root)
                _verify_registered_source(by_id[acquisition_id], entry, download)
            except (source_intake.SourceIntakeError, OSError, ValueError) as exc:
                record = {
                    "acquisition_id": acquisition_id,
                    "source_url": entry["source_url"],
                    "terminal_disposition": "REGISTRATION_FAILED",
                    "technical_failure": f"{type(exc).__name__}: {exc}",
                    "raw_sha256": None,
                }
        records.append(record)
        _atomic_json(path, _ledger("batch2dc_registration_ledger_v1", records))
    _validate_record_coverage(records, entries, REGISTRATION_TERMINAL, "Registration")
    manifest, by_id = _manifest_by_id(project_root)
    for entry, download, record in zip(entries, downloads, records):
        if record["terminal_disposition"] == "REGISTERED_HASH_VERIFIED":
            if entry["acquisition_id"] not in by_id:
                raise Batch2DCExecutionError("Registered ledger entry is absent from manifest")
            _verify_registered_source(by_id[entry["acquisition_id"]], entry, download)
        elif entry["acquisition_id"] in by_id:
            raise Batch2DCExecutionError("Failed registration unexpectedly appears in manifest")
    source_intake.inventory_independent_sources(project_root)
    return records


def run_extraction_stage(
    project_root: Path,
    entries: list[dict],
    downloads: list[dict],
    registrations: list[dict],
    run_dir: Path,
    *,
    extractor: Callable[..., dict] | None = None,
) -> list[dict]:
    _validate_record_coverage(registrations, entries, REGISTRATION_TERMINAL, "Registration")
    path = run_dir / "extraction_ledger.json"
    records = _load_records(path, "batch2dc_extraction_ledger_v1")
    _validate_prefix(records, entries, EXTRACTION_TERMINAL, "Extraction")
    extract = extractor or source_intake.extract_intake_source
    _, by_id = _manifest_by_id(project_root)
    for index, entry in enumerate(entries[len(records) :], start=len(records)):
        registration = registrations[index]
        acquisition_id = entry["acquisition_id"]
        if registration["terminal_disposition"] == "NOT_REGISTERED_DOWNLOAD_FAILURE":
            record = {
                "acquisition_id": acquisition_id,
                "source_url": entry["source_url"],
                "terminal_disposition": "NOT_EXTRACTED_DOWNLOAD_FAILURE",
                "technical_failure": registration["technical_failure"],
                "structured_path": None,
                "structured_sha256": None,
            }
        elif registration["terminal_disposition"] != "REGISTERED_HASH_VERIFIED":
            record = {
                "acquisition_id": acquisition_id,
                "source_url": entry["source_url"],
                "terminal_disposition": "NOT_EXTRACTED_REGISTRATION_FAILURE",
                "technical_failure": registration["technical_failure"],
                "structured_path": None,
                "structured_sha256": None,
            }
        else:
            source = by_id.get(acquisition_id)
            if source is None or source.get("acquisition_status") != "hash_verified":
                raise Batch2DCExecutionError("Extraction resume state mismatches manifest")
            try:
                result = extract(project_root, acquisition_id, False)
                manifest, by_id = _manifest_by_id(project_root)
                source = by_id[acquisition_id]
                record = {
                    "acquisition_id": acquisition_id,
                    "source_url": entry["source_url"],
                    "terminal_disposition": "EXTRACTED",
                    "technical_failure": None,
                    "structured_path": source["structured_path"],
                    "structured_sha256": source["structured_sha256"],
                    "source_id": result.get("source_id"),
                }
            except (source_intake.SourceIntakeError, OSError, ValueError) as exc:
                record = {
                    "acquisition_id": acquisition_id,
                    "source_url": entry["source_url"],
                    "terminal_disposition": "EXTRACTION_FAILED",
                    "technical_failure": f"{type(exc).__name__}: {exc}",
                    "structured_path": None,
                    "structured_sha256": None,
                }
        records.append(record)
        _atomic_json(path, _ledger("batch2dc_extraction_ledger_v1", records))
    _validate_record_coverage(records, entries, EXTRACTION_TERMINAL, "Extraction")
    source_intake.inventory_independent_sources(project_root)
    return records


def _new_run_id() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"


def _create_protocol(
    root: Path,
    run_dir: Path,
    cohort: dict,
    head: str,
    baseline_manifest_sha: str,
    tls_trust: Mapping[str, Any],
) -> dict:
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "parent_cohort_checkpoint_commit": COHORT_CHECKPOINT_COMMIT,
        "parent_cohort_fingerprint": COHORT_FINGERPRINT,
        "parent_cohort_file_sha256": COHORT_FILE_SHA256,
        "execution_git_commit": head,
        "runner_sha256": _runner_sha256(),
        "tls_protocol_schema_version": TLS_PROTOCOL_SCHEMA_VERSION,
        "tls_trust": dict(tls_trust),
        "baseline_manifest_sha256": baseline_manifest_sha,
        "baseline_source_count": BASELINE_SOURCE_COUNT,
        "frozen_source_count": len(cohort["entries"]),
        "allowed_source_urls": [entry["source_url"] for entry in cohort["entries"]],
        "retry_policy": {
            "maximum_attempts": MAX_ATTEMPTS,
            "retry_http_statuses": sorted(RETRY_HTTP_STATUSES),
            "deterministic_backoff_seconds": list(BACKOFF_SECONDS),
        },
        "registration_operation": "source_intake.register_source",
        "extraction_operation": "source_intake.extract_intake_source(force_ocr=False)",
        "identify_review_promote_forbidden": True,
        "selection_seed_query_gt_model_retrieval_forbidden": True,
    }
    _atomic_json(run_dir / "execution_protocol.json", protocol)
    return protocol


def _load_resume_protocol(
    run_dir: Path,
    head: str,
    tls_trust: Mapping[str, Any],
    entries: list[dict],
) -> dict:
    protocol = strict_json_load(run_dir / "execution_protocol.json")
    expected_keys = {
        "protocol_id",
        "run_id",
        "created_at",
        "parent_cohort_checkpoint_commit",
        "parent_cohort_fingerprint",
        "parent_cohort_file_sha256",
        "execution_git_commit",
        "runner_sha256",
        "tls_protocol_schema_version",
        "tls_trust",
        "baseline_manifest_sha256",
        "baseline_source_count",
        "frozen_source_count",
        "allowed_source_urls",
        "retry_policy",
        "registration_operation",
        "extraction_operation",
        "identify_review_promote_forbidden",
        "selection_seed_query_gt_model_retrieval_forbidden",
    }
    if set(protocol) != expected_keys:
        raise Batch2DCExecutionError("Resume execution protocol schema mismatch")
    if not isinstance(protocol.get("created_at"), str) or not protocol["created_at"]:
        raise Batch2DCExecutionError("Resume execution protocol timestamp is invalid")
    required = {
        "protocol_id": PROTOCOL_ID,
        "parent_cohort_checkpoint_commit": COHORT_CHECKPOINT_COMMIT,
        "parent_cohort_fingerprint": COHORT_FINGERPRINT,
        "parent_cohort_file_sha256": COHORT_FILE_SHA256,
        "runner_sha256": _runner_sha256(),
        "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "baseline_source_count": BASELINE_SOURCE_COUNT,
        "tls_protocol_schema_version": TLS_PROTOCOL_SCHEMA_VERSION,
        "frozen_source_count": 50,
        "allowed_source_urls": [entry["source_url"] for entry in entries],
        "retry_policy": {
            "maximum_attempts": MAX_ATTEMPTS,
            "retry_http_statuses": sorted(RETRY_HTTP_STATUSES),
            "deterministic_backoff_seconds": list(BACKOFF_SECONDS),
        },
        "registration_operation": "source_intake.register_source",
        "extraction_operation": "source_intake.extract_intake_source(force_ocr=False)",
        "identify_review_promote_forbidden": True,
        "selection_seed_query_gt_model_retrieval_forbidden": True,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise Batch2DCExecutionError(f"Resume protocol mismatch: {key}")
    if protocol.get("run_id") != run_dir.name:
        raise Batch2DCExecutionError("Resume run ID mismatch")
    if protocol.get("execution_git_commit") != head:
        raise Batch2DCExecutionError("Resume execution HEAD mismatch")
    if protocol.get("tls_trust") != dict(tls_trust):
        raise Batch2DCExecutionError("Resume protocol mismatch: tls_trust")
    return protocol


def _validate_failure_chain(
    entries: list[dict],
    downloads: list[dict],
    registrations: list[dict],
    extractions: list[dict],
) -> None:
    """Require terminal outcomes to propagate without replacement or skipping."""

    _validate_record_coverage(downloads, entries, DOWNLOAD_TERMINAL, "Download")
    _validate_record_coverage(
        registrations, entries, REGISTRATION_TERMINAL, "Registration"
    )
    _validate_record_coverage(extractions, entries, EXTRACTION_TERMINAL, "Extraction")
    for download, registration, extraction in zip(
        downloads, registrations, extractions
    ):
        downloaded = download["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
        registered = registration["terminal_disposition"] == "REGISTERED_HASH_VERIFIED"
        if not downloaded and registration["terminal_disposition"] != (
            "NOT_REGISTERED_DOWNLOAD_FAILURE"
        ):
            raise Batch2DCExecutionError("Download failure did not propagate to registration")
        if downloaded and registration["terminal_disposition"] == (
            "NOT_REGISTERED_DOWNLOAD_FAILURE"
        ):
            raise Batch2DCExecutionError("Successful download was marked as download failure")
        if not downloaded and extraction["terminal_disposition"] != (
            "NOT_EXTRACTED_DOWNLOAD_FAILURE"
        ):
            raise Batch2DCExecutionError("Download failure did not propagate to extraction")
        if downloaded and not registered and extraction["terminal_disposition"] != (
            "NOT_EXTRACTED_REGISTRATION_FAILURE"
        ):
            raise Batch2DCExecutionError(
                "Registration failure did not propagate to extraction"
            )
        if registered and extraction["terminal_disposition"] not in {
            "EXTRACTED",
            "EXTRACTION_FAILED",
        }:
            raise Batch2DCExecutionError("Registered source has invalid extraction outcome")


def validate_resume_state(
    project_root: Path, run_dir: Path, entries: list[dict]
) -> None:
    """Validate ledger barriers and manifest consistency before resume work."""

    downloads = _load_records(
        run_dir / "download_ledger.json", "batch2dc_download_ledger_v1"
    )
    registrations = _load_records(
        run_dir / "registration_ledger.json", "batch2dc_registration_ledger_v1"
    )
    extractions = _load_records(
        run_dir / "extraction_ledger.json", "batch2dc_extraction_ledger_v1"
    )
    _validate_prefix(downloads, entries, DOWNLOAD_TERMINAL, "Download")
    _validate_prefix(registrations, entries, REGISTRATION_TERMINAL, "Registration")
    _validate_prefix(extractions, entries, EXTRACTION_TERMINAL, "Extraction")
    if registrations and len(downloads) != len(entries):
        raise Batch2DCExecutionError(
            "Resume registration ledger crossed incomplete download barrier"
        )
    if extractions and len(registrations) != len(entries):
        raise Batch2DCExecutionError(
            "Resume extraction ledger crossed incomplete registration barrier"
        )
    for index, registration in enumerate(registrations):
        downloaded = (
            downloads[index]["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
        )
        disposition = registration["terminal_disposition"]
        if not downloaded and disposition != "NOT_REGISTERED_DOWNLOAD_FAILURE":
            raise Batch2DCExecutionError(
                "Resume registration does not propagate download failure"
            )
        if downloaded and disposition == "NOT_REGISTERED_DOWNLOAD_FAILURE":
            raise Batch2DCExecutionError(
                "Resume registration misclassifies successful download"
            )
    for index, extraction in enumerate(extractions):
        download_disposition = downloads[index]["terminal_disposition"]
        registration_disposition = registrations[index]["terminal_disposition"]
        extraction_disposition = extraction["terminal_disposition"]
        if download_disposition != "DOWNLOADED_VALID_PDF" and (
            extraction_disposition != "NOT_EXTRACTED_DOWNLOAD_FAILURE"
        ):
            raise Batch2DCExecutionError(
                "Resume extraction does not propagate download failure"
            )
        if (
            download_disposition == "DOWNLOADED_VALID_PDF"
            and registration_disposition != "REGISTERED_HASH_VERIFIED"
            and extraction_disposition != "NOT_EXTRACTED_REGISTRATION_FAILURE"
        ):
            raise Batch2DCExecutionError(
                "Resume extraction does not propagate registration failure"
            )
        if (
            registration_disposition == "REGISTERED_HASH_VERIFIED"
            and extraction_disposition not in {"EXTRACTED", "EXTRACTION_FAILED"}
        ):
            raise Batch2DCExecutionError(
                "Resume registered source has invalid extraction outcome"
            )

    _, by_id = _manifest_by_id(project_root)
    for index, record in enumerate(registrations):
        entry = entries[index]
        download = downloads[index]
        acquisition_id = entry["acquisition_id"]
        disposition = record["terminal_disposition"]
        if disposition == "REGISTERED_HASH_VERIFIED":
            source = by_id.get(acquisition_id)
            if source is None:
                raise Batch2DCExecutionError(
                    "Resume registered ledger entry is absent from manifest"
                )
            _verify_registered_source(source, entry, download)
        elif acquisition_id in by_id:
            raise Batch2DCExecutionError(
                "Resume failed registration unexpectedly appears in manifest"
            )
    for entry in entries[len(registrations) :]:
        if entry["acquisition_id"] in by_id:
            raise Batch2DCExecutionError(
                "Resume manifest is ahead of registration ledger"
            )

    downloads_dir = run_dir / "downloads"
    for record in downloads:
        staged = downloads_dir / f"{record['acquisition_id']}.pdf"
        if record["terminal_disposition"] == "DOWNLOADED_VALID_PDF":
            if staged.is_symlink() or not staged.is_file():
                raise Batch2DCExecutionError("Resume valid download artifact is missing")
            if sha256_file(staged) != record.get("staged_sha256"):
                raise Batch2DCExecutionError("Resume valid download SHA mismatch")
        elif staged.exists():
            raise Batch2DCExecutionError("Failed download left a final staged PDF")

    for index, record in enumerate(extractions):
        registration = registrations[index]
        acquisition_id = entries[index]["acquisition_id"]
        source = by_id.get(acquisition_id)
        disposition = record["terminal_disposition"]
        if disposition == "EXTRACTED":
            if source is None or source.get("acquisition_status") != "extracted":
                raise Batch2DCExecutionError(
                    "Resume extracted ledger entry mismatches manifest"
                )
            if (
                source.get("structured_path") != record.get("structured_path")
                or source.get("structured_sha256") != record.get("structured_sha256")
            ):
                raise Batch2DCExecutionError("Resume structured artifact metadata mismatch")
        elif registration["terminal_disposition"] == "REGISTERED_HASH_VERIFIED":
            if source is None or source.get("acquisition_status") != "hash_verified":
                raise Batch2DCExecutionError(
                    "Resume extraction failure state mismatches manifest"
                )
    for index in range(len(extractions), len(registrations)):
        registration = registrations[index]
        if registration["terminal_disposition"] == "REGISTERED_HASH_VERIFIED":
            source = by_id.get(entries[index]["acquisition_id"])
            if source is None or source.get("acquisition_status") != "hash_verified":
                raise Batch2DCExecutionError(
                    "Resume manifest is ahead of extraction ledger"
                )


def _finalize(
    root: Path,
    run_dir: Path,
    protocol: dict,
    entries: list[dict],
    downloads: list[dict],
    registrations: list[dict],
    extractions: list[dict],
) -> dict:
    _validate_failure_chain(entries, downloads, registrations, extractions)
    manifest = source_intake.load_acquisition_manifest(root)
    by_id = {source["acquisition_id"]: source for source in manifest["sources"]}
    registered_count = sum(
        record["terminal_disposition"] == "REGISTERED_HASH_VERIFIED"
        for record in registrations
    )
    extracted_count = sum(
        record["terminal_disposition"] == "EXTRACTED" for record in extractions
    )
    download_success_count = sum(
        record["terminal_disposition"] == "DOWNLOADED_VALID_PDF"
        for record in downloads
    )
    if registered_count > download_success_count or extracted_count > registered_count:
        raise Batch2DCExecutionError("Technical stage counts violate monotonic barriers")
    if len(manifest["sources"]) != BASELINE_SOURCE_COUNT + registered_count:
        raise Batch2DCExecutionError("Final manifest source count does not reconcile")
    cohort_sources = [by_id[item] for item in EXPECTED_IDS if item in by_id]
    status_counts = {
        status: sum(source["acquisition_status"] == status for source in cohort_sources)
        for status in ("hash_verified", "extracted", "eligible")
    }
    if status_counts != {
        "hash_verified": registered_count - extracted_count,
        "extracted": extracted_count,
        "eligible": 0,
    }:
        raise Batch2DCExecutionError("Final Batch 2D-C statuses do not reconcile")
    source_intake.inventory_independent_sources(root)
    summary = {
        "cohort_id": COHORT_ID,
        "technical_stages_closed": True,
        "download_attempted_count": len(entries),
        "download_success_count": download_success_count,
        "registered_count": registered_count,
        "registration_failed_count": sum(
            record["terminal_disposition"] == "REGISTRATION_FAILED"
            for record in registrations
        ),
        "extraction_attempted_count": registered_count,
        "extracted_count": extracted_count,
        "extraction_failed_count": sum(
            record["terminal_disposition"] == "EXTRACTION_FAILED"
            for record in extractions
        ),
        "final_manifest_source_count": len(manifest["sources"]),
        "batch2dc_status_counts": status_counts,
        "replacement_or_topup_count": 0,
        "identify_review_promote_count": 0,
    }
    _atomic_json(run_dir / "stage_summary.json", summary)
    output_hashes = {
        name: sha256_file(run_dir / name)
        for name in (
            "execution_protocol.json",
            "download_ledger.json",
            "registration_ledger.json",
            "extraction_ledger.json",
            "stage_summary.json",
        )
    }
    final_manifest_sha = sha256_file(root / MANIFEST_RELATIVE)
    failures = len(entries) - extracted_count
    run_manifest = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_dir.name,
        "completed_at": dt.datetime.now().astimezone().isoformat(),
        "execution_git_commit": protocol["execution_git_commit"],
        "runner_sha256": protocol["runner_sha256"],
        "tls_trust": protocol["tls_trust"],
        "cohort_fingerprint": COHORT_FINGERPRINT,
        "baseline_manifest_sha256": protocol["baseline_manifest_sha256"],
        "final_manifest_sha256": final_manifest_sha,
        "output_file_sha256": output_hashes,
        "status": "TECHNICAL_ACQUISITION_COMPLETE" if failures == 0 else "TECHNICAL_ACQUISITION_PARTIAL",
        "network_scope": "exact_50_frozen_source_urls_only",
        "pdf_download_count": summary["download_success_count"],
        "registered_count": registered_count,
        "extracted_count": extracted_count,
        "identity_review_promotion_executed": False,
        "preliminary_stratification_executed": False,
        "selection_seed_query_gt_model_retrieval_executed": False,
    }
    _atomic_json(run_dir / "run_manifest.json", run_manifest)
    return run_manifest


def run(
    *, project_root: Path, ca_bundle: Path, resume_run: Path | None = None
) -> Path:
    root = Path(project_root).resolve()
    cohort, intake, head = load_execution_inputs(root)
    context, tls_trust = build_verified_ssl_context(ca_bundle)
    downloader = make_tls_downloader(context)
    entries = cohort["entries"]
    if resume_run is None:
        validate_initial_baseline(root, intake)
        run_dir = root / RUNS_RELATIVE / _new_run_id()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "downloads").mkdir()
        protocol = _create_protocol(
            root,
            run_dir,
            cohort,
            head,
            sha256_file(root / MANIFEST_RELATIVE),
            tls_trust,
        )
    else:
        run_dir = Path(resume_run).resolve()
        try:
            run_dir.relative_to((root / RUNS_RELATIVE).resolve())
        except ValueError as exc:
            raise Batch2DCExecutionError("Resume run must remain under backend/data/runs") from exc
        protocol = _load_resume_protocol(run_dir, head, tls_trust, entries)
        validate_resume_state(root, run_dir, entries)
    downloads = run_download_stage(entries, run_dir, downloader=downloader)
    registrations = run_registration_stage(root, entries, downloads, run_dir)
    extractions = run_extraction_stage(
        root, entries, downloads, registrations, run_dir
    )
    _finalize(root, run_dir, protocol, entries, downloads, registrations, extractions)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Execute frozen Batch 2D-C technical acquisition",
        allow_abbrev=False,
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ca-bundle", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args(argv)
    result = run(
        project_root=args.project_root,
        ca_bundle=args.ca_bundle,
        resume_run=args.resume_run,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
