"""Fail-closed source intake for the independent retrieval benchmark.

This module handles provenance and source eligibility only.  It deliberately
does not generate queries or ground truth and has no retrieval/model imports.
Historical development judgments remain outside this contract.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


INTAKE_ROOT_RELATIVE = Path("backend/dataLaw/independent_intake")
ACQUISITION_MANIFEST_RELATIVE = INTAKE_ROOT_RELATIVE / "acquisition_manifest.json"
MANIFEST_VERSION = "independent_source_acquisition_v1"
STRUCTURED_SCHEMA_VERSION = "independent_structured_judgment_v1"
PROVENANCE_ANCHOR_VERSION = "independent_provenance_anchor_v1"
PUBLICATION_JOURNAL_VERSION = "independent_publication_journal_v1"
REGISTRATION_JOURNAL_VERSION = "independent_registration_journal_v1"
REVIEW_PAYLOAD_VERSION = "independent_source_review_v1"
IDENTITY_PAYLOAD_VERSION = "independent_source_identity_v1"
PROVENANCE_ROOT_RELATIVE = INTAKE_ROOT_RELATIVE / "provenance"
TRANSACTION_ROOT_RELATIVE = INTAKE_ROOT_RELATIVE / "transactions"
ACQUISITION_STATUSES = (
    "downloaded",
    "hash_verified",
    "extracted",
    "metadata_verified",
    "dedup_checked",
    "eligible",
    "quarantined",
    "rejected",
)
ELIGIBLE_STATUS_HISTORY = (
    "downloaded",
    "hash_verified",
    "extracted",
    "metadata_verified",
    "dedup_checked",
    "eligible",
)
DIRECT_ELIGIBLE_STATUS_HISTORY = (
    "downloaded",
    "hash_verified",
    "extracted",
    "eligible",
)
VERIFICATION_FIELDS = (
    "judgment_number",
    "case_id",
    "court",
    "decision_date",
    "procedure_stage",
    "development_overlap_check",
    "internal_dedup_check",
    "underlying_dispute_review",
)
FORBIDDEN_PROVENANCE_KEY_TOKENS = {
    "bm25",
    "candidate",
    "dense",
    "hybrid",
    "metric",
    "rank",
    "rerank",
    "retrieval",
    "score",
}
SOURCE_REQUIRED_FIELDS = {
    "acquisition_id",
    "source_id",
    "original_filename",
    "stored_pdf_path",
    "source_url",
    "offline_provenance",
    "source_domain",
    "download_timestamp",
    "sha256",
    "case_id",
    "judgment_number",
    "court",
    "decision_date",
    "procedure_stage",
    "underlying_dispute_id",
    "related_decision_ids",
    "structured_path",
    "structured_sha256",
    "metadata_verification",
    "acquisition_status",
    "status_history",
    "rejection_reasons",
    "notes",
}
IMMUTABLE_PROVENANCE_FIELDS = (
    "acquisition_id",
    "source_id",
    "original_filename",
    "stored_pdf_path",
    "source_url",
    "offline_provenance",
    "source_domain",
    "download_timestamp",
    "sha256",
)
JOURNAL_STATES = ("PREPARED", "ARTIFACT_PUBLISHED", "MANIFEST_PUBLISHED")
REGISTRATION_JOURNAL_STATES = (
    "PREPARED",
    "RAW_PUBLISHED",
    "ANCHOR_PUBLISHED",
    "MANIFEST_PUBLISHED",
)
PAGE_EXTRACTION_STATUSES = (
    "native_text",
    "ocr_text",
    "empty_after_native_and_ocr",
)
EMPTY_PAGE_EXTRACTION_STATUS = "empty_after_native_and_ocr"
REVIEW_PROCEDURE_STAGES = (
    "so_tham",
    "phuc_tham",
    "giam_doc_tham",
    "tai_tham",
)
REVIEW_METADATA_FIELDS = (
    "judgment_number",
    "case_id",
    "court",
    "decision_date",
    "procedure_stage",
)
REVIEW_COMPLETION_FIELDS = (
    "development_overlap_check",
    "internal_dedup_check",
    "underlying_dispute_review",
)


class SourceIntakeError(ValueError):
    """Raised when an independent-source intake invariant is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class _TrustedDirectoryChain:
    """Retain and revalidate every directory FD from the repository root."""

    def __init__(self, project_root: Path, relative_path: Path, *, create: bool):
        self.project_root = Path(project_root).resolve()
        self.relative = _require_repository_relative(
            relative_path.as_posix(), "intake directory"
        )
        intake_parts = PurePosixPath(INTAKE_ROOT_RELATIVE.as_posix()).parts
        if self.relative.parts[: len(intake_parts)] != intake_parts:
            raise SourceIntakeError(
                "Trusted directory must remain inside the intake root"
            )
        self.path = self.project_root / Path(*self.relative.parts)
        self._names = list(self.relative.parts)
        self._fds = []
        self._identities = []
        flags = _directory_open_flags()
        try:
            root_fd = os.open(self.project_root, flags)
            self._fds.append(root_fd)
            root_metadata = os.fstat(root_fd)
            self._identities.append((root_metadata.st_dev, root_metadata.st_ino))
            for index, part in enumerate(self._names):
                parent_fd = self._fds[-1]
                try:
                    child_fd = os.open(part, flags, dir_fd=parent_fd)
                except FileNotFoundError:
                    if not create or index < len(intake_parts):
                        raise SourceIntakeError(
                            f"Required intake directory is missing: {self.path}"
                        )
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    try:
                        child_fd = os.open(part, flags, dir_fd=parent_fd)
                    except OSError as exc:
                        raise SourceIntakeError(
                            "Cannot open newly created intake directory "
                            f"component: {part}"
                        ) from exc
                except OSError as exc:
                    raise SourceIntakeError(
                        f"Unsafe intake directory component: {part}"
                    ) from exc
                self._fds.append(child_fd)
                child_metadata = os.fstat(child_fd)
                self._identities.append(
                    (child_metadata.st_dev, child_metadata.st_ino)
                )
            self.assert_identity()
        except Exception:
            self.close()
            raise

    @property
    def fd(self) -> int:
        return self._fds[-1]

    def assert_identity(self) -> None:
        try:
            root_visible = os.stat(self.project_root, follow_symlinks=False)
            root_held = os.fstat(self._fds[0])
        except OSError as exc:
            raise SourceIntakeError(
                f"Trusted project root is unavailable: {self.project_root}"
            ) from exc
        if (
            not stat.S_ISDIR(root_visible.st_mode)
            or not stat.S_ISDIR(root_held.st_mode)
            or (root_visible.st_dev, root_visible.st_ino) != self._identities[0]
            or (root_held.st_dev, root_held.st_ino) != self._identities[0]
        ):
            raise SourceIntakeError("Trusted project-root identity changed")

        for index, name in enumerate(self._names, start=1):
            try:
                visible = os.stat(
                    name, dir_fd=self._fds[index - 1], follow_symlinks=False
                )
                held = os.fstat(self._fds[index])
            except OSError as exc:
                raise SourceIntakeError(
                    f"Trusted directory-chain entry is unavailable: {name}"
                ) from exc
            expected = self._identities[index]
            if (
                not stat.S_ISDIR(visible.st_mode)
                or not stat.S_ISDIR(held.st_mode)
                or (visible.st_dev, visible.st_ino) != expected
                or (held.st_dev, held.st_ino) != expected
            ):
                raise SourceIntakeError(
                    f"Trusted directory identity changed in chain: {name}"
                )

    def close(self) -> None:
        while self._fds:
            os.close(self._fds.pop())


def _open_trusted_directory(
    project_root: Path, relative_path: Path, *, create: bool
) -> tuple[Path, _TrustedDirectoryChain]:
    chain = _TrustedDirectoryChain(project_root, relative_path, create=create)
    return chain.path, chain


def _acquire_manifest_mutation_lock(intake_chain: _TrustedDirectoryChain) -> None:
    """Serialize cooperative manifest writers on the stable intake directory FD."""

    intake_chain.assert_identity()
    fcntl.flock(intake_chain.fd, fcntl.LOCK_EX)
    intake_chain.assert_identity()


def _release_manifest_mutation_lock(intake_chain: _TrustedDirectoryChain) -> None:
    fcntl.flock(intake_chain.fd, fcntl.LOCK_UN)


def _entry_lstat(directory_fd: int, name: str):
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _read_regular_bytes_at(directory_fd: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise SourceIntakeError(f"Cannot open regular intake file: {name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SourceIntakeError(f"Intake object is not a regular file: {name}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_bytes_create_once_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary_name = f".{name}.{uuid.uuid4().hex}.partial"
    descriptor = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SourceIntakeError(f"Refusing to overwrite create-once file: {name}") from exc
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _copy_fd_create_once_at(
    directory_fd: int,
    name: str,
    source_fd: int,
    expected_sha256: str,
    temporary_name: str,
) -> None:
    """Copy a held regular file into a trusted directory without overwriting."""

    descriptor = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        digest = hashlib.sha256()
        os.lseek(source_fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise SourceIntakeError("Raw PDF copy made no progress")
                view = view[written:]
        os.fsync(descriptor)
        if digest.hexdigest() != expected_sha256:
            raise SourceIntakeError("Raw PDF changed during registration copy")
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SourceIntakeError(
                f"Refusing to overwrite create-once raw PDF: {name}"
            ) from exc
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.lseek(source_fd, 0, os.SEEK_SET)


def _atomic_write_bytes_at(directory_fd: int, name: str, payload: bytes) -> None:
    temporary_name = f".{name}.{uuid.uuid4().hex}.partial"
    descriptor = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _remove_tree_at(parent_fd: int, name: str) -> None:
    metadata = _entry_lstat(parent_fd, name)
    if metadata is None:
        return
    if not stat.S_ISDIR(metadata.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    child_fd = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    try:
        for entry_name in os.listdir(child_fd):
            _remove_tree_at(child_fd, entry_name)
    finally:
        os.close(child_fd)
    os.rmdir(name, dir_fd=parent_fd)


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _parse_offset_timestamp(value, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise SourceIntakeError(f"{field_name} must be an offset-aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceIntakeError(
            f"{field_name} must be an offset-aware ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceIntakeError(f"{field_name} must be an offset-aware ISO-8601 timestamp")
    return parsed


def _require_offset_timestamp(value, field_name: str) -> None:
    _parse_offset_timestamp(value, field_name)


def _inspect_forbidden_keys(value, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if any(token in normalized for token in FORBIDDEN_PROVENANCE_KEY_TOKENS):
                raise SourceIntakeError(
                    f"Retrieval/ranking field is forbidden in intake provenance: {path}.{key}"
                )
            _inspect_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _inspect_forbidden_keys(nested, f"{path}[{index}]")


def _require_repository_relative(value, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceIntakeError(f"{field_name} must be a repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise SourceIntakeError(f"{field_name} must be a repository-relative POSIX path")
    return path


def _resolve_inside(project_root: Path, relative_path: str, approved_root: Path) -> Path:
    rel = _require_repository_relative(relative_path, "source path")
    resolved = (project_root / Path(*rel.parts)).resolve()
    try:
        resolved.relative_to(approved_root.resolve())
    except ValueError as exc:
        raise SourceIntakeError(
            f"Source path escapes approved intake root: {relative_path}"
        ) from exc
    return resolved


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _reject_symlink_components(path: Path, boundary: Path) -> None:
    """Reject every existing symlink from boundary through path."""

    boundary = Path(boundary).resolve()
    lexical = Path(path)
    try:
        relative = lexical.relative_to(boundary)
    except ValueError as exc:
        raise SourceIntakeError(f"Path is outside the approved boundary: {path}") from exc
    current = boundary
    for part in relative.parts:
        current = current / part
        if _lexists(current) and current.is_symlink():
            raise SourceIntakeError(f"Symlink path component is forbidden: {current}")


def _validated_acquisition_file_path(
    project_root: Path,
    source: dict,
    kind: str,
    *,
    require_regular_file: bool = True,
) -> Path:
    if kind not in {"raw", "structured"}:
        raise SourceIntakeError(f"Unsupported intake path kind: {kind}")
    expected_pdf, expected_structured = _expected_paths(source)
    field = "stored_pdf_path" if kind == "raw" else "structured_path"
    expected = expected_pdf if kind == "raw" else expected_structured
    if source[field] != expected:
        raise SourceIntakeError(f"{field} is not acquisition-scoped")
    relative = _require_repository_relative(source[field], field)
    project_root = Path(project_root).resolve()
    intake_root = project_root / INTAKE_ROOT_RELATIVE
    acquisition_dir = intake_root / kind / source["acquisition_id"]
    path = project_root / Path(*relative.parts)
    _reject_symlink_components(path, project_root)
    if path.parent != acquisition_dir:
        raise SourceIntakeError(f"{field} escapes its acquisition-specific directory")
    if path.resolve(strict=False).parent != acquisition_dir.resolve(strict=False):
        raise SourceIntakeError(f"{field} resolves outside its acquisition-specific directory")
    if require_regular_file:
        if not _lexists(path) or path.is_symlink() or not path.is_file():
            raise SourceIntakeError(f"{field} must be an existing regular non-symlink file")
    return path


def _prepare_output_directory(project_root: Path, source: dict) -> tuple[Path, Path]:
    """Resolve the exact acquisition-scoped output names without touching disk."""

    project_root = Path(project_root).resolve()
    _, expected_structured = _expected_paths(source)
    if source["structured_path"] != expected_structured:
        raise SourceIntakeError("structured_path is not acquisition-scoped")
    relative = _require_repository_relative(source["structured_path"], "structured_path")
    final_path = project_root / Path(*relative.parts)
    output_relative = (
        INTAKE_ROOT_RELATIVE / "structured" / source["acquisition_id"]
    )
    output_dir = project_root / output_relative
    if final_path.parent != output_dir:
        raise SourceIntakeError("Structured output escapes acquisition directory")

    return output_dir, final_path


def _assert_file_fd_identity(path: Path, descriptor: int) -> None:
    try:
        visible = os.stat(path, follow_symlinks=False)
        held = os.fstat(descriptor)
    except OSError as exc:
        raise SourceIntakeError(f"Trusted file path is unavailable: {path}") from exc
    if not stat.S_ISREG(visible.st_mode) or not stat.S_ISREG(held.st_mode):
        raise SourceIntakeError(f"Trusted file is not regular: {path}")
    if (visible.st_dev, visible.st_ino) != (held.st_dev, held.st_ino):
        raise SourceIntakeError(f"Trusted file identity changed: {path}")


def _open_acquisition_file_fd(
    project_root: Path, source: dict, kind: str
) -> tuple[_TrustedDirectoryChain, Path, int]:
    path = _validated_acquisition_file_path(project_root, source, kind)
    relative_dir = INTAKE_ROOT_RELATIVE / kind / source["acquisition_id"]
    _, directory_chain = _open_trusted_directory(
        project_root, relative_dir, create=False
    )
    filename = path.name
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(filename, flags, dir_fd=directory_chain.fd)
    except OSError as exc:
        directory_chain.close()
        raise SourceIntakeError(f"Cannot open trusted acquisition file: {path}") from exc
    try:
        directory_chain.assert_identity()
        _assert_file_fd_identity(path, file_fd)
    except Exception:
        os.close(file_fd)
        directory_chain.close()
        raise
    return directory_chain, path, file_fd


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _validate_pdf_fd(descriptor: int, expected_sha256: str) -> str:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise SourceIntakeError("PDF must be a regular file")
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, 5) != b"%PDF-":
            raise SourceIntakeError("PDF signature is invalid")
        actual_sha256 = _sha256_fd(descriptor)
        if actual_sha256 != expected_sha256:
            raise SourceIntakeError("PDF SHA256 mismatch")

        import pdfplumber

        with os.fdopen(os.dup(descriptor), "rb") as handle, pdfplumber.open(handle) as document:
            if len(document.pages) < 1:
                raise SourceIntakeError("PDF must contain at least one readable page")
            for page_number, page in enumerate(document.pages, start=1):
                if page.page_number != page_number or page.width <= 0 or page.height <= 0:
                    raise SourceIntakeError("PDF page structure is invalid")
    except SourceIntakeError:
        raise
    except Exception as exc:
        raise SourceIntakeError(f"PDF parser rejected document: {type(exc).__name__}") from exc
    os.lseek(descriptor, 0, os.SEEK_SET)
    return actual_sha256


def validate_pdf_source(path: Path, expected_sha256: str) -> str:
    """Authoritative PDF validity check used by intake and extraction."""

    path = Path(path)
    if not _lexists(path) or path.is_symlink() or not path.is_file():
        raise SourceIntakeError("PDF must be an existing regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceIntakeError("PDF must be an existing regular non-symlink file") from exc
    try:
        return _validate_pdf_fd(descriptor, expected_sha256)
    finally:
        os.close(descriptor)


def stable_source_id(acquisition_id: str) -> str:
    if not isinstance(acquisition_id, str) or not re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{2,127}", acquisition_id
    ):
        raise SourceIntakeError(
            "acquisition_id must be 3-128 lowercase ASCII letters, digits, dot, underscore, or hyphen"
        )
    return f"independent:{acquisition_id}"


def _require_review_text(value, field_name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value:
        raise SourceIntakeError(f"review.{field_name} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise SourceIntakeError(f"review.{field_name} must be Unicode NFC")
    if value != " ".join(value.split()) or len(value) > maximum:
        raise SourceIntakeError(
            f"review.{field_name} must be canonical normalized text"
        )
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise SourceIntakeError(f"review.{field_name} contains control characters")
    return value


def _require_review_decision_id(value, field_name: str) -> str:
    checked = _require_review_text(value, field_name, maximum=128)
    if not re.fullmatch(r"\d{1,4}/(?:19|20)\d{2}/[A-ZĐ0-9-]+", checked):
        raise SourceIntakeError(
            f"review.{field_name} must be a canonical Vietnamese decision identity"
        )
    return checked


def validate_source_review_payload(payload: dict) -> dict:
    """Validate the strict, non-patch input accepted by source review mutation."""

    combined_root_fields = {
        "review_version",
        "verified_metadata",
        "verified_fields",
        "related_decision_ids",
        "underlying_dispute_id",
        "reviews",
    }
    review_only_root_fields = {"review_version", "reviews"}
    if not isinstance(payload, dict) or frozenset(payload) not in {
        frozenset(combined_root_fields),
        frozenset(review_only_root_fields),
    }:
        raise SourceIntakeError("Source review payload has invalid root fields")
    if payload["review_version"] != REVIEW_PAYLOAD_VERSION:
        raise SourceIntakeError("Unsupported source review payload version")

    checked_identity = None
    if set(payload) == combined_root_fields:
        checked_identity = validate_source_identity_payload(
            {
                "identity_version": IDENTITY_PAYLOAD_VERSION,
                "verified_metadata": payload["verified_metadata"],
                "verified_fields": payload["verified_fields"],
                "related_decision_ids": payload["related_decision_ids"],
                "underlying_dispute_id": payload["underlying_dispute_id"],
            }
        )

    checked_reviews = _validate_review_completion_payload(payload["reviews"])
    return {
        "review_version": REVIEW_PAYLOAD_VERSION,
        "identity": checked_identity,
        "reviews": checked_reviews,
    }


def validate_source_identity_payload(payload: dict) -> dict:
    """Validate the strict, per-document identity mutation payload."""

    root_fields = {
        "identity_version",
        "verified_metadata",
        "verified_fields",
        "related_decision_ids",
        "underlying_dispute_id",
    }
    if not isinstance(payload, dict) or set(payload) != root_fields:
        raise SourceIntakeError("Source identity payload has invalid root fields")
    if payload["identity_version"] != IDENTITY_PAYLOAD_VERSION:
        raise SourceIntakeError("Unsupported source identity payload version")

    metadata = payload["verified_metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(REVIEW_METADATA_FIELDS):
        raise SourceIntakeError("Source identity verified_metadata fields are invalid")
    checked_metadata = {
        "judgment_number": _require_review_decision_id(
            metadata["judgment_number"], "verified_metadata.judgment_number"
        ),
        "case_id": _require_review_decision_id(
            metadata["case_id"], "verified_metadata.case_id"
        ),
        "court": _require_review_text(
            metadata["court"], "verified_metadata.court"
        ),
        "decision_date": _require_review_text(
            metadata["decision_date"], "verified_metadata.decision_date", maximum=10
        ),
        "procedure_stage": _require_review_text(
            metadata["procedure_stage"],
            "verified_metadata.procedure_stage",
            maximum=32,
        ),
    }
    try:
        parsed_date = datetime.strptime(
            checked_metadata["decision_date"], "%Y-%m-%d"
        )
    except ValueError as exc:
        raise SourceIntakeError(
            "review.verified_metadata.decision_date must use valid YYYY-MM-DD"
        ) from exc
    if parsed_date.strftime("%Y-%m-%d") != checked_metadata["decision_date"]:
        raise SourceIntakeError(
            "review.verified_metadata.decision_date must be canonical"
        )
    if checked_metadata["procedure_stage"] not in REVIEW_PROCEDURE_STAGES:
        raise SourceIntakeError(
            "review.verified_metadata.procedure_stage is not an allowed value"
        )

    verified_fields = payload["verified_fields"]
    if (
        not isinstance(verified_fields, dict)
        or set(verified_fields) != set(REVIEW_METADATA_FIELDS)
        or not all(verified_fields[field] is True for field in REVIEW_METADATA_FIELDS)
    ):
        raise SourceIntakeError(
            "Source identity requires every supplied legal metadata field to be verified=true"
        )

    related = payload["related_decision_ids"]
    if not isinstance(related, list):
        raise SourceIntakeError("review.related_decision_ids must be an array")
    checked_related = [
        _require_review_decision_id(value, f"related_decision_ids[{index}]")
        for index, value in enumerate(related)
    ]
    if len(checked_related) != len(set(checked_related)):
        raise SourceIntakeError("review.related_decision_ids contains duplicates")

    underlying = payload["underlying_dispute_id"]
    if underlying is not None:
        underlying = _require_review_text(
            underlying, "underlying_dispute_id", maximum=256
        )

    return {
        "identity_version": IDENTITY_PAYLOAD_VERSION,
        "verified_metadata": checked_metadata,
        "verified_fields": {
            field: True for field in REVIEW_METADATA_FIELDS
        },
        "related_decision_ids": checked_related,
        "underlying_dispute_id": underlying,
    }


def _validate_review_completion_payload(reviews: dict) -> dict:
    if not isinstance(reviews, dict) or set(reviews) != set(
        REVIEW_COMPLETION_FIELDS
    ):
        raise SourceIntakeError("Source review completion fields are invalid")
    checked_reviews = {}
    for field in REVIEW_COMPLETION_FIELDS:
        review = reviews[field]
        if not isinstance(review, dict) or set(review) != {"completed", "findings"}:
            raise SourceIntakeError(f"review.reviews.{field} fields are invalid")
        if not isinstance(review["completed"], bool):
            raise SourceIntakeError(
                f"review.reviews.{field}.completed must be boolean"
            )
        findings = review["findings"]
        if not isinstance(findings, list):
            raise SourceIntakeError(
                f"review.reviews.{field}.findings must be an array"
            )
        checked_findings = [
            _require_review_text(
                value, f"reviews.{field}.findings[{index}]", maximum=512
            )
            for index, value in enumerate(findings)
        ]
        if checked_findings != sorted(set(checked_findings)):
            raise SourceIntakeError(
                f"review.reviews.{field}.findings must be sorted and unique"
            )
        checked_reviews[field] = {
            "completed": review["completed"],
            "findings": checked_findings,
        }

    return checked_reviews


def _validate_source_url(source: dict, index: int) -> None:
    source_url = source["source_url"]
    offline = source["offline_provenance"]
    if not isinstance(source["source_domain"], str) or not source["source_domain"].strip():
        raise SourceIntakeError(f"sources[{index}].source_domain is required")
    if source_url is None:
        if not isinstance(offline, dict) or not offline.get("description"):
            raise SourceIntakeError(
                f"sources[{index}] requires source_url or explicit offline_provenance"
            )
    else:
        parsed = urlparse(source_url) if isinstance(source_url, str) else None
        if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SourceIntakeError(f"sources[{index}].source_url must be a canonical HTTP(S) URL")
        if offline is not None:
            raise SourceIntakeError(
                f"sources[{index}].offline_provenance must be null when source_url is supplied"
            )
        if source["source_domain"].casefold() != parsed.hostname.casefold():
            raise SourceIntakeError(
                f"sources[{index}].source_domain must match source_url hostname"
            )


def _validate_status_history(source: dict, index: int) -> None:
    history = source["status_history"]
    if not isinstance(history, list) or not history:
        raise SourceIntakeError(f"sources[{index}].status_history must be non-empty")
    statuses = []
    timestamps = []
    for history_index, entry in enumerate(history):
        if not isinstance(entry, dict) or set(entry) != {"status", "timestamp", "reasons"}:
            raise SourceIntakeError(
                f"sources[{index}].status_history[{history_index}] has invalid fields"
            )
        status = entry["status"]
        if status not in ACQUISITION_STATUSES:
            raise SourceIntakeError(
                f"sources[{index}].status_history[{history_index}] has invalid status"
            )
        timestamps.append(_parse_offset_timestamp(
            entry["timestamp"], f"sources[{index}].status_history[{history_index}].timestamp"
        ))
        if not isinstance(entry["reasons"], list) or not all(
            isinstance(reason, str) and reason for reason in entry["reasons"]
        ):
            raise SourceIntakeError(
                f"sources[{index}].status_history[{history_index}].reasons must be strings"
            )
        statuses.append(status)
    if timestamps != sorted(timestamps):
        raise SourceIntakeError(
            f"sources[{index}].status_history timestamps must be chronological"
        )
    download_timestamp = _parse_offset_timestamp(
        source["download_timestamp"], f"sources[{index}].download_timestamp"
    )
    if timestamps[0] < download_timestamp:
        raise SourceIntakeError(
            f"sources[{index}].status_history cannot predate download_timestamp"
        )
    if statuses[-1] != source["acquisition_status"]:
        raise SourceIntakeError(
            f"sources[{index}] current status must equal final status_history entry"
        )
    current_status = source["acquisition_status"]
    if current_status == "eligible":
        if tuple(statuses) not in {
            ELIGIBLE_STATUS_HISTORY,
            DIRECT_ELIGIBLE_STATUS_HISTORY,
        }:
            raise SourceIntakeError(
                f"sources[{index}] status requires the complete ordered status history"
            )
    elif current_status in ELIGIBLE_STATUS_HISTORY:
        expected = ELIGIBLE_STATUS_HISTORY[: ELIGIBLE_STATUS_HISTORY.index(current_status) + 1]
        if tuple(statuses) != expected:
            raise SourceIntakeError(
                f"sources[{index}] status requires the complete ordered status history"
            )
    else:
        prior = tuple(statuses[:-1])
        if not prior or prior != ELIGIBLE_STATUS_HISTORY[: len(prior)]:
            raise SourceIntakeError(
                f"sources[{index}] terminal status history is not a valid ordered prefix"
            )


def _validate_source_shape(source: dict, index: int) -> None:
    if not isinstance(source, dict):
        raise SourceIntakeError(f"sources[{index}] must be an object")
    missing = sorted(SOURCE_REQUIRED_FIELDS - set(source))
    extra = sorted(set(source) - SOURCE_REQUIRED_FIELDS)
    if missing or extra:
        raise SourceIntakeError(
            f"sources[{index}] fields mismatch; missing={missing}, extra={extra}"
        )
    expected_source_id = stable_source_id(source["acquisition_id"])
    if source["source_id"] != expected_source_id:
        raise SourceIntakeError(
            f"sources[{index}].source_id must equal {expected_source_id}"
        )
    if not isinstance(source["original_filename"], str) or (
        Path(source["original_filename"]).name != source["original_filename"]
        or Path(source["original_filename"]).suffix.casefold() != ".pdf"
    ):
        raise SourceIntakeError(f"sources[{index}].original_filename must be a PDF basename")
    _validate_source_url(source, index)
    _require_offset_timestamp(source["download_timestamp"], f"sources[{index}].download_timestamp")
    if not _is_sha256(source["sha256"]):
        raise SourceIntakeError(f"sources[{index}].sha256 must be lowercase 64-char hex")
    if source["structured_sha256"] is not None and not _is_sha256(source["structured_sha256"]):
        raise SourceIntakeError(
            f"sources[{index}].structured_sha256 must be lowercase 64-char hex"
        )
    if (
        source["acquisition_status"] in ELIGIBLE_STATUS_HISTORY[2:]
        and not _is_sha256(source["structured_sha256"])
    ):
        raise SourceIntakeError(
            f"sources[{index}].structured_sha256 is required from extracted status onward"
        )
    for field in (
        "case_id",
        "judgment_number",
        "court",
        "decision_date",
        "procedure_stage",
        "underlying_dispute_id",
        "notes",
    ):
        if source[field] is not None and not isinstance(source[field], str):
            raise SourceIntakeError(f"sources[{index}].{field} must be a string or null")
    if source["decision_date"] is not None:
        try:
            datetime.strptime(source["decision_date"], "%Y-%m-%d")
        except ValueError as exc:
            raise SourceIntakeError(
                f"sources[{index}].decision_date must use YYYY-MM-DD"
            ) from exc
    if not isinstance(source["related_decision_ids"], list) or not all(
        isinstance(value, str) and value for value in source["related_decision_ids"]
    ):
        raise SourceIntakeError(f"sources[{index}].related_decision_ids must be strings")
    if len(source["related_decision_ids"]) != len(set(source["related_decision_ids"])):
        raise SourceIntakeError(f"sources[{index}].related_decision_ids contains duplicates")
    verification = source["metadata_verification"]
    if not isinstance(verification, dict) or set(verification) != set(VERIFICATION_FIELDS):
        raise SourceIntakeError(f"sources[{index}].metadata_verification has invalid fields")
    if not all(isinstance(verification[field], bool) for field in VERIFICATION_FIELDS):
        raise SourceIntakeError(f"sources[{index}].metadata_verification values must be booleans")
    if source["acquisition_status"] not in ACQUISITION_STATUSES:
        raise SourceIntakeError(f"sources[{index}].acquisition_status is invalid")
    if not isinstance(source["rejection_reasons"], list) or not all(
        isinstance(reason, str) and reason for reason in source["rejection_reasons"]
    ):
        raise SourceIntakeError(f"sources[{index}].rejection_reasons must be strings")
    _validate_status_history(source, index)


def immutable_provenance_snapshot(source: dict) -> dict:
    """Create the canonical acquisition-time record for one source."""

    record = {field: source[field] for field in IMMUTABLE_PROVENANCE_FIELDS}
    return {
        **record,
        "immutable_fingerprint": _canonical_json_sha256(record),
    }


def _provenance_anchor_payload(source: dict) -> dict:
    record = {
        "anchor_version": PROVENANCE_ANCHOR_VERSION,
        "acquisition_id": source["acquisition_id"],
        "source_id": source["source_id"],
        "original_filename": source["original_filename"],
        "stored_pdf_path": source["stored_pdf_path"],
        "source_url": source["source_url"],
        "offline_provenance": source["offline_provenance"],
        "source_domain": source["source_domain"],
        "download_timestamp": source["download_timestamp"],
        "pdf_sha256": source["sha256"],
    }
    return {**record, "anchor_fingerprint": _canonical_json_sha256(record)}


def _validate_provenance_anchor(anchor: dict, source: dict) -> None:
    expected = _provenance_anchor_payload(source)
    if not isinstance(anchor, dict) or set(anchor) != set(expected):
        raise SourceIntakeError("Provenance anchor has invalid fields")
    record = {key: anchor[key] for key in anchor if key != "anchor_fingerprint"}
    if anchor["anchor_fingerprint"] != _canonical_json_sha256(record):
        raise SourceIntakeError("Provenance anchor fingerprint mismatch")
    if anchor != expected:
        raise SourceIntakeError(
            f"Immutable provenance differs from anchor: {source['acquisition_id']}"
        )


def create_provenance_anchor(project_root: Path, source: dict) -> Path:
    """Create the external acquisition anchor exactly once."""

    _validate_source_shape(source, 0)
    anchor_name = f"{source['acquisition_id']}.json"
    anchor_path, anchor_chain = _open_trusted_directory(
        project_root, PROVENANCE_ROOT_RELATIVE, create=True
    )
    try:
        anchor_chain.assert_identity()
        _write_bytes_create_once_at(
            anchor_chain.fd,
            anchor_name,
            _canonical_json_bytes(_provenance_anchor_payload(source)),
        )
        anchor_chain.assert_identity()
    finally:
        anchor_chain.close()
    return anchor_path / anchor_name


def _validate_manifest_anchors(project_root: Path, manifest: dict) -> None:
    _, anchor_chain = _open_trusted_directory(
        project_root, PROVENANCE_ROOT_RELATIVE, create=False
    )
    try:
        expected_names = {
            f"{source['acquisition_id']}.json" for source in manifest["sources"]
        }
        actual_names = {
            name for name in os.listdir(anchor_chain.fd) if name != ".gitkeep"
        }
        if actual_names != expected_names:
            raise SourceIntakeError(
                "Provenance anchors must contain exactly one anchor per source"
            )
        for source in manifest["sources"]:
            anchor_name = f"{source['acquisition_id']}.json"
            try:
                anchor = json.loads(
                    _read_regular_bytes_at(anchor_chain.fd, anchor_name)
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceIntakeError(
                    f"Invalid provenance anchor JSON: {anchor_name}"
                ) from exc
            _validate_provenance_anchor(anchor, source)
        anchor_chain.assert_identity()
    finally:
        anchor_chain.close()


def _validate_immutable_provenance(manifest: dict) -> None:
    snapshots = manifest["immutable_provenance"]
    if not isinstance(snapshots, list):
        raise SourceIntakeError("immutable_provenance must be an array")
    expected_fields = set(IMMUTABLE_PROVENANCE_FIELDS) | {"immutable_fingerprint"}
    by_id = {}
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or set(snapshot) != expected_fields:
            raise SourceIntakeError(
                f"immutable_provenance[{index}] has invalid fields"
            )
        acquisition_id = snapshot["acquisition_id"]
        if acquisition_id in by_id:
            raise SourceIntakeError("Duplicate immutable acquisition record")
        record = {field: snapshot[field] for field in IMMUTABLE_PROVENANCE_FIELDS}
        expected_fingerprint = _canonical_json_sha256(record)
        if snapshot["immutable_fingerprint"] != expected_fingerprint:
            raise SourceIntakeError(
                f"immutable_provenance[{index}] fingerprint mismatch"
            )
        by_id[acquisition_id] = snapshot

    source_ids = {source["acquisition_id"] for source in manifest["sources"]}
    if set(by_id) != source_ids:
        raise SourceIntakeError(
            "immutable_provenance must contain exactly one record per source"
        )
    for index, source in enumerate(manifest["sources"]):
        snapshot = by_id[source["acquisition_id"]]
        for field in IMMUTABLE_PROVENANCE_FIELDS:
            if source[field] != snapshot[field]:
                raise SourceIntakeError(
                    f"sources[{index}].{field} drifted from immutable provenance"
                )


def validate_acquisition_manifest(manifest: dict, prior_manifest: dict | None = None) -> None:
    """Validate manifest shape, recursive leakage guard, and immutable IDs."""

    if not isinstance(manifest, dict) or set(manifest) != {
        "manifest_version",
        "created_at",
        "sources",
        "immutable_provenance",
        "known_duplicate_links",
    }:
        raise SourceIntakeError("Acquisition manifest has invalid root fields")
    _inspect_forbidden_keys(manifest)
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise SourceIntakeError("Unsupported acquisition manifest version")
    _require_offset_timestamp(manifest["created_at"], "created_at")
    if not isinstance(manifest["sources"], list):
        raise SourceIntakeError("sources must be an array")
    acquisition_ids = []
    for index, source in enumerate(manifest["sources"]):
        _validate_source_shape(source, index)
        acquisition_ids.append(source["acquisition_id"])
    if len(acquisition_ids) != len(set(acquisition_ids)):
        raise SourceIntakeError("Duplicate acquisition_id values are forbidden")
    source_ids = [source["source_id"] for source in manifest["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise SourceIntakeError("Duplicate source_id values are forbidden")
    _validate_immutable_provenance(manifest)

    links = manifest["known_duplicate_links"]
    if not isinstance(links, list):
        raise SourceIntakeError("known_duplicate_links must be an array")
    known_ids = set(acquisition_ids)
    canonical_links = set()
    for index, link in enumerate(links):
        if not isinstance(link, dict) or set(link) != {
            "acquisition_id",
            "related_acquisition_id",
            "reason",
        }:
            raise SourceIntakeError(f"known_duplicate_links[{index}] has invalid fields")
        left = link["acquisition_id"]
        right = link["related_acquisition_id"]
        if left not in known_ids or right not in known_ids or left == right:
            raise SourceIntakeError(f"known_duplicate_links[{index}] references invalid source IDs")
        if not isinstance(link["reason"], str) or not link["reason"]:
            raise SourceIntakeError(f"known_duplicate_links[{index}].reason is required")
        pair = tuple(sorted((left, right)))
        if pair in canonical_links:
            raise SourceIntakeError("Duplicate known_duplicate_links pair")
        canonical_links.add(pair)

    if prior_manifest is not None:
        validate_acquisition_manifest(prior_manifest)
        prior_snapshots = {
            item["acquisition_id"]: item
            for item in prior_manifest["immutable_provenance"]
        }
        current_snapshots = {
            item["acquisition_id"]: item for item in manifest["immutable_provenance"]
        }
        for acquisition_id, prior_snapshot in prior_snapshots.items():
            if current_snapshots.get(acquisition_id) != prior_snapshot:
                raise SourceIntakeError(
                    f"Immutable acquisition record changed: {acquisition_id}"
                )


def _expected_paths(source: dict) -> tuple[str, str]:
    acquisition_id = source["acquisition_id"]
    filename = source["original_filename"]
    pdf_path = (INTAKE_ROOT_RELATIVE / "raw" / acquisition_id / filename).as_posix()
    structured_name = f"{Path(filename).stem}_structured.json"
    structured_path = (
        INTAKE_ROOT_RELATIVE / "structured" / acquisition_id / structured_name
    ).as_posix()
    return pdf_path, structured_path


def validate_structured_judgment(payload: dict, original_filename: str | None = None) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "judgment_name",
        "page_count",
        "pages",
        "sections",
    }:
        raise SourceIntakeError("Structured judgment has invalid root fields")
    if not isinstance(payload["judgment_name"], str) or not payload["judgment_name"]:
        raise SourceIntakeError("Structured judgment_name must be non-empty")
    if original_filename is not None and payload["judgment_name"] != original_filename:
        raise SourceIntakeError("Structured judgment_name does not match original_filename")
    pages = payload["pages"]
    if not isinstance(payload["page_count"], int) or payload["page_count"] <= 0:
        raise SourceIntakeError("Structured page_count must be a positive integer")
    if not isinstance(pages, list) or payload["page_count"] != len(pages):
        raise SourceIntakeError("Structured page_count must equal len(pages)")
    document_has_usable_text = False
    legacy_page_fields = {"page", "text"}
    classified_page_fields = {"page", "text", "extraction_status"}
    for index, page in enumerate(pages, start=1):
        page_fields = frozenset(page) if isinstance(page, dict) else frozenset()
        if not isinstance(page, dict) or page_fields not in {
            frozenset(legacy_page_fields),
            frozenset(classified_page_fields),
        }:
            raise SourceIntakeError(f"Structured page {index} has invalid fields")
        if page["page"] != index:
            raise SourceIntakeError("Structured page numbers must be consecutive and one-based")
        text = page["text"]
        if not isinstance(text, str):
            raise SourceIntakeError(f"Structured page {index} text must be a string")
        extraction_status = page.get("extraction_status")
        if extraction_status is not None and extraction_status not in PAGE_EXTRACTION_STATUSES:
            raise SourceIntakeError(
                f"Structured page {index} has unknown extraction_status"
            )
        if extraction_status == EMPTY_PAGE_EXTRACTION_STATUS:
            if text != "":
                raise SourceIntakeError(
                    f"Structured page {index} empty extraction status requires empty text"
                )
        elif not text.strip():
            raise SourceIntakeError(f"Structured page {index} must contain usable text")
        else:
            document_has_usable_text = True
        if unicodedata.normalize("NFC", text) != text:
            raise SourceIntakeError(f"Structured page {index} text must be Unicode NFC")
    if not document_has_usable_text:
        raise SourceIntakeError("Structured judgment must contain usable document text")
    sections = payload["sections"]
    required_sections = {"header", "facts", "reasoning", "decision"}
    if not isinstance(sections, dict) or set(sections) != required_sections:
        raise SourceIntakeError("Structured sections must contain exactly the required keys")
    for name, text in sections.items():
        if not isinstance(text, str):
            raise SourceIntakeError(f"Structured section {name} must be a string")
        if unicodedata.normalize("NFC", text) != text:
            raise SourceIntakeError(f"Structured section {name} must be Unicode NFC")


def structured_blank_page_numbers(payload: dict) -> list[int]:
    """Return explicitly classified blank page numbers in deterministic order."""

    return [
        page["page"]
        for page in payload["pages"]
        if page.get("extraction_status") == EMPTY_PAGE_EXTRACTION_STATUS
    ]


def _read_json_utf8(path: Path) -> dict:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        return json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError(f"Cannot read valid UTF-8 JSON: {path}") from exc


def _source_gate_failures(source: dict, project_root: Path) -> tuple[list[str], dict | None]:
    failures = []
    try:
        raw_chain, pdf_path, pdf_fd = _open_acquisition_file_fd(
            project_root, source, "raw"
        )
        try:
            _validate_pdf_fd(pdf_fd, source["sha256"])
            _assert_file_fd_identity(pdf_path, pdf_fd)
            raw_chain.assert_identity()
        finally:
            os.close(pdf_fd)
            raw_chain.close()
    except SourceIntakeError as exc:
        failures.append(f"invalid_pdf:{exc}")

    structured_payload = None
    try:
        structured_chain, structured_path, structured_fd = _open_acquisition_file_fd(
            project_root, source, "structured"
        )
    except SourceIntakeError as exc:
        failures.append(f"invalid_structured_path:{exc}")
    else:
        try:
            structured_bytes = _read_regular_bytes_at(
                structured_chain.fd, structured_path.name
            )
            structured_payload = json.loads(structured_bytes.decode("utf-8"))
            validate_structured_judgment(structured_payload, source["original_filename"])
        except (SourceIntakeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid_structured_json:{exc}")
        try:
            if (
                source["structured_sha256"] is None
                or _sha256_fd(structured_fd) != source["structured_sha256"]
            ):
                failures.append("structured_sha256_mismatch")
        except OSError:
            failures.append("unreadable_structured_json")
        finally:
            os.close(structured_fd)
            structured_chain.close()

    for field in ("judgment_number", "case_id", "court", "decision_date", "procedure_stage"):
        if not source[field]:
            failures.append(f"missing_{field}")
        if not source["metadata_verification"][field]:
            failures.append(f"unverified_{field}")
    for check in (
        "development_overlap_check",
        "internal_dedup_check",
        "underlying_dispute_review",
    ):
        if not source["metadata_verification"][check]:
            failures.append(f"incomplete_{check}")
    if source["rejection_reasons"]:
        failures.append("unresolved_rejection_reasons")
    return sorted(set(failures)), structured_payload


def _source_identity_failures(source: dict) -> list[str]:
    """Return only failures that mean a source is not identity-ready."""

    failures = []
    for field in REVIEW_METADATA_FIELDS:
        if not source[field]:
            failures.append(f"missing_{field}")
        if not source["metadata_verification"][field]:
            failures.append(f"unverified_{field}")
    return sorted(set(failures))


def _source_integrity_failures(
    source: dict, project_root: Path
) -> tuple[list[str], dict | None]:
    """Validate artifacts while excluding lifecycle gates from the result."""

    failures, structured_payload = _source_gate_failures(source, project_root)
    lifecycle_failures = set(_source_identity_failures(source)) | {
        f"incomplete_{field}" for field in REVIEW_COMPLETION_FIELDS
    } | {"unresolved_rejection_reasons"}
    return (
        [failure for failure in failures if failure not in lifecycle_failures],
        structured_payload,
    )


def _source_to_inventory_record(
    source: dict, structured_payload: dict, include_private_text: bool
) -> dict:
    full_text = "\n".join(page["text"] for page in structured_payload["pages"])
    decision_year = int(source["decision_date"][:4])
    record = {
        "source_id": source["source_id"],
        "acquisition_id": source["acquisition_id"],
        "case_id": source["case_id"],
        "judgment_number": source["judgment_number"],
        "underlying_dispute_id": source["underlying_dispute_id"],
        "related_decision_ids": list(source["related_decision_ids"]),
        "source_name": source["original_filename"],
        "source_path": source["stored_pdf_path"],
        "structured_path": source["structured_path"],
        "court": source["court"],
        "procedure_stage": source["procedure_stage"],
        "year": decision_year,
        "case_type": None,
        "preliminary_law_families": [],
        "preliminary_legal_functions": ["unknown"],
        "source_sha256": source["sha256"],
        "structured_sha256": source["structured_sha256"],
        "blank_page_numbers": structured_blank_page_numbers(structured_payload),
        "metadata_source": "verified_independent_acquisition_manifest_v1",
    }
    if include_private_text:
        record["_dedup_text"] = full_text
    return record


def _journal_payload(base: dict) -> dict:
    record = {key: value for key, value in base.items() if key != "journal_fingerprint"}
    return {**record, "journal_fingerprint": _canonical_json_sha256(record)}


def _registration_journal_payload(base: dict) -> dict:
    record = {
        key: value
        for key, value in base.items()
        if key != "journal_fingerprint"
    }
    return {**record, "journal_fingerprint": _canonical_json_sha256(record)}


def _registration_journal_name(acquisition_id: str) -> str:
    stable_source_id(acquisition_id)
    return f"{acquisition_id}.register.json"


def _validate_registration_journal(journal: dict, acquisition_id: str) -> None:
    expected_fields = {
        "journal_version",
        "transaction_id",
        "acquisition_id",
        "source_id",
        "state",
        "previous_manifest_sha256",
        "intended_manifest_sha256",
        "raw_directory_name",
        "raw_artifact_name",
        "raw_temporary_name",
        "raw_relative_path",
        "raw_sha256",
        "anchor_name",
        "anchor_sha256",
        "created_at",
        "updated_at",
        "journal_fingerprint",
    }
    if not isinstance(journal, dict) or set(journal) != expected_fields:
        raise SourceIntakeError("Registration journal has invalid fields")
    record = {key: journal[key] for key in journal if key != "journal_fingerprint"}
    if journal["journal_fingerprint"] != _canonical_json_sha256(record):
        raise SourceIntakeError("Registration journal fingerprint mismatch")
    if journal["journal_version"] != REGISTRATION_JOURNAL_VERSION:
        raise SourceIntakeError("Unsupported registration journal version")
    if journal["acquisition_id"] != acquisition_id:
        raise SourceIntakeError("Registration journal refers to wrong acquisition")
    if journal["source_id"] != stable_source_id(acquisition_id):
        raise SourceIntakeError("Registration journal source identity is invalid")
    if journal["state"] not in REGISTRATION_JOURNAL_STATES:
        raise SourceIntakeError("Registration journal state is invalid")
    transaction_id = journal["transaction_id"]
    if not isinstance(transaction_id, str) or not re.fullmatch(
        r"[0-9a-f]{32}", transaction_id
    ):
        raise SourceIntakeError("Registration journal transaction_id is invalid")
    if journal["raw_directory_name"] != acquisition_id:
        raise SourceIntakeError("Registration journal raw directory is invalid")
    raw_name = journal["raw_artifact_name"]
    if (
        not isinstance(raw_name, str)
        or Path(raw_name).name != raw_name
        or Path(raw_name).suffix.casefold() != ".pdf"
    ):
        raise SourceIntakeError("Registration journal raw artifact is invalid")
    if journal["raw_temporary_name"] != f".register-partial-{transaction_id}":
        raise SourceIntakeError("Registration journal temporary name is invalid")
    expected_raw_path = (
        INTAKE_ROOT_RELATIVE / "raw" / acquisition_id / raw_name
    ).as_posix()
    if journal["raw_relative_path"] != expected_raw_path:
        raise SourceIntakeError("Registration journal raw path is invalid")
    if journal["anchor_name"] != f"{acquisition_id}.json":
        raise SourceIntakeError("Registration journal anchor name is invalid")
    for field in (
        "previous_manifest_sha256",
        "intended_manifest_sha256",
        "raw_sha256",
        "anchor_sha256",
    ):
        if not _is_sha256(journal[field]):
            raise SourceIntakeError(
                f"Registration journal {field} is not a valid SHA256"
            )
    _require_offset_timestamp(journal["created_at"], "journal.created_at")
    _require_offset_timestamp(journal["updated_at"], "journal.updated_at")


def _read_registration_journal_at(
    transaction_fd: int, acquisition_id: str
) -> dict:
    name = _registration_journal_name(acquisition_id)
    try:
        journal = json.loads(_read_regular_bytes_at(transaction_fd, name))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError(f"Corrupt registration journal: {name}") from exc
    _validate_registration_journal(journal, acquisition_id)
    return journal


def _write_registration_journal_at(
    transaction_fd: int, journal: dict, *, create_once: bool
) -> dict:
    checked = _registration_journal_payload(journal)
    _validate_registration_journal(checked, checked.get("acquisition_id"))
    name = _registration_journal_name(checked["acquisition_id"])
    payload = _canonical_json_bytes(checked)
    if create_once:
        _write_bytes_create_once_at(transaction_fd, name, payload)
    else:
        _atomic_write_bytes_at(transaction_fd, name, payload)
    return checked


def _registration_raw_state(
    project_root: Path, journal: dict, *, rollback: bool
) -> bool:
    """Validate and optionally remove raw artifacts named by a journal."""

    _, raw_root_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE / "raw", create=False
    )
    try:
        acquisition_id = journal["acquisition_id"]
        exists = _entry_lstat(raw_root_chain.fd, acquisition_id)
        if exists is None:
            return False
        if not stat.S_ISDIR(exists.st_mode):
            raise SourceIntakeError(
                "Registration raw destination is not a directory"
            )
        _, raw_chain = _open_trusted_directory(
            project_root,
            INTAKE_ROOT_RELATIVE / "raw" / acquisition_id,
            create=False,
        )
        try:
            allowed = {
                journal["raw_artifact_name"],
                journal["raw_temporary_name"],
            }
            names = set(os.listdir(raw_chain.fd))
            unexpected = names - allowed
            if unexpected:
                raise SourceIntakeError(
                    f"Unexpected registration raw objects: {sorted(unexpected)}"
                )
            raw_name = journal["raw_artifact_name"]
            raw_metadata = _entry_lstat(raw_chain.fd, raw_name)
            raw_exists = raw_metadata is not None
            if raw_exists:
                if not stat.S_ISREG(raw_metadata.st_mode):
                    raise SourceIntakeError(
                        "Registration raw artifact is not regular"
                    )
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                raw_fd = os.open(raw_name, flags, dir_fd=raw_chain.fd)
                try:
                    _validate_pdf_fd(raw_fd, journal["raw_sha256"])
                finally:
                    os.close(raw_fd)
            temporary_name = journal["raw_temporary_name"]
            temporary_metadata = _entry_lstat(raw_chain.fd, temporary_name)
            if temporary_metadata is not None and not stat.S_ISREG(
                temporary_metadata.st_mode
            ):
                raise SourceIntakeError(
                    "Registration temporary raw artifact is unsafe"
                )
            raw_chain.assert_identity()
            raw_root_chain.assert_identity()
            if rollback:
                for name in (raw_name, temporary_name):
                    if _entry_lstat(raw_chain.fd, name) is not None:
                        os.unlink(name, dir_fd=raw_chain.fd)
                os.fsync(raw_chain.fd)
            elif temporary_metadata is not None:
                os.unlink(temporary_name, dir_fd=raw_chain.fd)
                os.fsync(raw_chain.fd)
            raw_chain.assert_identity()
        finally:
            raw_chain.close()
        if rollback:
            os.rmdir(acquisition_id, dir_fd=raw_root_chain.fd)
            os.fsync(raw_root_chain.fd)
        raw_root_chain.assert_identity()
        return raw_exists
    finally:
        raw_root_chain.close()


def _registration_anchor_state(
    project_root: Path, journal: dict, *, rollback: bool
) -> bool:
    """Validate and optionally remove the create-once registration anchor."""

    _, anchor_chain = _open_trusted_directory(
        project_root, PROVENANCE_ROOT_RELATIVE, create=False
    )
    try:
        anchor_name = journal["anchor_name"]
        metadata = _entry_lstat(anchor_chain.fd, anchor_name)
        if metadata is None:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise SourceIntakeError("Registration provenance anchor is not regular")
        if _hash_regular_file_at(anchor_chain.fd, anchor_name) != journal[
            "anchor_sha256"
        ]:
            raise SourceIntakeError("Registration provenance anchor SHA256 mismatch")
        anchor_chain.assert_identity()
        if rollback:
            os.unlink(anchor_name, dir_fd=anchor_chain.fd)
            os.fsync(anchor_chain.fd)
        anchor_chain.assert_identity()
        return True
    finally:
        anchor_chain.close()


def _rollback_registration_artifacts(project_root: Path, journal: dict) -> None:
    _registration_anchor_state(project_root, journal, rollback=True)
    _registration_raw_state(project_root, journal, rollback=True)


def _rollback_registration_artifacts_at(
    raw_root_fd: int, anchor_fd: int, journal: dict
) -> None:
    """Rollback through already trusted FDs even if visible parents moved."""

    anchor_name = journal["anchor_name"]
    anchor_metadata = _entry_lstat(anchor_fd, anchor_name)
    if anchor_metadata is not None:
        if not stat.S_ISREG(anchor_metadata.st_mode):
            raise SourceIntakeError("Registration provenance anchor is not regular")
        if _hash_regular_file_at(anchor_fd, anchor_name) != journal[
            "anchor_sha256"
        ]:
            raise SourceIntakeError("Registration provenance anchor SHA256 mismatch")
        os.unlink(anchor_name, dir_fd=anchor_fd)
        os.fsync(anchor_fd)

    acquisition_id = journal["acquisition_id"]
    raw_directory_metadata = _entry_lstat(raw_root_fd, acquisition_id)
    if raw_directory_metadata is None:
        return
    if not stat.S_ISDIR(raw_directory_metadata.st_mode):
        raise SourceIntakeError("Registration raw destination is not a directory")
    raw_fd = os.open(acquisition_id, _directory_open_flags(), dir_fd=raw_root_fd)
    try:
        raw_name = journal["raw_artifact_name"]
        temporary_name = journal["raw_temporary_name"]
        names = set(os.listdir(raw_fd))
        unexpected = names - {raw_name, temporary_name}
        if unexpected:
            raise SourceIntakeError(
                f"Unexpected registration raw objects: {sorted(unexpected)}"
            )
        raw_metadata = _entry_lstat(raw_fd, raw_name)
        if raw_metadata is not None:
            if not stat.S_ISREG(raw_metadata.st_mode):
                raise SourceIntakeError("Registration raw artifact is not regular")
            held_pdf_fd = os.open(
                raw_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=raw_fd,
            )
            try:
                _validate_pdf_fd(held_pdf_fd, journal["raw_sha256"])
            finally:
                os.close(held_pdf_fd)
        temporary_metadata = _entry_lstat(raw_fd, temporary_name)
        if temporary_metadata is not None and not stat.S_ISREG(
            temporary_metadata.st_mode
        ):
            raise SourceIntakeError("Registration temporary raw artifact is unsafe")
        for name in (raw_name, temporary_name):
            if _entry_lstat(raw_fd, name) is not None:
                os.unlink(name, dir_fd=raw_fd)
        os.fsync(raw_fd)
    finally:
        os.close(raw_fd)
    os.rmdir(acquisition_id, dir_fd=raw_root_fd)
    os.fsync(raw_root_fd)


def _reconcile_registration_journals(
    project_root: Path, manifest: dict, manifest_bytes: bytes
) -> None:
    """Recover interrupted raw/anchor/manifest registration transactions."""

    _, transaction_chain = _open_trusted_directory(
        project_root, TRANSACTION_ROOT_RELATIVE, create=False
    )
    try:
        journal_names = sorted(
            name
            for name in os.listdir(transaction_chain.fd)
            if name.endswith(".register.json")
        )
        for journal_name in journal_names:
            acquisition_id = journal_name[: -len(".register.json")]
            journal = _read_registration_journal_at(
                transaction_chain.fd, acquisition_id
            )
            current_hash = _sha256_bytes(manifest_bytes)
            matching = [
                source
                for source in manifest["sources"]
                if source["acquisition_id"] == acquisition_id
                or source["source_id"] == journal["source_id"]
            ]
            if current_hash == journal["intended_manifest_sha256"]:
                if len(matching) != 1:
                    raise SourceIntakeError(
                        "Manifest-published registration source is missing"
                    )
                source = matching[0]
                if (
                    source["acquisition_id"] != acquisition_id
                    or source["source_id"] != journal["source_id"]
                    or source["stored_pdf_path"] != journal["raw_relative_path"]
                    or source["sha256"] != journal["raw_sha256"]
                ):
                    raise SourceIntakeError(
                        "Manifest-published registration identity is inconsistent"
                    )
                if not _registration_raw_state(
                    project_root, journal, rollback=False
                ):
                    raise SourceIntakeError(
                        "Manifest-published registration raw PDF is missing"
                    )
                if not _registration_anchor_state(
                    project_root, journal, rollback=False
                ):
                    raise SourceIntakeError(
                        "Manifest-published registration anchor is missing"
                    )
                _, anchor_chain = _open_trusted_directory(
                    project_root, PROVENANCE_ROOT_RELATIVE, create=False
                )
                try:
                    anchor = json.loads(
                        _read_regular_bytes_at(
                            anchor_chain.fd, journal["anchor_name"]
                        )
                    )
                    anchor_chain.assert_identity()
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SourceIntakeError(
                        "Manifest-published registration anchor is invalid JSON"
                    ) from exc
                finally:
                    anchor_chain.close()
                _validate_provenance_anchor(anchor, source)
            elif current_hash == journal["previous_manifest_sha256"]:
                if matching:
                    raise SourceIntakeError(
                        "Previous registration manifest unexpectedly contains source"
                    )
                _rollback_registration_artifacts(project_root, journal)
            else:
                raise SourceIntakeError(
                    "Registration journal manifest fingerprint mismatch"
                )
            transaction_chain.assert_identity()
            os.unlink(journal_name, dir_fd=transaction_chain.fd)
            os.fsync(transaction_chain.fd)
        transaction_chain.assert_identity()
    finally:
        transaction_chain.close()


def _validate_journal(journal: dict, acquisition_id: str) -> None:
    expected_fields = {
        "journal_version",
        "transaction_id",
        "acquisition_id",
        "state",
        "previous_manifest_sha256",
        "intended_manifest_sha256",
        "previous_status",
        "intended_status",
        "temp_directory_name",
        "final_artifact_name",
        "final_relative_path",
        "structured_sha256",
        "created_at",
        "updated_at",
        "journal_fingerprint",
    }
    if not isinstance(journal, dict) or set(journal) != expected_fields:
        raise SourceIntakeError("Publication journal has invalid fields")
    record = {key: journal[key] for key in journal if key != "journal_fingerprint"}
    if journal["journal_fingerprint"] != _canonical_json_sha256(record):
        raise SourceIntakeError("Publication journal fingerprint mismatch")
    if journal["journal_version"] != PUBLICATION_JOURNAL_VERSION:
        raise SourceIntakeError("Unsupported publication journal version")
    if journal["acquisition_id"] != acquisition_id:
        raise SourceIntakeError("Publication journal refers to wrong acquisition")
    if journal["state"] not in JOURNAL_STATES:
        raise SourceIntakeError("Publication journal state is invalid")
    if not isinstance(journal["transaction_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", journal["transaction_id"]
    ):
        raise SourceIntakeError("Publication journal transaction_id is invalid")
    if journal["previous_status"] != "hash_verified" or journal[
        "intended_status"
    ] != "extracted":
        raise SourceIntakeError("Publication journal status transition is invalid")
    if not isinstance(journal["temp_directory_name"], str) or not re.fullmatch(
        r"\.extract-partial-[0-9a-f]{32}", journal["temp_directory_name"]
    ):
        raise SourceIntakeError("Publication journal temporary path is invalid")
    if (
        not isinstance(journal["final_artifact_name"], str)
        or not journal["final_artifact_name"]
        or Path(journal["final_artifact_name"]).name
        != journal["final_artifact_name"]
    ):
        raise SourceIntakeError("Publication journal final artifact name is invalid")
    _require_repository_relative(
        journal["final_relative_path"], "journal.final_relative_path"
    )
    if not all(
        _is_sha256(journal[field])
        for field in (
            "previous_manifest_sha256",
            "intended_manifest_sha256",
            "structured_sha256",
        )
    ):
        raise SourceIntakeError("Publication journal contains invalid SHA256")
    _require_offset_timestamp(journal["created_at"], "journal.created_at")
    _require_offset_timestamp(journal["updated_at"], "journal.updated_at")


def _read_journal_at(transaction_fd: int, acquisition_id: str) -> dict:
    name = f"{acquisition_id}.json"
    try:
        return json.loads(_read_regular_bytes_at(transaction_fd, name))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError(f"Corrupt publication journal: {name}") from exc


def _write_journal_at(
    transaction_fd: int, journal: dict, *, create_once: bool
) -> dict:
    checked = _journal_payload(journal)
    name = f"{checked['acquisition_id']}.json"
    payload = _canonical_json_bytes(checked)
    if create_once:
        _write_bytes_create_once_at(transaction_fd, name, payload)
    else:
        _atomic_write_bytes_at(transaction_fd, name, payload)
    return checked


def _hash_regular_file_at(directory_fd: int, name: str) -> str:
    payload = _read_regular_bytes_at(directory_fd, name)
    return _sha256_bytes(payload)


def _validate_structured_at(directory_fd: int, name: str, source: dict) -> str:
    payload_bytes = _read_regular_bytes_at(directory_fd, name)
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError(f"Cannot read valid structured JSON: {name}") from exc
    validate_structured_judgment(payload, source["original_filename"])
    return _sha256_bytes(payload_bytes)


def _reconcile_publication_journals(
    project_root: Path, manifest: dict, manifest_bytes: bytes
) -> None:
    _, transaction_chain = _open_trusted_directory(
        project_root, TRANSACTION_ROOT_RELATIVE, create=False
    )
    try:
        journal_names = sorted(
            name for name in os.listdir(transaction_chain.fd) if name != ".gitkeep"
        )
        for journal_name in journal_names:
            if not journal_name.endswith(".json"):
                raise SourceIntakeError(
                    f"Unexpected publication transaction object: {journal_name}"
                )
            acquisition_id = journal_name[:-5]
            journal = _read_journal_at(transaction_chain.fd, acquisition_id)
            _validate_journal(journal, acquisition_id)
            matching = [
                source
                for source in manifest["sources"]
                if source["acquisition_id"] == acquisition_id
            ]
            if len(matching) != 1:
                raise SourceIntakeError("Publication journal source is missing")
            source = matching[0]
            expected_pdf, expected_structured = _expected_paths(source)
            del expected_pdf
            if (
                journal["final_relative_path"] != expected_structured
                or journal["final_artifact_name"] != Path(expected_structured).name
            ):
                raise SourceIntakeError("Publication journal final path mismatch")
            output_relative = (
                INTAKE_ROOT_RELATIVE / "structured" / acquisition_id
            )
            _, output_chain = _open_trusted_directory(
                project_root, output_relative, create=False
            )
            try:
                final_name = journal["final_artifact_name"]
                final_metadata = _entry_lstat(output_chain.fd, final_name)
                final_exists = final_metadata is not None
                if final_exists and not stat.S_ISREG(final_metadata.st_mode):
                    raise SourceIntakeError("Journal final artifact is not regular")
                final_hash = (
                    _hash_regular_file_at(output_chain.fd, final_name)
                    if final_exists
                    else None
                )
                current_hash = _sha256_bytes(manifest_bytes)
                if current_hash == journal["intended_manifest_sha256"]:
                    if (
                        source["acquisition_status"] != journal["intended_status"]
                        or not final_exists
                        or final_hash != journal["structured_sha256"]
                        or source["structured_sha256"] != journal["structured_sha256"]
                    ):
                        raise SourceIntakeError(
                            "Manifest-published transaction is inconsistent"
                        )
                    output_chain.assert_identity()
                    transaction_chain.assert_identity()
                    _remove_tree_at(output_chain.fd, journal["temp_directory_name"])
                    os.unlink(journal_name, dir_fd=transaction_chain.fd)
                    os.fsync(transaction_chain.fd)
                elif current_hash == journal["previous_manifest_sha256"]:
                    if source["acquisition_status"] != journal["previous_status"]:
                        raise SourceIntakeError(
                            "Previous publication manifest status mismatch"
                        )
                    if final_exists:
                        if final_hash != journal["structured_sha256"]:
                            raise SourceIntakeError(
                                "Published artifact differs from recovery journal"
                            )
                        output_chain.assert_identity()
                        os.unlink(final_name, dir_fd=output_chain.fd)
                        os.fsync(output_chain.fd)
                    output_chain.assert_identity()
                    transaction_chain.assert_identity()
                    _remove_tree_at(output_chain.fd, journal["temp_directory_name"])
                    os.unlink(journal_name, dir_fd=transaction_chain.fd)
                    os.fsync(transaction_chain.fd)
                else:
                    raise SourceIntakeError(
                        "Publication journal manifest fingerprint mismatch"
                    )
                output_chain.assert_identity()
            finally:
                output_chain.close()
        transaction_chain.assert_identity()
    finally:
        transaction_chain.close()


def _validate_manifest_artifacts(project_root: Path, manifest: dict) -> None:
    for source in manifest["sources"]:
        acquisition_id = source["acquisition_id"]
        output_relative = INTAKE_ROOT_RELATIVE / "structured" / acquisition_id
        output_path = Path(project_root).resolve() / output_relative
        expected_name = Path(source["structured_path"]).name
        expects_artifact = source["structured_sha256"] is not None
        if not _lexists(output_path):
            if expects_artifact:
                raise SourceIntakeError(
                    f"Structured artifact missing for {acquisition_id}"
                )
            continue
        _, output_chain = _open_trusted_directory(
            project_root, output_relative, create=False
        )
        try:
            names = {
                name for name in os.listdir(output_chain.fd) if name != ".gitkeep"
            }
            unexpected = names - {expected_name}
            if unexpected:
                raise SourceIntakeError(
                    f"Unexpected structured objects for {acquisition_id}: {sorted(unexpected)}"
                )
            final_metadata = _entry_lstat(output_chain.fd, expected_name)
            if not expects_artifact:
                if final_metadata is not None:
                    raise SourceIntakeError(
                        f"Orphan structured artifact requires recovery: {acquisition_id}"
                    )
                continue
            if final_metadata is None or not stat.S_ISREG(final_metadata.st_mode):
                raise SourceIntakeError(
                    f"Structured artifact invalid for {acquisition_id}"
                )
            actual_hash = _validate_structured_at(
                output_chain.fd, expected_name, source
            )
            if actual_hash != source["structured_sha256"]:
                raise SourceIntakeError(
                    f"Structured artifact SHA256 mismatch for {acquisition_id}"
                )
            output_chain.assert_identity()
        finally:
            output_chain.close()


def _load_authoritative_manifest_locked(
    project_root: Path, intake_chain: _TrustedDirectoryChain
) -> tuple[dict, bytes]:
    """Load and reconcile the live manifest while its mutation lock is held."""

    project_root = Path(project_root).resolve()
    intake_chain.assert_identity()
    manifest_bytes = _read_regular_bytes_at(
        intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
    )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError("Acquisition manifest is not valid UTF-8 JSON") from exc
    validate_acquisition_manifest(manifest)
    _reconcile_registration_journals(project_root, manifest, manifest_bytes)
    _reconcile_publication_journals(project_root, manifest, manifest_bytes)
    _validate_manifest_anchors(project_root, manifest)
    _validate_manifest_artifacts(project_root, manifest)
    intake_chain.assert_identity()
    return manifest, manifest_bytes


def _load_authoritative_manifest(project_root: Path) -> tuple[dict, bytes]:
    project_root = Path(project_root).resolve()
    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    _acquire_manifest_mutation_lock(intake_chain)
    try:
        return _load_authoritative_manifest_locked(project_root, intake_chain)
    finally:
        _release_manifest_mutation_lock(intake_chain)
        intake_chain.close()


def load_acquisition_manifest(project_root: Path) -> dict:
    manifest, _ = _load_authoritative_manifest(project_root)
    return manifest


def inventory_independent_sources(
    project_root: Path, include_private_text: bool = False
) -> tuple[list[dict], dict, list[tuple[str, str]]]:
    """Return only fail-closed eligible sources and an intake validation report."""

    project_root = Path(project_root).resolve()
    manifest = load_acquisition_manifest(project_root)
    inventory = []
    source_results = []
    for source in sorted(manifest["sources"], key=lambda item: item["acquisition_id"]):
        failures, structured_payload = _source_gate_failures(source, project_root)
        claims_eligible = source["acquisition_status"] == "eligible"
        if claims_eligible and failures:
            raise SourceIntakeError(
                f"Eligible source {source['acquisition_id']} failed gates: {', '.join(failures)}"
            )
        if claims_eligible:
            inventory.append(
                _source_to_inventory_record(source, structured_payload, include_private_text)
            )
        blank_page_numbers = (
            structured_blank_page_numbers(structured_payload)
            if structured_payload is not None and not any(
                failure.startswith("invalid_structured_") for failure in failures
            )
            else []
        )
        source_results.append({
            "acquisition_id": source["acquisition_id"],
            "source_id": source["source_id"],
            "claimed_status": source["acquisition_status"],
            "eligible": claims_eligible and not failures,
            "gate_failures": failures,
            "blank_page_numbers": blank_page_numbers,
        })

    links = [
        (
            stable_source_id(link["acquisition_id"]),
            stable_source_id(link["related_acquisition_id"]),
        )
        for link in manifest["known_duplicate_links"]
    ]
    report = {
        "report_version": "independent_source_intake_validation_v1",
        "manifest_path": ACQUISITION_MANIFEST_RELATIVE.as_posix(),
        "source_count": len(manifest["sources"]),
        "eligible_source_count": len(inventory),
        "sources": source_results,
    }
    return inventory, report, sorted(links)


def _registration_publication_checkpoint(name: str) -> None:
    """Test seam for registration races and hard-stop recovery boundaries."""


def register_source(
    project_root: Path,
    acquisition_id: str,
    input_pdf: Path,
    source_url: str | None,
    source_domain: str,
    download_timestamp: str,
    original_filename: str,
    offline_provenance: dict | None = None,
) -> dict:
    """Register one validated raw PDF using a recoverable create-once transaction.

    Registration stops at ``hash_verified``.  It does not extract, attach legal
    identity metadata, run deduplication, or mutate benchmark selection state.
    """

    project_root = Path(project_root).resolve()
    source_id = stable_source_id(acquisition_id)
    if (
        not isinstance(original_filename, str)
        or Path(original_filename).name != original_filename
        or Path(original_filename).suffix.casefold() != ".pdf"
    ):
        raise SourceIntakeError("original_filename must be a PDF basename")
    parsed_download_timestamp = _parse_offset_timestamp(
        download_timestamp, "download_timestamp"
    )
    input_path = Path(input_pdf)
    if not _lexists(input_path) or input_path.is_symlink() or not input_path.is_file():
        raise SourceIntakeError(
            "Registration input PDF must be an existing regular non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        input_fd = os.open(input_path, flags)
    except OSError as exc:
        raise SourceIntakeError(
            "Registration input PDF must be an existing regular non-symlink file"
        ) from exc

    intake_chain = None
    transaction_chain = None
    raw_root_chain = None
    raw_chain = None
    anchor_chain = None
    structured_root_chain = None
    journal = None
    transaction_started = False
    journal_exists = False
    mutation_lock_held = False
    original_manifest_bytes = None
    intended_manifest_bytes = None
    try:
        input_metadata = os.fstat(input_fd)
        if not stat.S_ISREG(input_metadata.st_mode):
            raise SourceIntakeError("Registration input PDF must be a regular file")
        pdf_sha256 = _sha256_fd(input_fd)
        _validate_pdf_fd(input_fd, pdf_sha256)
        _assert_file_fd_identity(input_path, input_fd)

        registration_timestamp = datetime.now().astimezone()
        if registration_timestamp < parsed_download_timestamp:
            raise SourceIntakeError(
                "System clock precedes the source download_timestamp"
            )
        hash_verified_timestamp = registration_timestamp.isoformat()
        stored_pdf_path = (
            INTAKE_ROOT_RELATIVE
            / "raw"
            / acquisition_id
            / original_filename
        ).as_posix()
        structured_path = (
            INTAKE_ROOT_RELATIVE
            / "structured"
            / acquisition_id
            / f"{Path(original_filename).stem}_structured.json"
        ).as_posix()
        source = {
            "acquisition_id": acquisition_id,
            "source_id": source_id,
            "original_filename": original_filename,
            "stored_pdf_path": stored_pdf_path,
            "source_url": source_url,
            "offline_provenance": deepcopy(offline_provenance),
            "source_domain": source_domain,
            "download_timestamp": download_timestamp,
            "sha256": pdf_sha256,
            "case_id": None,
            "judgment_number": None,
            "court": None,
            "decision_date": None,
            "procedure_stage": None,
            "underlying_dispute_id": None,
            "related_decision_ids": [],
            "structured_path": structured_path,
            "structured_sha256": None,
            "metadata_verification": {
                field: False for field in VERIFICATION_FIELDS
            },
            "acquisition_status": "hash_verified",
            "status_history": [
                {
                    "status": "downloaded",
                    "timestamp": download_timestamp,
                    "reasons": [],
                },
                {
                    "status": "hash_verified",
                    "timestamp": hash_verified_timestamp,
                    "reasons": [],
                },
            ],
            "rejection_reasons": [],
            "notes": None,
        }
        _validate_source_shape(source, 0)
        expected_pdf, expected_structured = _expected_paths(source)
        if (
            source["stored_pdf_path"] != expected_pdf
            or source["structured_path"] != expected_structured
        ):
            raise SourceIntakeError("Registration paths do not match source identity")

        manifest, original_manifest_bytes = _load_authoritative_manifest(
            project_root
        )
        if any(
            item["acquisition_id"] == acquisition_id
            for item in manifest["sources"]
        ):
            raise SourceIntakeError(
                f"Duplicate acquisition_id is forbidden: {acquisition_id}"
            )
        if any(item["source_id"] == source_id for item in manifest["sources"]):
            raise SourceIntakeError(f"Duplicate source_id is forbidden: {source_id}")

        updated_manifest = deepcopy(manifest)
        updated_manifest["sources"].append(source)
        updated_manifest["sources"].sort(
            key=lambda item: item["acquisition_id"]
        )
        updated_manifest["immutable_provenance"].append(
            immutable_provenance_snapshot(source)
        )
        updated_manifest["immutable_provenance"].sort(
            key=lambda item: item["acquisition_id"]
        )
        validate_acquisition_manifest(updated_manifest, prior_manifest=manifest)
        intended_manifest_bytes = _canonical_json_bytes(updated_manifest)
        anchor_bytes = _canonical_json_bytes(_provenance_anchor_payload(source))

        _, intake_chain = _open_trusted_directory(
            project_root, INTAKE_ROOT_RELATIVE, create=False
        )
        _acquire_manifest_mutation_lock(intake_chain)
        mutation_lock_held = True
        _, transaction_chain = _open_trusted_directory(
            project_root, TRANSACTION_ROOT_RELATIVE, create=False
        )
        _, raw_root_chain = _open_trusted_directory(
            project_root, INTAKE_ROOT_RELATIVE / "raw", create=False
        )
        _, anchor_chain = _open_trusted_directory(
            project_root, PROVENANCE_ROOT_RELATIVE, create=False
        )
        _, structured_root_chain = _open_trusted_directory(
            project_root, INTAKE_ROOT_RELATIVE / "structured", create=False
        )
        _registration_publication_checkpoint("before_manifest_publication")
        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != original_manifest_bytes
        ):
            raise SourceIntakeError(
                "Acquisition manifest changed before registration"
            )
        if _entry_lstat(raw_root_chain.fd, acquisition_id) is not None:
            raise SourceIntakeError(
                f"Raw acquisition destination already exists: {acquisition_id}"
            )
        if _entry_lstat(structured_root_chain.fd, acquisition_id) is not None:
            raise SourceIntakeError(
                f"Structured acquisition destination already exists: {acquisition_id}"
            )
        anchor_name = f"{acquisition_id}.json"
        if _entry_lstat(anchor_chain.fd, anchor_name) is not None:
            raise SourceIntakeError(
                f"Provenance anchor already exists: {anchor_name}"
            )
        journal_name = _registration_journal_name(acquisition_id)
        if (
            _entry_lstat(transaction_chain.fd, journal_name) is not None
            or _entry_lstat(
                transaction_chain.fd, f"{acquisition_id}.json"
            )
            is not None
        ):
            raise SourceIntakeError(
                f"Source transaction already exists: {acquisition_id}"
            )

        transaction_id = uuid.uuid4().hex
        now = datetime.now().astimezone().isoformat()
        journal = _write_registration_journal_at(
            transaction_chain.fd,
            {
                "journal_version": REGISTRATION_JOURNAL_VERSION,
                "transaction_id": transaction_id,
                "acquisition_id": acquisition_id,
                "source_id": source_id,
                "state": "PREPARED",
                "previous_manifest_sha256": _sha256_bytes(
                    original_manifest_bytes
                ),
                "intended_manifest_sha256": _sha256_bytes(
                    intended_manifest_bytes
                ),
                "raw_directory_name": acquisition_id,
                "raw_artifact_name": original_filename,
                "raw_temporary_name": f".register-partial-{transaction_id}",
                "raw_relative_path": stored_pdf_path,
                "raw_sha256": pdf_sha256,
                "anchor_name": anchor_name,
                "anchor_sha256": _sha256_bytes(anchor_bytes),
                "created_at": now,
                "updated_at": now,
            },
            create_once=True,
        )
        transaction_started = True
        journal_exists = True
        _registration_publication_checkpoint("after_prepared")

        raw_root_chain.assert_identity()
        os.mkdir(acquisition_id, mode=0o700, dir_fd=raw_root_chain.fd)
        os.fsync(raw_root_chain.fd)
        _, raw_chain = _open_trusted_directory(
            project_root,
            INTAKE_ROOT_RELATIVE / "raw" / acquisition_id,
            create=False,
        )
        _copy_fd_create_once_at(
            raw_chain.fd,
            original_filename,
            input_fd,
            pdf_sha256,
            journal["raw_temporary_name"],
        )
        raw_chain.assert_identity()
        raw_root_chain.assert_identity()
        _validate_pdf_fd(input_fd, pdf_sha256)
        _assert_file_fd_identity(input_path, input_fd)
        journal["state"] = "RAW_PUBLISHED"
        journal["updated_at"] = datetime.now().astimezone().isoformat()
        journal = _write_registration_journal_at(
            transaction_chain.fd, journal, create_once=False
        )
        _registration_publication_checkpoint("after_raw_published")

        anchor_chain.assert_identity()
        _write_bytes_create_once_at(
            anchor_chain.fd, anchor_name, anchor_bytes
        )
        anchor_chain.assert_identity()
        journal["state"] = "ANCHOR_PUBLISHED"
        journal["updated_at"] = datetime.now().astimezone().isoformat()
        journal = _write_registration_journal_at(
            transaction_chain.fd, journal, create_once=False
        )
        _registration_publication_checkpoint("after_anchor_published")

        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != original_manifest_bytes
        ):
            raise SourceIntakeError(
                "Acquisition manifest changed during registration"
            )
        intake_chain.assert_identity()
        raw_root_chain.assert_identity()
        raw_chain.assert_identity()
        anchor_chain.assert_identity()
        transaction_chain.assert_identity()
        _atomic_write_bytes_at(
            intake_chain.fd,
            ACQUISITION_MANIFEST_RELATIVE.name,
            intended_manifest_bytes,
        )
        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != intended_manifest_bytes
        ):
            raise SourceIntakeError("Published registration manifest verification failed")
        journal["state"] = "MANIFEST_PUBLISHED"
        journal["updated_at"] = datetime.now().astimezone().isoformat()
        journal = _write_registration_journal_at(
            transaction_chain.fd, journal, create_once=False
        )
        _registration_publication_checkpoint("after_manifest_published")

        if _hash_regular_file_at(raw_chain.fd, original_filename) != pdf_sha256:
            raise SourceIntakeError("Final registered raw PDF SHA256 mismatch")
        if _hash_regular_file_at(anchor_chain.fd, anchor_name) != _sha256_bytes(
            anchor_bytes
        ):
            raise SourceIntakeError("Final registration anchor SHA256 mismatch")
        intake_chain.assert_identity()
        raw_chain.assert_identity()
        anchor_chain.assert_identity()
        transaction_chain.assert_identity()
        os.unlink(journal_name, dir_fd=transaction_chain.fd)
        os.fsync(transaction_chain.fd)
        journal_exists = False
        transaction_started = False
        return {
            "acquisition_id": acquisition_id,
            "source_id": source_id,
            "stored_pdf_path": stored_pdf_path,
            "sha256": pdf_sha256,
            "acquisition_status": "hash_verified",
        }
    except Exception as registration_error:
        if transaction_started and journal is not None:
            try:
                current_manifest_bytes = _read_regular_bytes_at(
                    intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                )
                if current_manifest_bytes == intended_manifest_bytes:
                    intake_chain.assert_identity()
                    _atomic_write_bytes_at(
                        intake_chain.fd,
                        ACQUISITION_MANIFEST_RELATIVE.name,
                        original_manifest_bytes,
                    )
                    current_manifest_bytes = _read_regular_bytes_at(
                        intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                    )
                if current_manifest_bytes != original_manifest_bytes:
                    try:
                        concurrent_manifest = json.loads(
                            current_manifest_bytes.decode("utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise SourceIntakeError(
                            "Registration failed after an invalid concurrent "
                            "manifest mutation; journal recovery is required"
                        ) from exc
                    validate_acquisition_manifest(
                        concurrent_manifest, prior_manifest=manifest
                    )
                    if any(
                        item["acquisition_id"] == acquisition_id
                        or item["source_id"] == source_id
                        for item in concurrent_manifest["sources"]
                    ):
                        raise SourceIntakeError(
                            "Concurrent manifest mutation claimed the registering "
                            "source; journal recovery is required"
                        )
                if raw_root_chain is not None and anchor_chain is not None:
                    _rollback_registration_artifacts_at(
                        raw_root_chain.fd, anchor_chain.fd, journal
                    )
                else:
                    _rollback_registration_artifacts(project_root, journal)
                if journal_exists:
                    checked_journal = _read_registration_journal_at(
                        transaction_chain.fd, acquisition_id
                    )
                    if checked_journal["transaction_id"] != journal[
                        "transaction_id"
                    ]:
                        raise SourceIntakeError(
                            "Registration journal identity changed during rollback"
                        )
                    os.unlink(
                        _registration_journal_name(acquisition_id),
                        dir_fd=transaction_chain.fd,
                    )
                    os.fsync(transaction_chain.fd)
                    journal_exists = False
                transaction_started = False
            except Exception as rollback_error:
                raise SourceIntakeError(
                    "Registration rollback failed; journal recovery is required"
                ) from rollback_error
        raise registration_error
    finally:
        if structured_root_chain is not None:
            structured_root_chain.close()
        if anchor_chain is not None:
            anchor_chain.close()
        if raw_chain is not None:
            raw_chain.close()
        if raw_root_chain is not None:
            raw_root_chain.close()
        if transaction_chain is not None:
            transaction_chain.close()
        if intake_chain is not None:
            if mutation_lock_held:
                _release_manifest_mutation_lock(intake_chain)
            intake_chain.close()
        os.close(input_fd)


def _eligibility_dedup_report(
    project_root: Path,
    manifest: dict,
) -> tuple[list[dict], dict]:
    """Evaluate the committed development/internal dedup contract."""

    from .independent_benchmark import (
        deduplicate_against_development,
        inventory_judgment_sources,
    )

    development_path = (
        Path(project_root).resolve() / "backend/data/judgments_evaluation.json"
    )
    if (
        not _lexists(development_path)
        or development_path.is_symlink()
        or not development_path.is_file()
    ):
        raise SourceIntakeError(
            "Development benchmark must be an existing regular non-symlink file"
        )
    try:
        development_records = json.loads(development_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceIntakeError("Development benchmark is not valid UTF-8 JSON") from exc
    if not isinstance(development_records, list):
        raise SourceIntakeError("Development benchmark must be a JSON array")

    development_inventory = inventory_judgment_sources(
        project_root, include_private_text=True
    )
    independent_inventory = []
    candidate_statuses = {
        "extracted",
        "metadata_verified",
        "dedup_checked",
        "eligible",
    }
    for source in sorted(
        manifest["sources"], key=lambda item: item["acquisition_id"]
    ):
        if source["acquisition_status"] not in candidate_statuses:
            continue
        integrity_failures, structured_payload = _source_integrity_failures(
            source, project_root
        )
        if integrity_failures or structured_payload is None:
            joined = (
                ", ".join(integrity_failures)
                or "structured_payload_unavailable"
            )
            raise SourceIntakeError(
                "Cannot evaluate independent-source dedup pool for "
                f"{source['acquisition_id']}: {joined}"
            )
        identity_failures = _source_identity_failures(source)
        if identity_failures:
            raise SourceIntakeError(
                "Peer identity not ready for independent-source dedup pool: "
                f"{source['acquisition_id']}: {', '.join(identity_failures)}"
            )
        independent_inventory.append(
            _source_to_inventory_record(
                source, structured_payload, include_private_text=True
            )
        )

    known_links = [
        (
            stable_source_id(link["acquisition_id"]),
            stable_source_id(link["related_acquisition_id"]),
        )
        for link in manifest["known_duplicate_links"]
    ]
    eligible, report = deduplicate_against_development(
        development_inventory + independent_inventory,
        development_records,
        known_duplicate_links=known_links,
    )
    return eligible, report


def _eligibility_review_findings(
    project_root: Path,
    manifest: dict,
    acquisition_id: str,
) -> dict[str, list[str]]:
    """Return deterministic findings grouped by the existing review gates."""

    eligible, report = _eligibility_dedup_report(project_root, manifest)
    target_source_id = stable_source_id(acquisition_id)
    reasons = {
        reason
        for exclusion in report["exclusions"]
        if exclusion["source_id"] == target_source_id
        for reason in exclusion["reasons"]
        if not reason.startswith("canonical_representative:")
    }
    for cluster in report["internal_duplicate_clusters"]:
        if target_source_id not in cluster["member_source_ids"]:
            continue
        member_ids = set(cluster["member_source_ids"])
        reasons.update(
            reason
            for exclusion in report["exclusions"]
            if exclusion["source_id"] in member_ids
            for reason in exclusion["reasons"]
            if not reason.startswith("canonical_representative:")
        )
    if (
        target_source_id not in {record["source_id"] for record in eligible}
        and not reasons
    ):
        reasons.add("source_absent_from_deduplicated_eligible_pool")

    underlying_prefixes = (
        "known_underlying_dispute:",
        "new_pool_known_duplicate_link",
        "new_pool_related_decision_ids",
        "new_pool_same_underlying_dispute",
    )
    development_prefixes = (
        "development_benchmark_",
        "near_duplicate_source_text:",
    )
    findings = {field: [] for field in REVIEW_COMPLETION_FIELDS}
    for reason in sorted(reasons):
        if reason.startswith(underlying_prefixes):
            field = "underlying_dispute_review"
        elif reason.startswith(development_prefixes):
            field = "development_overlap_check"
        else:
            field = "internal_dedup_check"
        findings[field].append(reason)
    return findings


def _eligibility_dedup_failures(
    project_root: Path,
    manifest: dict,
    acquisition_id: str,
) -> list[str]:
    findings = _eligibility_review_findings(
        project_root, manifest, acquisition_id
    )
    return [
        f"dedup_or_development_overlap:{reason}"
        for field in REVIEW_COMPLETION_FIELDS
        for reason in findings[field]
    ]


def _review_publication_checkpoint(name: str) -> None:
    """Test seam for review races and atomic-publication failures."""


def _identity_publication_checkpoint(name: str) -> None:
    """Test seam for identity races and atomic-publication failures."""


def _apply_verified_identity(source: dict, checked_identity: dict) -> None:
    """Apply one validated identity without allowing conflicting overwrites."""

    identity_was_ready = not _source_identity_failures(source)
    for field in REVIEW_METADATA_FIELDS:
        prior_value = source[field]
        identity_value = checked_identity["verified_metadata"][field]
        if prior_value is not None and prior_value != identity_value:
            raise SourceIntakeError(
                f"Source identity conflicts with existing {field}"
            )
        source[field] = identity_value
        source["metadata_verification"][field] = True

    prior_related = source["related_decision_ids"]
    identity_related = checked_identity["related_decision_ids"]
    if (
        (identity_was_ready and prior_related != identity_related)
        or (not identity_was_ready and prior_related and prior_related != identity_related)
    ):
        raise SourceIntakeError(
            "Source identity conflicts with existing related_decision_ids"
        )
    source["related_decision_ids"] = list(identity_related)

    prior_underlying = source["underlying_dispute_id"]
    identity_underlying = checked_identity["underlying_dispute_id"]
    if (
        (identity_was_ready and prior_underlying != identity_underlying)
        or (
            not identity_was_ready
            and prior_underlying is not None
            and prior_underlying != identity_underlying
        )
    ):
        raise SourceIntakeError(
            "Source identity conflicts with existing underlying_dispute_id"
        )
    source["underlying_dispute_id"] = identity_underlying


def _source_identity_result(source: dict, project_root: Path, *, idempotent_noop: bool) -> dict:
    failures, _ = _source_gate_failures(source, project_root)
    return {
        "acquisition_id": source["acquisition_id"],
        "source_id": source["source_id"],
        "acquisition_status": source["acquisition_status"],
        "metadata_verification": deepcopy(source["metadata_verification"]),
        "gate_failures": failures,
        "idempotent_noop": idempotent_noop,
    }


def record_source_identity(
    project_root: Path,
    acquisition_id: str,
    identity_payload: dict,
) -> dict:
    """Persist verified per-document identity without review or promotion."""

    project_root = Path(project_root).resolve()
    checked_identity = validate_source_identity_payload(identity_payload)
    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    _acquire_manifest_mutation_lock(intake_chain)
    try:
        manifest, original_manifest_bytes = _load_authoritative_manifest_locked(
            project_root, intake_chain
        )
        matches = [
            source
            for source in manifest["sources"]
            if source["acquisition_id"] == acquisition_id
        ]
        if len(matches) != 1:
            raise SourceIntakeError(f"Unknown acquisition_id: {acquisition_id}")
        source = matches[0]
        if source["acquisition_status"] != "extracted":
            raise SourceIntakeError(
                "Source identity requires extracted status; "
                f"found {source['acquisition_status']}"
            )
        integrity_failures, structured_payload = _source_integrity_failures(
            source, project_root
        )
        if integrity_failures or structured_payload is None:
            joined = (
                ", ".join(integrity_failures)
                or "structured_payload_unavailable"
            )
            raise SourceIntakeError(
                f"Source identity integrity gates failed for {acquisition_id}: {joined}"
            )

        updated_manifest = deepcopy(manifest)
        updated_source = next(
            item
            for item in updated_manifest["sources"]
            if item["acquisition_id"] == acquisition_id
        )
        _apply_verified_identity(updated_source, checked_identity)

        allowed_changes = set(REVIEW_METADATA_FIELDS) | {
            "underlying_dispute_id",
            "related_decision_ids",
            "metadata_verification",
        }
        for field in SOURCE_REQUIRED_FIELDS - allowed_changes:
            if updated_source[field] != source[field]:
                raise SourceIntakeError(
                    f"Source identity attempted forbidden mutation: {field}"
                )
        for field in REVIEW_COMPLETION_FIELDS:
            if (
                updated_source["metadata_verification"][field]
                != source["metadata_verification"][field]
            ):
                raise SourceIntakeError(
                    f"Source identity attempted review mutation: {field}"
                )

        validate_acquisition_manifest(updated_manifest, prior_manifest=manifest)
        _validate_manifest_anchors(project_root, updated_manifest)
        _validate_manifest_artifacts(project_root, updated_manifest)

        if updated_source == source:
            return _source_identity_result(
                updated_source, project_root, idempotent_noop=True
            )

        intended_manifest_bytes = _canonical_json_bytes(updated_manifest)
        try:
            _identity_publication_checkpoint("before_manifest_compare")
            current_manifest_bytes = _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_manifest_bytes != original_manifest_bytes:
                raise SourceIntakeError(
                    "Acquisition manifest changed during source identity"
                )
            intake_chain.assert_identity()
            _atomic_write_bytes_at(
                intake_chain.fd,
                ACQUISITION_MANIFEST_RELATIVE.name,
                intended_manifest_bytes,
            )
            _identity_publication_checkpoint("after_manifest_replacement")
            if (
                _read_regular_bytes_at(
                    intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                )
                != intended_manifest_bytes
            ):
                raise SourceIntakeError(
                    "Published source-identity manifest verification failed"
                )
            checked_manifest, checked_bytes = _load_authoritative_manifest_locked(
                project_root, intake_chain
            )
            checked_source = next(
                item
                for item in checked_manifest["sources"]
                if item["acquisition_id"] == acquisition_id
            )
            if checked_bytes != intended_manifest_bytes or checked_source != updated_source:
                raise SourceIntakeError(
                    "Published source identity failed authoritative reload"
                )
            result = _source_identity_result(
                checked_source, project_root, idempotent_noop=False
            )
        except Exception as publication_error:
            current_manifest_bytes = _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_manifest_bytes == intended_manifest_bytes:
                try:
                    _atomic_write_bytes_at(
                        intake_chain.fd,
                        ACQUISITION_MANIFEST_RELATIVE.name,
                        original_manifest_bytes,
                    )
                    if (
                        _read_regular_bytes_at(
                            intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                        )
                        != original_manifest_bytes
                    ):
                        raise SourceIntakeError(
                            "Source-identity rollback verification failed"
                        )
                    _load_authoritative_manifest_locked(project_root, intake_chain)
                except Exception as rollback_error:
                    raise SourceIntakeError(
                        "Source-identity publication rollback failed"
                    ) from rollback_error
                raise SourceIntakeError(
                    "Source-identity publication failed; previous manifest restored"
                ) from publication_error
            if current_manifest_bytes != original_manifest_bytes:
                raise SourceIntakeError(
                    "Source-identity publication failed with an unexpected manifest "
                    "state; refusing unsafe overwrite"
                ) from publication_error
            raise
        return result
    finally:
        _release_manifest_mutation_lock(intake_chain)
        intake_chain.close()


def _source_review_result(
    project_root: Path,
    manifest: dict,
    source: dict,
    *,
    idempotent_noop: bool,
) -> dict:
    failures, _ = _source_gate_failures(source, project_root)
    failures.extend(
        _eligibility_dedup_failures(
            project_root, manifest, source["acquisition_id"]
        )
    )
    return {
        "acquisition_id": source["acquisition_id"],
        "source_id": source["source_id"],
        "acquisition_status": source["acquisition_status"],
        "metadata_verification": deepcopy(source["metadata_verification"]),
        "gate_failures": sorted(set(failures)),
        "idempotent_noop": idempotent_noop,
    }


def record_source_review(
    project_root: Path,
    acquisition_id: str,
    review_payload: dict,
) -> dict:
    """Record one strict, independently verified review without promotion.

    The intake mutation lock covers authoritative loading, validation,
    comparison, publication, and authoritative reloading.  Only legal metadata,
    its verification state, related-decision metadata, and deterministic review
    findings can change; provenance, artifacts, status, and status history remain
    immutable.
    """

    project_root = Path(project_root).resolve()
    checked_review = validate_source_review_payload(review_payload)
    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    _acquire_manifest_mutation_lock(intake_chain)
    try:
        manifest, original_manifest_bytes = _load_authoritative_manifest_locked(
            project_root, intake_chain
        )
        matches = [
            source
            for source in manifest["sources"]
            if source["acquisition_id"] == acquisition_id
        ]
        if len(matches) != 1:
            raise SourceIntakeError(f"Unknown acquisition_id: {acquisition_id}")
        source = matches[0]
        if source["acquisition_status"] != "extracted":
            raise SourceIntakeError(
                "Source review requires extracted status; "
                f"found {source['acquisition_status']}"
            )

        updated_manifest = deepcopy(manifest)
        updated_source = next(
            item
            for item in updated_manifest["sources"]
            if item["acquisition_id"] == acquisition_id
        )
        if checked_review["identity"] is not None:
            _apply_verified_identity(updated_source, checked_review["identity"])
        else:
            identity_failures = _source_identity_failures(source)
            if identity_failures:
                raise SourceIntakeError(
                    "Source review target identity not ready: "
                    + ", ".join(identity_failures)
                )

        for field in REVIEW_COMPLETION_FIELDS:
            completed = checked_review["reviews"][field]["completed"]
            if source["metadata_verification"][field] and not completed:
                raise SourceIntakeError(
                    f"Source review cannot regress completed {field}"
                )
            updated_source["metadata_verification"][field] = completed

        computed_findings = _eligibility_review_findings(
            project_root, updated_manifest, acquisition_id
        )
        for field in REVIEW_COMPLETION_FIELDS:
            supplied_findings = checked_review["reviews"][field]["findings"]
            if supplied_findings != computed_findings[field]:
                raise SourceIntakeError(
                    f"Source review findings mismatch for {field}"
                )

        deterministic_rejections = sorted(
            f"dedup_or_development_overlap:{reason}"
            for field in REVIEW_COMPLETION_FIELDS
            for reason in computed_findings[field]
        )
        if (
            source["rejection_reasons"]
            and source["rejection_reasons"] != deterministic_rejections
        ):
            raise SourceIntakeError(
                "Source review conflicts with existing rejection_reasons"
            )
        updated_source["rejection_reasons"] = deterministic_rejections

        allowed_changes = set(REVIEW_METADATA_FIELDS) | {
            "underlying_dispute_id",
            "related_decision_ids",
            "metadata_verification",
            "rejection_reasons",
        }
        for field in SOURCE_REQUIRED_FIELDS - allowed_changes:
            if updated_source[field] != source[field]:
                raise SourceIntakeError(
                    f"Source review attempted forbidden mutation: {field}"
                )
        validate_acquisition_manifest(updated_manifest, prior_manifest=manifest)
        _validate_manifest_anchors(project_root, updated_manifest)
        _validate_manifest_artifacts(project_root, updated_manifest)

        if updated_source == source:
            return _source_review_result(
                project_root,
                updated_manifest,
                updated_source,
                idempotent_noop=True,
            )

        intended_manifest_bytes = _canonical_json_bytes(updated_manifest)
        try:
            _review_publication_checkpoint("before_manifest_compare")
            current_manifest_bytes = _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_manifest_bytes != original_manifest_bytes:
                raise SourceIntakeError(
                    "Acquisition manifest changed during source review"
                )
            intake_chain.assert_identity()
            _atomic_write_bytes_at(
                intake_chain.fd,
                ACQUISITION_MANIFEST_RELATIVE.name,
                intended_manifest_bytes,
            )
            _review_publication_checkpoint("after_manifest_replacement")
            if (
                _read_regular_bytes_at(
                    intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                )
                != intended_manifest_bytes
            ):
                raise SourceIntakeError(
                    "Published source-review manifest verification failed"
                )
            checked_manifest, checked_bytes = _load_authoritative_manifest_locked(
                project_root, intake_chain
            )
            checked_source = next(
                item
                for item in checked_manifest["sources"]
                if item["acquisition_id"] == acquisition_id
            )
            if checked_bytes != intended_manifest_bytes or checked_source != updated_source:
                raise SourceIntakeError(
                    "Published source review failed authoritative reload"
                )
            result = _source_review_result(
                project_root,
                checked_manifest,
                checked_source,
                idempotent_noop=False,
            )
        except Exception as publication_error:
            current_manifest_bytes = _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_manifest_bytes == intended_manifest_bytes:
                try:
                    _atomic_write_bytes_at(
                        intake_chain.fd,
                        ACQUISITION_MANIFEST_RELATIVE.name,
                        original_manifest_bytes,
                    )
                    if (
                        _read_regular_bytes_at(
                            intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                        )
                        != original_manifest_bytes
                    ):
                        raise SourceIntakeError(
                            "Source-review rollback verification failed"
                        )
                    _load_authoritative_manifest_locked(project_root, intake_chain)
                except Exception as rollback_error:
                    raise SourceIntakeError(
                        "Source-review publication rollback failed"
                    ) from rollback_error
                raise SourceIntakeError(
                    "Source-review publication failed; previous manifest restored"
                ) from publication_error
            if current_manifest_bytes != original_manifest_bytes:
                raise SourceIntakeError(
                    "Source-review publication failed with an unexpected manifest "
                    "state; refusing unsafe overwrite"
                ) from publication_error
            raise
        return result
    finally:
        _release_manifest_mutation_lock(intake_chain)
        intake_chain.close()


def promote_source_to_eligible(project_root: Path, acquisition_id: str) -> dict:
    """Atomically promote one fully validated extracted source to eligible."""

    project_root = Path(project_root).resolve()
    manifest, original_manifest_bytes = _load_authoritative_manifest(project_root)
    matches = [
        source
        for source in manifest["sources"]
        if source["acquisition_id"] == acquisition_id
    ]
    if len(matches) != 1:
        raise SourceIntakeError(f"Unknown acquisition_id: {acquisition_id}")
    source = matches[0]
    if source["acquisition_status"] != "extracted":
        raise SourceIntakeError(
            "Eligibility promotion requires extracted status; "
            f"found {source['acquisition_status']}"
        )

    failures, _ = _source_gate_failures(source, project_root)
    failures.extend(
        _eligibility_dedup_failures(project_root, manifest, acquisition_id)
    )
    failures = sorted(set(failures))
    if failures:
        raise SourceIntakeError(
            f"Eligibility gates failed for {acquisition_id}: {', '.join(failures)}"
        )

    updated_manifest = deepcopy(manifest)
    updated_source = next(
        item
        for item in updated_manifest["sources"]
        if item["acquisition_id"] == acquisition_id
    )
    timestamp = datetime.now().astimezone()
    prior_timestamp = _parse_offset_timestamp(
        updated_source["status_history"][-1]["timestamp"],
        "status_history final timestamp",
    )
    if timestamp < prior_timestamp:
        raise SourceIntakeError(
            "System clock precedes the final source status-history timestamp"
        )
    updated_source["acquisition_status"] = "eligible"
    updated_source["status_history"].append({
        "status": "eligible",
        "timestamp": timestamp.isoformat(),
        "reasons": [],
    })
    validate_acquisition_manifest(updated_manifest, prior_manifest=manifest)
    _validate_manifest_anchors(project_root, updated_manifest)
    intended_manifest_bytes = _canonical_json_bytes(updated_manifest)

    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    _acquire_manifest_mutation_lock(intake_chain)
    published = False
    try:
        _eligibility_publication_checkpoint("before_manifest_compare")
        current_manifest_bytes = _read_regular_bytes_at(
            intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
        )
        if current_manifest_bytes != original_manifest_bytes:
            raise SourceIntakeError(
                "Acquisition manifest changed during eligibility validation"
            )
        intake_chain.assert_identity()
        try:
            _atomic_write_bytes_at(
                intake_chain.fd,
                ACQUISITION_MANIFEST_RELATIVE.name,
                intended_manifest_bytes,
            )
        except Exception as publication_error:
            current_manifest_bytes = _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_manifest_bytes == intended_manifest_bytes:
                _atomic_write_bytes_at(
                    intake_chain.fd,
                    ACQUISITION_MANIFEST_RELATIVE.name,
                    original_manifest_bytes,
                )
                if (
                    _read_regular_bytes_at(
                        intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                    )
                    != original_manifest_bytes
                ):
                    raise SourceIntakeError(
                        "Eligibility publication rollback verification failed"
                    ) from publication_error
                raise SourceIntakeError(
                    "Eligibility publication failed after replacement; "
                    "previous manifest restored"
                ) from publication_error
            if current_manifest_bytes != original_manifest_bytes:
                raise SourceIntakeError(
                    "Eligibility publication failed with an unexpected manifest "
                    "state; refusing unsafe overwrite"
                ) from publication_error
            raise
        published = True
        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != intended_manifest_bytes
        ):
            raise SourceIntakeError("Published eligibility manifest verification failed")
        intake_chain.assert_identity()
    finally:
        _release_manifest_mutation_lock(intake_chain)
        intake_chain.close()

    try:
        checked_manifest = load_acquisition_manifest(project_root)
        checked_source = next(
            item
            for item in checked_manifest["sources"]
            if item["acquisition_id"] == acquisition_id
        )
        if checked_source["acquisition_status"] != "eligible":
            raise SourceIntakeError("Eligibility status was not published")
        checked_inventory, checked_report, _ = inventory_independent_sources(
            project_root
        )
        checked_result = next(
            item
            for item in checked_report["sources"]
            if item["acquisition_id"] == acquisition_id
        )
        if (
            not checked_result["eligible"]
            or checked_result["gate_failures"]
            or checked_source["source_id"]
            not in {record["source_id"] for record in checked_inventory}
        ):
            raise SourceIntakeError("Promoted source failed authoritative reload")
    except Exception as verification_error:
        if published:
            _, rollback_chain = _open_trusted_directory(
                project_root, INTAKE_ROOT_RELATIVE, create=False
            )
            try:
                current_manifest_bytes = _read_regular_bytes_at(
                    rollback_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                )
                if current_manifest_bytes != intended_manifest_bytes:
                    raise SourceIntakeError(
                        "Post-publication verification failed after a concurrent "
                        "manifest change; refusing unsafe rollback"
                    ) from verification_error
                rollback_chain.assert_identity()
                _atomic_write_bytes_at(
                    rollback_chain.fd,
                    ACQUISITION_MANIFEST_RELATIVE.name,
                    original_manifest_bytes,
                )
                if (
                    _read_regular_bytes_at(
                        rollback_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                    )
                    != original_manifest_bytes
                ):
                    raise SourceIntakeError(
                        "Eligibility manifest rollback verification failed"
                    ) from verification_error
                rollback_chain.assert_identity()
            finally:
                rollback_chain.close()
        raise SourceIntakeError(
            "Eligibility promotion verification failed; previous manifest restored"
        ) from verification_error

    return {
        "acquisition_id": acquisition_id,
        "source_id": checked_source["source_id"],
        "acquisition_status": "eligible",
        "status_history_length": len(checked_source["status_history"]),
    }


def _eligibility_publication_checkpoint(name: str) -> None:
    """Test seam for manifest races and publication failures."""


def extract_intake_source(
    project_root: Path, acquisition_id: str, force_ocr: bool = False
) -> dict:
    """Extract one manifest source without overwriting and record its hash.

    This is an explicit mutating operation intended for the future acquisition
    workflow.  It does not promote the source beyond ``extracted``.
    """

    project_root = Path(project_root).resolve()
    manifest, original_manifest_bytes = _load_authoritative_manifest(project_root)
    matches = [item for item in manifest["sources"] if item["acquisition_id"] == acquisition_id]
    if len(matches) != 1:
        raise SourceIntakeError(f"Unknown acquisition_id: {acquisition_id}")
    source = matches[0]
    if source["acquisition_status"] != "hash_verified":
        raise SourceIntakeError("Extraction requires hash_verified status")

    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    raw_chain, raw_path, raw_fd = _open_acquisition_file_fd(
        project_root, source, "raw"
    )
    _validate_pdf_fd(raw_fd, source["sha256"])
    output_dir, output_path = _prepare_output_directory(project_root, source)
    _, output_chain = _open_trusted_directory(
        project_root,
        INTAKE_ROOT_RELATIVE / "structured" / acquisition_id,
        create=True,
    )
    _, transaction_chain = _open_trusted_directory(
        project_root, TRANSACTION_ROOT_RELATIVE, create=False
    )
    temporary_name = f".extract-partial-{uuid.uuid4().hex}"
    os.mkdir(temporary_name, mode=0o700, dir_fd=output_chain.fd)
    temporary_fd = os.open(
        temporary_name, _directory_open_flags(), dir_fd=output_chain.fd
    )
    output_name = output_path.name
    journal_name = f"{acquisition_id}.json"
    journal = None
    journal_created = False
    final_published = False
    manifest_published = False
    _acquire_manifest_mutation_lock(intake_chain)
    try:
        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != original_manifest_bytes
        ):
            raise SourceIntakeError("Acquisition manifest changed before extraction")
        succeeded = _run_local_extractor(
            raw_fd,
            temporary_fd,
            source["original_filename"],
            force_ocr,
        )
        if not succeeded:
            raise SourceIntakeError("Local extraction failed")
        temporary_metadata = _entry_lstat(temporary_fd, output_name)
        if temporary_metadata is None or not stat.S_ISREG(temporary_metadata.st_mode):
            raise SourceIntakeError("Extractor did not produce one regular temporary output")
        structured_hash = _validate_structured_at(temporary_fd, output_name, source)

        _validate_pdf_fd(raw_fd, source["sha256"])
        _assert_file_fd_identity(raw_path, raw_fd)
        raw_chain.assert_identity()
        output_chain.assert_identity()

        updated_manifest = deepcopy(manifest)
        updated_source = next(
            item
            for item in updated_manifest["sources"]
            if item["acquisition_id"] == acquisition_id
        )
        timestamp = datetime.now().astimezone().isoformat()
        updated_source["structured_sha256"] = structured_hash
        updated_source["acquisition_status"] = "extracted"
        updated_source["status_history"].append(
            {"status": "extracted", "timestamp": timestamp, "reasons": []}
        )
        validate_acquisition_manifest(updated_manifest, prior_manifest=manifest)
        _validate_manifest_anchors(project_root, updated_manifest)
        intended_manifest_bytes = _canonical_json_bytes(updated_manifest)
        now = datetime.now().astimezone().isoformat()
        journal = _write_journal_at(
            transaction_chain.fd,
            {
                "journal_version": PUBLICATION_JOURNAL_VERSION,
                "transaction_id": uuid.uuid4().hex,
                "acquisition_id": acquisition_id,
                "state": "PREPARED",
                "previous_manifest_sha256": _sha256_bytes(original_manifest_bytes),
                "intended_manifest_sha256": _sha256_bytes(intended_manifest_bytes),
                "previous_status": source["acquisition_status"],
                "intended_status": "extracted",
                "temp_directory_name": temporary_name,
                "final_artifact_name": output_name,
                "final_relative_path": source["structured_path"],
                "structured_sha256": structured_hash,
                "created_at": now,
                "updated_at": now,
            },
            create_once=True,
        )
        journal_created = True
        _publication_checkpoint("after_prepared")

        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != original_manifest_bytes
        ):
            raise SourceIntakeError("Acquisition manifest changed during extraction")
        intake_chain.assert_identity()
        output_chain.assert_identity()
        transaction_chain.assert_identity()
        _publication_checkpoint("before_artifact_publication")
        intake_chain.assert_identity()
        output_chain.assert_identity()
        transaction_chain.assert_identity()
        try:
            os.link(
                output_name,
                output_name,
                src_dir_fd=temporary_fd,
                dst_dir_fd=output_chain.fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise SourceIntakeError(
                f"Refusing existing structured output: {output_path}"
            ) from exc
        final_published = True
        if _hash_regular_file_at(output_chain.fd, output_name) != structured_hash:
            raise SourceIntakeError("Published structured SHA256 mismatch")
        output_chain.assert_identity()
        journal["state"] = "ARTIFACT_PUBLISHED"
        journal["updated_at"] = datetime.now().astimezone().isoformat()
        journal = _write_journal_at(
            transaction_chain.fd, journal, create_once=False
        )
        _publication_checkpoint("after_artifact_published")

        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != original_manifest_bytes
        ):
            raise SourceIntakeError("Acquisition manifest changed before publication")
        intake_chain.assert_identity()
        output_chain.assert_identity()
        transaction_chain.assert_identity()
        _atomic_write_bytes_at(
            intake_chain.fd,
            ACQUISITION_MANIFEST_RELATIVE.name,
            intended_manifest_bytes,
        )
        manifest_published = True
        if (
            _read_regular_bytes_at(
                intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
            )
            != intended_manifest_bytes
        ):
            raise SourceIntakeError("Published manifest verification failed")
        journal["state"] = "MANIFEST_PUBLISHED"
        journal["updated_at"] = datetime.now().astimezone().isoformat()
        journal = _write_journal_at(
            transaction_chain.fd, journal, create_once=False
        )
        _publication_checkpoint("after_manifest_published")

        if _hash_regular_file_at(output_chain.fd, output_name) != structured_hash:
            raise SourceIntakeError("Final structured SHA256 mismatch")
        intake_chain.assert_identity()
        output_chain.assert_identity()
        transaction_chain.assert_identity()
        os.unlink(journal_name, dir_fd=transaction_chain.fd)
        os.fsync(transaction_chain.fd)
        journal_created = False
        return {
            "acquisition_id": acquisition_id,
            "structured_path": source["structured_path"],
            "structured_sha256": structured_hash,
            "acquisition_status": "extracted",
        }
    except Exception:
        if manifest_published:
            try:
                _atomic_write_bytes_at(
                    intake_chain.fd,
                    ACQUISITION_MANIFEST_RELATIVE.name,
                    original_manifest_bytes,
                )
                if (
                    _read_regular_bytes_at(
                        intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
                    )
                    != original_manifest_bytes
                ):
                    raise SourceIntakeError(
                        "Failed to verify manifest rollback after chain failure"
                    )
                manifest_published = False
            except Exception as rollback_error:
                raise SourceIntakeError(
                    "Manifest rollback failed; journal recovery is required"
                ) from rollback_error
        if not manifest_published:
            if (
                final_published
                and _entry_lstat(output_chain.fd, output_name) is not None
            ):
                os.unlink(output_name, dir_fd=output_chain.fd)
                os.fsync(output_chain.fd)
            if (
                journal_created
                and _entry_lstat(transaction_chain.fd, journal_name) is not None
            ):
                os.unlink(journal_name, dir_fd=transaction_chain.fd)
                os.fsync(transaction_chain.fd)
                journal_created = False
        raise
    finally:
        try:
            _remove_tree_at(output_chain.fd, temporary_name)
        finally:
            os.close(temporary_fd)
            transaction_chain.close()
            output_chain.close()
            os.close(raw_fd)
            raw_chain.close()
            _release_manifest_mutation_lock(intake_chain)
            intake_chain.close()


def _publication_checkpoint(name: str) -> None:
    """Test seam for simulating hard-stop publication boundaries."""


def _run_local_extractor(
    raw_fd: int,
    output_dir_fd: int,
    original_filename: str,
    force_ocr: bool,
) -> bool:
    from backend.scripts import ocrFullBanAn

    return ocrFullBanAn.process_judgment_file(
        Path(f"/dev/fd/{raw_fd}"),
        Path("."),
        force_ocr=force_ocr,
        overwrite=False,
        output_dir_fd=output_dir_fd,
        original_filename=original_filename,
    )


def _source_gate_failures_for_extraction(
    source: dict, project_root: Path
) -> tuple[list[str], None]:
    failures = []
    try:
        raw_chain, pdf_path, pdf_fd = _open_acquisition_file_fd(
            project_root, source, "raw"
        )
        try:
            _validate_pdf_fd(pdf_fd, source["sha256"])
            _assert_file_fd_identity(pdf_path, pdf_fd)
            raw_chain.assert_identity()
        finally:
            os.close(pdf_fd)
            raw_chain.close()
    except SourceIntakeError as exc:
        failures.append(f"invalid_pdf:{exc}")
    return sorted(set(failures)), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent judgment source intake", allow_abbrev=False
    )
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "register",
            "extract",
            "identify",
            "review",
            "promote-eligible",
        ),
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--acquisition-id")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--offline-description")
    parser.add_argument("--source-domain")
    parser.add_argument("--download-timestamp")
    parser.add_argument("--original-filename")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--force-ocr", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        if args.identity_file is not None:
            parser.error("--identity-file is only valid for identify")
        if args.review_file is not None:
            parser.error("--review-file is only valid for review")
        _, report, _ = inventory_independent_sources(args.project_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "register":
        if args.identity_file is not None:
            parser.error("--identity-file is only valid for identify")
        if args.review_file is not None:
            parser.error("--review-file is only valid for review")
        if args.force_ocr:
            parser.error("--force-ocr is not valid for register")
        required = {
            "--acquisition-id": args.acquisition_id,
            "--pdf": args.pdf,
            "--source-domain": args.source_domain,
            "--download-timestamp": args.download_timestamp,
            "--original-filename": args.original_filename,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"register requires {', '.join(missing)}")
        if (args.source_url is None) == (args.offline_description is None):
            parser.error(
                "register requires exactly one of --source-url or "
                "--offline-description"
            )
        result = register_source(
            args.project_root,
            args.acquisition_id,
            args.pdf,
            args.source_url,
            args.source_domain,
            args.download_timestamp,
            args.original_filename,
            (
                {"description": args.offline_description}
                if args.offline_description is not None
                else None
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "identify":
        forbidden = {
            "--pdf": args.pdf,
            "--source-url": args.source_url,
            "--offline-description": args.offline_description,
            "--source-domain": args.source_domain,
            "--download-timestamp": args.download_timestamp,
            "--original-filename": args.original_filename,
            "--review-file": args.review_file,
        }
        supplied_forbidden = [
            name for name, value in forbidden.items() if value is not None
        ]
        if args.force_ocr:
            supplied_forbidden.append("--force-ocr")
        if supplied_forbidden:
            parser.error(
                "identify rejects unrelated arguments: "
                + ", ".join(supplied_forbidden)
            )
        if not args.acquisition_id or args.identity_file is None:
            parser.error(
                "identify requires --acquisition-id and --identity-file"
            )
        identity_path = args.identity_file
        if (
            not _lexists(identity_path)
            or identity_path.is_symlink()
            or not identity_path.is_file()
        ):
            parser.error("--identity-file must be a regular non-symlink file")
        try:
            identity_payload = json.loads(
                identity_path.read_bytes().decode("utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parser.error(f"--identity-file must contain valid UTF-8 JSON: {exc}")
        result = record_source_identity(
            args.project_root, args.acquisition_id, identity_payload
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "review":
        forbidden = {
            "--pdf": args.pdf,
            "--source-url": args.source_url,
            "--offline-description": args.offline_description,
            "--source-domain": args.source_domain,
            "--download-timestamp": args.download_timestamp,
            "--original-filename": args.original_filename,
            "--identity-file": args.identity_file,
        }
        supplied_forbidden = [
            name for name, value in forbidden.items() if value is not None
        ]
        if args.force_ocr:
            supplied_forbidden.append("--force-ocr")
        if supplied_forbidden:
            parser.error(
                "review rejects unrelated arguments: "
                + ", ".join(supplied_forbidden)
            )
        if not args.acquisition_id or args.review_file is None:
            parser.error(
                "review requires --acquisition-id and --review-file"
            )
        review_path = args.review_file
        if (
            not _lexists(review_path)
            or review_path.is_symlink()
            or not review_path.is_file()
        ):
            parser.error("--review-file must be a regular non-symlink file")
        try:
            review_payload = json.loads(review_path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parser.error(f"--review-file must contain valid UTF-8 JSON: {exc}")
        result = record_source_review(
            args.project_root, args.acquisition_id, review_payload
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "promote-eligible":
        if args.identity_file is not None:
            parser.error("--identity-file is only valid for identify")
        if args.review_file is not None:
            parser.error("--review-file is only valid for review")
        if args.force_ocr:
            parser.error("--force-ocr is only valid for extract")
        if not args.acquisition_id:
            parser.error("promote-eligible requires --acquisition-id")
        result = promote_source_to_eligible(
            args.project_root, args.acquisition_id
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.identity_file is not None:
        parser.error("--identity-file is only valid for identify")
    if args.review_file is not None:
        parser.error("--review-file is only valid for review")
    if not args.acquisition_id:
        parser.error("extract requires --acquisition-id")
    result = extract_intake_source(args.project_root, args.acquisition_id, args.force_ocr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
