"""Fail-closed source intake for the independent retrieval benchmark.

This module handles provenance and source eligibility only.  It deliberately
does not generate queries or ground truth and has no retrieval/model imports.
Historical development judgments remain outside this contract.
"""

from __future__ import annotations

import argparse
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
    if current_status in ELIGIBLE_STATUS_HISTORY:
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
    for index, page in enumerate(pages, start=1):
        if not isinstance(page, dict) or set(page) != {"page", "text"}:
            raise SourceIntakeError(f"Structured page {index} has invalid fields")
        if page["page"] != index:
            raise SourceIntakeError("Structured page numbers must be consecutive and one-based")
        if not isinstance(page["text"], str) or not page["text"].strip():
            raise SourceIntakeError(f"Structured page {index} must contain usable text")
        if unicodedata.normalize("NFC", page["text"]) != page["text"]:
            raise SourceIntakeError(f"Structured page {index} text must be Unicode NFC")
    sections = payload["sections"]
    required_sections = {"header", "facts", "reasoning", "decision"}
    if not isinstance(sections, dict) or set(sections) != required_sections:
        raise SourceIntakeError("Structured sections must contain exactly the required keys")
    for name, text in sections.items():
        if not isinstance(text, str):
            raise SourceIntakeError(f"Structured section {name} must be a string")
        if unicodedata.normalize("NFC", text) != text:
            raise SourceIntakeError(f"Structured section {name} must be Unicode NFC")


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
        "metadata_source": "verified_independent_acquisition_manifest_v1",
    }
    if include_private_text:
        record["_dedup_text"] = full_text
    return record


def _journal_payload(base: dict) -> dict:
    record = {key: value for key, value in base.items() if key != "journal_fingerprint"}
    return {**record, "journal_fingerprint": _canonical_json_sha256(record)}


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


def _load_authoritative_manifest(project_root: Path) -> tuple[dict, bytes]:
    project_root = Path(project_root).resolve()
    _, intake_chain = _open_trusted_directory(
        project_root, INTAKE_ROOT_RELATIVE, create=False
    )
    try:
        manifest_bytes = _read_regular_bytes_at(
            intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
        )
        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceIntakeError("Acquisition manifest is not valid UTF-8 JSON") from exc
        validate_acquisition_manifest(manifest)
        _validate_manifest_anchors(project_root, manifest)
        _reconcile_publication_journals(project_root, manifest, manifest_bytes)
        _validate_manifest_artifacts(project_root, manifest)
        intake_chain.assert_identity()
        return manifest, manifest_bytes
    finally:
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
        source_results.append({
            "acquisition_id": source["acquisition_id"],
            "source_id": source["source_id"],
            "claimed_status": source["acquisition_status"],
            "eligible": claims_eligible and not failures,
            "gate_failures": failures,
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
    if (
        _read_regular_bytes_at(
            intake_chain.fd, ACQUISITION_MANIFEST_RELATIVE.name
        )
        != original_manifest_bytes
    ):
        intake_chain.close()
        raise SourceIntakeError("Acquisition manifest changed before extraction")

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
    try:
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
    parser = argparse.ArgumentParser(description="Independent judgment source intake")
    parser.add_argument("command", choices=("validate", "extract"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--acquisition-id")
    parser.add_argument("--force-ocr", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        _, report, _ = inventory_independent_sources(args.project_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.acquisition_id:
        parser.error("extract requires --acquisition-id")
    result = extract_intake_source(args.project_root, args.acquisition_id, args.force_ocr)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
