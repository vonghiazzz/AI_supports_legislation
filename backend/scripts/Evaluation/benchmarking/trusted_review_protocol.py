"""Inactive trusted activation and review-attestation protocol for R11.

The live registry intentionally contains no issuers or producers.  Verification
is based on code-controlled registry entries and Ed25519 signatures; source
data, caller strings, hashes without signatures, and intake-local JSON flags
cannot create authority.  Test registries exercise the protocol without
issuing a production authorization or review attestation.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REGISTRY_VERSION = "independent_trusted_review_registry_v1"
ISSUER_VERSION = "independent_review_issuer_v1"
PRODUCER_CONTRACT_VERSION = "independent_review_producer_v1"
ACTIVATION_AUTHORIZATION_VERSION = "independent_v21_activation_authorization_v1"
ATTESTATION_VERSION = "independent_source_review_attestation_v1"
REVIEW_CONTRACT_VERSION = "independent_source_review_contract_v2_1"
EVIDENCE_AMENDMENT_ID = "relation_specific_evidence_completeness_v2_1"
POLICY_ID = "independent_dedup_policy_v2"
POLICY_VERSION = 2
POLICY_FINGERPRINT = (
    "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
)
TARGET_MANIFEST_VERSION = "independent_source_acquisition_v2_1_scoped_bindings"
TARGET_REPOSITORY_ID = "vonghiazzz/AI_supports_legislation"
LIVE_LINEAGE_ROOT_SHA256 = (
    "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a"
)

REGISTRY_FIELDS = {
    "registry_version",
    "activation_state",
    "issuers",
    "producers",
    "approval_provenance",
    "registry_fingerprint",
}
ISSUER_FIELDS = {
    "issuer_id",
    "issuer_version",
    "ed25519_public_key_base64",
    "status",
    "approved_at",
    "approval_provenance",
    "revoked_at",
}
PRODUCER_FIELDS = {
    "producer_id",
    "producer_version",
    "ed25519_public_key_base64",
    "approved_code_sha256",
    "policy_id",
    "policy_version",
    "policy_fingerprint",
    "review_contract_version",
    "evidence_amendment_id",
    "trusted_execution_identity",
    "issuer_id",
    "approval_id",
    "approved_at",
    "approval_provenance",
    "status",
    "revoked_at",
}
ACTIVATION_FIELDS = {
    "authorization_version",
    "authorization_id",
    "issuer_id",
    "issuer_version",
    "target_repository_id",
    "target_manifest_pre_sha256",
    "target_manifest_version",
    "policy_id",
    "policy_version",
    "policy_fingerprint",
    "cutover_code_bundle_fingerprint",
    "issued_at",
    "expires_at",
    "nonce",
    "signature_base64",
}
ATTESTATION_FIELDS = {
    "attestation_version",
    "attestation_id",
    "producer_id",
    "producer_version",
    "issuer_id",
    "acquisition_id",
    "source_id",
    "source_local_evidence_fingerprint",
    "current_inventory_fingerprint",
    "candidate_set_fingerprint",
    "recomputed_findings_fingerprint",
    "adjudication_evidence_bindings",
    "review_decision_fingerprint",
    "producer_code_sha256",
    "policy_id",
    "policy_version",
    "policy_fingerprint",
    "review_contract_version",
    "evidence_amendment_id",
    "trusted_execution_identity",
    "issuance_provenance",
    "issued_at",
    "nonce",
    "signature_base64",
}


class TrustedReviewProtocolError(ValueError):
    """A registry, authorization, or attestation failed closed."""


def canonical_bytes(value: Any) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise TrustedReviewProtocolError("Value is not strict canonical JSON") from exc


def canonical_fingerprint(value: Any, *, excluded_key: str | None = None) -> str:
    checked = copy.deepcopy(value)
    if excluded_key is not None:
        if not isinstance(checked, dict):
            raise TrustedReviewProtocolError("Fingerprint exclusion requires an object")
        checked.pop(excluded_key, None)
    return hashlib.sha256(canonical_bytes(checked)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise TrustedReviewProtocolError(f"{field} must be an offset timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TrustedReviewProtocolError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TrustedReviewProtocolError(f"{field} must include an offset")
    return parsed.astimezone(timezone.utc)


def _public_key(value: Any, field: str) -> Ed25519PublicKey:
    if not isinstance(value, str):
        raise TrustedReviewProtocolError(f"{field} must be base64")
    try:
        raw = base64.b64decode(value, validate=True)
        if len(raw) != 32:
            raise ValueError
        return Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, TypeError) as exc:
        raise TrustedReviewProtocolError(f"{field} is not an Ed25519 public key") from exc


def _signature(value: Any) -> bytes:
    if not isinstance(value, str):
        raise TrustedReviewProtocolError("signature_base64 must be base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise TrustedReviewProtocolError("signature_base64 is invalid") from exc
    if len(raw) != 64:
        raise TrustedReviewProtocolError("signature_base64 is not an Ed25519 signature")
    return raw


def _verify_signature(payload: Mapping[str, Any], public_key: Ed25519PublicKey) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "signature_base64"}
    try:
        public_key.verify(_signature(payload.get("signature_base64")), canonical_bytes(unsigned))
    except InvalidSignature as exc:
        raise TrustedReviewProtocolError("Ed25519 signature verification failed") from exc


def _require_identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or value != "".join(value.split())
    ):
        raise TrustedReviewProtocolError(f"{field} is invalid")
    return value


def _validate_issuer(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or set(value) != ISSUER_FIELDS:
        raise TrustedReviewProtocolError("Issuer fields are invalid")
    checked = copy.deepcopy(dict(value))
    _require_identifier(checked["issuer_id"], "issuer_id")
    if checked["issuer_version"] != ISSUER_VERSION:
        raise TrustedReviewProtocolError("Issuer version is unsupported")
    _public_key(checked["ed25519_public_key_base64"], "issuer public key")
    if checked["status"] not in {"ACTIVE", "REVOKED"}:
        raise TrustedReviewProtocolError("Issuer status is invalid")
    _timestamp(checked["approved_at"], "issuer.approved_at")
    _require_identifier(checked["approval_provenance"], "issuer.approval_provenance")
    if checked["status"] == "ACTIVE" and checked["revoked_at"] is not None:
        raise TrustedReviewProtocolError("Active issuer cannot have revoked_at")
    if checked["status"] == "REVOKED":
        _timestamp(checked["revoked_at"], "issuer.revoked_at")
    return checked


def _validate_producer(value: Mapping[str, Any], issuer_ids: set[str]) -> dict:
    if not isinstance(value, Mapping) or set(value) != PRODUCER_FIELDS:
        raise TrustedReviewProtocolError("Producer fields are invalid")
    checked = copy.deepcopy(dict(value))
    _require_identifier(checked["producer_id"], "producer_id")
    _require_identifier(checked["producer_version"], "producer_version")
    _public_key(checked["ed25519_public_key_base64"], "producer public key")
    if not _is_sha256(checked["approved_code_sha256"]):
        raise TrustedReviewProtocolError("Producer code SHA256 is invalid")
    if (
        checked["policy_id"] != POLICY_ID
        or checked["policy_version"] != POLICY_VERSION
        or checked["policy_fingerprint"] != POLICY_FINGERPRINT
        or checked["review_contract_version"] != REVIEW_CONTRACT_VERSION
        or checked["evidence_amendment_id"] != EVIDENCE_AMENDMENT_ID
    ):
        raise TrustedReviewProtocolError("Producer policy/review contract binding is invalid")
    _require_identifier(
        checked["trusted_execution_identity"], "trusted_execution_identity"
    )
    if checked["issuer_id"] not in issuer_ids:
        raise TrustedReviewProtocolError("Producer references an unknown issuer")
    _require_identifier(checked["approval_id"], "approval_id")
    _timestamp(checked["approved_at"], "producer.approved_at")
    _require_identifier(checked["approval_provenance"], "producer.approval_provenance")
    if checked["status"] not in {"ACTIVE", "REVOKED"}:
        raise TrustedReviewProtocolError("Producer status is invalid")
    if checked["status"] == "ACTIVE" and checked["revoked_at"] is not None:
        raise TrustedReviewProtocolError("Active producer cannot have revoked_at")
    if checked["status"] == "REVOKED":
        _timestamp(checked["revoked_at"], "producer.revoked_at")
    return checked


def validate_registry(registry: Mapping[str, Any]) -> dict:
    if not isinstance(registry, Mapping) or set(registry) != REGISTRY_FIELDS:
        raise TrustedReviewProtocolError("Registry fields are invalid")
    checked = copy.deepcopy(dict(registry))
    if checked["registry_version"] != REGISTRY_VERSION:
        raise TrustedReviewProtocolError("Registry version is unsupported")
    if checked["activation_state"] not in {"INACTIVE", "ACTIVE"}:
        raise TrustedReviewProtocolError("Registry activation state is invalid")
    if not isinstance(checked["issuers"], list) or not isinstance(
        checked["producers"], list
    ):
        raise TrustedReviewProtocolError("Registry issuers/producers must be arrays")
    issuers = [_validate_issuer(item) for item in checked["issuers"]]
    issuer_ids = [item["issuer_id"] for item in issuers]
    if len(issuer_ids) != len(set(issuer_ids)):
        raise TrustedReviewProtocolError("Registry issuer IDs are not unique")
    producers = [_validate_producer(item, set(issuer_ids)) for item in checked["producers"]]
    producer_keys = [(item["producer_id"], item["producer_version"]) for item in producers]
    if len(producer_keys) != len(set(producer_keys)):
        raise TrustedReviewProtocolError("Registry producer identities are not unique")
    if checked["activation_state"] == "INACTIVE" and any(
        item["status"] == "ACTIVE" for item in issuers + producers
    ):
        raise TrustedReviewProtocolError("Inactive registry cannot contain active authority")
    _require_identifier(checked["approval_provenance"], "registry.approval_provenance")
    expected = canonical_fingerprint(checked, excluded_key="registry_fingerprint")
    if checked["registry_fingerprint"] != expected:
        raise TrustedReviewProtocolError("Registry fingerprint mismatch")
    return checked


def freeze_registry(
    *,
    activation_state: str,
    issuers: Sequence[Mapping[str, Any]],
    producers: Sequence[Mapping[str, Any]],
    approval_provenance: str,
) -> dict:
    registry = {
        "registry_version": REGISTRY_VERSION,
        "activation_state": activation_state,
        "issuers": sorted(
            [copy.deepcopy(dict(item)) for item in issuers],
            key=lambda item: item["issuer_id"],
        ),
        "producers": sorted(
            [copy.deepcopy(dict(item)) for item in producers],
            key=lambda item: (item["producer_id"], item["producer_version"]),
        ),
        "approval_provenance": approval_provenance,
        "registry_fingerprint": "PENDING",
    }
    registry["registry_fingerprint"] = canonical_fingerprint(
        registry, excluded_key="registry_fingerprint"
    )
    return validate_registry(registry)


LIVE_TRUST_REGISTRY = freeze_registry(
    activation_state="INACTIVE",
    issuers=[],
    producers=[],
    approval_provenance="R11_PROTOCOL_FREEZE_NO_LIVE_AUTHORITY",
)


def _registry_maps(registry: Mapping[str, Any]) -> tuple[dict, dict]:
    checked = validate_registry(registry)
    if checked["activation_state"] != "ACTIVE":
        raise TrustedReviewProtocolError("Trusted registry is inactive")
    issuers = {item["issuer_id"]: item for item in checked["issuers"]}
    producers = {
        (item["producer_id"], item["producer_version"]): item
        for item in checked["producers"]
    }
    return issuers, producers


def verify_activation_authorization(
    authorization: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    expected_code_bundle_fingerprint: str,
    now: datetime,
) -> dict:
    issuers, _ = _registry_maps(registry)
    if not isinstance(authorization, Mapping) or set(authorization) != ACTIVATION_FIELDS:
        raise TrustedReviewProtocolError("Activation authorization fields are invalid")
    checked = copy.deepcopy(dict(authorization))
    if checked["authorization_version"] != ACTIVATION_AUTHORIZATION_VERSION:
        raise TrustedReviewProtocolError("Activation authorization version is unsupported")
    _require_identifier(checked["authorization_id"], "authorization_id")
    _require_identifier(checked["nonce"], "authorization.nonce")
    issuer = issuers.get(checked["issuer_id"])
    if issuer is None or issuer["issuer_version"] != checked["issuer_version"]:
        raise TrustedReviewProtocolError("Activation issuer is unknown")
    if issuer["status"] != "ACTIVE":
        raise TrustedReviewProtocolError("Activation issuer is revoked")
    expected = {
        "target_repository_id": TARGET_REPOSITORY_ID,
        "target_manifest_pre_sha256": expected_manifest_sha256,
        "target_manifest_version": TARGET_MANIFEST_VERSION,
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "cutover_code_bundle_fingerprint": expected_code_bundle_fingerprint,
    }
    for field, value in expected.items():
        if checked[field] != value:
            raise TrustedReviewProtocolError(f"Activation binding mismatch: {field}")
    issued = _timestamp(checked["issued_at"], "authorization.issued_at")
    expires = _timestamp(checked["expires_at"], "authorization.expires_at")
    current = now.astimezone(timezone.utc)
    if expires <= issued or current < issued or current > expires:
        raise TrustedReviewProtocolError("Activation authorization is not currently valid")
    _verify_signature(
        checked,
        _public_key(issuer["ed25519_public_key_base64"], "issuer public key"),
    )
    return {
        "authorization_id": checked["authorization_id"],
        "issuer_id": checked["issuer_id"],
        "authorization_fingerprint": canonical_fingerprint(checked),
        "authorized": True,
    }


def cutover_code_bundle_fingerprint(project_root: Path) -> str:
    root = Path(project_root).resolve()
    relative_paths = (
        "backend/scripts/Evaluation/benchmarking/source_intake.py",
        "backend/scripts/Evaluation/benchmarking/dual_version_manifest_safety.py",
        "backend/scripts/Evaluation/benchmarking/trusted_review_protocol.py",
        "backend/scripts/Evaluation/benchmarking/live_cutover_readiness.py",
        "backend/scripts/Evaluation/benchmarking/cutover_authorization_lifecycle.py",
        "backend/scripts/Evaluation/benchmarking/dedup_performance_v1.py",
    )
    values = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise TrustedReviewProtocolError(f"Cutover code file is unsafe: {relative}")
        values[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_fingerprint(values)


def verify_live_activation_authorization(
    project_root: Path,
    authorization: Mapping[str, Any] | None,
) -> dict:
    """Production entry point bound to the approved V1 lineage root.

    The signed authorization remains valid across later atomic V2.1 mutations;
    each mutation separately validates exact pre-bytes, schema, provenance and
    artifacts.  R11's live registry is inactive, so this path cannot yet pass.
    """

    if authorization is None:
        raise TrustedReviewProtocolError("Live V2.1 activation authorization is missing")
    return verify_activation_authorization(
        authorization,
        LIVE_TRUST_REGISTRY,
        expected_manifest_sha256=LIVE_LINEAGE_ROOT_SHA256,
        expected_code_bundle_fingerprint=cutover_code_bundle_fingerprint(project_root),
        now=datetime.now(timezone.utc),
    )


def verify_review_attestation(
    attestation: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    authoritative_context: Mapping[str, Any],
    consumed_attestation_ids: set[str],
    consumed_nonces: set[str],
) -> dict:
    issuers, producers = _registry_maps(registry)
    if not isinstance(attestation, Mapping) or set(attestation) != ATTESTATION_FIELDS:
        raise TrustedReviewProtocolError("Attestation fields are invalid")
    checked = copy.deepcopy(dict(attestation))
    if checked["attestation_version"] != ATTESTATION_VERSION:
        raise TrustedReviewProtocolError("Attestation version is unsupported")
    _require_identifier(checked["attestation_id"], "attestation_id")
    _require_identifier(checked["nonce"], "attestation.nonce")
    if checked["attestation_id"] in consumed_attestation_ids:
        raise TrustedReviewProtocolError("Attestation replay detected")
    if checked["nonce"] in consumed_nonces:
        raise TrustedReviewProtocolError("Attestation nonce replay detected")
    producer = producers.get((checked["producer_id"], checked["producer_version"]))
    if producer is None:
        raise TrustedReviewProtocolError("Attestation producer is unknown")
    if producer["status"] != "ACTIVE":
        raise TrustedReviewProtocolError("Attestation producer is revoked")
    issuer = issuers.get(checked["issuer_id"])
    if issuer is None or issuer["status"] != "ACTIVE":
        raise TrustedReviewProtocolError("Attestation issuer is unknown or revoked")
    if producer["issuer_id"] != checked["issuer_id"]:
        raise TrustedReviewProtocolError("Attestation issuer does not approve producer")
    producer_bindings = {
        "producer_code_sha256": producer["approved_code_sha256"],
        "policy_id": producer["policy_id"],
        "policy_version": producer["policy_version"],
        "policy_fingerprint": producer["policy_fingerprint"],
        "review_contract_version": producer["review_contract_version"],
        "evidence_amendment_id": producer["evidence_amendment_id"],
        "trusted_execution_identity": producer["trusted_execution_identity"],
    }
    for field, value in producer_bindings.items():
        if checked[field] != value:
            raise TrustedReviewProtocolError(f"Attestation producer binding mismatch: {field}")
    expected_fields = {
        "acquisition_id",
        "source_id",
        "source_local_evidence_fingerprint",
        "current_inventory_fingerprint",
        "candidate_set_fingerprint",
        "recomputed_findings_fingerprint",
        "adjudication_evidence_bindings",
        "review_decision_fingerprint",
    }
    if not isinstance(authoritative_context, Mapping) or set(authoritative_context) != expected_fields:
        raise TrustedReviewProtocolError("Authoritative attestation context is invalid")
    for field in expected_fields:
        if checked[field] != authoritative_context[field]:
            raise TrustedReviewProtocolError(f"Attestation is stale or mismatched: {field}")
    for field in (
        "source_local_evidence_fingerprint",
        "current_inventory_fingerprint",
        "candidate_set_fingerprint",
        "recomputed_findings_fingerprint",
        "review_decision_fingerprint",
    ):
        if not _is_sha256(checked[field]):
            raise TrustedReviewProtocolError(f"Attestation {field} is invalid")
    bindings = checked["adjudication_evidence_bindings"]
    if (
        not isinstance(bindings, list)
        or bindings != sorted(set(bindings))
        or any(not _is_sha256(value) for value in bindings)
    ):
        raise TrustedReviewProtocolError("Attestation evidence bindings are invalid")
    _require_identifier(checked["issuance_provenance"], "issuance_provenance")
    _timestamp(checked["issued_at"], "attestation.issued_at")
    _verify_signature(
        checked,
        _public_key(producer["ed25519_public_key_base64"], "producer public key"),
    )
    return {
        "attestation_id": checked["attestation_id"],
        "nonce": checked["nonce"],
        "producer_id": checked["producer_id"],
        "attestation_fingerprint": canonical_fingerprint(checked),
        "verified": True,
        "authoritative_only_with_create_once_replay_ledger": True,
    }


def verify_live_review_attestation(*args, **kwargs) -> dict:
    """No live review attestation can pass while the live registry is inactive."""

    if LIVE_TRUST_REGISTRY["activation_state"] != "ACTIVE":
        raise TrustedReviewProtocolError("Live trusted producer registry is inactive")
    return verify_review_attestation(*args, registry=LIVE_TRUST_REGISTRY, **kwargs)


def record_attestation_once(
    ledger_directory: Path,
    verified_attestation: Mapping[str, Any],
) -> dict:
    """Record a verified attestation in a create-once, nonce-aware ledger.

    The caller must supply the result of ``verify_review_attestation``.  This
    primitive provides replay resistance only; it grants no review authority
    and is not connected to the inactive live registry in R11.
    """

    required = {
        "attestation_id",
        "nonce",
        "producer_id",
        "attestation_fingerprint",
        "verified",
        "authoritative_only_with_create_once_replay_ledger",
    }
    if not isinstance(verified_attestation, Mapping) or set(verified_attestation) != required:
        raise TrustedReviewProtocolError("Verified attestation result is invalid")
    if verified_attestation["verified"] is not True:
        raise TrustedReviewProtocolError("Unverified attestation cannot enter ledger")
    attestation_id = _require_identifier(
        verified_attestation["attestation_id"], "ledger.attestation_id"
    )
    nonce = _require_identifier(verified_attestation["nonce"], "ledger.nonce")
    fingerprint = verified_attestation["attestation_fingerprint"]
    if not _is_sha256(fingerprint):
        raise TrustedReviewProtocolError("Ledger attestation fingerprint is invalid")
    directory = Path(ledger_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise TrustedReviewProtocolError("Replay ledger must be a regular directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise TrustedReviewProtocolError("Cannot open trusted replay ledger") from exc
    lock_fd = None
    try:
        lock_fd = os.open(
            ".attestation-ledger.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        lock_metadata = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1:
            raise TrustedReviewProtocolError("Replay ledger lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for name in sorted(os.listdir(directory_fd)):
            if name == ".attestation-ledger.lock":
                continue
            if not name.endswith(".json"):
                raise TrustedReviewProtocolError("Replay ledger contains an unknown object")
            read_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                entry_metadata = os.fstat(read_fd)
                if (
                    not stat.S_ISREG(entry_metadata.st_mode)
                    or entry_metadata.st_nlink != 1
                ):
                    raise TrustedReviewProtocolError("Replay ledger entry is unsafe")
                with os.fdopen(os.dup(read_fd), "rb") as handle:
                    entry = json.loads(handle.read().decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise TrustedReviewProtocolError("Replay ledger entry is invalid") from exc
            finally:
                os.close(read_fd)
            if not isinstance(entry, dict) or set(entry) != {
                "ledger_version",
                "attestation_id",
                "nonce",
                "producer_id",
                "attestation_fingerprint",
                "ledger_fingerprint",
            }:
                raise TrustedReviewProtocolError("Replay ledger entry fields are invalid")
            if entry["ledger_version"] != "independent_attestation_replay_ledger_v1":
                raise TrustedReviewProtocolError("Replay ledger version is unsupported")
            if entry["ledger_fingerprint"] != canonical_fingerprint(
                entry, excluded_key="ledger_fingerprint"
            ):
                raise TrustedReviewProtocolError("Replay ledger entry fingerprint mismatch")
            if entry["attestation_id"] == attestation_id or entry["nonce"] == nonce:
                raise TrustedReviewProtocolError("Attestation replay detected by ledger")
        entry = {
            "ledger_version": "independent_attestation_replay_ledger_v1",
            "attestation_id": attestation_id,
            "nonce": nonce,
            "producer_id": verified_attestation["producer_id"],
            "attestation_fingerprint": fingerprint,
            "ledger_fingerprint": "PENDING",
        }
        entry["ledger_fingerprint"] = canonical_fingerprint(
            entry, excluded_key="ledger_fingerprint"
        )
        name = hashlib.sha256(attestation_id.encode("utf-8")).hexdigest() + ".json"
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            payload = canonical_bytes(entry)
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise TrustedReviewProtocolError("Replay ledger write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
        return entry
    except OSError as exc:
        raise TrustedReviewProtocolError("Replay ledger create-once write failed") from exc
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def evidence_amendment_specification() -> dict:
    specification = {
        "amendment_id": EVIDENCE_AMENDMENT_ID,
        "activation_state": "FROZEN_SPECIFICATION_NOT_POLICY_ACTIVATION",
        "exact_duplicate": {
            "required": ["authoritative_raw_or_structured_hash_equality"],
            "document_type_required": False,
            "weak_metadata_can_cancel_strong_positive": False,
        },
        "weak_distinct_control": {
            "required": [
                "source_and_counterpart_identity_verified",
                "complete_relation_evidence",
                "verified_contextual_conflict",
            ],
            "generic_override_allowed": False,
        },
        "related_decision_or_dispute": {
            "required": [
                "source_and_counterpart_identity_verified",
                "explicit_evidence_backed_adjudication",
                "exact_pair_evidence_binding",
            ],
            "generic_override_allowed": False,
        },
        "unresolved_behavior": "FAIL_CLOSED",
        "targeted_adjudications_preserved": ["027", "029"],
        "unresolved_sources_preserved": ["008", "014", "022", "024", "034"],
        "document_type_coverage_claim": "LIMITED_65_SOURCE_AUDIT_NOT_ALL_469",
        "frozen_policy_v2_modified": False,
    }
    specification["specification_fingerprint"] = canonical_fingerprint(specification)
    return specification
