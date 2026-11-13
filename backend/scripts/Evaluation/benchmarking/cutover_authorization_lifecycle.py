"""R12 authority-separation amendment for V2.1 cutover and operation.

One-time cutover approval, normal operational access, recovery-only access,
review production, and eligibility promotion are distinct authorities.  This
module contains only inactive verification/persistence support; it provisions
no production issuer, key, authorization, receipt, review, or promotion.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import trusted_review_protocol as trusted


AMENDMENT_VERSION = "cutover_operational_authority_separation_v1"
OPERATIONAL_ACCESS_VERSION = "independent_v21_operational_access_v1"
OPERATIONAL_AUTHORIZATION_VERSION = "independent_v21_operational_authorization_v1"
RECOVERY_ACCESS_VERSION = "independent_v21_recovery_access_v1"
RECOVERY_AUTHORIZATION_VERSION = "independent_v21_recovery_authorization_v1"
ACTIVATION_RECEIPT_VERSION = "independent_v21_activation_receipt_v1"
ACTIVATION_LEDGER_VERSION = "independent_v21_activation_ledger_v1"

OPERATIONAL_AUTHORIZATION_FIELDS = {
    "authorization_version",
    "authorization_id",
    "issuer_id",
    "issuer_version",
    "target_repository_id",
    "activation_receipt_fingerprint",
    "target_manifest_version",
    "policy_id",
    "policy_version",
    "policy_fingerprint",
    "operational_code_bundle_fingerprint",
    "issued_at",
    "not_before",
    "expires_at",
    "nonce",
    "signature_base64",
}
RECOVERY_AUTHORIZATION_FIELDS = {
    "authorization_version",
    "authorization_id",
    "issuer_id",
    "issuer_version",
    "target_repository_id",
    "activation_receipt_fingerprint",
    "target_manifest_version",
    "purpose",
    "issued_at",
    "not_before",
    "expires_at",
    "nonce",
    "signature_base64",
}
ACTIVATION_RECEIPT_FIELDS = {
    "receipt_version",
    "cutover_authorization_id",
    "cutover_authorization_fingerprint",
    "target_repository_id",
    "pre_manifest_sha256",
    "post_manifest_sha256",
    "target_manifest_version",
    "policy_id",
    "policy_version",
    "policy_fingerprint",
    "cutover_code_bundle_fingerprint",
    "published_at",
    "receipt_fingerprint",
}


class AuthorizationLifecycleError(ValueError):
    """An R12 authorization lifecycle invariant failed closed."""


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise AuthorizationLifecycleError(f"{field} must be an offset timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AuthorizationLifecycleError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuthorizationLifecycleError(f"{field} must include an offset")
    return parsed.astimezone(timezone.utc)


def _identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 180
        or value != "".join(value.split())
    ):
        raise AuthorizationLifecycleError(f"{field} is invalid")
    return value


def operational_code_bundle_fingerprint(project_root: Path) -> str:
    root = Path(project_root).resolve()
    relative_paths = (
        "backend/scripts/Evaluation/benchmarking/source_intake.py",
        "backend/scripts/Evaluation/benchmarking/independent_benchmark.py",
        "backend/scripts/Evaluation/benchmarking/dedup_performance_v1.py",
        "backend/scripts/Evaluation/benchmarking/trusted_review_protocol.py",
        "backend/scripts/Evaluation/benchmarking/cutover_authorization_lifecycle.py",
        "backend/scripts/Evaluation/benchmarking/live_cutover_readiness.py",
    )
    hashes = {}
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise AuthorizationLifecycleError(f"Operational code file is unsafe: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return trusted.canonical_fingerprint(hashes)


def validate_activation_receipt(receipt: Mapping[str, Any]) -> dict:
    if not isinstance(receipt, Mapping) or set(receipt) != ACTIVATION_RECEIPT_FIELDS:
        raise AuthorizationLifecycleError("Activation receipt fields are invalid")
    checked = copy.deepcopy(dict(receipt))
    if checked["receipt_version"] != ACTIVATION_RECEIPT_VERSION:
        raise AuthorizationLifecycleError("Activation receipt version is unsupported")
    _identifier(checked["cutover_authorization_id"], "receipt.authorization_id")
    if checked["target_repository_id"] != trusted.TARGET_REPOSITORY_ID:
        raise AuthorizationLifecycleError("Activation receipt repository binding mismatch")
    for field in (
        "cutover_authorization_fingerprint",
        "pre_manifest_sha256",
        "post_manifest_sha256",
        "cutover_code_bundle_fingerprint",
    ):
        if not _is_sha256(checked[field]):
            raise AuthorizationLifecycleError(f"Activation receipt {field} is invalid")
    if (
        checked["target_manifest_version"] != trusted.TARGET_MANIFEST_VERSION
        or checked["policy_id"] != trusted.POLICY_ID
        or checked["policy_version"] != trusted.POLICY_VERSION
        or checked["policy_fingerprint"] != trusted.POLICY_FINGERPRINT
    ):
        raise AuthorizationLifecycleError("Activation receipt schema/policy binding mismatch")
    _timestamp(checked["published_at"], "receipt.published_at")
    if checked["receipt_fingerprint"] != trusted.canonical_fingerprint(
        checked, excluded_key="receipt_fingerprint"
    ):
        raise AuthorizationLifecycleError("Activation receipt fingerprint mismatch")
    return checked


def build_activation_receipt(
    cutover_authorization: Mapping[str, Any],
    *,
    pre_manifest_sha256: str,
    post_manifest_sha256: str,
    published_at: str,
) -> dict:
    if not _is_sha256(pre_manifest_sha256) or not _is_sha256(post_manifest_sha256):
        raise AuthorizationLifecycleError("Receipt manifest SHA256 is invalid")
    receipt = {
        "receipt_version": ACTIVATION_RECEIPT_VERSION,
        "cutover_authorization_id": cutover_authorization["authorization_id"],
        "cutover_authorization_fingerprint": trusted.canonical_fingerprint(
            cutover_authorization
        ),
        "target_repository_id": cutover_authorization["target_repository_id"],
        "pre_manifest_sha256": pre_manifest_sha256,
        "post_manifest_sha256": post_manifest_sha256,
        "target_manifest_version": cutover_authorization["target_manifest_version"],
        "policy_id": cutover_authorization["policy_id"],
        "policy_version": cutover_authorization["policy_version"],
        "policy_fingerprint": cutover_authorization["policy_fingerprint"],
        "cutover_code_bundle_fingerprint": cutover_authorization[
            "cutover_code_bundle_fingerprint"
        ],
        "published_at": published_at,
        "receipt_fingerprint": "PENDING",
    }
    receipt["receipt_fingerprint"] = trusted.canonical_fingerprint(
        receipt, excluded_key="receipt_fingerprint"
    )
    return validate_activation_receipt(receipt)


def _registry_issuers(registry: Mapping[str, Any]) -> dict:
    try:
        checked = trusted.validate_registry(registry)
    except Exception as exc:
        raise AuthorizationLifecycleError("Trusted issuer registry is unavailable or invalid") from exc
    if checked["activation_state"] != "ACTIVE":
        raise AuthorizationLifecycleError("Trusted issuer registry is inactive")
    return {item["issuer_id"]: item for item in checked["issuers"]}


def _verify_issuer_signature(
    payload: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> None:
    issuers = _registry_issuers(registry)
    issuer = issuers.get(payload.get("issuer_id"))
    if issuer is None or issuer.get("issuer_version") != payload.get("issuer_version"):
        raise AuthorizationLifecycleError("Authorization issuer is unknown")
    if issuer["status"] != "ACTIVE":
        raise AuthorizationLifecycleError("Authorization issuer is revoked")
    try:
        trusted._verify_signature(
            payload,
            trusted._public_key(
                issuer["ed25519_public_key_base64"], "issuer public key"
            ),
        )
    except Exception as exc:
        raise AuthorizationLifecycleError("Authorization signature verification failed") from exc


def _validate_time_window(payload: Mapping[str, Any], now: datetime) -> None:
    issued = _timestamp(payload["issued_at"], "authorization.issued_at")
    not_before = _timestamp(payload["not_before"], "authorization.not_before")
    expires = _timestamp(payload["expires_at"], "authorization.expires_at")
    current = now.astimezone(timezone.utc)
    if not (issued <= not_before < expires):
        raise AuthorizationLifecycleError("Authorization time window is invalid")
    if current < not_before or current > expires:
        raise AuthorizationLifecycleError("Authorization is expired or not yet valid")


def verify_operational_authorization(
    authorization: Mapping[str, Any],
    receipt: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    project_root: Path,
    now: datetime,
) -> dict:
    checked_receipt = validate_activation_receipt(receipt)
    if not isinstance(authorization, Mapping) or set(authorization) != OPERATIONAL_AUTHORIZATION_FIELDS:
        raise AuthorizationLifecycleError("Operational authorization fields are invalid")
    checked = copy.deepcopy(dict(authorization))
    if checked["authorization_version"] != OPERATIONAL_AUTHORIZATION_VERSION:
        raise AuthorizationLifecycleError("Operational authorization version is unsupported")
    _identifier(checked["authorization_id"], "operational.authorization_id")
    _identifier(checked["nonce"], "operational.nonce")
    expected = {
        "target_repository_id": trusted.TARGET_REPOSITORY_ID,
        "activation_receipt_fingerprint": checked_receipt["receipt_fingerprint"],
        "target_manifest_version": trusted.TARGET_MANIFEST_VERSION,
        "policy_id": trusted.POLICY_ID,
        "policy_version": trusted.POLICY_VERSION,
        "policy_fingerprint": trusted.POLICY_FINGERPRINT,
        "operational_code_bundle_fingerprint": operational_code_bundle_fingerprint(
            project_root
        ),
    }
    for field, value in expected.items():
        if checked[field] != value:
            raise AuthorizationLifecycleError(f"Operational binding mismatch: {field}")
    _validate_time_window(checked, now)
    _verify_issuer_signature(checked, registry)
    return {
        "authorization_id": checked["authorization_id"],
        "activation_receipt_fingerprint": checked_receipt["receipt_fingerprint"],
        "operational_authorization_fingerprint": trusted.canonical_fingerprint(
            checked
        ),
        "authorized": True,
    }


def verify_recovery_authorization(
    authorization: Mapping[str, Any],
    receipt: Mapping[str, Any],
    registry: Mapping[str, Any],
    *,
    now: datetime,
) -> dict:
    checked_receipt = validate_activation_receipt(receipt)
    if not isinstance(authorization, Mapping) or set(authorization) != RECOVERY_AUTHORIZATION_FIELDS:
        raise AuthorizationLifecycleError("Recovery authorization fields are invalid")
    checked = copy.deepcopy(dict(authorization))
    if checked["authorization_version"] != RECOVERY_AUTHORIZATION_VERSION:
        raise AuthorizationLifecycleError("Recovery authorization version is unsupported")
    _identifier(checked["authorization_id"], "recovery.authorization_id")
    _identifier(checked["nonce"], "recovery.nonce")
    if (
        checked["target_repository_id"] != trusted.TARGET_REPOSITORY_ID
        or checked["activation_receipt_fingerprint"]
        != checked_receipt["receipt_fingerprint"]
        or checked["target_manifest_version"] != trusted.TARGET_MANIFEST_VERSION
        or checked["purpose"] != "READ_ONLY_EVIDENCE_RECOVERY"
    ):
        raise AuthorizationLifecycleError("Recovery authorization binding mismatch")
    _validate_time_window(checked, now)
    _verify_issuer_signature(checked, registry)
    return {
        "authorization_id": checked["authorization_id"],
        "activation_receipt_fingerprint": checked_receipt["receipt_fingerprint"],
        "recovery_authorization_fingerprint": trusted.canonical_fingerprint(checked),
        "read_only": True,
        "authorized": True,
    }


def verify_live_operational_access(
    project_root: Path,
    access: Mapping[str, Any] | None,
) -> dict:
    if not isinstance(access, Mapping) or set(access) != {
        "access_version",
        "operational_authorization",
        "activation_receipt",
    }:
        raise AuthorizationLifecycleError("Live operational access bundle is missing or invalid")
    if access["access_version"] != OPERATIONAL_ACCESS_VERSION:
        raise AuthorizationLifecycleError("Live operational access version is unsupported")
    return verify_operational_authorization(
        access["operational_authorization"],
        access["activation_receipt"],
        trusted.LIVE_TRUST_REGISTRY,
        project_root=project_root,
        now=datetime.now(timezone.utc),
    )


LIVE_RECOVERY_REGISTRY = trusted.freeze_registry(
    activation_state="INACTIVE",
    issuers=[],
    producers=[],
    approval_provenance="R12_RECOVERY_PROTOCOL_FREEZE_NO_LIVE_AUTHORITY",
)


def verify_live_recovery_access(access: Mapping[str, Any] | None) -> dict:
    if not isinstance(access, Mapping) or set(access) != {
        "access_version",
        "recovery_authorization",
        "activation_receipt",
    }:
        raise AuthorizationLifecycleError("Live recovery access bundle is missing or invalid")
    if access["access_version"] != RECOVERY_ACCESS_VERSION:
        raise AuthorizationLifecycleError("Live recovery access version is unsupported")
    return verify_recovery_authorization(
        access["recovery_authorization"],
        access["activation_receipt"],
        LIVE_RECOVERY_REGISTRY,
        now=datetime.now(timezone.utc),
    )


def record_activation_receipt_once(
    ledger_directory: Path,
    receipt: Mapping[str, Any],
) -> dict:
    """Persist one activation receipt; authorization IDs cannot be replayed."""

    checked = validate_activation_receipt(receipt)
    directory = Path(ledger_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise AuthorizationLifecycleError("Activation ledger must be a regular directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise AuthorizationLifecycleError("Cannot open trusted activation ledger") from exc
    lock_fd = None
    try:
        lock_fd = os.open(
            ".activation-ledger.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        lock_metadata = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_nlink != 1:
            raise AuthorizationLifecycleError("Activation ledger lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for name in sorted(os.listdir(directory_fd)):
            if name == ".activation-ledger.lock":
                continue
            if not name.endswith(".json"):
                raise AuthorizationLifecycleError("Activation ledger contains unknown object")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise AuthorizationLifecycleError("Activation ledger entry is unsafe")
                with os.fdopen(os.dup(descriptor), "rb") as handle:
                    existing = json.loads(handle.read().decode("utf-8"))
            finally:
                os.close(descriptor)
            existing = validate_activation_receipt(existing)
            if (
                existing["cutover_authorization_id"]
                == checked["cutover_authorization_id"]
                or existing["target_repository_id"] == checked["target_repository_id"]
            ):
                raise AuthorizationLifecycleError("One-time cutover authorization replay detected")
        name = hashlib.sha256(
            checked["cutover_authorization_id"].encode("utf-8")
        ).hexdigest() + ".json"
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        try:
            payload = trusted.canonical_bytes(checked)
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise AuthorizationLifecycleError("Activation receipt write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(directory_fd)
        return checked
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationLifecycleError("Activation ledger operation failed") from exc
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def execute_authorized_clone_cutover(
    project_root: Path,
    cutover_authorization: Mapping[str, Any],
    issuer_registry: Mapping[str, Any],
    activation_ledger_directory: Path,
    *,
    now: datetime,
) -> dict:
    """Execute one signed cutover on an isolated clone and persist its receipt.

    This inactive executor is structurally forbidden from the live repository.
    If receipt persistence fails after schema publication, the exact V1 bytes
    are restored so no clone is left activated without tamper-evident external
    provenance.
    """

    root = Path(project_root).resolve()
    if root == Path(__file__).resolve().parents[4]:
        raise AuthorizationLifecycleError("Live cutover execution is forbidden in R12")
    from . import live_cutover_readiness as r11
    from . import source_intake as intake

    manifest_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AuthorizationLifecycleError("Cutover manifest path is unsafe")
    pre_bytes = manifest_path.read_bytes()
    pre_sha256 = hashlib.sha256(pre_bytes).hexdigest()
    marker_path = root / intake.INACTIVE_V21_STAGING_MARKER_RELATIVE
    if marker_path.is_symlink():
        raise AuthorizationLifecycleError("Cutover staging marker is unsafe")
    marker_preexisting = marker_path.is_file()
    marker_before = marker_path.read_bytes() if marker_preexisting else None
    try:
        verified = trusted.verify_activation_authorization(
            cutover_authorization,
            issuer_registry,
            expected_manifest_sha256=pre_sha256,
            expected_code_bundle_fingerprint=trusted.cutover_code_bundle_fingerprint(
                root
            ),
            now=now,
        )
    except Exception as exc:
        raise AuthorizationLifecycleError("Cutover authorization verification failed") from exc

    migration = None
    try:
        migration = r11.migrate_clone_manifest(root, pre_sha256)
        if migration["result"] != "MIGRATED":
            raise AuthorizationLifecycleError("Cutover authorization is one-time only")
        receipt = build_activation_receipt(
            cutover_authorization,
            pre_manifest_sha256=pre_sha256,
            post_manifest_sha256=migration["post_sha256"],
            published_at=now.astimezone(timezone.utc).isoformat(),
        )
        record_activation_receipt_once(activation_ledger_directory, receipt)
        return {
            "authorization": verified,
            "migration": migration,
            "activation_receipt": receipt,
            "receipt_persisted": True,
        }
    except Exception as exc:
        if migration is not None and migration.get("result") == "MIGRATED":
            _, chain = intake._open_trusted_directory(
                root, intake.INTAKE_ROOT_RELATIVE, create=False
            )
            intake._acquire_manifest_mutation_lock(chain)
            try:
                current = intake._read_regular_bytes_at(
                    chain.fd, intake.ACQUISITION_MANIFEST_RELATIVE.name
                )
                if hashlib.sha256(current).hexdigest() != migration["post_sha256"]:
                    raise AuthorizationLifecycleError(
                        "Cutover receipt failure found an unexpected manifest state"
                    )
                intake._atomic_write_bytes_at(
                    chain.fd,
                    intake.ACQUISITION_MANIFEST_RELATIVE.name,
                    pre_bytes,
                )
                if intake._read_regular_bytes_at(
                    chain.fd, intake.ACQUISITION_MANIFEST_RELATIVE.name
                ) != pre_bytes:
                    raise AuthorizationLifecycleError("Cutover rollback verification failed")
                marker_name = intake.INACTIVE_V21_STAGING_MARKER_RELATIVE.name
                marker_metadata = intake._entry_lstat(chain.fd, marker_name)
                if not marker_preexisting and marker_metadata is not None:
                    os.unlink(marker_name, dir_fd=chain.fd)
                    os.fsync(chain.fd)
                elif marker_preexisting:
                    if marker_metadata is None or intake._read_regular_bytes_at(
                        chain.fd, marker_name
                    ) != marker_before:
                        raise AuthorizationLifecycleError(
                            "Cutover marker rollback verification failed"
                        )
            finally:
                intake._release_manifest_mutation_lock(chain)
                chain.close()
        if isinstance(exc, AuthorizationLifecycleError):
            raise
        raise AuthorizationLifecycleError("Authorized clone cutover failed closed") from exc


def authority_separation_specification() -> dict:
    specification = {
        "amendment_version": AMENDMENT_VERSION,
        "activation_state": "INACTIVE_PROTOCOL_FREEZE",
        "authorities": {
            "CUTOVER_AUTHORIZATION": {
                "scope": "ONE_EXACT_V1_TO_V2_1_SCHEMA_TRANSITION",
                "reusable_for_operations": False,
                "create_once_receipt_required": True,
            },
            "OPERATIONAL_AUTHORIZATION": {
                "scope": "NORMAL_V2_1_READ_WRITE_WITH_EXISTING_SCIENTIFIC_GATES",
                "implies_review_authority": False,
                "implies_promotion_authority": False,
                "expired_or_revoked_behavior": "SUSPEND_NORMAL_OPERATIONS_FAIL_CLOSED",
            },
            "REVIEW_PRODUCER_AUTHORIZATION": {
                "scope": "AUTHORITATIVE_REVIEW_ATTESTATION_ISSUANCE_ONLY",
                "implies_operational_authority": False,
                "implies_promotion_authority": False,
            },
            "PROMOTION_AUTHORIZATION": {
                "scope": "SEPARATE_V2_REVIEW_BASED_ELIGIBILITY_TRANSITION",
                "active": False,
            },
            "RECOVERY_AUTHORIZATION": {
                "scope": "READ_ONLY_EVIDENCE_AND_TRANSACTION_INSPECTION",
                "normal_writes_allowed": False,
                "automatic_v1_rollback_allowed": False,
            },
        },
        "intake_local_json_is_trust_root": False,
        "private_keys_committed": False,
        "live_authority_issued": False,
    }
    specification["specification_fingerprint"] = trusted.canonical_fingerprint(
        specification
    )
    return specification
