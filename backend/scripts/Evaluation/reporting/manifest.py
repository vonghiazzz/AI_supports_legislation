import hashlib
import importlib.metadata
import json
import subprocess


def fingerprint_records(records):
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_provenance(project_root, baseline_source_paths):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ["git", "status", "--porcelain", "--", *baseline_source_paths],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "git_commit": commit,
            "git_dirty": bool(status),
            "tracked_code_dirty": bool(tracked_status),
            "baseline_source_dirty": bool(source_status),
            "baseline_source_status": source_status.splitlines(),
        }
    except Exception:
        return {
            "git_commit": None,
            "git_dirty": None,
            "tracked_code_dirty": None,
            "baseline_source_dirty": None,
            "baseline_source_status": [],
        }


def resolved_model_revision(embedder):
    try:
        config = embedder._first_module().auto_model.config
        return getattr(config, "_commit_hash", None)
    except Exception:
        return None


def verify_model_revision(requested_revision, resolved_revision):
    if not requested_revision:
        raise ValueError("Baseline embedding_revision must be explicitly pinned")
    if not resolved_revision:
        raise RuntimeError("Could not resolve the loaded embedding model revision")
    if requested_revision != resolved_revision:
        raise RuntimeError(
            "Embedding revision mismatch: "
            f"requested {requested_revision}, resolved {resolved_revision}"
        )


def enforce_full_run_provenance(max_queries, benchmark_count, provenance):
    is_full_run = max_queries is None or int(max_queries) >= int(benchmark_count)
    if is_full_run and provenance.get("baseline_source_dirty") is not False:
        status = provenance.get("baseline_source_status") or ["source status unavailable"]
        raise RuntimeError(
            "Full baseline requires committed evaluator/config/test/experiment-lock files. "
            f"Current baseline source status: {status}"
        )
