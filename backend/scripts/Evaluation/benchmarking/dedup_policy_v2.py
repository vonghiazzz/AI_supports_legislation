"""Pure specification scaffold for independent-source Dedup Policy V2.

V2 is additive and inactive.  It does not replace the historical V1 dedup
implementation, mutate source review state, or promote eligibility.  The
module validates one frozen policy/fixture pair and exposes deterministic
pair, graph, canonical-retention, and bounded similarity helpers for tests and
future review planning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import independent_benchmark as v1


POLICY_ID = "independent_dedup_policy_v2"
POLICY_VERSION = 2
POLICY_STATUS = "FROZEN_SPECIFICATION_NOT_ACTIVATED"
DEFAULT_POLICY_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/independent_dedup_policy_v2.yaml"
)
EVIDENCE_FIXTURE_RELATIVE = Path(
    "backend/scripts/Tests/fixtures/independent_dedup_policy_v2_evidence.json"
)
PARENT_GIT_SHA = "13a283e1fd315a1a7dfb9a98018a75c693ba6ed0"
PARENT_MANIFEST_SHA256 = (
    "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a"
)
PARENT_R2_PLAN_FINGERPRINT = (
    "8fca6c2db743dc94c4ea9f479c2de27abc89739cfbd8b83efea0612e8aea5297"
)
PARENT_R3_REPORT_FINGERPRINT = (
    "20d403b32175a62c45a2a3ec03dc2d85b62db38d64b9376c1de6027a667ab0ac"
)
V1_DEDUP_REPORT_FINGERPRINT = (
    "6de3f1edb83eee4f61c8f724d872e719e9bd823ca383ef13b6e9b27cdf539c8c"
)
V1_ALGORITHM = "source_name_sha_case_id_text_shingle_v2"
NEAR_DUPLICATE_THRESHOLD = 0.72
EVIDENCE_FIXTURE_SHA256 = (
    "2083bb5da3af87301ab3ab54c51079890a83036d9feb04c19d5627207d96d4ef"
)
EVIDENCE_FIXTURE_FINGERPRINT = (
    "8a2c7e1e5e48f76a5ac5c481a13e798b229657225031ffc99ddfd5e8fdb5e05b"
)

ROOT_FIELDS = {
    "policy_id",
    "policy_version",
    "status",
    "policy_fingerprint",
    "bindings",
    "activation",
    "evidence_model",
    "development_overlap",
    "dispute_leakage",
    "canonical_retention",
    "relation_graphs",
    "missing_evidence",
    "counterfactual_reference",
    "prohibitions",
}
PAIR_FIELDS = {
    "left_source_id",
    "right_source_id",
    "scope",
    "raw_sha_equal",
    "structured_sha_equal",
    "case_id_equal",
    "normalized_judgment_equal",
    "filename_equal",
    "court_equal",
    "decision_date_equal",
    "decision_year_equal",
    "document_type_equal",
    "procedure_stage_equal",
    "text_similarity",
    "related_decision",
    "same_underlying_dispute",
    "findings_provenance",
}
CONTEXT_FIELDS = (
    "court_equal",
    "decision_date_equal",
    "decision_year_equal",
    "document_type_equal",
    "procedure_stage_equal",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DedupPolicyV2Error(ValueError):
    """V2 policy, evidence, or pure evaluation violated the specification."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise DedupPolicyV2Error(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any, *, excluded_key: str | None = None) -> str:
    payload = copy.deepcopy(value)
    if excluded_key is not None:
        if not isinstance(payload, dict):
            raise DedupPolicyV2Error("Fingerprint exclusion requires a mapping")
        payload.pop(excluded_key, None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_policy(path: Path) -> dict:
    try:
        value = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except DedupPolicyV2Error:
        raise
    except Exception as exc:
        raise DedupPolicyV2Error("V2 policy is not valid strict UTF-8 YAML") from exc
    if not isinstance(value, dict):
        raise DedupPolicyV2Error("V2 policy root must be a mapping")
    return value


def _exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise DedupPolicyV2Error(f"{label} differs from frozen V2 specification")


def validate_policy(policy: dict, project_root: Path | None = None) -> dict:
    if not isinstance(policy, dict) or set(policy) != ROOT_FIELDS:
        raise DedupPolicyV2Error("V2 policy top-level fields are invalid")
    _exact(policy["policy_id"], POLICY_ID, "policy_id")
    _exact(policy["policy_version"], POLICY_VERSION, "policy_version")
    _exact(policy["status"], POLICY_STATUS, "status")
    expected_fingerprint = canonical_fingerprint(
        policy, excluded_key="policy_fingerprint"
    )
    _exact(policy["policy_fingerprint"], expected_fingerprint, "policy fingerprint")
    _exact(
        policy["bindings"],
        {
            "parent_git_sha": PARENT_GIT_SHA,
            "parent_manifest_path": (
                "backend/dataLaw/independent_intake/acquisition_manifest.json"
            ),
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "parent_r2_plan_fingerprint": PARENT_R2_PLAN_FINGERPRINT,
            "parent_r3_report_fingerprint": PARENT_R3_REPORT_FINGERPRINT,
            "v1_algorithm": V1_ALGORITHM,
            "v1_near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "v1_dedup_report_fingerprint": V1_DEDUP_REPORT_FINGERPRINT,
            "evidence_fixture_path": EVIDENCE_FIXTURE_RELATIVE.as_posix(),
            "evidence_fixture_sha256": EVIDENCE_FIXTURE_SHA256,
            "evidence_fixture_fingerprint": EVIDENCE_FIXTURE_FINGERPRINT,
        },
        "bindings",
    )
    _exact(
        policy["activation"],
        {
            "active": False,
            "replaces_v1": False,
            "source_review_authorized": False,
            "eligibility_promotion_authorized": False,
            "migration_authorized": False,
        },
        "activation",
    )
    evidence = policy["evidence_model"]
    _exact(evidence["near_duplicate_threshold"], NEAR_DUPLICATE_THRESHOLD, "threshold")
    _exact(
        evidence["confirmed_exact_duplicate_any_of"],
        ["raw_pdf_sha_equal", "structured_artifact_sha_equal"],
        "exact duplicate signals",
    )
    _exact(evidence["strong_positive_not_cancelled_by_negative_metadata"], True, "positive evidence priority")
    _exact(evidence["case_id_alone_nonblocking"], True, "case ID semantics")
    _exact(
        evidence["weak_nonblocking_audit_signals"],
        ["filename_equal", "normalized_judgment_equal", "case_id_equal"],
        "weak signals",
    )
    distinct = evidence["distinct_control_requirements"]
    _exact(
        distinct,
        {
            "raw_sha_equal": False,
            "structured_sha_equal": False,
            "text_similarity_lt": NEAR_DUPLICATE_THRESHOLD,
            "minimum_verified_contextual_conflicts": 1,
            "contextual_fields": list(CONTEXT_FIELDS),
        },
        "distinct control requirements",
    )
    _exact(
        policy["development_overlap"],
        {
            "confirmed_same_document_blocks": True,
            "weak_filename_only_may_clear_with_complete_negative_evidence": True,
            "ambiguous_metadata_overlap_blocks": True,
            "known_or_unresolved_dispute_relation_blocks": True,
            "filename_mismatch_never_waives_other_evidence": True,
        },
        "development overlap",
    )
    _exact(
        policy["dispute_leakage"],
        {
            "separate_from_same_document": True,
            "related_decision_is_not_document_equivalence": True,
            "unresolved_related_decision_blocks": True,
            "confirmed_same_underlying_dispute_blocks": True,
            "reviewed_distinct_dispute_may_clear": True,
            "never_auto_promote_acquisition_ids": [
                "batch2dc-20261105-008",
                "batch2dc-20261105-014",
                "batch2dc-20261105-022",
                "batch2dc-20261105-024",
                "batch2dc-20261105-034",
            ],
        },
        "dispute leakage",
    )
    _exact(
        policy["canonical_retention"],
        {
            "exact_raw_sha_clusters_only": True,
            "ordering": ["case_id", "source_id", "raw_sha256"],
            "at_most_one_canonical": True,
            "noncanonical_rejected": True,
            "preserve_duplicate_provenance": True,
            "preserve_other_blockers": True,
            "positive_control_canonical_source_id": (
                "independent:batch2dc-20261105-025"
            ),
            "positive_control_noncanonical_source_id": (
                "independent:batch2dc-20261105-026"
            ),
        },
        "canonical retention",
    )
    _exact(
        policy["relation_graphs"],
        {
            "strong_same_document": {
                "transitive": True,
                "purpose": "DOCUMENT_EQUIVALENCE_ONLY",
            },
            "weak_metadata_association": {
                "transitive": False,
                "purpose": "AUDIT_SIGNAL_ONLY",
            },
            "underlying_dispute": {
                "transitive_for_review_grouping": True,
                "implies_same_document": False,
                "purpose": "FAIL_CLOSED_LEAKAGE_REVIEW",
            },
        },
        "relation graphs",
    )
    _exact(
        policy["missing_evidence"],
        {
            "weak_collision_with_missing_required_evidence": "BLOCK_AMBIGUOUS",
            "nonfinite_similarity": "REJECT_INPUT",
            "conflicting_strong_positive_and_negative_metadata": (
                "STRONG_POSITIVE_WINS_AND_CONFLICT_IS_RETAINED"
            ),
        },
        "missing evidence",
    )
    _exact(
        policy["counterfactual_reference"],
        {
            "S0_batch2dc_potential": 23,
            "S1_batch2dc_potential": 39,
            "S2_batch2dc_potential": 40,
            "reference_only_not_acceptance_target": True,
        },
        "counterfactual reference",
    )
    required_prohibitions = {
        "no_v1_modification",
        "no_historical_artifact_rewrite",
        "no_source_review",
        "no_eligibility_promotion",
        "no_benchmark_or_sampling_mutation",
        "no_network",
        "no_threshold_tuning_for_land_2024",
        "no_generalized_weak_signal_waiver",
    }
    _exact(set(policy["prohibitions"]), required_prohibitions, "prohibitions")
    if not all(policy["prohibitions"].values()):
        raise DedupPolicyV2Error("Every V2 prohibition must remain enabled")

    if project_root is not None:
        root = Path(project_root).resolve()
        fixture_path = root / EVIDENCE_FIXTURE_RELATIVE
        _exact(sha256_file(fixture_path), EVIDENCE_FIXTURE_SHA256, "fixture SHA256")
        validate_evidence_fixture(load_evidence_fixture(fixture_path))
    return copy.deepcopy(policy)


def load_and_validate_policy(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    return validate_policy(load_policy(root / DEFAULT_POLICY_RELATIVE), root)


def load_evidence_fixture(path: Path) -> dict:
    try:
        fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DedupPolicyV2Error("V2 evidence fixture is not strict JSON") from exc
    if not isinstance(fixture, dict):
        raise DedupPolicyV2Error("V2 evidence fixture root must be an object")
    return fixture


def validate_evidence_fixture(fixture: dict) -> dict:
    expected_fields = {
        "fixture_version",
        "fixture_role",
        "parent_git_sha",
        "parent_manifest_sha256",
        "parent_r2_plan_fingerprint",
        "parent_r3_report_fingerprint",
        "v1_dedup_report_fingerprint",
        "v1_algorithm",
        "v1_near_duplicate_threshold",
        "clean_batch2dc_acquisition_ids",
        "source_adjudications",
        "pair_controls",
        "r3_counterfactual_reference",
        "fixture_fingerprint",
    }
    if not isinstance(fixture, dict) or set(fixture) != expected_fields:
        raise DedupPolicyV2Error("V2 evidence fixture fields are invalid")
    _exact(fixture["fixture_version"], "independent_dedup_policy_v2_evidence_v1", "fixture version")
    _exact(fixture["fixture_role"], "FROZEN_LOCAL_CONTROLS_NOT_SOURCE_STATE", "fixture role")
    _exact(fixture["parent_git_sha"], PARENT_GIT_SHA, "fixture parent Git")
    _exact(fixture["parent_manifest_sha256"], PARENT_MANIFEST_SHA256, "fixture manifest")
    _exact(fixture["parent_r2_plan_fingerprint"], PARENT_R2_PLAN_FINGERPRINT, "fixture R2")
    _exact(fixture["parent_r3_report_fingerprint"], PARENT_R3_REPORT_FINGERPRINT, "fixture R3")
    _exact(fixture["v1_dedup_report_fingerprint"], V1_DEDUP_REPORT_FINGERPRINT, "fixture V1 report")
    _exact(fixture["v1_algorithm"], V1_ALGORITHM, "fixture V1 algorithm")
    _exact(fixture["v1_near_duplicate_threshold"], NEAR_DUPLICATE_THRESHOLD, "fixture threshold")
    _exact(
        fixture["fixture_fingerprint"],
        canonical_fingerprint(fixture, excluded_key="fixture_fingerprint"),
        "fixture fingerprint",
    )
    if len(fixture["clean_batch2dc_acquisition_ids"]) != 23:
        raise DedupPolicyV2Error("Fixture must bind exactly 23 clean sources")
    adjudications = fixture["source_adjudications"]
    if not isinstance(adjudications, list) or len(adjudications) != 23:
        raise DedupPolicyV2Error("Fixture must bind exactly 23 flagged adjudications")
    counts = {}
    for item in adjudications:
        label = item.get("analytical_classification")
        counts[label] = counts.get(label, 0) + 1
    _exact(
        counts,
        {
            "CONFIRMED_SAME_DOCUMENT": 2,
            "DISTINCT_DOCUMENT_WITH_SHARED_METADATA": 16,
            "GRAPH_RELATED_BUT_DOCUMENT_IDENTITY_UNPROVEN": 5,
        },
        "fixture classification counts",
    )
    controls = fixture["pair_controls"]
    if not isinstance(controls, list) or len(controls) != 23:
        raise DedupPolicyV2Error("Fixture must contain exactly 23 pair controls")
    if len({item.get("control_id") for item in controls}) != len(controls):
        raise DedupPolicyV2Error("Fixture control IDs must be unique")
    _exact(
        fixture["r3_counterfactual_reference"],
        [
            {
                "scenario": "S0_EXISTING_STRICT_POLICY",
                "batch2dc_potential_eligible_count": 23,
                "global_potential_eligible_count": 280,
                "land_2024_compatible_count": 31,
                "maximum_filled_minimum_slots": 181,
                "family_shortage": 4,
            },
            {
                "scenario": "S1_JUSTIFIED_WEAK_SIGNAL_RECLASSIFICATION",
                "batch2dc_potential_eligible_count": 39,
                "global_potential_eligible_count": 296,
                "land_2024_compatible_count": 31,
                "maximum_filled_minimum_slots": 181,
                "family_shortage": 4,
            },
            {
                "scenario": "S2_S1_PLUS_EXACT_DUPLICATE_CANONICAL_RETENTION",
                "batch2dc_potential_eligible_count": 40,
                "global_potential_eligible_count": 297,
                "land_2024_compatible_count": 32,
                "maximum_filled_minimum_slots": 182,
                "family_shortage": 3,
            },
        ],
        "R3 counterfactual reference",
    )
    return copy.deepcopy(fixture)


def pair_input_from_fixture(control: Mapping[str, Any]) -> dict:
    return {
        "left_source_id": control["left_source_id"],
        "right_source_id": control["right_source_id"],
        "scope": control["scope"],
        "raw_sha_equal": control["raw_sha_equal"],
        "structured_sha_equal": control["structured_sha_equal"],
        "case_id_equal": control["case_id_equal"],
        "normalized_judgment_equal": control["normalized_judgment_equal"],
        "filename_equal": control["filename_equal"],
        "court_equal": control["court_equal"],
        "decision_date_equal": control["decision_date_equal"],
        "decision_year_equal": control["decision_year_equal"],
        "document_type_equal": control["document_type_equal"],
        "procedure_stage_equal": control["procedure_stage_equal"],
        "text_similarity": control["text_similarity"],
        "related_decision": control["related_decision"],
        "same_underlying_dispute": control["same_underlying_dispute"],
        "findings_provenance": list(control["finding_reasons"]),
    }


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise DedupPolicyV2Error(f"{field} must be boolean or null")
    return value


def validate_pair_evidence(evidence: Mapping[str, Any]) -> dict:
    if not isinstance(evidence, Mapping) or set(evidence) != PAIR_FIELDS:
        raise DedupPolicyV2Error("V2 pair evidence fields are invalid")
    checked = dict(evidence)
    for field in ("left_source_id", "right_source_id"):
        if not isinstance(checked[field], str) or not checked[field]:
            raise DedupPolicyV2Error(f"{field} must be a non-empty string")
    if checked["left_source_id"] == checked["right_source_id"]:
        raise DedupPolicyV2Error("Pair evidence must compare different sources")
    if checked["scope"] not in {"DEVELOPMENT", "INTERNAL"}:
        raise DedupPolicyV2Error("Pair evidence scope is invalid")
    for field in (
        "raw_sha_equal",
        "structured_sha_equal",
        "case_id_equal",
        "normalized_judgment_equal",
        "filename_equal",
        *CONTEXT_FIELDS,
        "same_underlying_dispute",
    ):
        checked[field] = _optional_bool(checked[field], field)
    if not isinstance(checked["related_decision"], bool):
        raise DedupPolicyV2Error("related_decision must be boolean")
    similarity = checked["text_similarity"]
    if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
        raise DedupPolicyV2Error("text_similarity must be numeric")
    similarity = float(similarity)
    if not (0.0 <= similarity <= 1.0):
        raise DedupPolicyV2Error("text_similarity must be finite in [0, 1]")
    checked["text_similarity"] = similarity
    provenance = checked["findings_provenance"]
    if (
        not isinstance(provenance, list)
        or any(not isinstance(item, str) or not item for item in provenance)
        or provenance != sorted(set(provenance))
    ):
        raise DedupPolicyV2Error("findings_provenance must be sorted unique text")
    return checked


def evaluate_pair(policy: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict:
    validate_policy(dict(policy))
    item = validate_pair_evidence(evidence)
    exact = item["raw_sha_equal"] is True or item["structured_sha_equal"] is True
    positive_context = sum(item[field] is True for field in CONTEXT_FIELDS)
    contextual_conflicts = sum(item[field] is False for field in CONTEXT_FIELDS)
    context_complete = all(item[field] is not None for field in CONTEXT_FIELDS)
    weak_signal = any(
        item[field] is True
        for field in ("filename_equal", "normalized_judgment_equal", "case_id_equal")
    )
    high_text = item["text_similarity"] >= NEAR_DUPLICATE_THRESHOLD
    corroborated = (
        high_text
        and (
            item["case_id_equal"] is True
            or item["normalized_judgment_equal"] is True
            or positive_context >= 2
        )
    ) or (
        item["case_id_equal"] is True
        and item["normalized_judgment_equal"] is True
        and positive_context >= 2
    )
    complete_negative = (
        item["raw_sha_equal"] is False
        and item["structured_sha_equal"] is False
        and item["text_similarity"] < NEAR_DUPLICATE_THRESHOLD
        and contextual_conflicts >= 1
    )

    if exact:
        same_document_relation = "CONFIRMED_EXACT_DUPLICATE"
    elif corroborated:
        same_document_relation = "CORROBORATED_SAME_DOCUMENT"
    elif high_text:
        same_document_relation = "AMBIGUOUS_STRONG_TEXT_MATCH"
    elif weak_signal and complete_negative:
        same_document_relation = "WEAK_ASSOCIATION_DISTINCT_CONTROL"
    elif weak_signal:
        same_document_relation = "AMBIGUOUS_WEAK_ASSOCIATION"
    else:
        same_document_relation = "NO_SAME_DOCUMENT_RELATION"

    same_document_blocking = same_document_relation in {
        "CONFIRMED_EXACT_DUPLICATE",
        "CORROBORATED_SAME_DOCUMENT",
        "AMBIGUOUS_STRONG_TEXT_MATCH",
        "AMBIGUOUS_WEAK_ASSOCIATION",
    }

    if exact or corroborated:
        dispute_relation = (
            "DUPLICATE_INTERNAL_RELATION_NOT_INDEPENDENT_BLOCKER"
            if item["related_decision"] or item["same_underlying_dispute"] is True
            else "NO_DISPUTE_RELATION"
        )
        dispute_blocking = False
    elif item["same_underlying_dispute"] is True:
        dispute_relation = "CONFIRMED_SAME_UNDERLYING_DISPUTE"
        dispute_blocking = True
    elif item["related_decision"] and item["same_underlying_dispute"] is False:
        dispute_relation = "REVIEWED_DISTINCT_DISPUTE_RELATION"
        dispute_blocking = False
    elif item["related_decision"]:
        dispute_relation = "DISPUTE_RELATION_UNRESOLVED"
        dispute_blocking = True
    else:
        dispute_relation = "NO_DISPUTE_RELATION"
        dispute_blocking = False

    if item["scope"] == "DEVELOPMENT":
        if same_document_relation in {
            "CONFIRMED_EXACT_DUPLICATE",
            "CORROBORATED_SAME_DOCUMENT",
        }:
            development_relation = "CONFIRMED_DEVELOPMENT_SAME_DOCUMENT"
            development_blocking = True
        elif same_document_relation == "WEAK_ASSOCIATION_DISTINCT_CONTROL":
            development_relation = "WEAK_DEVELOPMENT_COLLISION_CLEARED"
            development_blocking = dispute_blocking
        else:
            development_relation = "AMBIGUOUS_DEVELOPMENT_OVERLAP"
            development_blocking = True
    else:
        development_relation = "NOT_DEVELOPMENT_SCOPE"
        development_blocking = False

    return {
        "left_source_id": item["left_source_id"],
        "right_source_id": item["right_source_id"],
        "scope": item["scope"],
        "same_document_relation": same_document_relation,
        "same_document_blocking": same_document_blocking,
        "development_relation": development_relation,
        "development_blocking": development_blocking,
        "dispute_relation": dispute_relation,
        "dispute_blocking": dispute_blocking,
        "weak_signal": weak_signal,
        "complete_negative_evidence": complete_negative,
        "context_evidence_complete": context_complete,
        "contextual_conflict_count": contextual_conflicts,
        "overall_blocking": (
            same_document_blocking or development_blocking or dispute_blocking
        ),
        "findings_provenance": list(item["findings_provenance"]),
    }


def _components(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    components = []
    unseen = set(adjacency)
    while unseen:
        first = min(unseen)
        stack = [first]
        members = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members, reverse=True))
        unseen -= members
        components.append(sorted(members))
    return sorted(components, key=lambda values: values[0])


def relation_views(results: Sequence[Mapping[str, Any]]) -> dict:
    strong_edges = []
    dispute_edges = []
    weak = []
    for result in results:
        left = result["left_source_id"]
        right = result["right_source_id"]
        relation = result["same_document_relation"]
        if relation in {"CONFIRMED_EXACT_DUPLICATE", "CORROBORATED_SAME_DOCUMENT"}:
            strong_edges.append((left, right))
        elif relation in {
            "WEAK_ASSOCIATION_DISTINCT_CONTROL",
            "AMBIGUOUS_WEAK_ASSOCIATION",
        }:
            weak.append(
                {
                    "left_source_id": left,
                    "right_source_id": right,
                    "relation": relation,
                }
            )
        if result["dispute_relation"] in {
            "CONFIRMED_SAME_UNDERLYING_DISPUTE",
            "DISPUTE_RELATION_UNRESOLVED",
        }:
            dispute_edges.append((left, right))
    return {
        "strong_same_document_components": _components(strong_edges),
        "weak_associations_nontransitive": sorted(
            weak, key=lambda item: (item["left_source_id"], item["right_source_id"])
        ),
        "underlying_dispute_review_components": _components(dispute_edges),
    }


def canonicalize_exact_duplicate_cluster(records: Sequence[Mapping[str, Any]]) -> dict:
    if not isinstance(records, Sequence) or len(records) < 2:
        raise DedupPolicyV2Error("Exact duplicate cluster requires at least two records")
    expected_fields = {
        "source_id",
        "case_id",
        "raw_sha256",
        "other_blockers",
        "findings_provenance",
    }
    checked = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise DedupPolicyV2Error("Exact duplicate cluster record fields are invalid")
        if not isinstance(record["source_id"], str) or not record["source_id"]:
            raise DedupPolicyV2Error("Exact duplicate source_id is invalid")
        if not isinstance(record["case_id"], str) or not record["case_id"]:
            raise DedupPolicyV2Error("Exact duplicate case_id is invalid")
        if not isinstance(record["raw_sha256"], str) or not SHA256_RE.fullmatch(record["raw_sha256"]):
            raise DedupPolicyV2Error("Exact duplicate SHA256 is invalid")
        if not isinstance(record["other_blockers"], list) or record["other_blockers"] != sorted(set(record["other_blockers"])):
            raise DedupPolicyV2Error("other_blockers must be sorted and unique")
        if not isinstance(record["findings_provenance"], list) or record["findings_provenance"] != sorted(set(record["findings_provenance"])):
            raise DedupPolicyV2Error("findings_provenance must be sorted and unique")
        checked.append(copy.deepcopy(dict(record)))
    if len({item["source_id"] for item in checked}) != len(checked):
        raise DedupPolicyV2Error("Exact duplicate cluster source IDs must be unique")
    if len({item["raw_sha256"] for item in checked}) != 1:
        raise DedupPolicyV2Error("Canonical retention is restricted to exact raw-SHA clusters")
    ordered = sorted(
        checked,
        key=lambda item: (item["case_id"], item["source_id"], item["raw_sha256"]),
    )
    canonical = ordered[0]
    return {
        "canonical_source_id": canonical["source_id"],
        "canonical_blocked_by_other_findings": bool(canonical["other_blockers"]),
        "canonical_other_blockers": list(canonical["other_blockers"]),
        "noncanonical_rejected_source_ids": sorted(
            item["source_id"] for item in ordered[1:]
        ),
        "findings_provenance_by_source": {
            item["source_id"]: list(item["findings_provenance"])
            for item in sorted(ordered, key=lambda value: value["source_id"])
        },
        "ordering": ["case_id", "source_id", "raw_sha256"],
    }


def cached_text_similarities(
    text_by_source_id: Mapping[str, str],
    pairs: Sequence[tuple[str, str]],
    *,
    max_sources: int = 512,
    max_pairs: int = 131072,
) -> dict:
    """Compute exact V1 shingle similarity with one shingle build per source."""

    if not isinstance(text_by_source_id, Mapping) or not text_by_source_id:
        raise DedupPolicyV2Error("text_by_source_id must be a non-empty mapping")
    if len(text_by_source_id) > max_sources:
        raise DedupPolicyV2Error("Bounded similarity source limit exceeded")
    if not isinstance(pairs, Sequence) or len(pairs) > max_pairs:
        raise DedupPolicyV2Error("Bounded similarity pair limit exceeded")
    if any(not isinstance(key, str) or not isinstance(text, str) for key, text in text_by_source_id.items()):
        raise DedupPolicyV2Error("Similarity cache keys/text must be strings")
    shingles = {
        source_id: v1._dedup_shingles(text)
        for source_id, text in sorted(text_by_source_id.items())
    }
    scores = {}
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise DedupPolicyV2Error("Similarity pairs must be two-tuples")
        left, right = pair
        if left not in shingles or right not in shingles or left == right:
            raise DedupPolicyV2Error("Similarity pair references invalid source IDs")
        key = tuple(sorted((left, right)))
        if key in scores:
            raise DedupPolicyV2Error("Duplicate similarity pair")
        left_values = shingles[key[0]]
        right_values = shingles[key[1]]
        union = left_values | right_values
        scores[key] = (
            1.0 if not union else len(left_values & right_values) / len(union)
        )
    return {
        "scores": scores,
        "source_count": len(shingles),
        "pair_count": len(scores),
        "shingle_build_count": len(shingles),
        "naive_shingle_build_count": 2 * len(scores),
        "semantics": "EXACT_V1_DEDUP_TEXT_SIMILARITY",
    }


def counterfactual_membership(fixture: Mapping[str, Any]) -> dict:
    checked = validate_evidence_fixture(dict(fixture))
    clean = set(checked["clean_batch2dc_acquisition_ids"])
    distinct = {
        item["acquisition_id"]
        for item in checked["source_adjudications"]
        if item["analytical_classification"]
        == "DISTINCT_DOCUMENT_WITH_SHARED_METADATA"
        and item["evidence_sufficient_for_policy_decision"] is True
    }
    graph = {
        item["acquisition_id"]
        for item in checked["source_adjudications"]
        if item["analytical_classification"]
        == "GRAPH_RELATED_BUT_DOCUMENT_IDENTITY_UNPROVEN"
    }
    duplicates = {
        item["acquisition_id"]
        for item in checked["source_adjudications"]
        if item["analytical_classification"] == "CONFIRMED_SAME_DOCUMENT"
    }
    if duplicates != {
        "batch2dc-20261105-025",
        "batch2dc-20261105-026",
    }:
        raise DedupPolicyV2Error("Fixture exact duplicate control changed")
    return {
        "S0": {
            "included": sorted(clean),
            "unresolved": sorted(distinct | graph | duplicates),
            "rejected": [],
        },
        "S1": {
            "included": sorted(clean | distinct),
            "unresolved": sorted(graph | duplicates),
            "rejected": [],
        },
        "S2": {
            "included": sorted(clean | distinct | {"batch2dc-20261105-025"}),
            "unresolved": sorted(graph),
            "rejected": ["batch2dc-20261105-026"],
        },
    }
