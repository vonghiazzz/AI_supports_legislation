"""Deterministic family-quota planning for an independent benchmark.

Legal-family membership is multi-label documentary metadata.  Quota-family
assignment is a separate sampling-only decision: a candidate may fill at most
one compatible family slot, or remain unassigned.  Neither concept is legal
ground truth, and this module never reads query, annotation, retrieval, model,
or LLM output.

The solver is deliberately generic over an explicitly named candidate scope.
It does not decide whether future planning bounds apply to the main slice, the
whole selected benchmark, or another preregistered subset.
"""

from __future__ import annotations

from collections import Counter, deque
from itertools import combinations

from .preliminary_stratification import (
    PRELIMINARY_FAMILIES,
    TARGET_FAMILIES,
)


class FamilyQuotaError(ValueError):
    """Raised when family-quota planning inputs violate the protocol."""


def compatible_families(record: dict) -> tuple[str, ...]:
    """Return target-family memberships already present in one record.

    This is a projection of deterministic preliminary metadata, not a new
    classifier.  UNKNOWN and OTHER intentionally map to no target family.
    """

    if not isinstance(record, dict):
        raise FamilyQuotaError("Preliminary family record must be an object")
    primary = record.get("primary_family_preliminary")
    if primary not in PRELIMINARY_FAMILIES:
        raise FamilyQuotaError(f"Unknown preliminary family: {primary}")
    if primary in TARGET_FAMILIES:
        return (primary,)
    if primary in {"UNKNOWN", "OTHER"}:
        return ()
    if primary != "MULTI_FAMILY":
        raise FamilyQuotaError(
            f"Preliminary family cannot enter quota planning: {primary}"
        )

    secondary = record.get("secondary_families_preliminary")
    if not isinstance(secondary, list):
        raise FamilyQuotaError(
            "MULTI_FAMILY requires secondary_families_preliminary"
        )
    if len(secondary) != len(set(secondary)):
        raise FamilyQuotaError("MULTI_FAMILY contains duplicate memberships")
    unsupported = sorted(set(secondary) - set(TARGET_FAMILIES))
    if unsupported:
        raise FamilyQuotaError(
            f"MULTI_FAMILY contains unsupported memberships: {unsupported}"
        )
    memberships = tuple(family for family in TARGET_FAMILIES if family in secondary)
    if len(memberships) < 2:
        raise FamilyQuotaError(
            "MULTI_FAMILY requires at least two deterministic target memberships"
        )
    return memberships


def _validate_bounds(
    lower_bounds: dict[str, int],
    upper_bounds: dict[str, int] | None,
) -> tuple[dict[str, int], dict[str, int] | None]:
    expected = set(TARGET_FAMILIES)
    if not isinstance(lower_bounds, dict) or set(lower_bounds) != expected:
        raise FamilyQuotaError(
            "Lower bounds must specify exactly the five target families"
        )
    checked_lower = {}
    for family in TARGET_FAMILIES:
        value = lower_bounds[family]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FamilyQuotaError(f"Invalid lower bound for {family}")
        checked_lower[family] = value

    if upper_bounds is None:
        return checked_lower, None
    if not isinstance(upper_bounds, dict) or set(upper_bounds) != expected:
        raise FamilyQuotaError(
            "Upper bounds must specify exactly the five target families"
        )
    checked_upper = {}
    for family in TARGET_FAMILIES:
        value = upper_bounds[family]
        if not isinstance(value, int) or isinstance(value, bool):
            raise FamilyQuotaError(f"Invalid upper bound for {family}")
        if value < checked_lower[family]:
            raise FamilyQuotaError(
                f"Upper bound is below lower bound for {family}"
            )
        checked_upper[family] = value
    return checked_lower, checked_upper


def _candidate_memberships(
    records: list[dict],
    candidate_scope: str,
) -> tuple[list[dict], dict[str, tuple[str, ...]]]:
    if not isinstance(candidate_scope, str) or not candidate_scope.strip():
        raise FamilyQuotaError("An explicit candidate_scope is required")
    if not isinstance(records, list):
        raise FamilyQuotaError("Candidate records must be a list")

    source_ids = []
    case_ids = []
    memberships = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise FamilyQuotaError(f"Candidate record {index} must be an object")
        source_id = record.get("source_id")
        case_id = record.get("case_id")
        if not isinstance(source_id, str) or not source_id.startswith("independent:"):
            raise FamilyQuotaError(
                "Family-quota planning accepts independent candidates only"
            )
        if not isinstance(case_id, str) or not case_id:
            raise FamilyQuotaError(f"Candidate {source_id} requires case_id")
        source_ids.append(source_id)
        case_ids.append(case_id)
        memberships[source_id] = compatible_families(record)

    if len(source_ids) != len(set(source_ids)):
        raise FamilyQuotaError("Duplicate source_id in candidate scope")
    if len(case_ids) != len(set(case_ids)):
        raise FamilyQuotaError("Duplicate case_id in candidate scope")
    return records, memberships


class _DeterministicDinic:
    def __init__(self, size: int):
        self.graph: list[list[list[int]]] = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int) -> int:
        forward_index = len(self.graph[source])
        reverse_index = len(self.graph[target])
        self.graph[source].append([target, capacity, reverse_index, capacity])
        self.graph[target].append([source, 0, forward_index, 0])
        return forward_index

    def maximum_flow(self, source: int, sink: int) -> int:
        total = 0
        while True:
            levels = [-1] * len(self.graph)
            levels[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for target, capacity, _, _ in self.graph[node]:
                    if capacity and levels[target] < 0:
                        levels[target] = levels[node] + 1
                        queue.append(target)
            if levels[sink] < 0:
                return total

            cursors = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursors[node] < len(self.graph[node]):
                    edge = self.graph[node][cursors[node]]
                    target, capacity, reverse_index, _ = edge
                    if capacity and levels[target] == levels[node] + 1:
                        sent = send(target, min(amount, capacity))
                        if sent:
                            edge[1] -= sent
                            self.graph[target][reverse_index][1] += sent
                            return sent
                    cursors[node] += 1
                return 0

            while True:
                sent = send(source, 10**9)
                if not sent:
                    break
                total += sent


def _hall_bottlenecks(
    memberships: dict[str, tuple[str, ...]],
    lower_bounds: dict[str, int],
) -> list[dict]:
    diagnostics = []
    for size in range(1, len(TARGET_FAMILIES) + 1):
        for families in combinations(TARGET_FAMILIES, size):
            family_set = set(families)
            compatible_sources = sorted(
                source_id
                for source_id, values in memberships.items()
                if family_set.intersection(values)
            )
            required = sum(lower_bounds[family] for family in families)
            shortage = max(0, required - len(compatible_sources))
            if shortage:
                diagnostics.append(
                    {
                        "families": list(families),
                        "compatible_source_count": len(compatible_sources),
                        "required_slots": required,
                        "shortage": shortage,
                    }
                )
    diagnostics.sort(
        key=lambda item: (
            -item["shortage"],
            len(item["families"]),
            item["families"],
        )
    )
    return diagnostics


def solve_family_quota_assignment(
    records: list[dict],
    lower_bounds: dict[str, int],
    *,
    candidate_scope: str,
    upper_bounds: dict[str, int] | None = None,
) -> dict:
    """Fill as many preliminary family lower-bound slots as possible.

    Returned assignments exist only in memory.  A source is connected to all
    deterministic compatible families but has source capacity one, which is
    the no-double-count proof.  Because assignment is optional, a feasible
    lower-bound solution can always remain within any validated upper bounds.

    Deterministic traversal orders candidates by fewer compatible families and
    then source_id.  Candidate-to-family edges prefer lower raw slack
    (compatible count minus lower bound), then target-family declaration order.
    Dinic residual reassignment preserves maximum-flow completeness, so these
    preferences choose among solutions rather than replacing feasibility with
    a greedy decision.
    """

    checked_lower, checked_upper = _validate_bounds(lower_bounds, upper_bounds)
    _, memberships = _candidate_memberships(records, candidate_scope)
    compatible_counts = Counter(
        family for values in memberships.values() for family in values
    )
    family_order = {
        family: index for index, family in enumerate(TARGET_FAMILIES)
    }
    scarcity_order = sorted(
        TARGET_FAMILIES,
        key=lambda family: (
            compatible_counts[family] - checked_lower[family],
            compatible_counts[family],
            family_order[family],
        ),
    )
    scarcity_rank = {
        family: index for index, family in enumerate(scarcity_order)
    }
    candidates = sorted(
        (source_id for source_id, values in memberships.items() if values),
        key=lambda source_id: (len(memberships[source_id]), source_id),
    )

    source_node = 0
    first_candidate_node = 1
    first_family_node = first_candidate_node + len(candidates)
    sink_node = first_family_node + len(TARGET_FAMILIES)
    network = _DeterministicDinic(sink_node + 1)
    family_nodes = {
        family: first_family_node + index
        for index, family in enumerate(TARGET_FAMILIES)
    }
    assignment_edges = {}
    for offset, source_id in enumerate(candidates):
        candidate_node = first_candidate_node + offset
        network.add_edge(source_node, candidate_node, 1)
        ordered_families = sorted(
            memberships[source_id],
            key=lambda family: (scarcity_rank[family], family_order[family]),
        )
        for family in ordered_families:
            edge_index = network.add_edge(
                candidate_node,
                family_nodes[family],
                1,
            )
            assignment_edges[(source_id, family)] = (
                candidate_node,
                edge_index,
            )
    for family in TARGET_FAMILIES:
        network.add_edge(
            family_nodes[family],
            sink_node,
            checked_lower[family],
        )

    maximum_flow = network.maximum_flow(source_node, sink_node)
    assignment = {}
    for source_id in sorted(memberships):
        assigned = None
        for family in TARGET_FAMILIES:
            edge_location = assignment_edges.get((source_id, family))
            if edge_location is None:
                continue
            node, edge_index = edge_location
            edge = network.graph[node][edge_index]
            if edge[3] == 1 and edge[1] == 0:
                if assigned is not None:
                    raise FamilyQuotaError(
                        f"Source was assigned twice: {source_id}"
                    )
                assigned = family
        assignment[source_id] = assigned

    family_counts = {
        family: sum(value == family for value in assignment.values())
        for family in TARGET_FAMILIES
    }
    if checked_upper is not None and any(
        family_counts[family] > checked_upper[family]
        for family in TARGET_FAMILIES
    ):
        raise FamilyQuotaError("Solver assignment exceeded an upper bound")
    required_slots = sum(checked_lower.values())
    return {
        "candidate_scope": candidate_scope,
        "candidate_count": len(records),
        "feasible": maximum_flow == required_slots,
        "lower_bounds": checked_lower,
        "upper_bounds": checked_upper,
        "required_slots": required_slots,
        "maximum_filled_minimum_slots": maximum_flow,
        "shortage": required_slots - maximum_flow,
        "compatible_membership_counts": {
            family: compatible_counts[family] for family in TARGET_FAMILIES
        },
        "family_counts": family_counts,
        "quota_family_assignment": assignment,
        "unassigned_sources": sorted(
            source_id for source_id, family in assignment.items() if family is None
        ),
        "hall_bottlenecks": _hall_bottlenecks(memberships, checked_lower),
        "tie_break_rule": (
            "fewer compatible families first; then source_id ASC; family edges "
            "by raw slack ASC then declared target-family order; deterministic "
            "Dinic residual reassignment"
        ),
        "assignment_role": "SAMPLING_METADATA_NOT_GROUND_TRUTH",
        "blttds_guardrail_separate": True,
    }


def family_quota_status_report(
    records: list[dict],
    lower_bounds: dict[str, int],
    *,
    candidate_scope: str,
    upper_bounds: dict[str, int] | None = None,
) -> dict:
    """Distinguish primary, compatible, and unique-assignment quantities."""

    result = solve_family_quota_assignment(
        records,
        lower_bounds,
        candidate_scope=candidate_scope,
        upper_bounds=upper_bounds,
    )
    single_counts = Counter(
        record.get("primary_family_preliminary")
        for record in records
        if record.get("primary_family_preliminary") in TARGET_FAMILIES
    )
    return {
        "candidate_scope": candidate_scope,
        "families": {
            family: {
                "SINGLE_FAMILY_COUNT": single_counts[family],
                "COMPATIBLE_MEMBERSHIP_COUNT": result[
                    "compatible_membership_counts"
                ][family],
                "UNIQUE_ASSIGNMENT_CAPACITY": result["family_counts"][family],
            }
            for family in TARGET_FAMILIES
        },
        "solver": result,
    }
