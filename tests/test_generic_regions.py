"""FE-027 P2 -- executable region membership + discovery/relevance separation.

Proves, on the same synthetic non-SiO2 pool used in P1, that:
  * membership of discovered regimes is EXECUTABLE and total over the pool (frame -> regime_id),
    reconstructed from discovery evidence, not by parsing any free-text rule;
  * region DISCOVERY (deterministic) is cleanly separated from target RELEVANCE (an Agent
    proposal), which is gated by a purely deterministic validator that returns an issue list;
  * a valid proposal binds into a TargetRegimeModel carrying the discovered EXECUTABLE membership
    rules and the proposed roles, and an invalid proposal fails closed.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import ase  # noqa: F401
    import pydantic  # noqa: F401
    _HAS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_DEPS = False

from test_generic_representation import (  # reuse the P1 synthetic-pool fixtures
    _FakeController,
    _build_pool,
    _scope_contract,
)


def _representation(discriminative: bool = True):
    from framework_v2.acquisition.generic_representation import (
        build_adequate_representation, load_pool)
    tmp = Path(tempfile.mkdtemp())
    ctrl = _FakeController(_build_pool(tmp, discriminative=discriminative))
    scope = _scope_contract()
    pool = load_pool(ctrl)
    result = build_adequate_representation(
        pool, id_prefix="p2-run", scope_contract=scope,
        deployment_claim="generic deployment claim")
    return result.representation, pool, scope


def _valid_proposal(representation, scope):
    """Assign the first discovered regime CORE_TARGET, the rest BOUNDARY_GUARDRAIL."""
    from framework_v2.acquisition.contracts import RelevanceRole
    from framework_v2.acquisition.generic_regions import (
        RegimeRelevanceAssignment, RegimeRelevanceProposal)
    assignments = []
    for i, regime in enumerate(representation.regimes):
        role = RelevanceRole.CORE_TARGET if i == 0 else RelevanceRole.BOUNDARY_GUARDRAIL
        assignments.append(RegimeRelevanceAssignment(
            regime_id=regime.regime_id, relevance_role=role, rationale="test"))
    return RegimeRelevanceProposal(
        proposal_id="p2-proposal",
        representation_sha256=representation.content_sha256(),
        scope_contract_sha256=scope.content_sha256(),
        assignments=assignments)


@unittest.skipUnless(_HAS_DEPS, "pydantic/ase not installed")
class GenericRegionsP2(unittest.TestCase):
    def test_membership_is_executable_and_total(self):
        from framework_v2.acquisition.generic_regions import (
            build_frame_regime_classifier, build_regime_membership)

        representation, pool, _ = _representation()
        membership = build_regime_membership(representation)
        all_members = [fid for members in membership.values() for fid in members]
        # No pool-provenance ref leaks in as a member id.
        self.assertFalse(any(m.startswith("pool_manifest:") for m in all_members))
        # Every member is a real pool frame id; membership is disjoint across regimes.
        pool_ids = {f.item_id for f in pool.frames}
        self.assertTrue(set(all_members).issubset(pool_ids))
        self.assertEqual(len(all_members), len(set(all_members)))

        classify = build_frame_regime_classifier(representation)
        for regime_id, members in membership.items():
            for fid in members:
                self.assertEqual(classify(fid), regime_id)
        self.assertIsNone(classify("does-not-exist#999"))

    def test_valid_relevance_proposal_passes_validator(self):
        from framework_v2.acquisition.generic_regions import validate_relevance_proposal

        representation, _, scope = _representation()
        proposal = _valid_proposal(representation, scope)
        issues = validate_relevance_proposal(
            representation, proposal, scope_contract=scope)
        self.assertEqual(issues, [])

    def test_validator_detects_incomplete_and_unbound_proposals(self):
        from framework_v2.acquisition.contracts import RelevanceRole
        from framework_v2.acquisition.generic_regions import (
            RegimeRelevanceAssignment, RegimeRelevanceProposal,
            validate_relevance_proposal)

        representation, _, scope = _representation()
        self.assertGreater(len(representation.regimes), 1)

        # (a) missing an assignment for a discovered regime + no CORE_TARGET.
        partial = RegimeRelevanceProposal(
            proposal_id="p",
            representation_sha256=representation.content_sha256(),
            scope_contract_sha256=scope.content_sha256(),
            assignments=[RegimeRelevanceAssignment(
                regime_id=representation.regimes[0].regime_id,
                relevance_role=RelevanceRole.BOUNDARY_GUARDRAIL)])
        issues = validate_relevance_proposal(representation, partial, scope_contract=scope)
        self.assertTrue(any("no relevance assignment" in i for i in issues))
        self.assertTrue(any("CORE_TARGET" in i for i in issues))

        # (b) unknown regime + duplicate + unbound representation.
        rid0 = representation.regimes[0].regime_id
        bad = RegimeRelevanceProposal(
            proposal_id="p",
            representation_sha256="0" * 64,
            scope_contract_sha256=scope.content_sha256(),
            assignments=[
                RegimeRelevanceAssignment(regime_id=rid0, relevance_role=RelevanceRole.CORE_TARGET),
                RegimeRelevanceAssignment(regime_id=rid0, relevance_role=RelevanceRole.CORE_TARGET),
                RegimeRelevanceAssignment(regime_id="ghost", relevance_role=RelevanceRole.CORE_TARGET),
            ])
        issues = validate_relevance_proposal(representation, bad, scope_contract=scope)
        self.assertTrue(any("not bound to the provided representation" in i for i in issues))
        self.assertTrue(any("assigned more than once" in i for i in issues))
        self.assertTrue(any("unknown regime" in i for i in issues))

    def test_assemble_binds_executable_rules_and_roles(self):
        from framework_v2.acquisition.generic_regions import assemble_target_regime_model

        representation, _, scope = _representation()
        proposal = _valid_proposal(representation, scope)
        model = assemble_target_regime_model(
            representation, proposal, scope_contract=scope,
            objective_sha256="obj-sha", model_id="p2-trm")

        self.assertEqual(len(model.regimes), len(representation.regimes))
        self.assertTrue(model.core_regimes())
        # Each TargetRegime carries the DISCOVERED (executable interval-box) membership rule,
        # never a freshly-authored predicate.
        rule_of_discovered = {r.regime_id: r.membership_rule for r in representation.regimes}
        for tr in model.regimes:
            self.assertEqual(tr.membership_rule, rule_of_discovered[tr.regime_id])

    def test_assemble_fails_closed_on_invalid_proposal(self):
        from framework_v2.acquisition.contracts import RelevanceRole
        from framework_v2.acquisition.generic_regions import (
            RegimeRelevanceAssignment, RegimeRelevanceProposal,
            RelevanceProposalInvalid, assemble_target_regime_model)

        representation, _, scope = _representation()
        # Only one regime assigned; the rest unassigned -> fail closed.
        proposal = RegimeRelevanceProposal(
            proposal_id="p",
            representation_sha256=representation.content_sha256(),
            scope_contract_sha256=scope.content_sha256(),
            assignments=[RegimeRelevanceAssignment(
                regime_id=representation.regimes[0].regime_id,
                relevance_role=RelevanceRole.CORE_TARGET)])
        with self.assertRaises(RelevanceProposalInvalid) as cm:
            assemble_target_regime_model(
                representation, proposal, scope_contract=scope,
                objective_sha256="obj-sha", model_id="p2-trm")
        self.assertTrue(cm.exception.issues)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
