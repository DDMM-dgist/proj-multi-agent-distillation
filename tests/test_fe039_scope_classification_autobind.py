"""FE-039 pre-run wiring closure regression: AUTOMATIC resolution + propagation of the frozen
config_type->canonical structure-class ``label_map`` to the Stage-4 data-coverage adequacy gate.

The FE-039 gate needs the frozen ``label_map`` to fold acquired per-config_type occupancy into
per-declared-class PRESENCE, but a fresh successor workflow must reach Stage 4 WITHOUT a human
manually passing ``scope_classification_evidence_path``. ``_bind_scope_classification_for_data_coverage``
resolves the SAME artifact ``acquisition_readiness`` (FE-033) resolves -- from the closure-bound
``DeploymentScopeContractV2`` regions' ``membership_evidence`` -- and injects it into the
``build_data_coverage_report`` proposal, failing closed on a conflicting frozen classification.

Deliberately material-agnostic (synthetic ``raw_*`` / ``domain_*`` labels + temp paths) to prove the
binder hard-codes no SiO2-x name, no frame count, and no filesystem path; one test additionally drives
the REAL frozen ``02_deployment_scope_v2.json`` to prove the exact fresh-successor artifact resolves.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import pydantic  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False

_PROJECT = Path(__file__).resolve().parents[1]
_FROZEN_SCOPE = (_PROJECT / "docs"
                 / "FRESH_CAMPAIGN_FROZEN_POLICIES_v2_fresh_frozen_framework_validation"
                 / "02_deployment_scope_v2.json")


class _Controller:
    """Minimal controller stub exposing exactly what the binder reads/writes: project_dir, the bound
    V2 scope contract via _v2_state()/v2_contract(), a mutable state['events'], and a no-op save()."""

    def __init__(self, project_dir: Path, scope_sha, scope_dict):
        self.state = {"project_dir": str(project_dir), "events": []}
        self._scope_sha = scope_sha
        self._scope_dict = scope_dict
        self.saved = 0

    def _v2_state(self):
        return {"scope_contract_sha256": self._scope_sha}

    def v2_contract(self, sha):
        return self._scope_dict if sha == self._scope_sha else None

    def save(self):
        self.saved += 1


def _write_scope_classification(path: Path, label_map, primary_domains, *,
                                contract_id="scope::synthetic_v1"):
    """Write a DeploymentScopeContractV2-shaped scope-classification evidence doc carrying label_map."""
    doc = {
        "contract_id": contract_id,
        "objective": "synthetic FE-039 auto-bind regression scope",
        "primary_domains": list(primary_domains),
        "label_map": [
            {"raw_label": rl, "canonical_domain": cd, "claim_role": cr, "rationale": "synthetic"}
            for rl, cd, cr in label_map],
        "representative_deployment_points": [],
    }
    path.write_text(json.dumps(doc, indent=2))
    return path


def _scope_contract_dict(evidence_rels, *, contract_id="scope::synthetic_v1"):
    """A regions-form bound scope contract whose regions reference the label_map evidence file(s),
    exactly how the closure-bound contract points at its scope-classification evidence."""
    return {
        "contract_id": contract_id,
        "objective": "synthetic",
        "regions": [
            {"region_id": f"r{i}", "category": "PRIMARY_DEPLOYMENT",
             "membership_rule": "synthetic", "membership_evidence": [rel]}
            for i, rel in enumerate(evidence_rels)],
    }


def _proposal(**params):
    return {"action_type": "build_data_coverage_report", "stage": "data_coverage",
            "parameters": dict(params)}


def _bind(controller, proposal):
    from runtimes.pydantic_ai.cli import _bind_scope_classification_for_data_coverage
    return _bind_scope_classification_for_data_coverage(controller, proposal)


@unittest.skipUnless(_HAS_PYDANTIC, "scope contract validation requires pydantic")
class CanonicalAutoResolutionTests(unittest.TestCase):
    def test_binder_injects_frozen_label_map_path_from_bound_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = _write_scope_classification(
                root / "scope_classification.json",
                label_map=[("raw_a", "domain_one", "primary_claim"),
                           ("raw_b", "domain_two", "primary_claim")],
                primary_domains=["domain_one", "domain_two"])
            scope = _scope_contract_dict(["scope_classification.json"])
            c = _Controller(root, "sha_scope", scope)

            out = _bind(c, _proposal())
            bound = out["parameters"].get("scope_classification_evidence_path")
            self.assertEqual(Path(bound).resolve(), evidence.resolve())
            # provenance is audited
            self.assertTrue(c.saved >= 1)
            ev = [e for e in c.state["events"]
                  if e["type"] == "scope_classification_auto_bound_for_data_coverage"]
            self.assertEqual(len(ev), 1)
            self.assertEqual(ev[0]["scope_contract_sha256"], "sha_scope")
            self.assertEqual(ev[0]["scope_contract_id"], "scope::synthetic_v1")
            self.assertTrue(ev[0]["classification_evidence_sha256"])

    def test_injected_path_propagates_to_stage4_occupancy_assessment(self):
        """Stage3->Stage4 integration: the executor's own resolver reads the auto-bound path and
        produces frozen_deployment_domain occupancy dimensions (NOT NOT_ASSESSABLE)."""
        from runtimes.pydantic_ai.executors import _resolve_frozen_structure_class_label_map
        from validation.coverage_gap_assessment import build_structure_class_dimensions
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_scope_classification(
                root / "scope_classification.json",
                label_map=[("raw_a", "domain_one", "primary_claim"),
                           ("raw_b", "domain_two", "primary_claim")],
                primary_domains=["domain_one", "domain_two"])
            scope = _scope_contract_dict(["scope_classification.json"])
            c = _Controller(root, "sha_scope", scope)
            out = _bind(c, _proposal(project_dir=str(root)))

            p = out["parameters"]
            label_map = _resolve_frozen_structure_class_label_map(p, deployment_domain={})
            self.assertIsInstance(label_map, list)
            dims = build_structure_class_dimensions(
                ["domain_one", "domain_two"], {"raw_a": 3}, label_map)
            provs = {d["criterion_provenance"] for d in dims}
            self.assertEqual(provs, {"frozen_deployment_domain"})
            by_class = {d["declared_target"]["structure_class"]: d["assessment_status"]
                        for d in dims}
            self.assertEqual(by_class["domain_one"], "PASS")   # raw_a occupied
            self.assertEqual(by_class["domain_two"], "FAIL")   # zero-occupancy => UNSUPPORTED

    def test_real_frozen_deployment_scope_artifact_resolves(self):
        """Requirement 6: the EXACT fresh-successor frozen artifact (02_deployment_scope_v2.json)
        resolves automatically from a bound contract that references it, no manual path."""
        if not _FROZEN_SCOPE.is_file():
            self.skipTest("frozen deployment scope artifact not present")
        rel = str(_FROZEN_SCOPE.relative_to(_PROJECT))
        scope = _scope_contract_dict([rel, rel, rel])  # repeated identical ref, as in the real run
        c = _Controller(_PROJECT, "sha_real", scope)
        out = _bind(c, _proposal())
        bound = out["parameters"].get("scope_classification_evidence_path")
        self.assertEqual(Path(bound).resolve(), _FROZEN_SCOPE.resolve())
        # the resolved artifact really carries a non-empty label_map
        doc = json.loads(Path(bound).read_text())
        self.assertTrue(isinstance(doc.get("label_map"), list) and doc["label_map"])


@unittest.skipUnless(_HAS_PYDANTIC, "scope contract validation requires pydantic")
class FailClosedTests(unittest.TestCase):
    def test_no_bound_mapping_injects_nothing(self):
        """No authoritative mapping resolvable -> bind nothing; Stage 4 then reports NOT_ASSESSABLE."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # bound contract references a non-existent evidence file
            scope = _scope_contract_dict(["missing_scope.json"])
            c = _Controller(root, "sha_scope", scope)
            out = _bind(c, _proposal())
            self.assertNotIn("scope_classification_evidence_path", out["parameters"])
            self.assertEqual(c.state["events"], [])

    def test_no_scope_contract_bound_injects_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            c = _Controller(root, None, None)
            out = _bind(c, _proposal())
            self.assertNotIn("scope_classification_evidence_path", out["parameters"])

    def test_missing_mapping_yields_not_assessable_downstream(self):
        """The contract behavior a missing map must preserve: NOT_ASSESSABLE, never fabricated."""
        from validation.coverage_gap_assessment import build_structure_class_dimensions
        dims = build_structure_class_dimensions(["domain_one"], {"raw_a": 5}, None)
        self.assertEqual([d["assessment_status"] for d in dims], ["NOT_ASSESSABLE"])

    def test_conflicting_frozen_mappings_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_scope_classification(
                root / "map_a.json",
                label_map=[("raw_a", "domain_one", "primary_claim")],
                primary_domains=["domain_one"], contract_id="scope::a")
            _write_scope_classification(
                root / "map_b.json",
                label_map=[("raw_a", "domain_two", "primary_claim")],
                primary_domains=["domain_two"], contract_id="scope::b")
            scope = _scope_contract_dict(["map_a.json", "map_b.json"])
            c = _Controller(root, "sha_scope", scope)
            with self.assertRaises(ValueError) as ctx:
                _bind(c, _proposal())
            self.assertIn("SCOPE_CLASSIFICATION_CONFLICT", str(ctx.exception))


@unittest.skipUnless(_HAS_PYDANTIC, "scope contract validation requires pydantic")
class ExplicitOverrideTests(unittest.TestCase):
    def test_explicit_evidence_path_not_overridden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_scope_classification(
                root / "scope_classification.json",
                label_map=[("raw_a", "domain_one", "primary_claim")],
                primary_domains=["domain_one"])
            scope = _scope_contract_dict(["scope_classification.json"])
            c = _Controller(root, "sha_scope", scope)
            out = _bind(c, _proposal(scope_classification_evidence_path="human/chosen.json"))
            self.assertEqual(out["parameters"]["scope_classification_evidence_path"],
                             "human/chosen.json")
            self.assertEqual(c.state["events"], [])

    def test_inline_structure_class_label_map_not_overridden(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_scope_classification(
                root / "scope_classification.json",
                label_map=[("raw_a", "domain_one", "primary_claim")],
                primary_domains=["domain_one"])
            scope = _scope_contract_dict(["scope_classification.json"])
            c = _Controller(root, "sha_scope", scope)
            dd = {"structure_class_label_map": [{"raw_label": "x", "canonical_domain": "y",
                                                 "claim_role": "primary_claim"}]}
            out = _bind(c, _proposal(deployment_domain=dd))
            self.assertNotIn("scope_classification_evidence_path", out["parameters"])
            self.assertEqual(c.state["events"], [])


@unittest.skipUnless(_HAS_PYDANTIC, "scope contract validation requires pydantic")
class NonApplicabilityTests(unittest.TestCase):
    def test_non_data_coverage_action_untouched(self):
        c = _Controller(Path("."), "sha", {"regions": []})
        proposal = {"action_type": "acquire_structures", "stage": "acquisition", "parameters": {}}
        out = _bind(c, proposal)
        self.assertIs(out, proposal)
        self.assertNotIn("scope_classification_evidence_path", out["parameters"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
