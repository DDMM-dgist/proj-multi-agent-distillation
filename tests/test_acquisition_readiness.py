"""FE-033 regression: deterministic, Teacher-free ``acquisition`` (EXISTING_POOL_SELECTION)
criterion-evidence surfacing.

Root cause these tests lock down: the acquisition gate turns on (A) every selected parent joining a
real sanitized-pool seed, (B) seed selection being documented against the frozen deployment domain,
and (C) explicit selection-control attestations -- but the raw acquisition manifest's semantic
join/mapping/attestation fields were dropped by the generic JSON summariser, so ffv4k's three Judges
REVISEd on parent->pool join = null, domain = unknown, and attestation values = null even though the
values existed on disk. ``compute_acquisition_evidence`` surfaces those deterministic values into the
gate packet WITHOUT re-selecting structures or changing any AcquisitionPlan field.

Deliberately uses synthetic, material-agnostic category/domain names (``cat_alpha`` ...,
``domain_one`` ...) to prove the surfacer is generic and hardcodes no material-specific label.

Network- and Teacher-free; skips pydantic-dependent scope validation if the extra is absent.
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


class _Controller:
    """Minimal controller stub exposing exactly what the surfacer reads: project_dir, run_dir, and
    the bound V2 scope contract (regions form) accessed via _v2_state()/v2_contract()."""

    def __init__(self, root: Path, scope_sha, scope_dict):
        self.run_dir = root
        self.state = {"project_dir": str(root)}
        self._scope_sha = scope_sha
        self._scope_dict = scope_dict

    def _v2_state(self):
        return {"scope_contract_sha256": self._scope_sha}

    def v2_contract(self, sha):
        return self._scope_dict if sha == self._scope_sha else None


def _write_pool(root: Path, categories):
    """categories: list of (name, n_frames). Writes a sanitized pool manifest and returns its path
    and its self-declared sha (the framework's manifest identity, not a file hash)."""
    sha = "poolsha_" + "_".join(f"{n}{c}" for c, n in categories)
    manifest = {
        "total_frames": sum(n for _, n in categories),
        "n_categories": len(categories),
        "sanitized_pool_manifest_sha256": sha,
        "categories": [
            {"category": c, "sanitized_file": f"{c}.sanitized.xyz", "n_frames": n}
            for c, n in categories],
    }
    path = root / "sanitized_pool_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path, sha


def _write_scope_classification(root: Path, label_map, primary_domains):
    """label_map: list of (raw_label, canonical_domain, claim_role). Writes the label_map scope
    classification evidence (DeploymentScopeContractV2 shape)."""
    doc = {
        "contract_id": "scope::synthetic_v1",
        "objective": "synthetic generic acquisition-surfacer regression scope",
        "primary_domains": primary_domains,
        "label_map": [
            {"raw_label": rl, "canonical_domain": cd, "claim_role": cr,
             "rationale": "synthetic"} for rl, cd, cr in label_map],
        "representative_deployment_points": [],
    }
    path = root / "scope_classification.json"
    path.write_text(json.dumps(doc, indent=2))
    return path


def _scope_contract_dict(evidence_rel: str):
    """A regions-form DeploymentScopeContract whose regions reference the label_map evidence file
    (exactly how the closure-bound contract points at its scope-classification evidence)."""
    return {
        "contract_id": "scope::synthetic_v1",
        "objective": "synthetic",
        "regions": [
            {"region_id": "r0", "category": "PRIMARY_DEPLOYMENT",
             "membership_rule": "synthetic", "membership_evidence": [evidence_rel]}],
    }


def _write_acquisition_manifest(root: Path, *, pool_path, pool_sha, selected,
                                dft_used=False, teacher_inf=False, exclusion_status="PASS"):
    """selected: list of (structure_id, source_index, [categories])."""
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1,
        "operation": "select_existing_pool",
        "stage": "acquisition",
        "pool_path": str(pool_path),
        "pool_manifest_sha256": pool_sha,
        "selected_parent_structure_ids": [s[0] for s in selected],
        "selected_source_global_indices": [s[1] for s in selected],
        "selected_source_records": [
            {"source_index": idx, "structure_id": sid, "categories": cats}
            for sid, idx, cats in selected],
        "dft_labels_used_as_selection_scores": dft_used,
        "performs_teacher_inference": teacher_inf,
        "protected_reference_exclusion_report": {
            "status": exclusion_status, "n_checked": len(selected), "n_overlaps": 0,
            "dft_labels_used_as_selection_scores": dft_used},
    }
    path = artifacts / "acquisition.manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _proposal(**extra):
    p = {"action_type": "acquire_structures", "stage": "acquisition",
         "expected_outputs": ["artifacts/acquisition.manifest.json"]}
    p.update(extra)
    return p


def _compute(controller, proposal):
    from runtimes.pydantic_ai.acquisition_readiness import compute_acquisition_evidence
    return compute_acquisition_evidence(controller, proposal)


@unittest.skipUnless(_HAS_PYDANTIC, "pydantic not installed")
class AcquisitionReadinessTests(unittest.TestCase):
    def _standard(self, root: Path, *, selected=None, dft_used=False, teacher_inf=False,
                  label_map=None, primary_domains=None, categories=None):
        categories = categories or [("cat_alpha", 100), ("cat_beta", 50)]
        pool_path, pool_sha = _write_pool(root, categories)
        label_map = label_map or [
            ("cat_alpha", "domain_one", "primary_claim"),
            ("cat_beta", "domain_two", "primary_claim")]
        primary_domains = primary_domains or ["domain_one", "domain_two", "domain_three"]
        evidence = _write_scope_classification(root, label_map, primary_domains)
        scope_dict = _scope_contract_dict(evidence.name)
        selected = selected if selected is not None else [
            ("cat_alpha#0", 0, ["cat_alpha"]), ("cat_alpha#5", 5, ["cat_alpha"])]
        _write_acquisition_manifest(root, pool_path=pool_path, pool_sha=pool_sha,
                                    selected=selected, dft_used=dft_used, teacher_inf=teacher_inf)
        controller = _Controller(root, "synthsha", scope_dict)
        return controller

    # (1) acquisition Judges receive explicit parent->pool join counts
    def test_parent_pool_join_counts_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root)
            out = _compute(c, _proposal())
            self.assertIsNotNone(out)
            join = out["criteria"]["parent_pool_join"]
            self.assertEqual(join["status"], "COMPLETE")
            self.assertEqual(join["joined"], 2)
            self.assertEqual(join["unmatched"], 0)
            self.assertEqual(join["foreign"], 0)
            self.assertEqual(join["duplicate"], 0)
            self.assertTrue(join["pool_sha_matches"])
            self.assertTrue(out["ready"], out["blocking_gaps"])

    # (2) deployment-domain mapping resolved from the canonical bound scope evidence
    def test_deployment_domain_resolved_from_bound_scope_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root)
            out = _compute(c, _proposal())
            dom = out["criteria"]["deployment_domain_mapping"]
            self.assertEqual(dom["status"], "RESOLVED")
            self.assertEqual(dom["scope_contract_sha256"], "synthsha")
            self.assertTrue(dom["classification_evidence_path"].endswith("scope_classification.json"))
            self.assertEqual(dom["aggregate_domain_counts"], {"domain_one": 2})
            self.assertEqual({p["canonical_domain"] for p in dom["per_parent"]}, {"domain_one"})
            # narrowness stays VISIBLE (uncovered primary domains surfaced), not treated as a failure
            self.assertIn("domain_two", dom["uncovered_primary_domains"])
            self.assertIn("domain_three", dom["uncovered_primary_domains"])
            self.assertEqual(dom["covered_primary_domains"], ["domain_one"])

    # (3) unresolved mapping fails closed rather than silently "unknown"
    def test_unresolved_mapping_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # cat_beta is NOT in the label_map -> must be AMBIGUOUS, not guessed / not "unknown"
            c = self._standard(
                root,
                selected=[("cat_alpha#0", 0, ["cat_alpha"]), ("cat_beta#0", 100, ["cat_beta"])],
                label_map=[("cat_alpha", "domain_one", "primary_claim")],
                primary_domains=["domain_one"])
            out = _compute(c, _proposal())
            dom = out["criteria"]["deployment_domain_mapping"]
            self.assertEqual(dom["status"], "AMBIGUOUS")
            self.assertEqual(dom["unresolved_parents"], 1)
            ambiguous = [p for p in dom["per_parent"] if p["resolution"] == "AMBIGUOUS"]
            self.assertEqual(len(ambiguous), 1)
            self.assertIsNone(ambiguous[0]["canonical_domain"])  # never fabricated to "unknown"
            self.assertFalse(out["ready"])
            self.assertTrue(any("deterministically" in g for g in out["blocking_gaps"]))

    # (4) dft_labels_used_as_selection_scores=False reaches the packet as False
    def test_dft_labels_attestation_false_reaches_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root, dft_used=False)
            out = _compute(c, _proposal())
            att = out["criteria"]["selection_control_attestations"]
            self.assertEqual(att["status"], "ATTESTED")
            self.assertIs(att["dft_labels_used_as_selection_scores"], False)

    # (5) performs_teacher_inference=False reaches the packet as False
    def test_teacher_inference_attestation_false_reaches_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root, teacher_inf=False)
            out = _compute(c, _proposal())
            att = out["criteria"]["selection_control_attestations"]
            self.assertIs(att["performs_teacher_inference"], False)

    # (6) the surfacer changes NO scientific AcquisitionPlan / manifest field (read-only)
    def test_surfacer_mutates_no_scientific_artifact(self):
        from workflow.integrity import sha256_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root)
            files = sorted(p for p in root.rglob("*") if p.is_file())
            before = {str(p): sha256_file(p) for p in files}
            _compute(c, _proposal())
            after = {str(p): sha256_file(p) for p in files}
            self.assertEqual(before, after)

    # (7) protected-reference isolation intact: exclusion status surfaced; a foreign seed breaks
    # exclusive sanitized-pool membership (fail closed) rather than being silently admitted.
    def test_protected_reference_isolation_and_foreign_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(
                root,
                selected=[("cat_alpha#0", 0, ["cat_alpha"]),
                          ("cat_foreign#0", 0, ["cat_foreign"])])  # not a pool category
            out = _compute(c, _proposal())
            join = out["criteria"]["parent_pool_join"]
            self.assertEqual(join["foreign"], 1)
            self.assertEqual(join["status"], "INCOMPLETE")
            att = out["criteria"]["selection_control_attestations"]
            self.assertFalse(att["exclusive_sanitized_pool_membership"])
            self.assertEqual(att["protected_reference_exclusion_status"], "PASS")
            self.assertFalse(out["ready"])

    # (8) applicability + determinism: no-op for non-acquisition actions; canonical packet
    def test_not_applicable_for_non_acquisition_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root)
            self.assertIsNone(_compute(c, _proposal(action_type="validate_teacher_reference")))

    def test_canonical_record_identical_across_computations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(root)
            first = _compute(c, _proposal())
            second = _compute(c, _proposal())
            self.assertEqual(json.dumps(first, sort_keys=True, default=str),
                             json.dumps(second, sort_keys=True, default=str))

    # extra: pool SHA mismatch fails closed (hash-bound pool identity)
    def test_pool_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path, _pool_sha = _write_pool(root, [("cat_alpha", 100)])
            evidence = _write_scope_classification(
                root, [("cat_alpha", "domain_one", "primary_claim")], ["domain_one"])
            scope_dict = _scope_contract_dict(evidence.name)
            _write_acquisition_manifest(root, pool_path=pool_path, pool_sha="WRONG_SHA",
                                        selected=[("cat_alpha#0", 0, ["cat_alpha"])])
            c = _Controller(root, "synthsha", scope_dict)
            out = _compute(c, _proposal())
            join = out["criteria"]["parent_pool_join"]
            self.assertFalse(join["pool_sha_matches"])
            self.assertEqual(join["status"], "INCOMPLETE")
            self.assertFalse(out["ready"])
            self.assertTrue(any("SHA mismatch" in g for g in out["blocking_gaps"]))

    # extra: duplicate selection is counted and fails closed
    def test_duplicate_selection_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self._standard(
                root,
                selected=[("cat_alpha#0", 0, ["cat_alpha"]), ("cat_alpha#0", 0, ["cat_alpha"])])
            out = _compute(c, _proposal())
            join = out["criteria"]["parent_pool_join"]
            self.assertEqual(join["duplicate"], 1)
            self.assertFalse(out["ready"])

    # extra: pre-execution (manifest not yet produced) marks criteria pending, blocks nothing
    def test_pending_execution_when_manifest_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = _write_scope_classification(
                root, [("cat_alpha", "domain_one", "primary_claim")], ["domain_one"])
            scope_dict = _scope_contract_dict(evidence.name)
            c = _Controller(root, "synthsha", scope_dict)
            out = _compute(c, _proposal())  # no acquisition.manifest.json written
            self.assertIsNotNone(out)
            self.assertTrue(out["pending_execution"])
            self.assertTrue(out["ready"])  # nothing to block pre-execution (geometry-only stage)
            self.assertEqual(out["criteria"]["parent_pool_join"]["status"], "PENDING_EXECUTION")

    # extra: unbound scope contract -> domain mapping unresolvable, fails closed (never guesses)
    def test_no_bound_scope_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path, pool_sha = _write_pool(root, [("cat_alpha", 100)])
            _write_acquisition_manifest(root, pool_path=pool_path, pool_sha=pool_sha,
                                        selected=[("cat_alpha#0", 0, ["cat_alpha"])])
            c = _Controller(root, None, None)  # no bound scope contract
            out = _compute(c, _proposal())
            dom = out["criteria"]["deployment_domain_mapping"]
            self.assertEqual(dom["status"], "UNRESOLVABLE_SCOPE")
            self.assertFalse(out["ready"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
