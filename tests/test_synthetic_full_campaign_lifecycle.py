"""Part C, Paths A/B/C: a synthetic campaign matching the REAL 12-stage production topology
(``configs/runs/sio2-sox-allegro-simplenn-r17/workflow.yaml``'s stage order:
teacher_baseline, reference_validation, acquisition, data_coverage, teacher_labeling,
dataset_split, training, evaluation, uncertainty, deployment_md, physical_validation, analysis)
reaches ``COMPLETED`` with every stage's gate at ``PASS``, driven entirely through the real
``runtimes.pydantic_ai.cli.run_campaign``/``run_production_stage``/``RunController`` production
path -- no parallel/bypass implementation.

Four of the twelve stages -- ``data_coverage``, ``uncertainty``, ``physical_validation``,
``analysis`` (``TARGET_STAGES`` below) -- historically carried NO ``pydantic_ai`` role/action in
the real R17 config, so in that frozen production run those four are completed by a human/analyst
script calling ``complete_external_stage``+``record_gate`` directly, never by ``run-campaign``.
Real, deterministic, self-validating composite executors now exist for all four
(``build_data_coverage_report``, ``build_uncertainty_report``, ``build_physical_validation_report``,
``generate_run_summary``), and ``configs/templates/workflow.yaml`` -- the authoritative template a
fresh campaign (e.g. R18) is initialized from -- now carries their canonical ``pydantic_ai``
role/action pair AND output contract for all four. A workflow config MAY opt any of them into
automation via that same template shape -- exactly like any other stage -- while a config with no
such block for one of them (like R17's) is completely unaffected and still fails closed. This file
proves all three regimes with the SAME real dispatch/gate/controller/contract machinery, never a
parallel/bypass implementation:

  * Path A -- ``test_path_a_all_target_stages_automated``: all four target stages carry the SAME
    ``pydantic_ai`` role/action AND ``contract`` declarations as ``configs/templates/workflow.yaml``
    (only their run-specific ``parameters``/evidence paths differ, exactly as the template's own
    README documents). A SINGLE ``run_campaign()`` invocation drives the entire 12-stage campaign
    -- stage selection, routing, artifact registration, contract validation (including
    ``data_coverage``'s real write-once ``validation_contract`` cross-check), Judge gating, and
    Controller advancement -- to ``COMPLETED`` with ZERO manual intervention: no
    ``complete_external_stage()``, ``record_gate()``, stage pre-completion, or synthetic manual
    routing of any kind.
  * Path B -- ``test_path_b_mixed_automated_and_manual_target_stages``: two of the four
    (``data_coverage``, ``uncertainty``) are automated (same production route+contract
    declarations as Path A) and two (``physical_validation``, ``analysis``) remain manual, proving
    automated and manual target stages can be freely interleaved in one real campaign.
  * Path C -- ``test_path_c_all_target_stages_manual_matches_real_r17_config``: all four target
    stages carry NO ``pydantic_ai`` block at all (byte-for-byte the same shape as the real R17
    config for these four stages) and are completed manually; this also proves (rather than
    assumes) that ``run_campaign`` fails closed with a precise ``ValueError`` -- never a silent
    skip or a wrong dispatch -- whenever it is asked to advance past one of them before a human has
    completed it.

Only Path A's four target stages are asserted, one-by-one, to reuse the exact
``role``/``action``/``contract.kind``/``contract.validator`` strings ``configs/templates/
workflow.yaml`` declares (``test_path_a_stage_routes_match_the_production_template``) -- so this
file cannot silently drift from the template while still claiming to exercise "the real production
route". Paths B/C's manual variants are never presented as evidence that the automated production
topology is complete (see each path's docstring above).

dataset_split/training/evaluation reuse the SAME real action names production uses
(``generate_group_split``, ``train_committee``, ``evaluate_heldout_fidelity`` -- the latter two
exactly as ``tests/test_run_campaign.py`` already exercises, mock committee, real approval boundary
``costly_training``). The other four automated-in-all-paths stages
(``teacher_baseline``, ``reference_validation``, ``acquisition``, ``teacher_labeling``,
``deployment_md``) stand in for what would be heavier Teacher/MD adapters in production with the
SAME already-proven, deterministic, ungated ``build_dataset_manifest`` executor
``tests/test_run_campaign_recovery.py`` uses -- avoiding scientific compute and optional heavy
dependencies while still exercising the identical real dispatch/gate/controller machinery for each
stage slot. No OpenAI network call anywhere (mock runtime only).

Contract wiring notes (why each automated target stage's evidence resolves for real, not just by
skipping the check):
  * ``data_coverage`` establishes a real write-once ``validation_contract`` (via
    ``validation_contract_sources``, exactly like a real campaign) and its own
    ``deployment_domain``/``dataset_policy`` parameters are the SAME values/file used to build
    that contract, so the contract's hash cross-check in ``validation.data_coverage`` genuinely
    passes rather than being skipped. Its evidence files (candidate dataset, acquisition manifest,
    dataset policy) are declared workflow ``inputs:`` so the Controller's evidence allowlist binds
    them for real.
  * ``uncertainty`` consumes ``training``'s real committee manifest and ``evaluation``'s real
    labeled held-out population (with per-seed ``student_forces_seedNN`` already embedded by
    ``workflow.steps.evaluate_committee``) as its evidence -- genuine upstream run artifacts, never
    separately fabricated stand-ins -- so the Controller's evidence allowlist binds them as
    upstream artifacts with no extra wiring.
  * ``physical_validation``'s ``validation_profile``/``frames_path`` evidence files are declared
    workflow ``inputs:``, and its own ``validation_profile.yaml`` doubles as the
    ``validation_contract_sources.validation_profile`` source (one real file serving both roles,
    like a real campaign's).
  * ``analysis``'s own auto-injected ``run_state.snapshot.json`` evidence file is declared as one
    of the stage's own ``outputs`` (matching the template's fix for this), so it is bound as a
    just-submitted artifact.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_WORKFLOW = ROOT / "configs" / "templates" / "workflow.yaml"

# The real production stage order (see module docstring).
TARGET_STAGES = {"data_coverage", "uncertainty", "physical_validation", "analysis"}
STAGE_ORDER = [
    "teacher_baseline", "reference_validation", "acquisition", "data_coverage",
    "teacher_labeling", "dataset_split", "training", "evaluation", "uncertainty",
    "deployment_md", "physical_validation", "analysis",
]
# Automated-in-all-paths stages standing in for a heavier real Teacher/MD action with the same
# already-proven ungated build_dataset_manifest executor (see module docstring).
_MANIFEST_STAGES = ["teacher_baseline", "reference_validation", "acquisition", "teacher_labeling",
                    "deployment_md"]


def _tiny_dataset(path: Path, tag: str) -> Path:
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.info["structure_id"] = tag
    write(str(path), [atoms])
    return path


def _manifest_stage_cfg(root: Path, name: str) -> dict:
    dataset = _tiny_dataset(root / f"{name}_dataset.extxyz", name)
    params = {"dataset": str(dataset), "manifest_path": f"{{artifacts_dir}}/{name}_manifest.json"}
    if name == "teacher_baseline":
        # cli._fill_default_parameters hardcodes this requirement by STAGE NAME regardless of
        # the configured action (see cli.py:210-211) -- harmless extra key for
        # build_dataset_manifest's free-form `parameters` dict (Part B audit: ActionProposalBase.
        # parameters is deliberately unconstrained; per-action shape isn't enforced pre-Phase 4).
        params["structures_path"] = str(dataset)
    return {
        "name": name, "command": None, "outputs": [f"artifacts/{name}_manifest.json"],
        "gate": {"criteria": [f"{name} manifest is complete"]},
        "pydantic_ai": {
            "role": "data-curator", "action": "build_dataset_manifest",
            "idempotency_key": f"synthetic-12stage:{name}:001",
            "parameters": params,
        },
    }


def _dataset_split_stage_cfg(root: Path) -> dict:
    dataset = root / "dataset_split_dataset.extxyz"
    frames = []
    for i in range(6):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"ds{i}"
        a.info["parent_structure_id"] = f"seed:{i}"
        frames.append(a)
    write(str(dataset), frames)
    return {
        "name": "dataset_split", "command": None,
        "outputs": ["artifacts/dataset_split/split_manifest.json"],
        "gate": {"criteria": ["split manifest is complete"]},
        "pydantic_ai": {
            "role": "data-curator", "action": "generate_group_split",
            "idempotency_key": "synthetic-12stage:dataset_split:001",
            "parameters": {"dataset": str(dataset),
                          "output_dir": "{artifacts_dir}/dataset_split",
                          "manifest": "{artifacts_dir}/dataset_split/split_manifest.json"},
        },
    }


def _training_evaluation_stage_cfgs(root: Path) -> tuple:
    dataset = root / "training_dataset.extxyz"
    frames = []
    for i in range(3):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"t{i}"
        a.info["parent_structure_id"] = f"p{i}"
        frames.append(a)
    write(str(dataset), frames)
    student_cfg = root / "student.yaml"
    student_cfg.write_text(
        "kind: mock\ncommittee:\n  seeds: [1, 2, 3]\n"
        "predict:\n  factory: adapters.mock_model.MockCheckpointCalculator\n"
        "  checkpoint_arg: checkpoint\n  kwargs: {}\n")
    training = {
        "name": "training", "command": None,
        "outputs": ["artifacts/student_committee.manifest.json", "artifacts/committee"],
        "pydantic_ai": {
            "role": "ml-trainer", "action": "train_committee",
            "approval_boundary": "costly_training",
            "idempotency_key": "synthetic-12stage-training-001",
            "parameters": {
                "student_config": str(student_cfg), "dataset": str(dataset),
                "output_dir": "{artifacts_dir}/committee",
                "manifest_path": "{artifacts_dir}/student_committee.manifest.json",
            },
        },
        "gate": {"criteria": ["committee manifest is complete"]},
    }
    evaluation = {
        "name": "evaluation", "command": None,
        "outputs": ["artifacts/heldout_labeled.extxyz", "artifacts/heldout_report.json"],
        "pydantic_ai": {
            "role": "ml-trainer", "action": "evaluate_heldout_fidelity",
            "approval_boundary": "costly_training",
            "idempotency_key": "synthetic-12stage-evaluation-001",
            "parameters": {
                "student_config": str(student_cfg),
                "committee_manifest": "{artifacts_dir}/student_committee.manifest.json",
                "frames_path": str(dataset),
                "labeled_output": "{artifacts_dir}/heldout_labeled.extxyz",
                "report_path": "{artifacts_dir}/heldout_report.json",
            },
        },
        "gate": {"criteria": ["fidelity report is complete"]},
    }
    return training, evaluation


# --- Shared campaign-level configs backing the four automated target stages' real contracts ---

def _shared_campaign_configs(root: Path) -> dict:
    """Build the real files a campaign binds before any scientific stage runs: the write-once
    validation_contract sources (``distillation_scope``, ``validation_profile``,
    ``dataset_policy``). ``validation_profile.yaml`` doubles as ``physical_validation``'s own
    executor input (its ``checks`` list) -- one real file serving both roles, like a real
    campaign's."""
    deployment_domain = {"structure_classes": ["bulk_cu_synthetic"]}
    distillation_scope = root / "distillation_scope.yaml"
    distillation_scope.write_text(yaml.safe_dump({"deployment_domain": deployment_domain}))

    validation_profile = root / "validation_profile.yaml"
    validation_profile.write_text(yaml.safe_dump({
        "kind": "project-validation",
        "deployment_domain": deployment_domain,
        "checks": [
            {"name": "rdf_Si_Si", "category": "structure", "required": True, "threshold": None},
            {"name": "rdf_Si_O", "category": "structure", "required": True, "threshold": None},
            {"name": "rdf_O_O", "category": "structure", "required": True, "threshold": None},
            {"name": "coordination_Si", "category": "structure", "required": True, "threshold": None},
            {"name": "coordination_O", "category": "structure", "required": True, "threshold": None},
            {"name": "density", "category": "structure", "required": True, "threshold": None},
            {"name": "nve_drift", "category": "dynamics", "required": True,
             "threshold": {"operator": "max_abs", "threshold": 1000.0}},
            {"name": "msd_selfdiffusion", "category": "dynamics", "required": True, "threshold": None},
        ],
    }))

    dataset_policy = root / "dataset_policy.yaml"
    dataset_policy.write_text(yaml.safe_dump(
        {"split_policy": {"method": "by_parent_structure_id", "seed": 7}}))

    return {"deployment_domain": deployment_domain, "distillation_scope": distillation_scope,
            "validation_profile": validation_profile, "dataset_policy": dataset_policy}


# --- Automated configs for the four historically-manual target stages -----------------------
# Each returns (stage_cfg, extra_workflow_inputs) -- role/action/contract match
# configs/templates/workflow.yaml exactly (see test_path_a_stage_routes_match_the_production_
# template); only parameters/evidence paths are run-specific, per the template's own README.

def _data_coverage_stage_cfg(root: Path, shared: dict) -> tuple:
    candidate = root / "data_coverage_candidate.extxyz"
    frames = []
    for i in range(4):
        a = Atoms("Cu", positions=[[i, 0, 0]], cell=[10, 10, 10], pbc=True)
        a.info["structure_id"] = f"dc{i}"
        a.info["parent_structure_id"] = f"dcparent{i}"
        a.info["config_type"] = "bulk"
        frames.append(a)
    write(str(candidate), frames)
    acquisition_manifest = root / "data_coverage_acquisition.manifest.json"
    acquisition_manifest.write_text(json.dumps({"n_frames": 10, "elements": ["Cu"]}))
    cfg = {
        "name": "data_coverage", "command": None,
        "outputs": ["artifacts/data_coverage_report.json"],
        "gate": {"criteria": ["data coverage report is complete"]},
        "contract": {
            "kind": "validation_manifest",
            "manifest": "artifacts/data_coverage_report.json",
            "validator": "validation.data_coverage.validate_data_coverage_report",
            "options": {"validation_contract_path": "{run_dir}/validation_contract.json"},
        },
        "pydantic_ai": {
            "role": "data-curator", "action": "build_data_coverage_report",
            "idempotency_key": "synthetic-path:data_coverage:001",
            "parameters": {
                "candidate_dataset": str(candidate),
                "acquisition_manifest": str(acquisition_manifest),
                "report_path": "{artifacts_dir}/data_coverage_report.json",
                # SAME deployment_domain / dataset_policy file used to establish this run's
                # write-once validation_contract below, so the contract's hash cross-check
                # genuinely passes rather than being skipped.
                "deployment_domain": shared["deployment_domain"],
                "dataset_policy": str(shared["dataset_policy"]),
            },
        },
    }
    return cfg, [candidate, acquisition_manifest, shared["dataset_policy"]]


def _uncertainty_stage_cfg(root: Path, shared: dict) -> tuple:
    cfg = {
        "name": "uncertainty", "command": None,
        "outputs": ["artifacts/uncertainty_report.json"],
        "gate": {"criteria": ["uncertainty report is complete"]},
        "contract": {
            "kind": "validation_manifest",
            "manifest": "artifacts/uncertainty_report.json",
            "validator": "validation.uncertainty.validate_uncertainty_report",
        },
        "pydantic_ai": {
            "role": "ml-trainer", "action": "build_uncertainty_report",
            "idempotency_key": "synthetic-path:uncertainty:001",
            "parameters": {
                # The REAL upstream committee manifest (training) and labeled held-out
                # population (evaluation) -- genuine upstream run artifacts, never separately
                # fabricated stand-ins -- so the Controller's evidence allowlist binds them as
                # upstream artifacts with no extra wiring.
                "committee_manifest": "{artifacts_dir}/student_committee.manifest.json",
                "population_frames": "{artifacts_dir}/heldout_labeled.extxyz",
                "report_path": "{artifacts_dir}/uncertainty_report.json",
            },
        },
    }
    return cfg, []


def _physical_validation_stage_cfg(root: Path, shared: dict) -> tuple:
    frames_path = root / "physical_validation_frames.extxyz"
    frames = []
    for i in range(3):
        positions = [[0, 0, 0], [1.6 + 0.01 * i, 0, 0], [0, 1.6, 0], [0, 0, 1.6]]
        a = Atoms("SiSiOO", positions=positions, cell=[10, 10, 10], pbc=True)
        frames.append(a)
    write(str(frames_path), frames)
    cfg = {
        "name": "physical_validation", "command": None,
        "outputs": ["artifacts/physical_validation_report.json"],
        "gate": {"criteria": ["physical validation report is complete"]},
        "contract": {
            "kind": "validation_manifest",
            "manifest": "artifacts/physical_validation_report.json",
            "validator": "validation.report.validate_validation_report",
        },
        "pydantic_ai": {
            "role": "simulation", "action": "build_physical_validation_report",
            "idempotency_key": "synthetic-path:physical_validation:001",
            "parameters": {
                "validation_profile": str(shared["validation_profile"]),
                "frames_path": str(frames_path),
                "r_max": 4.0,
                "cutoffs": {"Si-O": 2.2, "default": 3.0},
                "energies": [-10.0, -10.0001, -10.0002],
                "n_atoms": 4,
                "report_path": "{artifacts_dir}/physical_validation_report.json",
            },
        },
    }
    return cfg, [frames_path, shared["validation_profile"]]


_LAMMPS_DUMP_TEMPLATE = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
{n_atoms}
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type x y z fx fy fz
{atom_lines}
"""


def _physical_validation_lammps_stage_cfg(root: Path, shared: dict, student_config: Path) -> tuple:
    """Same production route/contract as ``_physical_validation_stage_cfg``, but
    ``frames_path`` is a REAL raw LAMMPS dump (integer atom types, no element column
    -- see ``templates/lammps/prod_md.in.template``) with a bound ``student_config``,
    proving cli.py resolves the authoritative specorder from
    ``student_config.deploy.elements`` end to end through the real production dispatch."""
    frames_path = root / "physical_validation_trajectory.dump"
    positions = [(0, 0, 0), (1.6, 0, 0), (0, 1.6, 0), (0, 0, 1.6)]
    # deploy.elements == [O, Si] -> type 1 = O, type 2 = Si; types below match the
    # Si,Si,O,O composition _physical_validation_stage_cfg already exercises via extxyz.
    types = [2, 2, 1, 1]
    lines = [f"{i} {t} {pos[0]} {pos[1]} {pos[2]} 0.0 0.0 0.0"
            for i, (t, pos) in enumerate(zip(types, positions), start=1)]
    frames_path.write_text(_LAMMPS_DUMP_TEMPLATE.format(n_atoms=len(types),
                                                        atom_lines="\n".join(lines)))
    cfg = {
        "name": "physical_validation", "command": None,
        "outputs": ["artifacts/physical_validation_report.json"],
        "gate": {"criteria": ["physical validation report is complete"]},
        "contract": {
            "kind": "validation_manifest",
            "manifest": "artifacts/physical_validation_report.json",
            "validator": "validation.report.validate_validation_report",
        },
        "pydantic_ai": {
            "role": "simulation", "action": "build_physical_validation_report",
            "idempotency_key": "synthetic-path:physical_validation_lammps:001",
            "parameters": {
                "validation_profile": str(shared["validation_profile"]),
                "frames_path": str(frames_path),
                "student_config": str(student_config),
                "r_max": 4.0,
                "cutoffs": {"Si-O": 2.2, "default": 3.0},
                "energies": [-10.0, -10.0001, -10.0002],
                "n_atoms": 4,
                "report_path": "{artifacts_dir}/physical_validation_report.json",
            },
        },
    }
    return cfg, [frames_path, shared["validation_profile"], student_config]


def _analysis_stage_cfg(root: Path, shared: dict) -> tuple:
    cfg = {
        "name": "analysis", "command": None,
        # artifacts/analysis/run_state.snapshot.json is the state snapshot cli.py::
        # _proposal_from_stage always writes to this exact fixed location and cites as the
        # report's run_state_snapshot evidence (see configs/templates/workflow.yaml's matching
        # fix) -- it must be declared here too or the contract's evidence-allowlist check
        # rejects it as unbound.
        "outputs": ["artifacts/analysis/run_summary.json",
                   "artifacts/analysis/run_state.snapshot.json"],
        "gate": {"criteria": ["run summary is complete"]},
        "contract": {
            "kind": "validation_manifest",
            "manifest": "artifacts/analysis/run_summary.json",
            "validator": "validation.run_summary.validate_run_summary_report",
        },
        "pydantic_ai": {
            "role": "analyst", "action": "generate_run_summary",
            "idempotency_key": "synthetic-path:analysis:001",
            # run_state_path is deliberately NOT set here -- cli._proposal_from_stage always
            # freshly assembles and injects it from CURRENT Controller state before dispatch.
            "parameters": {"report_path": "{artifacts_dir}/analysis/run_summary.json"},
        },
    }
    return cfg, []


_AUTOMATED_STAGE_CFG = {
    "data_coverage": _data_coverage_stage_cfg,
    "uncertainty": _uncertainty_stage_cfg,
    "physical_validation": _physical_validation_stage_cfg,
    "analysis": _analysis_stage_cfg,
}
# Which of the four automated configs actually reads `shared` (so we only pay for building the
# shared campaign-config files when at least one such stage is automated for this path).
_NEEDS_SHARED_CONFIGS = {"data_coverage", "physical_validation"}


def _manual_stage_cfg(name: str) -> dict:
    # No "pydantic_ai" key at all -- exactly matches the real R17 production config for these
    # four stages, so run_campaign can NEVER auto-dispatch them (see cli._proposal_from_stage).
    return {"name": name, "command": None, "outputs": [f"artifacts/{name}.json"],
           "gate": {"criteria": [f"{name} is complete"]}}


def _twelve_stage_workflow(root: Path, automated_stage_names, extra_workflow_inputs=(),
                          stage_overrides=None) -> Path:
    """``stage_overrides`` (optional ``{stage_name: callable(root, shared) -> (cfg,
    inputs)}``) lets a caller substitute one target stage's automated config for a
    variant that still shares the SAME campaign-level ``shared`` configs (e.g. a real
    LAMMPS-dump ``physical_validation`` variant) without duplicating this function."""
    training, evaluation = _training_evaluation_stage_cfgs(root)
    by_name = {"dataset_split": _dataset_split_stage_cfg(root), "training": training,
              "evaluation": evaluation}
    for name in _MANIFEST_STAGES:
        by_name[name] = _manifest_stage_cfg(root, name)

    shared = (_shared_campaign_configs(root)
             if automated_stage_names & _NEEDS_SHARED_CONFIGS else None)
    extra_inputs = list(extra_workflow_inputs)
    for name in TARGET_STAGES:
        if stage_overrides and name in stage_overrides:
            cfg, stage_inputs = stage_overrides[name](root, shared)
            by_name[name] = cfg
            extra_inputs.extend(stage_inputs)
        elif name in automated_stage_names:
            cfg, stage_inputs = _AUTOMATED_STAGE_CFG[name](root, shared)
            by_name[name] = cfg
            extra_inputs.extend(stage_inputs)
        else:
            by_name[name] = _manual_stage_cfg(name)

    workflow_dict = {"run_id": "synthetic-12stage",
                    "stages": [by_name[name] for name in STAGE_ORDER],
                    "inputs": [str(p) for p in extra_inputs]}
    if "data_coverage" in automated_stage_names:
        # Establishes the SAME write-once validation_contract a real campaign binds before any
        # scientific stage runs -- required by data_coverage's own production contract.options
        # (validation_contract_path).
        workflow_dict["validation_contract_sources"] = {
            "distillation_scope": str(shared["distillation_scope"]),
            "validation_profile": str(shared["validation_profile"]),
            "dataset_policy": str(shared["dataset_policy"]),
        }
    workflow = root / "workflow.yaml"
    workflow.write_text(yaml.safe_dump(workflow_dict))
    return workflow


def _drive_campaign_to_completion(root: Path, workflow: Path, manual_stage_names):
    """Shared driver for Paths A/B/C: dispatches automated stages through the real
    run_campaign/run_production_stage path, and completes any stage named in
    ``manual_stage_names`` directly via complete_external_stage+record_gate (exactly matching
    how a human/analyst script would, since those stages carry no pydantic_ai block at all).
    """
    from runtimes.pydantic_ai import cli
    from workflow.controller import RunController

    run_dir = root / "run"
    RunController.initialize(workflow, run_dir)
    assert cli.main(["approve", "--run-dir", str(run_dir), "--boundary", "costly_training",
                    "--note", "pre-approved for synthetic campaign test"]) == cli.EXIT_SUCCESS

    c = RunController(run_dir)
    guard = 0
    while True:
        guard += 1
        if guard >= 30:
            raise AssertionError("test driver looped without making progress")
        pending_name = next((s["name"] for s in c.state["stages"] if s["gate"] != "PASS"), None)
        if pending_name is None:
            break
        if pending_name in manual_stage_names:
            artifact = c.run_dir / f"artifacts/{pending_name}.json"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text('{"status": "ok"}')
            c.complete_external_stage(pending_name, [artifact])
            votes = c.run_dir / "gates" / f"{pending_name}.votes.json"
            criteria = c.stage(pending_name)["gate_criteria"]
            lenses = c.stage(pending_name)["gate_review_lenses"]
            votes.write_text(json.dumps({
                "stage": pending_name, "criteria": criteria, "review_lenses": lenses,
                "artifact_sha256": {a["path"]: a["sha256"]
                                    for a in c.stage_artifacts(pending_name)},
                "decision": "PASS",
                "votes": [{"judge_id": f"judge-{i}", "review_lens": lens["id"], "verdict": "PASS",
                          "criteria_checked": [{"criterion": cr, "value_read": "checked",
                                                "ok": True} for cr in criteria],
                          "rationale": "ok", "required_fix": ""}
                         for i, lens in enumerate(lenses, 1)]}))
            c.record_gate(pending_name, votes_path=votes)
            c = RunController(run_dir)
            continue
        try:
            result = cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                                      auto_mock_judges=True, max_iterations=20)
        except ValueError as exc:
            # A workflow-authoring/operator error, not a hidden contract defect: proves
            # run_campaign fails closed with a precise message rather than silently skipping
            # or mis-dispatching a stage that has no pydantic_ai route. Only expected when this
            # campaign actually has manual target stages left to reach.
            if not manual_stage_names:
                raise
            assert "pydantic_ai role/action metadata" in str(exc), str(exc)
            c = RunController(run_dir)
            continue
        c = RunController(run_dir)
        if result.outcome == cli.CAMPAIGN_COMPLETED:
            break
        if result.outcome == cli.CAMPAIGN_FAILED:
            raise AssertionError(f"unexpected terminal failure: {result.message}")

    c = RunController(run_dir)
    for name in STAGE_ORDER:
        assert c.stage(name)["gate"] == "PASS", name
        assert c.stage(name)["status"] == "completed", name
    assert c.state.get("pending_recovery") is None

    # Idempotent resume: re-running after terminal completion must not error or mutate.
    from runtimes.pydantic_ai import cli as _cli
    before = c.state
    final = _cli.run_campaign(c, runtime="mock", repo_root=str(ROOT),
                              auto_mock_judges=True, max_iterations=20)
    assert final.outcome == _cli.CAMPAIGN_COMPLETED
    after = RunController(run_dir).state
    assert before["stages"] == after["stages"]
    return c


class SyntheticCampaignLifecyclePathsTests(unittest.TestCase):
    def test_path_a_stage_routes_match_the_production_template(self):
        """Guards against silent drift: Path A's four target-stage configs must declare the SAME
        role/action/contract-kind/contract-validator the authoritative
        configs/templates/workflow.yaml declares for them -- so a future template edit that
        changes one of these canonical bindings cannot go unnoticed here."""
        template = yaml.safe_load(TEMPLATE_WORKFLOW.read_text())
        template_stages = {stage["name"]: stage for stage in template["stages"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = _shared_campaign_configs(root)
            for name in TARGET_STAGES:
                cfg, _ = _AUTOMATED_STAGE_CFG[name](root, shared)
                template_stage = template_stages[name]
                self.assertEqual(cfg["pydantic_ai"]["role"], template_stage["pydantic_ai"]["role"],
                                 name)
                self.assertEqual(cfg["pydantic_ai"]["action"],
                                 template_stage["pydantic_ai"]["action"], name)
                self.assertEqual(cfg["contract"]["kind"], template_stage["contract"]["kind"], name)
                self.assertEqual(cfg["contract"]["validator"],
                                 template_stage["contract"]["validator"], name)

    def test_path_a_all_target_stages_automated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _twelve_stage_workflow(root, automated_stage_names=set(TARGET_STAGES))
            _drive_campaign_to_completion(root, workflow, manual_stage_names=set())

    def test_path_b_mixed_automated_and_manual_target_stages(self):
        automated = {"data_coverage", "uncertainty"}
        manual = TARGET_STAGES - automated
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _twelve_stage_workflow(root, automated_stage_names=automated)
            _drive_campaign_to_completion(root, workflow, manual_stage_names=manual)

    def test_path_c_all_target_stages_manual_matches_real_r17_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow = _twelve_stage_workflow(root, automated_stage_names=set())
            _drive_campaign_to_completion(root, workflow, manual_stage_names=set(TARGET_STAGES))

    def test_path_d_physical_validation_reads_lammps_dump_via_bound_student_config(self):
        """The EXECUTOR_SPECIES_MAPPING_BUG fix, exercised through the real 12-stage
        production dispatch: physical_validation's frames_path is a genuine raw LAMMPS
        dump (integer atom types, no element column) instead of the self-describing
        extxyz the other paths use, with a bound student_config declaring deploy.
        elements == [O, Si]. Proves the campaign still reaches COMPLETED and that the
        resolved species_mapping (Controller-bound, not model-supplied) is recorded in
        the report with the correct O/Si species -- not the H/He an unmapped read
        would silently produce."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            student_config = root / "student.simple-nn.yaml"
            student_config.write_text(yaml.safe_dump(
                {"kind": "mock", "deploy": {"lammps_pair_style": "nn", "elements": ["O", "Si"]}}))

            def _lammps_override(root, shared):
                return _physical_validation_lammps_stage_cfg(root, shared, student_config)

            workflow = _twelve_stage_workflow(
                root, automated_stage_names=set(TARGET_STAGES),
                extra_workflow_inputs=[student_config],
                stage_overrides={"physical_validation": _lammps_override})
            c = _drive_campaign_to_completion(root, workflow, manual_stage_names=set())
            report = json.loads(
                (c.run_dir / "artifacts" / "physical_validation_report.json").read_text())
            self.assertEqual(report["species_mapping"]["specorder"], ["O", "Si"])
            self.assertEqual(report["species_mapping"]["source"], "student_config.deploy.elements")
            names = {chk["observable"] for chk in report["checks"]}
            self.assertIn("rdf_Si_Si", names)
            self.assertIn("coordination_O", names)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
