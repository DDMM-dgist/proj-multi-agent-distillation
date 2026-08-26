"""FE-049 proofs -- two independent, self-contained parts (no cross-test imports; tests/ has no
__init__.py).

Part 1 -- recovery corrective-action parameter-contract validation at ACCEPTANCE:
a corrective_action whose parameters would make its deterministic executor raise ``KeyError`` at
dispatch is rejected by ``validate_recovery_plan_proposal`` BEFORE a human approves it, single-
sourcing each executor's required parameters from the registry ``input_contract`` via
``executors.required_parameters_for_action``. This is the exact gap that failed the eng5 resume:
an approved ``validate_label_preservation`` corrective action carried ``labeled_output_path`` while
the executor reads ``labeled_path`` -> uncaught KeyError -> campaign exit 2.

Part 2 -- ``validate_species_mapping_consistency`` deterministic executor: exposes the concrete
element->type-index mapping recorded in a Teacher labeling manifest and fails closed unless every
independently-sourced mapping (declared config / constructed-calculator runtime / compiled-model
metadata / optional fresh on-disk config) agrees and the mapping is attested. This gives a recovery
targeting a species-mapping evidence-exposure REVISE a corrective action that can actually converge.
"""
import json
from pathlib import Path

import pytest

from runtimes.pydantic_ai.executors import required_parameters_for_action
from runtimes.pydantic_ai.recovery_bridge import (
    CorrectiveAction, RecoveryPlanProposal, RecoveryPlanValidationError,
    validate_recovery_plan_proposal)
from runtimes.pydantic_ai.deterministic_executors import (
    validate_species_mapping_consistency, _ValidationFailure)
from adapters.teacher import SpeciesMappingConflictError
from workflow.integrity import sha256_file

DIAG = "a" * 64
ROSTER = {"data_repair": "data-curator"}


def _proposal(action_type, parameters, *, return_stage="teacher_labeling"):
    return RecoveryPlanProposal(
        run_id="run-x", failed_stage="teacher_labeling", diagnosis_artifact_sha256=DIAG,
        capability="data_repair", return_stage=return_stage,
        proposed_changes=[{"type": "evidence_exposure"}],
        labeling={"teacher_relabel": False, "new_dft": False},
        student_training={"retrain": False, "mode": "none"},
        revalidation={"reuse_profile": True, "targets": ["teacher_labeling"]},
        rationale="expose+validate species mapping",
        corrective_action=CorrectiveAction(action_type=action_type, parameters=parameters))


def _validate(proposal, *, route_action=None, route_parameters=None, replans=None):
    return validate_recovery_plan_proposal(
        proposal, expected_failed_stage="teacher_labeling", expected_diagnosis_sha256=DIAG,
        capability_roster=ROSTER, valid_stage_names=["teacher_labeling"],
        dft_comparison_evidence_present=True, gate_alleges_accuracy_disagreement=True,
        return_stage_route_action=route_action, return_stage_route_parameters=route_parameters,
        return_stage_replans=replans,
        action_required_parameters={
            "validate_label_preservation": required_parameters_for_action(
                "validate_label_preservation"),
            "validate_species_mapping_consistency": required_parameters_for_action(
                "validate_species_mapping_consistency"),
            "acquire_structures": required_parameters_for_action("acquire_structures"),
        })


# --- Part 1: parameter-contract acceptance validation ------------------------------------------

def test_required_parameters_parsed_single_source_from_contract():
    assert required_parameters_for_action("validate_label_preservation") == frozenset(
        {"labeled_path"})
    assert required_parameters_for_action("validate_species_mapping_consistency") == frozenset(
        {"manifest_path"})
    # HPC/approval-gated action has no parseable READY-executor parameter contract -> None.
    assert required_parameters_for_action("acquire_structures") is None
    assert required_parameters_for_action("does_not_exist") is None


def test_reproduces_eng5_bug_wrong_param_name_rejected_at_acceptance():
    # The exact eng5 failure: validate_label_preservation with labeled_output_path (executor reads
    # labeled_path). action_type differs from the return stage's label_with_teacher route, so no
    # base params are inherited -> the missing labeled_path must be caught at acceptance.
    prop = _proposal("validate_label_preservation", {"labeled_output_path": "/x.extxyz"})
    with pytest.raises(RecoveryPlanValidationError) as exc:
        _validate(prop, route_action="label_with_teacher",
                  route_parameters={"structures_path": "/s.extxyz"})
    assert "labeled_path" in str(exc.value)


def test_missing_required_param_for_different_action_type_rejected():
    prop = _proposal("validate_species_mapping_consistency", {})  # manifest_path missing
    with pytest.raises(RecoveryPlanValidationError) as exc:
        _validate(prop, route_action="label_with_teacher",
                  route_parameters={"structures_path": "/s.extxyz"})
    assert "manifest_path" in str(exc.value)


def test_present_required_param_accepted():
    prop = _proposal("validate_species_mapping_consistency", {"manifest_path": "/m.json"})
    out = _validate(prop, route_action="label_with_teacher",
                    route_parameters={"structures_path": "/s.extxyz"})
    assert out is prop


def test_required_param_satisfied_by_inherited_base_route_params_when_same_action_type():
    # When the corrective action re-runs the return stage's OWN route action, dispatch merges the
    # stage's route parameters; acceptance must mirror that and NOT demand the param be re-listed.
    prop = _proposal("validate_label_preservation", {})
    out = _validate(prop, route_action="validate_label_preservation",
                    route_parameters={"labeled_path": "/labeled.extxyz"})
    assert out is prop


def test_action_without_parseable_contract_not_falsely_rejected():
    # acquire_structures -> required_parameters_for_action is None -> no requirement enforced.
    prop = _proposal("acquire_structures", {})
    out = _validate(prop, route_action="acquire_structures",
                    route_parameters={"acquisition_config": "/a.json"})
    assert out is prop


# --- Part 2: validate_species_mapping_consistency executor -------------------------------------

def _manifest(tmp_path, *, declared=("O", "Si"), runtime=None, compiled=None,
              fallback_applied=False, teacher_config_sha256=None, name="teacher_labels.manifest.json"):
    evidence = {
        "declared_chemical_symbols": list(declared) if declared is not None else None,
        "declared_chemical_species_to_atom_type_map": None,
        "runtime_chemical_species_to_atom_type_map": runtime,
        "runtime_mapping_source": "calculator.transforms[0]" if runtime else None,
        "compiled_model_type_names_map": compiled,
        "compiled_model_metadata_source": "compiled_model[x].type_names" if compiled else None,
        "fallback_applied": fallback_applied,
        "fallback_reason": None,
    }
    manifest = {"schema_version": 1, "species_mapping_evidence": evidence}
    if teacher_config_sha256 is not None:
        manifest["teacher_config_sha256"] = teacher_config_sha256
    p = tmp_path / name
    p.write_text(json.dumps(manifest, indent=2))
    return p


def test_species_mapping_valid_exposes_and_cross_checks(tmp_path):
    mp = _manifest(tmp_path, declared=("O", "Si"), runtime={"O": 0, "Si": 1},
                   compiled={"O": 0, "Si": 1})
    res = validate_species_mapping_consistency({"parameters": {"manifest_path": str(mp)}})
    m = res["metrics"]
    assert m["ok"] is True and m["attested"] is True
    assert m["species_to_type_index_map"] == {"O": 0, "Si": 1}
    assert set(m["sources_cross_checked"]) == {
        "declared_config", "constructed_calculator_runtime", "compiled_model_metadata"}


def test_species_mapping_conflict_fails_closed(tmp_path):
    mp = _manifest(tmp_path, declared=("O", "Si"), runtime={"O": 1, "Si": 0},
                   compiled={"O": 0, "Si": 1})
    with pytest.raises(SpeciesMappingConflictError):
        validate_species_mapping_consistency({"parameters": {"manifest_path": str(mp)}})


def test_species_mapping_not_attested_fails_closed(tmp_path):
    # Declares a chemical_symbols convention but records no resolved runtime mapping -> not attested.
    mp = _manifest(tmp_path, declared=("O", "Si"), runtime=None, compiled=None)
    with pytest.raises(_ValidationFailure) as exc:
        validate_species_mapping_consistency({"parameters": {"manifest_path": str(mp)}})
    assert "species_mapping_not_attested" in str(exc.value)


def test_species_mapping_manifest_sha_mismatch_fails_closed(tmp_path):
    mp = _manifest(tmp_path, runtime={"O": 0, "Si": 1}, compiled={"O": 0, "Si": 1})
    with pytest.raises(_ValidationFailure) as exc:
        validate_species_mapping_consistency(
            {"parameters": {"manifest_path": str(mp), "expected_manifest_sha256": "b" * 64}})
    assert "manifest_sha256_mismatch" in str(exc.value)


def test_species_mapping_teacher_config_binding_matches(tmp_path):
    cfg = tmp_path / "teacher.yaml"
    cfg.write_text("calculator:\n  kwargs:\n    chemical_symbols: [O, Si]\n")
    cfg_sha = sha256_file(cfg)
    mp = _manifest(tmp_path, declared=("O", "Si"), runtime={"O": 0, "Si": 1},
                   compiled={"O": 0, "Si": 1}, teacher_config_sha256=cfg_sha)
    res = validate_species_mapping_consistency(
        {"parameters": {"manifest_path": str(mp), "teacher_config": str(cfg)}})
    binding = res["metrics"]["teacher_config_binding"]
    assert binding["sha256_matches_manifest"] is True
    assert binding["config_species_to_type_index_map"] == {"O": 0, "Si": 1}
    assert "reread_teacher_config" in res["metrics"]["sources_cross_checked"]


def test_species_mapping_teacher_config_sha_mismatch_fails_closed(tmp_path):
    cfg = tmp_path / "teacher.yaml"
    cfg.write_text("calculator:\n  kwargs:\n    chemical_symbols: [O, Si]\n")
    mp = _manifest(tmp_path, declared=("O", "Si"), runtime={"O": 0, "Si": 1},
                   compiled={"O": 0, "Si": 1}, teacher_config_sha256="c" * 64)
    with pytest.raises(_ValidationFailure) as exc:
        validate_species_mapping_consistency(
            {"parameters": {"manifest_path": str(mp), "teacher_config": str(cfg)}})
    assert "teacher_config_sha256_mismatch" in str(exc.value)
