"""Build deterministic scientific-contract reports for the packaged mock run."""
import argparse
import json
from pathlib import Path

from ase.io import read

from validation.report import evidence_record


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def teacher_baseline(teacher_config, distillation_scope, validation_profile, evidence, output):
    write_json(output, {
        "schema_version": 1,
        "profile": "mock-deployment-v1",
        "teacher": {"config": str(Path(teacher_config).resolve())},
        "distillation_scope": str(Path(distillation_scope).resolve()),
        "validation_profile": str(Path(validation_profile).resolve()),
        "deployment_domain": {"structure_classes": ["mock-seed-neighborhood"]},
        "applicability": {"status": "CONDITIONAL",
                          "limitations": ["mock evidence only"]},
        "species_mapping": {
            "declared_chemical_symbols": None,
            "declared_chemical_species_to_atom_type_map": None,
            "runtime_chemical_species_to_atom_type_map": None,
            "fallback_applied": False,
            "fallback_reason": None,
        },
        "checks": [{
            "domain": "accuracy", "observable": "mock_teacher_reference",
            "status": "RECORDED", "value": 0.0, "unit": "mock-unit",
            "criterion": None, "purpose": "student_teacher_fidelity",
            "reference_source": "teacher", "protocol": "mock-static-v1",
        }],
        "evidence": [evidence_record("teacher_config", teacher_config),
                     evidence_record("distillation_scope", distillation_scope),
                     evidence_record("validation_profile", validation_profile),
                     evidence_record("teacher_scope_input", evidence)],
    })


def data_coverage(dataset_policy, dataset, output):
    frames = read(dataset, index=":")
    parents = {str(frame.info["parent_structure_id"]) for frame in frames}
    write_json(output, {
        "schema_version": 1,
        "teacher_training_data_access": "unavailable",
        "dataset_policy": str(Path(dataset_policy).resolve()),
        "coverage_status": "NOT_ASSESSABLE",
        "deployment_domain": {"structure_classes": ["mock-seed-neighborhood"]},
        "dataset_sources": [{
            "category": "proposed_acquisition", "n_parents": len(parents),
            "n_frames": len(frames), "fraction": 1.0, "label_sources": ["unlabeled"],
            "evidence_role": "proposed_distillation_structures",
            "statistics": {"kind": "ase", "grouping_key": "parent_structure_id"},
        }],
        "coverage_dimensions": {},
        "replay_policy": {"enabled": False},
        "identified_gaps": ["Teacher training distribution is unavailable"],
        "limitations": ["Mock run does not claim quantitative Teacher-set coverage"],
        "evidence": [evidence_record("dataset_policy", dataset_policy),
                     evidence_record("proposed_distillation_structures", dataset)],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    baseline = sub.add_parser("teacher-baseline")
    baseline.add_argument("teacher_config"); baseline.add_argument("distillation_scope")
    baseline.add_argument("validation_profile")
    baseline.add_argument("evidence")
    baseline.add_argument("output")
    coverage = sub.add_parser("data-coverage")
    coverage.add_argument("dataset_policy"); coverage.add_argument("dataset")
    coverage.add_argument("output")
    args = parser.parse_args()
    if args.action == "teacher-baseline":
        teacher_baseline(args.teacher_config, args.distillation_scope,
                         args.validation_profile, args.evidence, args.output)
    else:
        data_coverage(args.dataset_policy, args.dataset, args.output)


if __name__ == "__main__":
    main()
