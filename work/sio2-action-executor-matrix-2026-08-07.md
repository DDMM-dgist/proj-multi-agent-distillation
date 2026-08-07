# SiO2 action -> trusted executor matrix (2026-08-07)

Honest audit of in-scope producer/analyst actions against EXISTING repository implementations.
Idempotency: all actions use the controller-backed idempotency store. Capability-registry
(out-of-scope) actions are excluded here; see runtimes/pydantic_ai/actions.py.

| action_type | role | status | trusted executor (backing) | input contract | output artifact | validator | approval | dry-run tested | sandbox-primary tested | real exec later |
|---|---|---|---|---|---|---|---|---|---|---|
| inspect_dataset | data-curator | NOT_IMPLEMENTED | - | no standalone dataset-inspection function | - | - | - | yes | no | no |
| summarize_source_categories | data-curator | NOT_IMPLEMENTED | - | only internal data_coverage._source_statistics; no public producer | - | - | - | yes | no | no |
| sample_seed_pool | data-curator | NOT_IMPLEMENTED | - | no deterministic seed-pool sampler exists | - | - | - | yes | no | no |
| reconstruct_lineage | data-curator | NOT_IMPLEMENTED | - | validate_lineage only asserts presence; no reconstruction | - | - | - | yes | no | no |
| generate_group_split | data-curator | AVAILABLE | workflow.steps.split_dataset | dataset,output_dir,manifest | split manifest (sha256-bound) | workflow.steps split integrity | - | yes | yes | no |
| label_with_teacher | data-curator | AVAILABLE_HPC | adapters.acquisition.label_with_teacher | teacher_cfg,structures,out,manifest | labeled extxyz + labeling manifest | labeling manifest integrity | costly_teacher_labeling | gated | no | yes |
| validate_label_preservation | data-curator | NOT_IMPLEMENTED | - | no label-preservation validator | - | - | - | yes | no | no |
| build_dataset_manifest | data-curator | NOT_IMPLEMENTED | - | manifests are byproducts of split/merge; no standalone builder | - | - | - | yes | no | no |
| compare_deployment_coverage | data-curator | AVAILABLE | validation.data_coverage.validate_data_coverage_report | coverage manifest + required categories | validated coverage report | data_coverage validator | - | yes | yes | no |
| detect_duplicates | data-curator | NOT_IMPLEMENTED | - | exact dedup only inside steps.merge_datasets; no standalone action | - | - | - | yes | no | no |
| detect_atomic_overlap | data-curator | NOT_IMPLEMENTED | - | no minimum-distance / atomic-overlap function exists | - | - | - | yes | no | no |
| prepare_student_inputs | ml-trainer | NOT_IMPLEMENTED | - | only internal student._render_simple_nn_config; no standalone action | - | - | - | yes | no | no |
| train_committee | ml-trainer | AVAILABLE_HPC | workflow.steps.train_committee | student_config,dataset,output_dir,manifest | committee manifest + checkpoints | training-completion (n/a) | costly_training | gated | no | yes |
| collect_checkpoints | ml-trainer | NOT_IMPLEMENTED | - | checkpoints are train_committee output; no standalone collector | - | - | - | yes | no | no |
| evaluate_heldout_fidelity | ml-trainer | AVAILABLE_HPC | workflow.steps.evaluate_committee | student_config,committee_manifest,frames | 3-channel fidelity report | four_channel_audit | - | gated | no | yes |
| summarize_seed_variation | ml-trainer | NOT_IMPLEMENTED | - | committee_force_std gives per-seed std; no seed-variation summarizer action | - | - | - | yes | no | no |
| compute_committee_disagreement | ml-trainer | AVAILABLE | adapters.uncertainty.committee_force_std | forces_per_seed[,aggregate] | committee u_per_atom + u_frame | - | - | yes | yes | no |
| validate_training_completion | ml-trainer | NOT_IMPLEMENTED | - | no training-completion validator function | - | - | - | yes | no | no |
| run_teacher_md | simulation | AVAILABLE_HPC | adapters.acquisition.run_teacher_md | cfg,teacher_cfg,seed,out | teacher MD snapshots | - | production_md | gated | no | yes |
| run_student_md | simulation | AVAILABLE_HPC | workflow.steps.run_md / adapters.md_backend.run | md_cfg,student_cfg,checkpoint,... | MD trajectory + manifest | - | production_md | gated | no | yes |
| compute_rdf | simulation | AVAILABLE | validation.structure_dynamics.compute_rdf | frames_path,elements[,r_max,nbins] | partial RDF peaks | - | - | yes | yes | no |
| compute_coordination | simulation | AVAILABLE | validation.structure_dynamics.compute_coordination | frames_path,elements,cutoffs | mean coordination | - | - | yes | yes | no |
| compute_minimum_distance | simulation | NOT_IMPLEMENTED | - | no minimum-distance function in validation/structure_dynamics | - | - | - | yes | no | no |
| detect_force_spike | simulation | NOT_IMPLEMENTED | - | no force-spike detector function | - | - | - | yes | no | no |
| compute_nve_drift | simulation | AVAILABLE | validation.structure_dynamics.compute_nve_drift | energies,timestep_fs,n_atoms | NVE drift meV/atom/ns | - | - | yes | yes | no |
| validate_simulation_completion | simulation | NOT_IMPLEMENTED | - | structure_dynamics CLI emits a report; no standalone completion validator action | - | - | - | yes | no | no |
| submit_scheduler_job | simulation | NOT_IMPLEMENTED | - | typed scheduler bridge interface only; no scheduler backend (approval-gated) | - | - | scheduler_submission | yes | no | no |
| query_scheduler_job | simulation | NOT_IMPLEMENTED | - | no scheduler backend | - | - | - | yes | no | no |
| collect_scheduler_artifact | simulation | NOT_IMPLEMENTED | - | no scheduler backend | - | - | - | yes | no | no |
| compare_force_errors | analyst | AVAILABLE | validation.four_channel_audit.channel | frames_path,ref_prefix,pred_prefix | force error channel metrics | - | - | yes | yes | no |
| compare_energy_errors | analyst | AVAILABLE | validation.four_channel_audit.channel | frames_path,ref_prefix,pred_prefix | energy error channel metrics | - | - | yes | yes | no |
| summarize_committee_disagreement | analyst | AVAILABLE | adapters.uncertainty.committee_force_std | forces_per_seed | u summary | - | - | yes | yes | no |
| compare_rdf | analyst | AVAILABLE | validation.structure_dynamics.compute_rdf | frames_path,elements | RDF peaks | - | - | yes | yes | no |
| compare_coordination | analyst | AVAILABLE | validation.structure_dynamics.compute_coordination | frames_path,elements,cutoffs | coordination | - | - | yes | yes | no |
| fit_nve_drift | analyst | AVAILABLE | validation.structure_dynamics.compute_nve_drift | energies,timestep_fs,n_atoms | NVE drift fit | - | - | yes | yes | no |
| summarize_md_stability | analyst | NOT_IMPLEMENTED | - | composition of nve/msd; no standalone stability-summary action | - | - | - | yes | no | no |
| classify_root_cause | analyst | NOT_IMPLEMENTED | - | root-cause is a reasoning output, not a deterministic executor | - | - | - | yes | no | no |

## Summary
- AVAILABLE (wired to existing code, sandbox-tested): 12
- AVAILABLE_HPC (wired conceptually; real Teacher/Student/MD; approval-gated; NOT run in tests): 5
- NOT_IMPLEMENTED (no backing; never mocked): 20

