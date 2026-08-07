# SiO2 action -> trusted executor matrix (2026-08-07, updated)

Every in-scope action is bound; readiness statuses: READY_EXECUTOR / READY_HPC_APPROVAL_GATED /
READY_INTERFACE_BACKEND_NOT_CONFIGURED / READY_REASONING_OUTPUT. No in-scope action is
NOT_IMPLEMENTED. Idempotency: controller-backed for all.

| action_type | role | status | trusted executor (backing) | input contract | output artifact | validator | approval | real exec later |
|---|---|---|---|---|---|---|---|---|
| inspect_dataset | data-curator | READY_EXECUTOR | ase.io.read + metadata | frames_path | dataset summary | - | - | no |
| summarize_source_categories | data-curator | READY_EXECUTOR | frame metadata (cf. validation.data_coverage) | frames_path[,category_key] | category counts + fractions | - | - | no |
| sample_seed_pool | data-curator | READY_EXECUTOR | deterministic policy seed_pool_v1 | frames_path,count,seed | selection manifest (ids + hashes) | - | - | no |
| reconstruct_lineage | data-curator | READY_EXECUTOR | adapters.acquisition.validate_lineage grouping | frames_path[,group_key] | lineage groups | - | - | no |
| generate_group_split | data-curator | READY_EXECUTOR | workflow.steps.split_dataset | dataset,output_dir,manifest | split manifest (sha256-bound) | workflow.steps split integrity | - | no |
| label_with_teacher | data-curator | READY_HPC_APPROVAL_GATED | adapters.acquisition.label_with_teacher | teacher_cfg,structures,out,manifest | labeled extxyz + labeling manifest | labeling manifest integrity | costly_teacher_labeling | yes |
| validate_label_preservation | data-curator | READY_EXECUTOR | ase.io.read + acquisition.validate_lineage | labeled_path[,n_source_frames] | label-preservation report | artifact completeness | - | no |
| build_dataset_manifest | data-curator | READY_EXECUTOR | workflow.integrity.artifact_digest | dataset[,manifest_path] | hash-bound dataset manifest | - | - | no |
| compare_deployment_coverage | data-curator | READY_EXECUTOR | validation.data_coverage.validate_data_coverage_report | manifest_path[,required_source_categories] | validated coverage report | data_coverage validator | - | no |
| detect_duplicates | data-curator | READY_EXECUTOR | workflow.steps._structure_fingerprint | frames_path | duplicate indices | - | - | no |
| detect_atomic_overlap | data-curator | READY_EXECUTOR | ASE get_all_distances(mic=True) | frames_path[,min_distance_threshold] | overlapping frame indices | - | - | no |
| prepare_student_inputs | ml-trainer | READY_EXECUTOR | adapters.student._render_simple_nn_config | student_config,out_dir | rendered SIMPLE-NN config | - | - | no |
| train_committee | ml-trainer | READY_HPC_APPROVAL_GATED | workflow.steps.train_committee | student_config,dataset,output_dir,manifest | committee manifest + checkpoints | - | costly_training | yes |
| collect_checkpoints | ml-trainer | READY_EXECUTOR | committee manifest convention | committee_manifest | checkpoint paths + integrity | - | - | no |
| evaluate_heldout_fidelity | ml-trainer | READY_HPC_APPROVAL_GATED | workflow.steps.evaluate_committee | student_config,committee_manifest,frames | 3-channel fidelity report | four_channel_audit | - | yes |
| summarize_seed_variation | ml-trainer | READY_EXECUTOR | adapters.uncertainty.committee_force_std | forces_per_seed | seed-variation summary | - | - | no |
| compute_committee_disagreement | ml-trainer | READY_EXECUTOR | adapters.uncertainty.committee_force_std | forces_per_seed[,aggregate] | committee u_per_atom + u_frame | - | - | no |
| validate_training_completion | ml-trainer | READY_EXECUTOR | workflow.integrity.artifact_digest | committee_manifest[,expected_seeds] | training-completeness report | artifact completeness | - | no |
| run_teacher_md | simulation | READY_HPC_APPROVAL_GATED | adapters.acquisition.run_teacher_md | cfg,teacher_cfg,seed,out | teacher MD snapshots | - | production_md | yes |
| run_student_md | simulation | READY_HPC_APPROVAL_GATED | workflow.steps.run_md / adapters.md_backend.run | md_cfg,student_cfg,checkpoint,... | MD trajectory + manifest | - | production_md | yes |
| compute_rdf | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_rdf | frames_path,elements[,r_max,nbins] | partial RDF peaks | - | - | no |
| compute_coordination | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_coordination | frames_path,elements,cutoffs | mean coordination | - | - | no |
| compute_minimum_distance | simulation | READY_EXECUTOR | ASE get_all_distances(mic=True) | frames_path | min distance per frame (A) | - | - | no |
| detect_force_spike | simulation | READY_EXECUTOR | ASE forces + norm | frames_path[,force_threshold] | force-spike frame indices (eV/A) | - | - | no |
| compute_nve_drift | simulation | READY_EXECUTOR | validation.structure_dynamics.compute_nve_drift | energies,timestep_fs,n_atoms | NVE drift meV/atom/ns | - | - | no |
| validate_simulation_completion | simulation | READY_EXECUTOR | artifact existence + finiteness | md_manifest[,trajectory_path,energies] | simulation-completeness report | artifact completeness | - | no |
| submit_scheduler_job | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | SchedulerSubmissionProposal (protocol+config hash, idempotency) | job identity / status / collected artifact reference | - | scheduler_submission | yes |
| query_scheduler_job | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | job identity | job identity / status / collected artifact reference | - | - | yes |
| collect_scheduler_artifact | simulation | READY_INTERFACE_BACKEND_NOT_CONFIGURED | scheduler interface (no HPC backend configured) | job identity -> artifact reference | job identity / status / collected artifact reference | - | - | yes |
| compare_force_errors | analyst | READY_EXECUTOR | validation.four_channel_audit.channel | frames_path,ref_prefix,pred_prefix | force error channel metrics | - | - | no |
| compare_energy_errors | analyst | READY_EXECUTOR | validation.four_channel_audit.channel | frames_path,ref_prefix,pred_prefix | energy error channel metrics | - | - | no |
| summarize_committee_disagreement | analyst | READY_EXECUTOR | adapters.uncertainty.committee_force_std | forces_per_seed | u summary | - | - | no |
| compare_rdf | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_rdf | frames_path,elements | RDF peaks | - | - | no |
| compare_coordination | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_coordination | frames_path,elements,cutoffs | coordination | - | - | no |
| fit_nve_drift | analyst | READY_EXECUTOR | validation.structure_dynamics.compute_nve_drift | energies,timestep_fs,n_atoms | NVE drift fit | - | - | no |
| summarize_md_stability | analyst | READY_EXECUTOR | compose NVE drift + min distance (validation.structure_dynamics) | energies|frames_path | MD-stability summary | - | - | no |
| classify_root_cause | analyst | READY_REASONING_OUTPUT | Analyst typed reasoning output (not a deterministic executor) | deterministic analysis artifacts | RootCauseClassification (typed) | - | - | no |

## Summary
- READY_EXECUTOR: 28
- READY_HPC_APPROVAL_GATED: 5
- READY_INTERFACE_BACKEND_NOT_CONFIGURED: 3
- READY_REASONING_OUTPUT: 1
