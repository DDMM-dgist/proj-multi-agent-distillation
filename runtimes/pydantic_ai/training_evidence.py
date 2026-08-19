"""Deterministic compact training-evidence summary for the training gate.

The training stage's declared output ``artifacts/committee/`` is a multi-seed directory holding
~1,500 intermediate feature-cache files whose full per-file digest is filesystem noise to a Judge
and, even after ``_compact_directory_integrity`` truncation, tells the Judge nothing about *what
was trained*. This module builds a small, semantic, deterministic evidence artifact from
already-produced R31 outputs only (the committee manifest, each seed's SIMPLE_NN ``LOG``, and the
accepted ``dataset_split`` provenance artifacts). It NEVER retrains, relabels, or mutates any
scientific artifact -- it only reads existing logs and re-expresses them compactly so the training
gate's Judges can see: dataset provenance, committee/checkpoint identity, per-seed training
dynamics that actually exist in the logs, and deterministic verification of the checkpoint- and
provenance-integrity claims the Judges are asked to check.

Every field either carries a real recovered value or the explicit sentinel ``NOT_RECORDED`` -- no
value is ever invented. ``build_training_evidence_summary`` is pure (read-only);
``write_training_evidence_summary`` is the only function that writes, and it writes a single new
derived-evidence JSON, not any training artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from workflow.integrity import sha256_file

NOT_RECORDED = "NOT_RECORDED"
TRAINING_EVIDENCE_PROFILE = "training_evidence_summary"
TRAINING_EVIDENCE_FILENAME = "training_evidence_summary.json"

_EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)\s+E RMSE\(T V\)\s+(\S+)\s+(\S+)\s+"
    r"F RMSE\(T V\)\s+(\S+)\s+(\S+)\s+learning_rate:\s+(\S+)")
_TOTAL_EPOCH_RE = re.compile(r"Total\s+tran?ing epoch\s*:\s*(\d+)")
_BEST_EPOCH_RE = re.compile(r"Best loss.*written at\s+(\d+)\s+epoch")
_WALL_TIME_RE = re.compile(r"Total wall time:\s*([\d.]+)\s*s")
_TRAIN_ELAPSED_RE = re.compile(r"Elapsed time in training:\s*([\d.]+)\s*s")
_SEED_RE = re.compile(r"SEED:\s*(\d+)")


def _to_float(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return NOT_RECORDED


def parse_simple_nn_log(text: str) -> dict:
    """Extract only the training-dynamics values that genuinely appear in a SIMPLE_NN ``LOG``.

    Any field absent from the log is returned as ``NOT_RECORDED`` -- never guessed. The
    per-epoch ``E/F RMSE (train, valid)`` lines drive ``epochs_completed`` (the max epoch logged)
    and the final-epoch metrics; ``Total traning epoch`` gives the requested budget (the upstream
    tool's own spelling is matched); ``Best loss ... written at N epoch`` gives the checkpoint's
    best epoch; the two elapsed-time lines give wall/training seconds.
    """
    seed = NOT_RECORDED
    epochs_requested = NOT_RECORDED
    best_epoch = NOT_RECORDED
    wall_time_s = NOT_RECORDED
    training_elapsed_s = NOT_RECORDED
    last_epoch = None
    last_metrics = None
    for line in text.splitlines():
        m = _EPOCH_RE.match(line)
        if m:
            last_epoch = int(m.group(1))
            last_metrics = {
                "final_train_energy_rmse": _to_float(m.group(2)),
                "final_valid_energy_rmse": _to_float(m.group(3)),
                "final_train_force_rmse": _to_float(m.group(4)),
                "final_valid_force_rmse": _to_float(m.group(5)),
                "final_learning_rate": _to_float(m.group(6)),
            }
            continue
        if seed == NOT_RECORDED:
            ms = _SEED_RE.search(line)
            if ms:
                seed = int(ms.group(1))
        if epochs_requested == NOT_RECORDED:
            mt = _TOTAL_EPOCH_RE.search(line)
            if mt:
                epochs_requested = int(mt.group(1))
        if best_epoch == NOT_RECORDED:
            mb = _BEST_EPOCH_RE.search(line)
            if mb:
                best_epoch = int(mb.group(1))
        if wall_time_s == NOT_RECORDED:
            mw = _WALL_TIME_RE.search(line)
            if mw:
                wall_time_s = _to_float(mw.group(1))
        if training_elapsed_s == NOT_RECORDED:
            me = _TRAIN_ELAPSED_RE.search(line)
            if me:
                training_elapsed_s = _to_float(me.group(1))
    epochs_completed = last_epoch if last_epoch is not None else NOT_RECORDED
    metrics = last_metrics or {
        "final_train_energy_rmse": NOT_RECORDED,
        "final_valid_energy_rmse": NOT_RECORDED,
        "final_train_force_rmse": NOT_RECORDED,
        "final_valid_force_rmse": NOT_RECORDED,
        "final_learning_rate": NOT_RECORDED,
    }
    if (isinstance(epochs_completed, int) and isinstance(epochs_requested, int)):
        stopping_reason = ("completed_all_requested_epochs"
                           if epochs_completed >= epochs_requested
                           else f"stopped_at_epoch_{epochs_completed}_of_{epochs_requested}")
    else:
        stopping_reason = NOT_RECORDED
    return {
        "seed": seed,
        "epochs_requested": epochs_requested,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "stopping_reason": stopping_reason,
        "wall_time_s": wall_time_s,
        "training_elapsed_s": training_elapsed_s,
        "rmse_note": ("raw per-epoch train/valid RMSE as emitted verbatim by the SIMPLE_NN LOG; "
                      "these are training-optimization diagnostics only, NOT Student-vs-Teacher "
                      "physical-unit evaluation metrics (those are produced by the evaluation "
                      "stage) -- units are the training tool's own configured units, not asserted "
                      "here"),
        **metrics,
    }


def _committee_dir_sha256(run_dir: Path, committee_dir: Path) -> str:
    """The committee tree's aggregate digest, read from the Controller's ALREADY-registered
    artifact record -- never recomputed here (the tree is ~15 GB; re-hashing it would be a costly
    no-op, and the registered digest is the authoritative value the gate binds anyway)."""
    state_path = run_dir / "manifest.json"
    if not state_path.exists():
        return NOT_RECORDED
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return NOT_RECORDED
    target = str(committee_dir.resolve())
    for record in state.get("artifacts", []):
        if str(Path(record.get("path", "")).resolve()) == target and record.get("sha256"):
            return record["sha256"]
    return NOT_RECORDED


def build_training_evidence_summary(run_dir: str | Path) -> dict:
    """Assemble the compact, deterministic training-evidence summary from existing artifacts.

    Read-only. Reads the committee manifest, each declared seed's ``LOG``, and the accepted
    ``dataset_split`` provenance artifacts; computes deterministic verification outcomes for the
    exact claims the training-gate criteria ask about (committee completeness, checkpoint
    distinctness/existence, dataset-hash binding to the accepted split, protected-reference and
    parent-family leakage). Missing inputs degrade to ``NOT_RECORDED``/``false`` outcomes rather
    than raising, so a partial run still produces auditable evidence.
    """
    run_dir = Path(run_dir).resolve()
    artifacts = run_dir / "artifacts"
    manifest_path = artifacts / "student_committee.manifest.json"
    committee_dir = artifacts / "committee"
    split_manifest_path = artifacts / "dataset" / "split_manifest.json"
    protection_audit_path = artifacts / "dataset_split_protection_audit.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models", []) if isinstance(manifest.get("models"), list) else []
    dataset_integrity = manifest.get("dataset_integrity") or {}
    dataset_path = manifest.get("dataset")

    split_manifest = {}
    if split_manifest_path.exists():
        try:
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            split_manifest = {}
    split_train = ((split_manifest.get("splits") or {}).get("train") or {})
    split_train_sha = split_train.get("sha256")
    split_train_frames = split_train.get("n_frames")
    overlap_checks = split_manifest.get("overlap_checks") or {}

    protection_audit = {}
    if protection_audit_path.exists():
        try:
            protection_audit = json.loads(protection_audit_path.read_text(encoding="utf-8"))
        except Exception:
            protection_audit = {}
    protection_checks = protection_audit.get("checks") or {}

    # -- committee / checkpoints -------------------------------------------------------------
    seeds = []
    checkpoint_hashes = []
    all_checkpoints_exist = True
    committee = []
    for model in models:
        seed = model.get("seed")
        seeds.append(seed)
        ck_path = model.get("path")
        integ = model.get("integrity") or {}
        ck_sha = integ.get("sha256")
        checkpoint_hashes.append(ck_sha)
        exists = bool(ck_path) and Path(ck_path).exists()
        all_checkpoints_exist = all_checkpoints_exist and exists
        committee.append({
            "seed": seed,
            "checkpoint_path": ck_path,
            "checkpoint_sha256": ck_sha or NOT_RECORDED,
            "checkpoint_exists": exists,
            "checkpoint_size": integ.get("size", NOT_RECORDED),
            "dataset_sha256": dataset_integrity.get("sha256", NOT_RECORDED),
        })

    # -- per-seed training dynamics from the real LOGs ---------------------------------------
    dynamics = []
    for model in models:
        seed = model.get("seed")
        log_path = committee_dir / f"seed-{seed}" / "LOG"
        if log_path.exists():
            parsed = parse_simple_nn_log(log_path.read_text(encoding="utf-8", errors="replace"))
            parsed["log_path"] = str(log_path)
            parsed["log_present"] = True
        else:
            parsed = {"seed": seed, "log_present": False, "log_path": str(log_path)}
        dynamics.append(parsed)

    # -- deterministic verification of the exact gate claims ---------------------------------
    n_models = len(models)
    distinct_hashes = [h for h in checkpoint_hashes if h]
    dataset_hash = dataset_integrity.get("sha256")
    dataset_in_run = bool(dataset_path) and str(Path(dataset_path)).startswith(
        str((artifacts / "dataset").resolve()))
    protection_all_pass = bool(protection_checks) and all(
        v == "PASS" for v in protection_checks.values())
    overlap_all_zero = bool(overlap_checks) and all(v == 0 for v in overlap_checks.values())

    verification_outcomes = [
        {"check": "committee_size_is_frozen_4", "expected": 4, "observed": n_models,
         "ok": n_models == 4},
        {"check": "all_checkpoints_registered", "observed": len(distinct_hashes),
         "expected": n_models, "ok": len(distinct_hashes) == n_models and n_models > 0},
        {"check": "all_checkpoint_files_exist", "observed": all_checkpoints_exist,
         "ok": all_checkpoints_exist and n_models > 0},
        {"check": "checkpoint_hashes_distinct",
         "observed": len(set(distinct_hashes)), "expected": n_models,
         "ok": len(set(distinct_hashes)) == n_models and n_models > 0},
        {"check": "training_dataset_hash_matches_accepted_split_train",
         "observed": dataset_hash, "expected": split_train_sha,
         "ok": bool(dataset_hash) and dataset_hash == split_train_sha},
        {"check": "training_dataset_belongs_to_this_run",
         "observed": dataset_path, "ok": dataset_in_run},
        {"check": "protected_reference_overlap_zero",
         "observed": dict(protection_checks), "ok": protection_all_pass},
        {"check": "train_val_test_parent_family_leakage_zero",
         "observed": dict(overlap_checks), "ok": overlap_all_zero},
    ]

    return {
        "schema_version": 1,
        "profile": TRAINING_EVIDENCE_PROFILE,
        "run_id": manifest.get("run_id") or split_manifest.get("run_id") or run_dir.name,
        "dataset_provenance": {
            "path": dataset_path,
            "sha256": dataset_hash or NOT_RECORDED,
            "size": dataset_integrity.get("size", NOT_RECORDED),
            "n_frames": split_train_frames if split_train_frames is not None else NOT_RECORDED,
            "producing_stage": "dataset_split",
            "belongs_to_this_run": dataset_in_run,
            "accepted_split_train_sha256": split_train_sha or NOT_RECORDED,
            "parent_family_overlap_checks": dict(overlap_checks) or NOT_RECORDED,
            "protected_reference_checks": dict(protection_checks) or NOT_RECORDED,
            "dataset_split_gate_result": "PASS" if (protection_all_pass and overlap_all_zero)
            else NOT_RECORDED,
        },
        "committee": {
            "n_models": n_models,
            "frozen_n_models": 4,
            "seeds": seeds,
            "student_config": manifest.get("student_config"),
            "student_config_sha256": (manifest.get("student_config_integrity") or {}).get(
                "sha256", NOT_RECORDED),
            "committee_dir_sha256": _committee_dir_sha256(run_dir, committee_dir),
            "members": committee,
        },
        "training_dynamics": dynamics,
        "verification_outcomes": verification_outcomes,
        "all_verifications_passed": all(o["ok"] for o in verification_outcomes),
    }


def write_training_evidence_summary(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Build and write the compact training-evidence summary; return its path.

    This is the ONLY function here that writes, and it writes a single derived-evidence JSON to
    ``artifacts/training_evidence_summary.json`` (never any training artifact)."""
    run_dir = Path(run_dir).resolve()
    summary = build_training_evidence_summary(run_dir)
    out = Path(out_path).resolve() if out_path else (
        run_dir / "artifacts" / TRAINING_EVIDENCE_FILENAME)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
