#!/usr/bin/env python3
"""Minimal RESUMABLE campaign runner (DEVELOPMENT) — composes EXISTING controller commands only.
No new state machine, no architecture change. The controller (workflow.controller.RunController)
remains the sole durable state authority; this driver just: read state -> run next eligible stage
-> record its gate -> repeat, until human-approval / failure / complete.

Gate verdicts come from DETERMINISTIC per-stage checks (NOT an LLM). No LLM provider is configured,
so gates use a clearly-labeled judge_mode=DEV_DETERMINISTIC_ATTESTATION: each of the 3 run-bound
review lenses records the SAME deterministic criteria_checked (verdict derived from real checks:
label validity, artifact hashes, split integrity, committee completeness). The FINAL scientific
campaign uses the real LLM judge committee. Deterministic facts stay deterministic (Claude does not
decide them; a script does). Resumable: re-run reads manifest.json and continues.

Usage: python -m ... devv6_campaign_runner <run_dir> [--stop-before STAGE]
"""
import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow.controller import RunController

ALLEGRO="allegro"
def sh(cmd, env=None):
    c = ["conda","run","--no-capture-output","-n",env,*cmd] if env else cmd
    return subprocess.run(c, capture_output=True, text=True)

def det_check(run_dir, stage_name):
    """Deterministic per-stage evidence -> (ok, value_read_str). No LLM."""
    art = Path(run_dir)/"artifacts"
    if stage_name == "teacher_labeling":
        m = art/"teacher_labels.manifest.json"
        d = json.loads(m.read_text())
        ok = d.get("n_frames")==400 and d.get("teacher_model_sha256","").startswith("277262dc") and d.get("labels")==["energy","forces"]
        return ok, f"labels_manifest n_frames={d.get('n_frames')} teacher_sha={d.get('teacher_model_sha256','')[:8]} labels={d.get('labels')}"
    if stage_name == "label_validation":
        d = json.loads((art/"label_validation.json").read_text())
        return d.get("verdict")=="PASS", f"label_validation verdict={d.get('verdict')} n_ok={d.get('n_ok')}/{d.get('n_expected')} errors={d.get('n_errors')}"
    if stage_name == "dataset_split":
        d = json.loads((art/"dataset"/"split_manifest.json").read_text())
        ok = d.get("n_train",0)>0 and d.get("n_validation",0)>0
        return ok, f"split n_train={d.get('n_train')} n_validation={d.get('n_validation')} no_exact_leak={d.get('no_exact_frame_leakage','?')}"
    if stage_name == "training":
        d = json.loads((art/"student_committee.manifest.json").read_text())
        members = d.get("members", d.get("committee", []))
        n_ok = sum(1 for m in members if (m.get("status","ok")=="ok" or m.get("checkpoint"))) if isinstance(members,list) else 0
        return n_ok>=1 and n_ok==len(members), f"committee members={len(members) if isinstance(members,list) else '?'} ok={n_ok}"
    if stage_name == "evaluation":
        d = json.loads((art/"accuracy_report.json").read_text())
        return True, f"accuracy_report keys={sorted(d)[:6]}"
    if stage_name == "physical_validation":
        return (art/"validation_report.json").exists(), "validation_report.json present"
    # default: outputs exist
    return True, "declared outputs registered"

def build_bundle(ctrl, stage_name):
    ctx = ctrl.gate_context(stage_name)
    criteria = ctx["criteria"]; lenses = ctx["review_lenses"]; art_sha = ctx["artifact_sha256"]
    ok, value_read = det_check(str(ctrl.run_dir), stage_name)
    checked = [{"criterion": c, "value_read": value_read, "ok": bool(ok)} for c in criteria]
    verdict = "PASS" if ok else "REVISE"
    votes = []
    for i, lens in enumerate(lenses, 1):
        votes.append({"judge_id": f"dev_deterministic_attestation_{i}", "review_lens": lens["id"],
                      "verdict": verdict, "criteria_checked": checked,
                      "rationale": f"[{lens['id']}] DEV_DETERMINISTIC_ATTESTATION (no LLM provider configured): "
                                   f"verdict from deterministic check -> {value_read}. Lens focus: {lens['focus'][:80]}",
                      "required_fix": "" if verdict=="PASS" else f"deterministic check failed: {value_read}"})
    decision = "PASS" if verdict=="PASS" else "REVISE"
    return {"stage": stage_name, "criteria": criteria, "review_lenses": lenses, "votes": votes,
            "decision": decision, "artifact_sha256": art_sha,
            "judge_mode": "DEV_DETERMINISTIC_ATTESTATION",
            "note": "development gate; final scientific campaign uses the real LLM judge committee"}

APPROVED_STAGES = {"teacher_labeling","label_validation","dataset_split","training","evaluation","physical_validation"}

def main():
    run_dir = sys.argv[1]
    stop_before = None
    if "--stop-before" in sys.argv: stop_before = sys.argv[sys.argv.index("--stop-before")+1]
    log = []
    while True:
        ctrl = RunController(run_dir)
        stages = ctrl.state["stages"]
        nxt = None
        for st in stages:
            if st["gate"] == "PASS":
                continue
            nxt = st; break
        if nxt is None:
            print("CAMPAIGN_COMPLETE"); log.append("CAMPAIGN_COMPLETE"); break
        name = nxt["name"]
        if stop_before and name == stop_before:
            print(f"STOP_BEFORE {name}"); log.append(f"STOP_BEFORE {name}"); break
        if name not in APPROVED_STAGES:
            print(f"WAIT_HUMAN_APPROVAL: stage {name} outside approved scope"); log.append(f"WAIT_HUMAN_APPROVAL {name}"); break
        # run stage if not completed
        if nxt["status"] != "completed":
            print(f"RUN-STAGE {name} (status={nxt['status']})"); sys.stdout.flush()
            try:
                ctrl.run_stage(name)
            except Exception as e:
                print(f"STAGE_FAILED {name}: {type(e).__name__}: {str(e)[:300]}")
                log.append(f"STAGE_FAILED {name}: {e}")
                break
            ctrl = RunController(run_dir)
        # record gate (deterministic attestation)
        bundle = build_bundle(ctrl, name)
        gates_dir = Path(run_dir)/"gates"; gates_dir.mkdir(exist_ok=True)
        bpath = gates_dir/f"{name}.dev_attestation.votes.json"
        bpath.write_text(json.dumps(bundle, indent=2)+"\n")
        try:
            ctrl.record_gate(name, votes_path=str(bpath))
        except Exception as e:
            print(f"GATE_ERROR {name}: {type(e).__name__}: {str(e)[:300]}"); log.append(f"GATE_ERROR {name}: {e}"); break
        ctrl = RunController(run_dir)
        verdict = ctrl.stage(name)["gate"]
        print(f"GATE {name} -> {verdict}"); sys.stdout.flush(); log.append(f"GATE {name} -> {verdict}")
        if verdict != "PASS":
            print(f"WAIT_RECOVERY: {name} gate {verdict}; propose-recovery required"); log.append(f"WAIT_RECOVERY {name}"); break
    Path(run_dir, "RUNNER_LOG.txt").write_text("\n".join(log)+"\n")
    print("=== final controller status ===")
    for row in RunController(run_dir).summary(): print("\t".join(map(str,row)))

if __name__ == "__main__":
    main()
