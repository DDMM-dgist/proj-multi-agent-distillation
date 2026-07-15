"""Run commands stage-by-stage and require a recorded PASS before advancing."""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RunController:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.state_path = self.run_dir / "manifest.json"
        if not self.state_path.exists():
            raise FileNotFoundError(f"run is not initialized: {self.run_dir}")
        self.state = json.loads(self.state_path.read_text())

    @classmethod
    def initialize(cls, workflow_config, run_dir):
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir()
        (run_dir / "artifacts").mkdir()
        (run_dir / "gates").mkdir()
        cfg = yaml.safe_load(Path(workflow_config).read_text())
        snapshot = run_dir / "workflow.yaml"
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
        stages = []
        for item in cfg["stages"]:
            stages.append({"name": item["name"], "status": "pending", "gate": "pending",
                           "command": item.get("command"), "outputs": item.get("outputs", []),
                           "started_at": None, "completed_at": None, "attempts": 0})
        state = {"schema_version": 1, "run_id": cfg["run_id"], "created_at": now(),
                 "updated_at": now(), "workflow_config": str(snapshot), "artifacts": [],
                 "project_dir": str(Path.cwd().resolve()), "events": [], "stages": stages}
        (run_dir / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")
        return cls(run_dir)

    def save(self):
        self.state["updated_at"] = now()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.state_path)

    def stage(self, name):
        for stage in self.state["stages"]:
            if stage["name"] == name:
                return stage
        raise KeyError(f"unknown stage: {name}")

    def _previous_passed(self, name):
        for stage in self.state["stages"]:
            if stage["name"] == name:
                return
            if stage["gate"] != "PASS":
                raise RuntimeError(f"stage {name!r} blocked: {stage['name']!r} gate is {stage['gate']}")

    def run_stage(self, name):
        self._previous_passed(name)
        stage = self.stage(name)
        if not stage["command"]:
            raise ValueError(f"stage {name!r} has no command")
        context = {"run_dir": str(self.run_dir), "artifacts_dir": str(self.run_dir / "artifacts"),
                   "project_dir": self.state["project_dir"]}
        command = [str(x).format(**context) for x in stage["command"]]
        stage.update(status="running", started_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        self.save()
        log_path = self.run_dir / "logs" / f"{name}.attempt-{stage['attempts']}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=self.run_dir, stdout=log, stderr=subprocess.STDOUT)
        stage["completed_at"] = now()
        if result.returncode != 0:
            stage["status"] = "failed"
            self.state["events"].append({"at": now(), "type": "stage_failed", "stage": name,
                                         "returncode": result.returncode, "log": str(log_path)})
            self.save()
            raise RuntimeError(f"stage {name!r} failed; see {log_path}")
        stage["status"] = "completed"
        for relative in stage["outputs"]:
            path = (self.run_dir / relative).resolve()
            if not path.exists():
                stage["status"] = "failed"
                self.save()
                raise FileNotFoundError(f"declared output missing: {path}")
            self.register_artifact(name, path)
        self.save()

    def register_artifact(self, stage, path):
        path = Path(path).resolve()
        record = {"stage": stage, "path": str(path), "size": path.stat().st_size,
                  "sha256": sha256(path), "registered_at": now()}
        self.state["artifacts"].append(record)
        return record

    def complete_external_stage(self, name, artifacts):
        """Register artifacts produced by an agent, scheduler, or external tool."""
        self._previous_passed(name)
        stage = self.stage(name)
        if not artifacts:
            raise ValueError("at least one artifact is required")
        stage.update(status="completed", started_at=stage.get("started_at") or now(),
                     completed_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        for path in artifacts:
            path = Path(path)
            if not path.is_absolute():
                path = self.run_dir / path
            if not path.is_file():
                raise FileNotFoundError(f"external artifact is missing or not a file: {path}")
            self.register_artifact(name, path)
        self.state["events"].append({"at": now(), "type": "external_stage_completed",
                                     "stage": name, "artifacts": [str(x) for x in artifacts]})
        self.save()

    def record_gate(self, name, verdict, evidence=None):
        if verdict not in {"PASS", "REVISE", "FAIL"}:
            raise ValueError("verdict must be PASS, REVISE, or FAIL")
        stage = self.stage(name)
        if stage["status"] != "completed":
            raise RuntimeError("a gate can only judge a completed stage")
        stage["gate"] = verdict
        self.state["events"].append({"at": now(), "type": "gate", "stage": name,
                                     "verdict": verdict, "evidence": evidence})
        self.save()

    def summary(self):
        return [(s["name"], s["status"], s["gate"], s["attempts"]) for s in self.state["stages"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init")
    init.add_argument("workflow_config")
    init.add_argument("run_dir")
    run = sub.add_parser("run-stage")
    run.add_argument("run_dir")
    run.add_argument("stage")
    complete = sub.add_parser("complete-stage")
    complete.add_argument("run_dir")
    complete.add_argument("stage")
    complete.add_argument("--artifact", action="append", required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("run_dir")
    gate.add_argument("stage")
    gate.add_argument("verdict", choices=["PASS", "REVISE", "FAIL"])
    gate.add_argument("--evidence")
    status = sub.add_parser("status")
    status.add_argument("run_dir")
    args = parser.parse_args()
    if args.action == "init":
        controller = RunController.initialize(args.workflow_config, args.run_dir)
    else:
        controller = RunController(args.run_dir)
    if args.action == "run-stage":
        controller.run_stage(args.stage)
    elif args.action == "complete-stage":
        controller.complete_external_stage(args.stage, args.artifact)
    elif args.action == "gate":
        controller.record_gate(args.stage, args.verdict, args.evidence)
    for row in controller.summary():
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
