"""Run commands stage-by-stage and require a recorded PASS before advancing."""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from workflow.integrity import artifact_digest, sha256_file, verify_artifact


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


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
        workflow_config = Path(workflow_config).resolve()
        snapshot = run_dir / "workflow.yaml"
        snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
        snapshot_dir = run_dir / "inputs"
        snapshot_dir.mkdir()
        input_records = []
        for raw in cfg.get("inputs", []):
            spec = raw if isinstance(raw, dict) else {"path": raw, "copy": True}
            source = Path(str(spec["path"]).format(project_dir=str(Path.cwd().resolve())))
            if not source.is_absolute():
                source = (workflow_config.parent / source).resolve()
            if not source.exists():
                raise FileNotFoundError(f"declared workflow input is missing: {source}")
            source_integrity = artifact_digest(source)
            destination = None
            if spec.get("copy", True):
                if not source.is_file():
                    raise ValueError("directory inputs must use copy: false and are hash-bound in place")
                destination = snapshot_dir / f"{len(input_records):03d}-{source.name}"
                shutil.copy2(source, destination)
            input_records.append({"source": str(source),
                                  "snapshot": str(destination) if destination else None,
                                  "copy": bool(spec.get("copy", True)),
                                  "source_integrity": source_integrity,
                                  "size": source_integrity["size"],
                                  "sha256": source_integrity["sha256"],
                                  "source_sha256": source_integrity["sha256"]})
        stages = []
        names = [item.get("name") for item in cfg.get("stages", [])]
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("workflow stages must have unique non-empty names")
        for item in cfg["stages"]:
            stages.append({"name": item["name"], "status": "pending", "gate": "pending",
                           "command": item.get("command"), "outputs": item.get("outputs", []),
                           "env": item.get("env"),
                           "started_at": None, "completed_at": None, "attempts": 0})
        state = {"schema_version": 2, "run_id": cfg["run_id"], "created_at": now(),
                 "updated_at": now(), "workflow_config": str(snapshot), "artifacts": [],
                 "project_dir": str(Path.cwd().resolve()), "inputs": input_records,
                 "events": [], "stages": stages}
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
            self.verify_stage_artifacts(stage["name"])

    def verify_inputs(self):
        for record in self.state.get("inputs", []):
            source = Path(record["source"])
            if record.get("snapshot"):
                snapshot = Path(record["snapshot"])
                if not snapshot.is_file() or sha256_file(snapshot) != record["sha256"]:
                    raise RuntimeError(f"run input snapshot integrity check failed: {snapshot}")
            try:
                verify_artifact(source, record.get("source_integrity", {"kind": "file",
                                                                        "size": record["size"],
                                                                        "sha256": record["source_sha256"]}))
            except (FileNotFoundError, RuntimeError):
                raise RuntimeError(f"declared workflow input changed after initialization: {source}")

    def rebind_inputs(self):
        """Explicitly accept changed inputs and invalidate all prior stage results."""
        revisions = sum(1 for event in self.state["events"] if event["type"] == "inputs_rebound") + 1
        revision_dir = self.run_dir / "inputs" / f"revision-{revisions:03d}"
        revision_dir.mkdir(parents=True, exist_ok=False)
        changes = []
        for index, record in enumerate(self.state.get("inputs", [])):
            source = Path(record["source"])
            integrity = artifact_digest(source)
            old_sha = record.get("source_integrity", {}).get("sha256", record["source_sha256"])
            snapshot = None
            if record.get("copy", True):
                if not source.is_file():
                    raise ValueError("copied input became a directory; declare it with copy: false")
                snapshot = revision_dir / f"{index:03d}-{source.name}"
                shutil.copy2(source, snapshot)
            record.update(snapshot=str(snapshot) if snapshot else None,
                          source_integrity=integrity, size=integrity["size"],
                          sha256=integrity["sha256"], source_sha256=integrity["sha256"])
            changes.append({"source": str(source), "old_sha256": old_sha,
                            "new_sha256": integrity["sha256"]})
        if self.state["stages"]:
            self.invalidate_from(self.state["stages"][0]["name"], include_stage=True)
        self.state["events"].append({"at": now(), "type": "inputs_rebound",
                                     "revision": revisions, "changes": changes})
        self.save()
        return changes

    def _stage_index(self, name):
        return next(i for i, stage in enumerate(self.state["stages"]) if stage["name"] == name)

    def invalidate_from(self, name, include_stage=False):
        """Invalidate stale downstream state and remove its artifact records."""
        start = self._stage_index(name) + (0 if include_stage else 1)
        affected = {s["name"] for s in self.state["stages"][start:]}
        if not affected:
            return
        self.quarantine_artifacts(affected)
        for stage in self.state["stages"][start:]:
            stage.update(status="pending", gate="pending", started_at=None, completed_at=None)
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] not in affected]
        self.state["events"].append({"at": now(), "type": "downstream_invalidated",
                                     "after": name, "stages": sorted(affected)})

    def quarantine_artifacts(self, stage_names):
        """Move invalidated run-local outputs aside so they cannot be re-registered by accident."""
        records = [a for a in self.state["artifacts"] if a["stage"] in set(stage_names)]
        if not records:
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for record in sorted(records, key=lambda a: len(Path(a["path"]).parts)):
            source = Path(record["path"])
            if not source.exists() or not source.is_relative_to(self.run_dir):
                continue
            destination = self.run_dir / "stale" / stamp / record["stage"] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.move(str(source), str(destination))

    def stage_artifacts(self, name):
        return [a for a in self.state["artifacts"] if a["stage"] == name]

    def verify_stage_artifacts(self, name):
        records = self.stage_artifacts(name)
        if not records:
            raise RuntimeError(f"stage {name!r} has no registered artifacts")
        for record in records:
            path = Path(record["path"])
            try:
                verify_artifact(path, record)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(f"artifact integrity check failed for stage {name!r}: {path}") from exc
        return records

    def run_stage(self, name):
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        if not stage["command"]:
            raise ValueError(f"stage {name!r} has no command")
        self.invalidate_from(name)
        self.quarantine_artifacts({name})
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        context = {"run_dir": str(self.run_dir), "artifacts_dir": str(self.run_dir / "artifacts"),
                   "project_dir": self.state["project_dir"], "python": sys.executable}
        command = [str(x).format(**context) for x in stage["command"]]
        if stage.get("env"):
            if command and Path(command[0]).resolve() == Path(sys.executable).resolve():
                command[0] = "python"
            command = ["conda", "run", "--no-capture-output", "-n", stage["env"], *command]
        stage.update(status="running", started_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        self.save()
        log_path = self.run_dir / "logs" / f"{name}.attempt-{stage['attempts']}.log"
        environment = os.environ.copy()
        project_dir = self.state["project_dir"]
        environment["PYTHONPATH"] = project_dir + os.pathsep + environment.get("PYTHONPATH", "")
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=self.run_dir, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT)
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
        digest = artifact_digest(path)
        record = {"stage": stage, "path": str(path), **digest, "registered_at": now()}
        self.state["artifacts"].append(record)
        return record

    def complete_external_stage(self, name, artifacts):
        """Register artifacts produced by an agent, scheduler, or external tool."""
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        if not artifacts:
            raise ValueError("at least one artifact is required")
        self.invalidate_from(name)
        self.quarantine_artifacts({name})
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        stage.update(status="completed", started_at=stage.get("started_at") or now(),
                     completed_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        for path in artifacts:
            path = Path(path)
            if not path.is_absolute():
                path = self.run_dir / path
            if not path.exists():
                raise FileNotFoundError(f"external artifact is missing: {path}")
            self.register_artifact(name, path)
        self.state["events"].append({"at": now(), "type": "external_stage_completed",
                                     "stage": name, "artifacts": [str(x) for x in artifacts]})
        self.save()

    def _validate_vote_bundle(self, name, votes_path):
        bundle = json.loads(Path(votes_path).read_text())
        criteria = bundle.get("criteria")
        votes = bundle.get("votes")
        if bundle.get("stage", bundle.get("gate")) != name:
            raise ValueError("vote bundle gate/stage does not match the controller stage")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("vote bundle must contain non-empty criteria")
        if not isinstance(votes, list) or len(votes) != 3:
            raise ValueError("exactly three judge votes are required")
        verdicts = []
        judge_ids = set()
        for index, vote in enumerate(votes, 1):
            verdict = vote.get("verdict")
            checked = vote.get("criteria_checked")
            judge_id = str(vote.get("judge_id", vote.get("id", index)))
            if judge_id in judge_ids:
                raise ValueError("judge identifiers must be unique")
            judge_ids.add(judge_id)
            if verdict not in {"PASS", "REVISE", "FAIL"}:
                raise ValueError("judge vote has an invalid verdict")
            if not isinstance(checked, list) or len(checked) != len(criteria):
                raise ValueError("every judge must report every criterion")
            if [item.get("criterion") for item in checked] != criteria:
                raise ValueError("judge criteria must exactly match the ordered gate criteria")
            if verdict == "PASS" and not all(item.get("ok") is True for item in checked):
                raise ValueError("a PASS vote requires every criterion to be explicitly true")
            verdicts.append(verdict)
        decision = "FAIL" if "FAIL" in verdicts else ("PASS" if verdicts == ["PASS"] * 3 else "REVISE")
        if bundle.get("decision") != decision:
            raise ValueError("vote bundle decision does not match the recomputed decision")
        expected = {a["path"]: a["sha256"] for a in self.verify_stage_artifacts(name)}
        if bundle.get("artifact_sha256") != expected:
            raise ValueError("vote bundle artifact hashes do not match current registered artifacts")
        return decision, bundle

    def record_gate(self, name, verdict=None, evidence=None, votes_path=None):
        bundle = None
        if votes_path:
            verdict, bundle = self._validate_vote_bundle(name, votes_path)
        elif verdict == "PASS":
            raise ValueError("PASS requires --votes with three validated judge votes")
        if verdict not in {"PASS", "REVISE", "FAIL"}:
            raise ValueError("verdict must be PASS, REVISE, or FAIL")
        stage = self.stage(name)
        if stage["status"] != "completed":
            raise RuntimeError("a gate can only judge a completed stage")
        stage["gate"] = verdict
        if verdict != "PASS":
            self.invalidate_from(name)
        saved_votes = None
        if votes_path:
            saved_votes = self.run_dir / "gates" / f"{name}.votes.json"
            saved_votes.write_text(json.dumps(bundle, indent=2) + "\n")
        self.state["events"].append({"at": now(), "type": "gate", "stage": name,
                                     "verdict": verdict, "evidence": evidence,
                                     "votes": str(saved_votes) if saved_votes else None,
                                     "vote_bundle": bundle})
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
    gate.add_argument("verdict", nargs="?", choices=["REVISE", "FAIL"])
    gate.add_argument("--evidence")
    gate.add_argument("--votes")
    rebind = sub.add_parser("rebind-inputs")
    rebind.add_argument("run_dir")
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
        controller.record_gate(args.stage, args.verdict, args.evidence, args.votes)
    elif args.action == "rebind-inputs":
        controller.rebind_inputs()
    for row in controller.summary():
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
