"""Run commands stage-by-stage and require a recorded PASS before advancing."""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from workflow.integrity import artifact_digest, sha256_file, verify_artifact
from workflow.contracts import validate_md_manifest, validate_validation_manifest


RECOVERY_CATEGORIES = {
    "data_quality", "dataset_coverage", "student_fidelity", "teacher_applicability",
    "physical_validation", "simulation_protocol", "evidence_gap", "other",
}
RECOVERY_AGENTS = {"data-curator", "ml-trainer", "simulation", "analyst", "director"}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def format_context(value, context):
    """Format controller placeholders recursively in contract options."""
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [format_context(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_context(item, context) for key, item in value.items()}
    return value


def git_revision(project_dir):
    """Return the Git commit and a content hash for any tracked/untracked changes."""
    project_dir = Path(project_dir).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_dir), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "-C", str(project_dir), "diff", "--binary", "HEAD"], check=True,
            capture_output=True,
        ).stdout
        untracked_raw = subprocess.run(
            ["git", "-C", str(project_dir), "ls-files", "--others", "--exclude-standard", "-z"],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "git_commit": None, "dirty": None, "diff_sha256": None}
    untracked = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        path = project_dir / os.fsdecode(raw)
        if path.exists():
            untracked.append({"path": os.fsdecode(raw), **artifact_digest(path)})
    dirty = bool(status.strip())
    payload = diff + status + json.dumps(untracked, sort_keys=True, default=str).encode()
    return {"available": True, "git_commit": commit, "dirty": dirty,
            "diff_sha256": hashlib.sha256(payload).hexdigest() if dirty else None}


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
        workflow_config = Path(workflow_config).resolve()
        cfg = yaml.safe_load(workflow_config.read_text())
        if (not isinstance(cfg, dict) or not isinstance(cfg.get("run_id"), str) or
                not cfg["run_id"].strip()):
            raise ValueError("workflow config requires a non-empty run_id")
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        project_dir = Path.cwd().resolve()
        prepared_inputs = []
        for raw in cfg.get("inputs", []):
            spec = raw if isinstance(raw, dict) else {"path": raw, "copy": True}
            if not isinstance(spec.get("path"), (str, os.PathLike)):
                raise ValueError("every workflow input requires a path")
            source = Path(str(spec["path"]).format(project_dir=str(project_dir)))
            if not source.is_absolute():
                source = (workflow_config.parent / source).resolve()
            if not source.exists():
                raise FileNotFoundError(f"declared workflow input is missing: {source}")
            source_integrity = artifact_digest(source)
            if spec.get("copy", True):
                if not source.is_file():
                    raise ValueError("directory inputs must use copy: false and are hash-bound in place")
            prepared_inputs.append((source, bool(spec.get("copy", True)), source_integrity))
        stages = []
        raw_stages = cfg.get("stages", [])
        if not isinstance(raw_stages, list) or any(not isinstance(item, dict)
                                                   for item in raw_stages):
            raise ValueError("workflow stages must be a list of mappings")
        names = [item.get("name") for item in raw_stages]
        if not names or any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("workflow stages must have unique non-empty names")
        for item in raw_stages:
            command = item.get("command")
            if command is not None and (not isinstance(command, list) or not command):
                raise ValueError(f"stage {item['name']!r} command must be a non-empty list or null")
            outputs = item.get("outputs", [])
            if (not isinstance(outputs, list) or
                    any(not isinstance(value, str) or not value.strip() for value in outputs) or
                    len(outputs) != len(set(outputs))):
                raise ValueError(f"stage {item['name']!r} outputs must be unique non-empty paths")
            for value in outputs:
                output = Path(value)
                if output.is_absolute() or ".." in output.parts:
                    raise ValueError(f"stage {item['name']!r} output must stay inside the run: {value}")
            env = item.get("env")
            if env is not None and (not isinstance(env, str) or not env.strip()):
                raise ValueError(f"stage {item['name']!r} env must be a non-empty string")
            gate_config = item.get("gate")
            if gate_config is not None and not isinstance(gate_config, dict):
                raise ValueError(f"stage {item['name']!r} gate must be a mapping")
            gate_criteria = (gate_config or {}).get("criteria")
            if (gate_criteria is not None and
                    (not isinstance(gate_criteria, list) or not gate_criteria or
                     any(not isinstance(value, str) or not value.strip()
                         for value in gate_criteria) or
                     len(gate_criteria) != len(set(gate_criteria)))):
                raise ValueError(
                    f"stage {item['name']!r} gate criteria must be unique non-empty strings"
                )
            contract = item.get("contract")
            if contract is not None and not isinstance(contract, dict):
                raise ValueError(f"stage {item['name']!r} contract must be a mapping")
            if contract is not None:
                contract_kind = contract.get("kind")
                required_fields = {
                    "md_manifest": ("manifest", "committee_manifest"),
                    "validation_manifest": ("manifest", "validator"),
                }
                if contract_kind not in required_fields:
                    raise ValueError(
                        f"stage {item['name']!r} has unknown contract kind: {contract_kind!r}"
                    )
                missing = [field for field in required_fields[contract_kind]
                           if not isinstance(contract.get(field), str) or
                           not contract[field].strip()]
                if missing:
                    raise ValueError(
                        f"stage {item['name']!r} contract is missing: " + ", ".join(missing)
                    )
                if (contract_kind == "validation_manifest" and
                        "." not in contract["validator"]):
                    raise ValueError("validation contract validator must be a dotted callable path")
                if "options" in contract and not isinstance(contract["options"], dict):
                    raise ValueError("validation contract options must be a mapping")
                required_evidence = contract.get("required_evidence")
                if (required_evidence is not None and
                        (not isinstance(required_evidence, list) or
                         any(not isinstance(role, str) or not role.strip()
                             for role in required_evidence) or
                         len(required_evidence) != len(set(required_evidence)))):
                    raise ValueError("contract required_evidence must list unique non-empty roles")
            stages.append({"name": item["name"], "status": "pending", "gate": "pending",
                           "command": command, "outputs": outputs,
                           "env": env, "contract": contract,
                           "gate_criteria": gate_criteria,
                           "started_at": None, "completed_at": None, "attempts": 0})
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.init-", dir=run_dir.parent))
        try:
            for name in ("logs", "artifacts", "gates", "inputs"):
                (temporary / name).mkdir()
            (temporary / "workflow.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            input_records = []
            for index, (source, copy_input, source_integrity) in enumerate(prepared_inputs):
                destination = None
                if copy_input:
                    temporary_destination = temporary / "inputs" / f"{index:03d}-{source.name}"
                    shutil.copy2(source, temporary_destination)
                    destination = run_dir / "inputs" / temporary_destination.name
                input_records.append({"source": str(source),
                                      "snapshot": str(destination) if destination else None,
                                      "copy": copy_input, "source_integrity": source_integrity,
                                      "size": source_integrity["size"],
                                      "sha256": source_integrity["sha256"],
                                      "source_sha256": source_integrity["sha256"]})
            created_at = now()
            state = {"schema_version": 5, "run_id": cfg["run_id"], "created_at": created_at,
                     "updated_at": created_at, "workflow_config": str(run_dir / "workflow.yaml"),
                     "artifacts": [], "project_dir": str(project_dir), "inputs": input_records,
                     "code_revision": git_revision(project_dir), "events": [], "stages": stages,
                     "iterations": [{"id": 1, "parent_iteration": None, "status": "active",
                                     "started_at": created_at, "trigger": None}],
                     "recoveries": [], "pending_recovery": None}
            (temporary / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")
            temporary.rename(run_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
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
        expected_revision = self.state.get("code_revision")
        if expected_revision and expected_revision.get("available"):
            current_revision = git_revision(self.state["project_dir"])
            if current_revision != expected_revision:
                raise RuntimeError("project code changed after run initialization; start a new run")
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
        self._ensure_no_pending_recovery()
        revisions = sum(1 for event in self.state["events"] if event["type"] == "inputs_rebound") + 1
        revision_dir = self.run_dir / "inputs" / f"revision-{revisions:03d}"
        if revision_dir.exists():
            raise FileExistsError(f"input revision already exists: {revision_dir}")
        prepared = []
        for index, record in enumerate(self.state.get("inputs", [])):
            source = Path(record["source"])
            integrity = artifact_digest(source)
            if record.get("copy", True) and not source.is_file():
                raise ValueError("copied input became a directory; declare it with copy: false")
            prepared.append((index, record, source, integrity))

        temporary = Path(tempfile.mkdtemp(prefix=f".revision-{revisions:03d}-",
                                          dir=self.run_dir / "inputs"))
        try:
            new_records, changes = [], []
            for index, record, source, integrity in prepared:
                old_snapshot = record.get("snapshot")
                snapshot = None
                if record.get("copy", True):
                    temporary_snapshot = temporary / f"{index:03d}-{source.name}"
                    shutil.copy2(source, temporary_snapshot)
                    snapshot = revision_dir / temporary_snapshot.name
                updated = dict(record)
                updated.update(snapshot=str(snapshot) if snapshot else None,
                               source_integrity=integrity, size=integrity["size"],
                               sha256=integrity["sha256"], source_sha256=integrity["sha256"])
                new_records.append(updated)
                old_sha = record.get("source_integrity", {}).get("sha256",
                                                                  record["source_sha256"])
                changes.append({"source": str(source), "old_sha256": old_sha,
                                "new_sha256": integrity["sha256"],
                                "old_snapshot": old_snapshot,
                                "new_snapshot": str(snapshot) if snapshot else None})
            temporary.rename(revision_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.state["inputs"] = new_records
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

    def quarantine_artifacts(self, stage_names, exclude_paths=None):
        """Move invalidated run-local outputs aside so they cannot be re-registered by accident."""
        excluded = {Path(path).resolve() for path in (exclude_paths or [])}
        records = [a for a in self.state["artifacts"] if a["stage"] in set(stage_names)]
        if not records:
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for record in sorted(records, key=lambda a: len(Path(a["path"]).parts)):
            source = Path(record["path"])
            if (source.resolve() in excluded or not source.exists() or
                    not source.is_relative_to(self.run_dir)):
                continue
            destination = self.run_dir / "stale" / stamp / record["stage"] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.move(str(source), str(destination))

    def quarantine_declared_outputs(self, stage):
        """Move unregistered leftovers from a failed attempt out of the output paths."""
        paths = sorted({(self.run_dir / relative).resolve()
                        for relative in stage.get("outputs", [])},
                       key=lambda path: len(path.parts))
        existing = [path for path in paths
                    if path.exists() and path.is_relative_to(self.run_dir)]
        if not existing:
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        for source in existing:
            if not source.exists():
                continue
            relative = source.relative_to(self.run_dir)
            destination = self.run_dir / "stale" / stamp / stage["name"] / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
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
        self._ensure_no_pending_recovery()
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        if not stage["command"]:
            raise ValueError(f"stage {name!r} has no command")
        self.invalidate_from(name)
        self.quarantine_artifacts({name})
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        self.quarantine_declared_outputs(stage)
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
            try:
                result = subprocess.run(command, cwd=self.run_dir, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                log.write(f"stage launch failed: {exc}\n")
                stage.update(status="failed", completed_at=now())
                self.state["events"].append({"at": now(), "type": "stage_failed",
                                             "stage": name, "returncode": None,
                                             "error": str(exc), "log": str(log_path)})
                self.save()
                raise RuntimeError(f"stage {name!r} could not be launched; see {log_path}") from exc
        stage["completed_at"] = now()
        if result.returncode != 0:
            stage["status"] = "failed"
            self.state["events"].append({"at": now(), "type": "stage_failed", "stage": name,
                                         "returncode": result.returncode, "log": str(log_path)})
            self.save()
            raise RuntimeError(f"stage {name!r} failed; see {log_path}")
        output_paths = []
        for relative in stage["outputs"]:
            path = (self.run_dir / relative).resolve()
            if not path.exists():
                stage["status"] = "failed"
                self.save()
                raise FileNotFoundError(f"declared output missing: {path}")
            output_paths.append(path)
        try:
            self._validate_external_contract(stage, output_paths)
        except Exception as exc:
            stage["status"] = "failed"
            self.state["events"].append({"at": now(), "type": "stage_contract_failed",
                                         "stage": name, "error": str(exc)})
            self.save()
            raise
        stage["status"] = "completed"
        for path in output_paths:
            self.register_artifact(name, path)
        self.save()

    def register_artifact(self, stage, path):
        path = Path(path).resolve()
        digest = artifact_digest(path)
        record = {"stage": stage, "path": str(path), **digest, "registered_at": now()}
        self.state["artifacts"].append(record)
        return record

    def _registered_artifact(self, path):
        path = str(Path(path).resolve())
        matches = [record for record in self.state["artifacts"] if record["path"] == path]
        if len(matches) != 1:
            raise ValueError(f"required upstream artifact is not uniquely registered: {path}")
        verify_artifact(path, matches[0])
        return matches[0]

    def _validation_evidence_allowlist(self, current_artifacts, current_stage=None):
        """Paths that a validation report may cite as run-bound evidence."""
        paths = {Path(path).resolve() for path in current_artifacts}
        for record in self.state.get("inputs", []):
            paths.add(Path(record["source"]).resolve())
            if record.get("snapshot"):
                paths.add(Path(record["snapshot"]).resolve())
        if current_stage is not None:
            upstream = {stage["name"] for stage in
                        self.state["stages"][:self._stage_index(current_stage)]}
            paths.update(Path(record["path"]).resolve()
                         for record in self.state["artifacts"]
                         if record.get("stage") in upstream)
        return sorted(paths)

    def _validate_external_contract(self, stage, artifacts, enforce_required_pass=False):
        contract = stage.get("contract")
        if not contract:
            return None
        context = {"run_dir": str(self.run_dir), "artifacts_dir": str(self.run_dir / "artifacts"),
                   "project_dir": self.state["project_dir"]}
        manifest = Path(str(contract["manifest"]).format(**context))
        if not manifest.is_absolute():
            manifest = self.run_dir / manifest
        manifest = manifest.resolve()
        if manifest not in {Path(path).resolve() for path in artifacts}:
            raise ValueError("external contract manifest must be included in --artifact")
        kind = contract.get("kind")
        if kind == "md_manifest":
            committee = Path(str(contract["committee_manifest"]).format(**context))
            if not committee.is_absolute():
                committee = self.run_dir / committee
            self._registered_artifact(committee)
            return validate_md_manifest(manifest, committee, artifacts,
                                        contract.get("required_evidence"))
        if kind == "validation_manifest":
            return validate_validation_manifest(manifest, contract.get("validator"),
                                                format_context(contract.get("options"), context), artifacts,
                                                self._validation_evidence_allowlist(
                                                    artifacts, stage.get("name")),
                                                enforce_required_pass)
        raise ValueError(f"unknown external stage contract: {kind!r}")

    def complete_external_stage(self, name, artifacts):
        """Register artifacts produced by an agent, scheduler, or external tool."""
        self._ensure_no_pending_recovery()
        self.verify_inputs()
        self._previous_passed(name)
        stage = self.stage(name)
        if not artifacts:
            raise ValueError("at least one artifact is required")
        resolved = []
        for path in artifacts:
            path = Path(path)
            if not path.is_absolute():
                path = self.run_dir / path
            if not path.exists():
                raise FileNotFoundError(f"external artifact is missing: {path}")
            resolved.append(path.resolve())
        submitted = set(resolved)
        declared = {(self.run_dir / relative).resolve() for relative in stage.get("outputs", [])}
        missing_outputs = declared - submitted
        if missing_outputs:
            raise ValueError("external stage is missing declared outputs: " +
                             ", ".join(map(str, sorted(missing_outputs))))
        contract_result = self._validate_external_contract(stage, resolved)
        self.invalidate_from(name)
        self.quarantine_artifacts({name}, exclude_paths=resolved)
        self.state["artifacts"] = [a for a in self.state["artifacts"] if a["stage"] != name]
        stage.update(status="completed", started_at=stage.get("started_at") or now(),
                     completed_at=now(), attempts=stage["attempts"] + 1, gate="pending")
        for path in resolved:
            self.register_artifact(name, path)
        self.state["events"].append({"at": now(), "type": "external_stage_completed",
                                     "stage": name, "artifacts": [str(x) for x in resolved],
                                     "contract": stage.get("contract"),
                                     "contract_validated": contract_result is not None})
        self.save()

    def _validate_vote_bundle(self, name, votes_path):
        bundle = json.loads(Path(votes_path).read_text())
        criteria = bundle.get("criteria")
        votes = bundle.get("votes")
        if bundle.get("stage", bundle.get("gate")) != name:
            raise ValueError("vote bundle gate/stage does not match the controller stage")
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("vote bundle must contain non-empty criteria")
        bound_criteria = self.stage(name).get("gate_criteria")
        if not bound_criteria:
            raise ValueError(
                "Judge PASS/REVISE bundle requires gate.criteria bound at run initialization"
            )
        if criteria != bound_criteria:
            raise ValueError("vote bundle criteria do not match the run-bound gate criteria")
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
        if not expected:
            raise ValueError("a Judge gate requires at least one registered artifact")
        if bundle.get("artifact_sha256") != expected:
            raise ValueError("vote bundle artifact hashes do not match current registered artifacts")
        return decision, bundle

    def gate_context(self, name):
        """Return the verified artifact hashes and run-bound Judge criteria."""
        stage = self.stage(name)
        if stage["status"] != "completed":
            raise RuntimeError("gate context requires a completed stage")
        if not stage.get("gate_criteria"):
            raise ValueError(
                "Judge gate requires gate.criteria bound at run initialization"
            )
        hashes = {record["path"]: record["sha256"]
                  for record in self.verify_stage_artifacts(name)}
        if not hashes:
            raise ValueError("a Judge gate requires at least one registered artifact")
        return {"stage": name, "criteria": list(stage["gate_criteria"]),
                "artifact_sha256": hashes}

    def record_gate(self, name, verdict=None, evidence=None, votes_path=None):
        self._ensure_no_pending_recovery()
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
        if verdict == "PASS":
            self._require_verified_recovery_for_pass(name)
        if verdict == "PASS" and (stage.get("contract") or {}).get("kind") == "validation_manifest":
            self._validate_external_contract(
                stage, [record["path"] for record in self.stage_artifacts(name)],
                enforce_required_pass=True,
            )
        saved_votes = None
        if votes_path:
            iteration_id = self._current_iteration()["id"]
            saved_votes = self.run_dir / "gates" / f"{name}.iteration-{iteration_id:03d}.votes.json"
            saved_votes.write_text(json.dumps(bundle, indent=2) + "\n")
        stage["gate"] = verdict
        if verdict == "PASS":
            iteration = self._current_iteration()
            trigger = iteration.get("trigger")
            if trigger and trigger.get("failed_stage") == name:
                recovery = next(item for item in self.state.get("recoveries", [])
                                if item.get("id") == trigger["recovery_id"])
                recovery.update(status="resolved", resolved_at=now())
                iteration["recovery_execution"]["status"] = "resolved"
        if verdict != "PASS":
            self.invalidate_from(name)
        gate_time = now()
        self.state["events"].append({"at": gate_time, "type": "gate", "stage": name,
                                     "verdict": verdict, "evidence": evidence,
                                     "votes": str(saved_votes) if saved_votes else None,
                                     "vote_bundle": bundle})
        if verdict != "PASS":
            self.state["pending_recovery"] = {
                "status": "required", "failed_stage": name, "verdict": verdict,
                "gate_recorded_at": gate_time,
                "artifact_sha256": {record["path"]: record["sha256"]
                                    for record in self.verify_stage_artifacts(name)},
                "votes_integrity": artifact_digest(saved_votes) if saved_votes else None,
            }
        self.save()

    def _ensure_no_pending_recovery(self):
        pending = self.state.get("pending_recovery")
        if pending:
            raise RuntimeError(
                "a REVISE/FAIL recovery is pending; propose, approve, and start the next iteration"
            )

    def _current_iteration(self):
        iterations = self.state.setdefault("iterations", [])
        if not iterations:
            iterations.append({"id": 1, "parent_iteration": None, "status": "active",
                               "started_at": self.state.get("created_at", now()),
                               "trigger": None})
        return iterations[-1]

    def propose_recovery(self, plan_path):
        """Bind a scientific recovery proposal to the failed gate and its evidence."""
        pending = self.state.get("pending_recovery")
        if not pending or pending.get("status") != "required":
            raise RuntimeError("no REVISE/FAIL gate is waiting for a recovery proposal")
        source = Path(plan_path).resolve()
        plan = json.loads(source.read_text())
        if plan.get("schema_version") != 1:
            raise ValueError("recovery plan requires schema_version=1")
        failed_stage = pending["failed_stage"]
        if plan.get("failed_stage") != failed_stage:
            raise ValueError("recovery plan failed_stage does not match the pending gate")
        category = plan.get("failure_category")
        if category not in RECOVERY_CATEGORIES:
            raise ValueError(f"recovery plan has invalid failure_category: {category!r}")
        for field in ("root_cause", "responsible_agent", "return_stage"):
            if not isinstance(plan.get(field), str) or not plan[field].strip():
                raise ValueError(f"recovery plan requires non-empty {field}")
        if plan["responsible_agent"] not in RECOVERY_AGENTS:
            raise ValueError("recovery responsible_agent is not a registered recovery role")
        try:
            return_index = self._stage_index(plan["return_stage"])
        except StopIteration as exc:
            raise ValueError(f"recovery return_stage is unknown: {plan['return_stage']}") from exc
        if return_index > self._stage_index(failed_stage):
            raise ValueError("recovery return_stage cannot be downstream of the failed stage")
        changes = plan.get("proposed_changes")
        if (not isinstance(changes, list) or not changes or
                any(not isinstance(item, dict) or not isinstance(item.get("type"), str) or
                    not item["type"].strip() for item in changes)):
            raise ValueError("recovery plan requires proposed_changes with non-empty type")
        labeling = plan.get("labeling")
        if (not isinstance(labeling, dict) or
                any(not isinstance(labeling.get(key), bool)
                    for key in ("teacher_relabel", "new_dft"))):
            raise ValueError("recovery labeling requires boolean teacher_relabel and new_dft")
        training = plan.get("student_training")
        if (not isinstance(training, dict) or not isinstance(training.get("retrain"), bool) or
                not isinstance(training.get("mode"), str) or not training["mode"].strip()):
            raise ValueError("recovery student_training requires retrain and mode")
        if training["retrain"] == (training["mode"] == "none"):
            raise ValueError("recovery student_training retrain and mode are inconsistent")
        revalidation = plan.get("revalidation")
        if (not isinstance(revalidation, dict) or
                not isinstance(revalidation.get("reuse_profile"), bool) or
                not isinstance(revalidation.get("targets"), list) or
                not revalidation["targets"] or
                any(not isinstance(item, str) or not item.strip()
                    for item in revalidation["targets"])):
            raise ValueError("recovery revalidation requires reuse_profile and non-empty targets")
        if "estimated_cost" not in plan or not isinstance(plan["estimated_cost"], dict):
            raise ValueError("recovery estimated_cost must be an object")

        recovery_id = len(self.state.setdefault("recoveries", [])) + 1
        recovery_dir = self.run_dir / "recovery"
        recovery_dir.mkdir(exist_ok=True)
        destination = recovery_dir / f"recovery-{recovery_id:03d}.json"
        record = {
            "id": recovery_id, "iteration": self._current_iteration()["id"],
            "status": "proposed", "proposed_at": now(), "source": str(source),
            "failed_stage": failed_stage, "verdict": pending["verdict"],
            "gate_binding": {
                "recorded_at": pending["gate_recorded_at"],
                "artifact_sha256": pending["artifact_sha256"],
                "votes_integrity": pending.get("votes_integrity"),
            },
            "plan": plan, "human_approval": None,
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        record["path"] = str(destination)
        record["integrity"] = artifact_digest(destination)
        self.state["recoveries"].append(record)
        self.state["pending_recovery"] = {"status": "proposed", "recovery_id": recovery_id}
        self.state["events"].append({"at": now(), "type": "recovery_proposed",
                                     "recovery_id": recovery_id, "path": str(destination),
                                     "integrity": record["integrity"]})
        self.save()
        return record

    def _pending_recovery_record(self, expected_status):
        pending = self.state.get("pending_recovery")
        if not pending or pending.get("status") != expected_status:
            raise RuntimeError(f"no recovery is waiting in {expected_status!r} state")
        matches = [item for item in self.state.get("recoveries", [])
                   if item.get("id") == pending.get("recovery_id")]
        if len(matches) != 1:
            raise RuntimeError("pending recovery record is missing or ambiguous")
        try:
            verify_artifact(matches[0]["path"], matches[0]["integrity"])
        except (KeyError, FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError("pending recovery proposal integrity check failed") from exc
        return matches[0]

    def approve_recovery(self, approved_by, note=None):
        """Record explicit human approval without claiming identity verification."""
        recovery = self._pending_recovery_record("proposed")
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise ValueError("recovery approval requires approved_by")
        approval = {"approved_at": now(), "approved_by": approved_by.strip(),
                    "note": note or ""}
        recovery.update(status="approved", human_approval=approval)
        self.state["pending_recovery"] = {"status": "approved",
                                          "recovery_id": recovery["id"]}
        self.state["events"].append({"at": now(), "type": "recovery_approved",
                                     "recovery_id": recovery["id"], **approval})
        self.save()
        return recovery

    def start_iteration(self):
        """Activate an approved recovery and invalidate from its declared return stage."""
        recovery = self._pending_recovery_record("approved")
        old_iteration = self._current_iteration()
        old_iteration.update(status="superseded", completed_at=now())
        return_stage = recovery["plan"]["return_stage"]
        return_index = self._stage_index(return_stage)
        baseline_artifacts = [dict(record) for record in self.state["artifacts"]
                              if self._stage_index(record["stage"]) >= return_index]
        self.invalidate_from(return_stage, include_stage=True)
        new_iteration = old_iteration["id"] + 1
        self.state["iterations"].append({
            "id": new_iteration, "parent_iteration": old_iteration["id"],
            "status": "active", "started_at": now(),
            "trigger": {"recovery_id": recovery["id"],
                        "failed_stage": recovery["failed_stage"],
                        "return_stage": return_stage},
            "baseline_artifacts": baseline_artifacts,
            "recovery_execution": {"status": "required"},
        })
        recovery.update(status="activated", activated_at=now(),
                        new_iteration=new_iteration)
        self.state["pending_recovery"] = None
        self.state["events"].append({"at": now(), "type": "iteration_started",
                                     "iteration": new_iteration,
                                     "recovery_id": recovery["id"],
                                     "return_stage": return_stage})
        self.save()
        return recovery

    def verify_recovery_execution(self, report_path):
        """Verify that an approved recovery produced changed, registered artifacts."""
        iteration = self._current_iteration()
        trigger = iteration.get("trigger")
        if not trigger:
            raise RuntimeError("the current iteration was not started by a recovery")
        if "baseline_artifacts" not in iteration:
            raise RuntimeError(
                "this recovery iteration has no artifact baseline; start a new recovery iteration"
            )
        execution = iteration.get("recovery_execution", {})
        if execution.get("status") != "required":
            raise RuntimeError("the current recovery execution is not waiting for verification")
        matches = [item for item in self.state.get("recoveries", [])
                   if item.get("id") == trigger.get("recovery_id")]
        if len(matches) != 1 or matches[0].get("status") != "activated":
            raise RuntimeError("the activated recovery record is missing or ambiguous")
        recovery = matches[0]
        source = Path(report_path).resolve()
        report = json.loads(source.read_text())
        if report.get("schema_version") != 1:
            raise ValueError("recovery execution report requires schema_version=1")
        if report.get("recovery_id") != recovery["id"]:
            raise ValueError("recovery execution report has the wrong recovery_id")
        if (report.get("previous_iteration") != iteration["parent_iteration"] or
                report.get("current_iteration") != iteration["id"]):
            raise ValueError("recovery execution report has the wrong iteration binding")

        planned_changes = recovery["plan"]["proposed_changes"]
        applied_changes = report.get("changes")
        if not isinstance(applied_changes, list) or len(applied_changes) != len(planned_changes):
            raise ValueError("recovery execution must report every proposed change exactly once")
        baseline = iteration.get("baseline_artifacts", [])

        def validate_stage(stage_name):
            if not isinstance(stage_name, str) or not stage_name.strip():
                raise ValueError("recovery execution evidence stage must be non-empty")
            stage = self.stage(stage_name)
            if self._stage_index(stage_name) < self._stage_index(trigger["return_stage"]):
                raise ValueError(
                    f"recovery execution stage precedes the approved return stage: {stage_name}"
                )
            if stage["status"] != "completed":
                raise ValueError(f"recovery execution stage is not completed: {stage_name}")
            current = self.verify_stage_artifacts(stage_name)
            previous = [item for item in baseline if item["stage"] == stage_name]
            if previous:
                old_hashes = {item["sha256"] for item in previous}
                new_hashes = {item["sha256"] for item in current}
                if old_hashes == new_hashes:
                    raise ValueError(
                        f"recovery execution did not change artifacts for stage: {stage_name}"
                    )
            return stage_name

        def validate_changed_artifact(raw_path):
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError("recovery execution evidence artifact must be non-empty")
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.run_dir / path
            path = path.resolve()
            current = [item for item in self.state["artifacts"]
                       if Path(item["path"]).resolve() == path]
            if len(current) != 1:
                raise ValueError(f"recovery execution artifact is not registered: {path}")
            validate_stage(current[0]["stage"])
            previous = [item for item in baseline
                        if Path(item["path"]).resolve() == path]
            if previous and previous[0]["sha256"] == current[0]["sha256"]:
                raise ValueError(f"recovery execution artifact did not change: {path}")
            return current[0]["stage"]

        change_types = []
        evidence_stages = set()
        for planned, applied in zip(planned_changes, applied_changes):
            if not isinstance(applied, dict) or applied.get("type") != planned["type"]:
                raise ValueError("recovery execution change order/type differs from the approved plan")
            if applied.get("status") != "APPLIED":
                raise ValueError("every recovery execution change must have status APPLIED")
            artifacts = applied.get("evidence_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("every applied recovery change requires evidence_artifacts")
            evidence_stages.update(validate_changed_artifact(path) for path in artifacts)
            change_types.append(applied["type"])

        labeling = recovery["plan"]["labeling"]
        label_report = report.get("labeling")
        if not isinstance(label_report, dict):
            raise ValueError("recovery execution requires labeling")
        for flag, stage_field in (("teacher_relabel", "teacher_relabel_stage"),
                                  ("new_dft", "new_dft_stage")):
            if label_report.get(flag) != labeling[flag]:
                raise ValueError(f"recovery execution labeling.{flag} differs from the plan")
            stage_name = label_report.get(stage_field)
            if labeling[flag]:
                evidence_stages.add(validate_stage(stage_name))
            elif stage_name is not None:
                raise ValueError(f"recovery execution labeling.{stage_field} must be null")

        training = recovery["plan"]["student_training"]
        training_report = report.get("student_training")
        if not isinstance(training_report, dict):
            raise ValueError("recovery execution requires student_training")
        for field in ("retrain", "mode"):
            if training_report.get(field) != training[field]:
                raise ValueError(f"recovery execution student_training.{field} differs from the plan")
        training_stage = training_report.get("stage")
        if training["retrain"]:
            evidence_stages.add(validate_stage(training_stage))
        elif training_stage is not None:
            raise ValueError("recovery execution student_training.stage must be null")

        revalidation = recovery["plan"]["revalidation"]
        revalidation_report = report.get("revalidation")
        if not isinstance(revalidation_report, dict):
            raise ValueError("recovery execution requires revalidation")
        if revalidation_report.get("targets") != revalidation["targets"]:
            raise ValueError("recovery execution revalidation targets differ from the plan")
        stages = revalidation_report.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("recovery execution revalidation requires evidence stages")
        evidence_stages.update(validate_stage(name) for name in stages)

        destination = self.run_dir / "recovery" / f"recovery-{recovery['id']:03d}.execution.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        record = {"status": "verified", "verified_at": now(), "path": str(destination),
                  "integrity": artifact_digest(destination),
                  "change_types": change_types, "evidence_stages": sorted(evidence_stages)}
        iteration["recovery_execution"] = record
        recovery["execution"] = record
        self.state["events"].append({"at": now(), "type": "recovery_execution_verified",
                                     "recovery_id": recovery["id"], **record})
        self.save()
        return record

    def _require_verified_recovery_for_pass(self, stage_name):
        iteration = self._current_iteration()
        trigger = iteration.get("trigger")
        if trigger and trigger.get("failed_stage") == stage_name:
            if iteration.get("recovery_execution", {}).get("status") != "verified":
                raise RuntimeError(
                    "the recovered stage cannot PASS until recovery execution is verified"
                )

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
    propose = sub.add_parser("propose-recovery")
    propose.add_argument("run_dir")
    propose.add_argument("plan")
    approve = sub.add_parser("approve-recovery")
    approve.add_argument("run_dir")
    approve.add_argument("--approved-by", required=True)
    approve.add_argument("--note")
    iteration = sub.add_parser("start-iteration")
    iteration.add_argument("run_dir")
    verify_recovery = sub.add_parser("verify-recovery")
    verify_recovery.add_argument("run_dir")
    verify_recovery.add_argument("report")
    context = sub.add_parser("gate-context")
    context.add_argument("run_dir")
    context.add_argument("stage")
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
    elif args.action == "propose-recovery":
        controller.propose_recovery(args.plan)
    elif args.action == "approve-recovery":
        controller.approve_recovery(args.approved_by, args.note)
    elif args.action == "start-iteration":
        controller.start_iteration()
    elif args.action == "verify-recovery":
        controller.verify_recovery_execution(args.report)
    elif args.action == "gate-context":
        print(json.dumps(controller.gate_context(args.stage), indent=2))
        return
    for row in controller.summary():
        print("\t".join(map(str, row)))
    if args.action == "status" and controller.state.get("pending_recovery"):
        print("RECOVERY\t" + json.dumps(controller.state["pending_recovery"], sort_keys=True))
    if args.action == "status":
        execution = controller._current_iteration().get("recovery_execution")
        if execution:
            print("RECOVERY_EXECUTION\t" + json.dumps(execution, sort_keys=True))


if __name__ == "__main__":
    main()
