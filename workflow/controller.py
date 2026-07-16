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
            state = {"schema_version": 3, "run_id": cfg["run_id"], "created_at": now(),
                     "updated_at": now(), "workflow_config": str(run_dir / "workflow.yaml"),
                     "artifacts": [], "project_dir": str(project_dir), "inputs": input_records,
                     "code_revision": git_revision(project_dir), "events": [], "stages": stages}
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
        if verdict == "PASS" and (stage.get("contract") or {}).get("kind") == "validation_manifest":
            self._validate_external_contract(
                stage, [record["path"] for record in self.stage_artifacts(name)],
                enforce_required_pass=True,
            )
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
    elif args.action == "gate-context":
        print(json.dumps(controller.gate_context(args.stage), indent=2))
        return
    for row in controller.summary():
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
