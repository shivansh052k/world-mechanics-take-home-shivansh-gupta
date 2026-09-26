"""Checks for result provenance: the code-only dirty flag.

Run from the repo root, with .venv active:
    python scripts/check_evidence.py <check>

Each check prints its result and stores it under its own key in results/evidence/checks.json.
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from vjepa_physics.evidence import code_changes, git, repo_root, save_result

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/evidence/checks.json"

# Files of the scratch repository; mirrors the real layout that matters for the flag.
SCRATCH_FILES = (
    "src/pkg/module.py", "scripts/run.py", "pyproject.toml", "requirements.lock.txt",
    "results/out.json", "docs/notes.md",
)

# Scenario -> (edits applied to a clean committed tree, expected dirty flag).
SCENARIOS = {
    "committed": ({}, False),
    "results_changed": ({"results/out.json": "changed\n", "results/new.json": "new\n"}, False),
    "docs_changed": ({"docs/notes.md": "changed\n"}, False),
    "src_edited": ({"src/pkg/module.py": "changed\n"}, True),
    "src_new_untracked": ({"src/pkg/extra.py": "new\n"}, True),
    "scripts_edited": ({"scripts/run.py": "changed\n"}, True),
    "pyproject_changed": ({"pyproject.toml": "changed\n"}, True),
    "lock_changed": ({"requirements.lock.txt": "changed\n"}, True),
}

# Data-audit result subjects, re-run from a clean tree before the audit report.
AUDIT_SUBJECTS = ("data_files", "metadata", "design", "videos", "tracking")
# Committed outputs the audit checks write besides their checks.json.
AUDIT_OUTPUTS = (
    "results/tracking/clip_flags.csv",
    "results/videos/contact_sheet.png",
    "results/design/design.png",
    "results/tracking/tracking.png",
)

# Split, extraction and joined results, re-run from a clean tree before their report. The full extraction is not
# re-run: `verify` re-extracts a seeded subset and requires it bit-identical to the stored activations instead.
GATE_SUBJECTS = ("splits", "extraction", "joined")
GATE_OUTPUTS = ("artifacts/manifests/splits.csv",)
NOT_RERUN = {("extraction", key) for key in ("extract_direction", "extract_speed", "extract_acceleration")}
SUBSET_REEXTRACTION = ("extraction", "verify")
# Result fields that describe the machine at run time, not the data or the code: left out of the comparison.
VOLATILE_FIELDS = {("joined", "storage"): ("free_disk_bytes", "memory")}

# Probing-stage checks judged by the code-hash rule: a key passes if some commit at or after its recorded commit
# holds exactly the file hashes it recorded; otherwise it is re-run from committed code.
HASH_CHECK_SUBJECTS = ("probes", "baselines", "layer_curves", "patches", "nullspace")
CODE_PREFIXES = ("src/", "scripts/")  # the recorded code files live here

# Keys re-run at the probing gate because their code changed before it was committed (code_hash_check).
PROBING_RERUNS = (("baselines", "pixel_grams"), ("baselines", "pixel_floor"), ("probes", "shuffled_labels"),
                  ("layer_curves", "bootstrap"), ("patches", "patch_alpha_diagnostic"))


def scratch_git(root: Path, *args: str) -> None:
    """Git in the scratch repository, with a fixed identity and no commit signing."""
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=check", "-c", "user.email=check@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def changes_from(cwd: Path, root: Path) -> list[str]:
    """code_changes(root) evaluated with the process working directory set to `cwd`."""
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        return code_changes(root)
    finally:
        os.chdir(previous)


def check_dirty_flag() -> dict:
    """The dirty flag reacts to code and environment changes only, from any working directory.

    Every scenario runs in a scratch git repository in a temporary folder, so the real
    repository is never modified. Also records the real repository's current state.
    """
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        scratch_git(root, "init", "--quiet")
        for rel in SCRATCH_FILES:
            write(root / rel, "original\n")
        scratch_git(root, "add", "--all")
        scratch_git(root, "commit", "--quiet", "--message", "initial")

        for name, (edits, expected) in SCENARIOS.items():
            scratch_git(root, "reset", "--quiet", "--hard")
            scratch_git(root, "clean", "--quiet", "-fd")
            for rel, text in edits.items():
                write(root / rel, text)
            lines = code_changes(root)
            results[name] = {"dirty": bool(lines), "expected": expected, "paths": lines}

        # Same answer whatever the working directory (last scenario's edit is still in place).
        from_elsewhere = {
            "repo_root": changes_from(root, root),
            "subfolder": changes_from(root / "docs", root),
            "outside": changes_from(Path(tempfile.gettempdir()), root),
        }

    real_root_from_elsewhere = repo_root() == REPO
    previous = Path.cwd()
    os.chdir(tempfile.gettempdir())
    try:
        real_root_from_outside = repo_root() == REPO
    finally:
        os.chdir(previous)

    same_everywhere = len({tuple(v) for v in from_elsewhere.values()}) == 1 and bool(from_elsewhere["outside"])
    return {
        "scenarios": results,
        "same_answer_from_any_working_directory": same_everywhere,
        "real_repo_root_found_from_repo_and_outside": real_root_from_elsewhere and real_root_from_outside,
        "real_repo_code_changes_now": code_changes(REPO),
        "passed": bool(
            all(r["dirty"] == r["expected"] for r in results.values())
            and same_everywhere
            and real_root_from_elsewhere
            and real_root_from_outside
        ),
    }

def check_rerun_identical() -> dict:
    """A clean re-run of the data-audit checks reproduces the committed results exactly.

    For every key of results/<subject>/checks.json (AUDIT_SUBJECTS) as committed at HEAD: the key must
    be present in the working tree, its provenance must say it was produced at HEAD with no code
    changes (git_dirty False), and its "result" must equal the committed one exactly (provenance may
    differ). The committed outputs in AUDIT_OUTPUTS must be byte-identical to HEAD. Passes if all hold.
    """
    head = git(REPO, "rev-parse", "HEAD").strip()
    per_subject, missing, not_clean, differing = {}, [], [], []
    for subject in AUDIT_SUBJECTS:
        path = f"results/{subject}/checks.json"
        committed = json.loads(git(REPO, "show", f"HEAD:{path}"))
        current = json.loads((REPO / path).read_text())
        keys = {}
        for key, entry in committed.items():
            now = current.get(key)
            if now is None:
                missing.append(f"{subject}/{key}")
                keys[key] = "missing"
                continue
            clean = now["provenance"].get("git_commit") == head and now["provenance"].get("git_dirty") is False
            same = now["result"] == entry["result"]
            keys[key] = {"rerun_from_clean_head": clean, "result_identical": same}
            if not clean:
                not_clean.append(f"{subject}/{key}")
            if not same:
                differing.append(f"{subject}/{key}")
        per_subject[subject] = keys

    changed_outputs = [p for p in AUDIT_OUTPUTS if git(REPO, "status", "--porcelain", "--", p).strip()]
    compared = sum(len(k) for k in per_subject.values())
    criteria = {
        "keys_compared": compared > 0,
        "no_key_missing": not missing,
        "every_key_rerun_from_clean_head": not not_clean,
        "every_result_identical": not differing,
        "committed_outputs_unchanged": not changed_outputs,
    }
    return {
        "criteria": criteria,
        "head": head,
        "keys_compared": compared,
        "missing": missing,
        "not_rerun_from_clean_head": not_clean,
        "results_differing": differing,
        "outputs_changed": changed_outputs,
        "per_subject": per_subject,
        "passed": all(criteria.values()),
    }

def without(result: dict, fields: tuple[str, ...]) -> dict:
    """`result` minus the top-level `fields`."""
    return {key: value for key, value in result.items() if key not in fields}


def check_rerun_identical_splits_extraction() -> dict:
    """A clean re-run of the split, extraction and joined checks reproduces the committed results.

    For every key of results/<subject>/checks.json (GATE_SUBJECTS) as committed at HEAD:
    - keys in NOT_RERUN must be present and exactly equal to the committed entry, provenance included (they are
      deliberately not re-run, so they must not have been touched);
    - every other key must be present, produced at HEAD with no code changes (git_dirty False), and its result
      must equal the committed one exactly, except the fields in VOLATILE_FIELDS, which are listed per key.
    The subset re-extraction (SUBSET_REEXTRACTION) must be among the re-run keys, and the committed outputs in
    GATE_OUTPUTS must be byte-identical to HEAD. Passes if all hold.
    """
    head = git(REPO, "rev-parse", "HEAD").strip()
    per_subject, missing, touched, not_clean, differing, rerun = {}, [], [], [], [], []
    for subject in GATE_SUBJECTS:
        path = f"results/{subject}/checks.json"
        committed = json.loads(git(REPO, "show", f"HEAD:{path}"))
        current = json.loads((REPO / path).read_text())
        keys = {}
        for key, entry in committed.items():
            name = f"{subject}/{key}"
            now = current.get(key)
            if now is None:
                missing.append(name)
                keys[key] = "missing"
                continue
            if (subject, key) in NOT_RERUN:
                unchanged = now == entry
                keys[key] = {"not_rerun_by_design": True, "unchanged": unchanged}
                if not unchanged:
                    touched.append(name)
                continue
            ignored = VOLATILE_FIELDS.get((subject, key), ())
            clean = now["provenance"].get("git_commit") == head and now["provenance"].get("git_dirty") is False
            same = without(now["result"], ignored) == without(entry["result"], ignored)
            keys[key] = {"rerun_from_clean_head": clean, "result_identical": same, "fields_ignored": list(ignored)}
            rerun.append((subject, key))
            if not clean:
                not_clean.append(name)
            if not same:
                differing.append(name)
        per_subject[subject] = keys

    changed_outputs = [p for p in GATE_OUTPUTS if git(REPO, "status", "--porcelain", "--", p).strip()]
    criteria = {
        "keys_compared": bool(rerun),
        "subset_reextraction_rerun": SUBSET_REEXTRACTION in rerun,
        "no_key_missing": not missing,
        "not_rerun_keys_untouched": not touched,
        "every_rerun_key_from_clean_head": not not_clean,
        "every_rerun_result_identical": not differing,
        "committed_outputs_unchanged": not changed_outputs,
    }
    return {
        "criteria": criteria,
        "head": head,
        "keys_rerun": len(rerun),
        "missing": missing,
        "not_rerun_but_changed": touched,
        "not_rerun_from_clean_head": not_clean,
        "results_differing": differing,
        "outputs_changed": changed_outputs,
        "per_subject": per_subject,
        "passed": all(criteria.values()),
    }
    
def git_bytes(root: Path, *args: str) -> bytes:
    """Raw stdout of git run against `root` (bytes, no newline translation)."""
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


def committed_code_hashes(root: Path, commit: str, blob_cache: dict[str, str]) -> dict[str, str]:
    """SHA-256 of every file under CODE_PREFIXES in `commit`'s tree, by repo-relative path.

    Contents come from git's object store (never the working tree); each distinct blob is hashed once.
    """
    hashes = {}
    for line in git_bytes(root, "ls-tree", "-r", commit).decode().splitlines():
        meta, path = line.split("\t", 1)
        if not path.startswith(CODE_PREFIXES):
            continue
        blob = meta.split()[2]
        if blob not in blob_cache:
            blob_cache[blob] = hashlib.sha256(git_bytes(root, "cat-file", "blob", blob)).hexdigest()
        hashes[path] = blob_cache[blob]
    return hashes


def check_code_hash_check() -> dict:
    """Gate check: was the code behind every probing-stage result committed unchanged?

    For every key in results/<subject>/checks.json (HASH_CHECK_SUBJECTS), walks from the key's recorded
    git_commit forward to HEAD and looks for the first commit whose files carry exactly the SHA-256s the key
    recorded (running script + every package module). Passes only if every key has such a commit; otherwise
    `rerun_needed` lists the keys to re-run from committed code.
    """
    root = repo_root()
    head = git(root, "rev-parse", "HEAD").strip()
    blob_cache: dict[str, str] = {}
    tree_cache: dict[str, dict[str, str]] = {}
    keys = []
    for subject in HASH_CHECK_SUBJECTS:
        records = json.loads((REPO / "results" / subject / "checks.json").read_text())
        for key, entry in records.items():
            provenance = entry["provenance"]
            recorded = provenance["git_commit"]
            files = provenance["code"]["files"]
            later = git(root, "rev-list", "--reverse", f"{recorded}..HEAD").split()
            match = None
            for commit in [recorded, *later]:
                if commit not in tree_cache:
                    tree_cache[commit] = committed_code_hashes(root, commit, blob_cache)
                if all(tree_cache[commit].get(path) == digest for path, digest in files.items()):
                    match = commit
                    break
            keys.append({
                "subject": subject, "key": key, "recorded_commit": recorded,
                "git_dirty": provenance.get("git_dirty"), "files": len(files), "matching_commit": match,
            })
    rerun = [f"{k['subject']}/{k['key']}" for k in keys if k["matching_commit"] is None]
    return {"head": head, "keys_checked": len(keys), "commits_searched": len(tree_cache), "keys": keys,
            "rerun_needed": rerun, "passed": not rerun}   


def without_machine_state(subject: str, key: str, result: dict) -> dict:
    """A copy of `result` without timing and memory fields (they describe the machine, not the result)."""
    r = copy.deepcopy(result)
    if (subject, key) == ("baselines", "pixel_grams"):
        r.pop("peak_rss_bytes", None)
        for variable in ("direction", "speed", "acceleration"):
            r[variable].pop("decode_seconds", None)
            for gram in r[variable]["grams"].values():
                gram.pop("gram_seconds", None)
    return r


def check_rerun_identical_probing() -> dict:
    """The probing-stage keys re-run from committed code reproduce their committed results exactly.

    For every key in PROBING_RERUNS, compares the result now on disk with the one committed at HEAD (read with
    git show), minus machine-state fields. Passes if every result is identical, every re-run was saved from
    committed code (git_dirty false) and at a different commit than the committed result.
    """
    root = repo_root()
    head = git(root, "rev-parse", "HEAD").strip()
    rows = []
    for subject, key in PROBING_RERUNS:
        path = f"results/{subject}/checks.json"
        committed = json.loads(git(root, "show", f"HEAD:{path}"))[key]
        current = json.loads((REPO / path).read_text())[key]
        rows.append({
            "subject": subject, "key": key,
            "identical": without_machine_state(subject, key, current["result"])
            == without_machine_state(subject, key, committed["result"]),
            "rerun_commit": current["provenance"]["git_commit"],
            "rerun_dirty": current["provenance"]["git_dirty"],
            "committed_commit": committed["provenance"]["git_commit"],
        })
    criteria = {
        "results_identical": all(r["identical"] for r in rows),
        "reruns_from_committed_code": all(r["rerun_dirty"] is False for r in rows),
        "actually_rerun": all(r["rerun_commit"] != r["committed_commit"] for r in rows),
    }
    return {"head": head, "criteria": criteria, "keys": rows, "passed": all(criteria.values())}


CHECKS = {
    "dirty_flag": check_dirty_flag,
    "rerun_identical": check_rerun_identical,
    "rerun_identical_splits_extraction": check_rerun_identical_splits_extraction,
    "code_hash_check": check_code_hash_check,
    "rerun_identical_probing": check_rerun_identical_probing,
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    name = parser.parse_args().check
    result = CHECKS[name]()
    print(json.dumps(result, indent=2))
    save_result(OUT, name, result)
    print(f"saved '{name}' -> {OUT.relative_to(REPO)}")
    if result.get("passed") is False:
        raise SystemExit(f"check '{name}' FAILED")


if __name__ == "__main__":
    main()