from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

STATE_PATH = ROOT / "STATE.md"
HANDOFF_STATE = ROOT / "handoff/STATE.md"
GIT_SYNC = ROOT / "scripts/git_sync.py"

AUDIT_DIR = ROOT / "artifacts/audit"
REPORT_DIR = ROOT / "reports"

AUDIT_PATH = AUDIT_DIR / "notebook05d_segct_clip_reproduction_audit.json"
INVENTORY_PATH = AUDIT_DIR / "notebook05d_segct_clip_author_repo_inventory.json"
REPORT_PATH = REPORT_DIR / "notebook05d_segct_clip_reproduction_audit.md"

AUTHOR_CLONE_DIR = WORK / "CT_Insight_Author_Repo"

DOI = "10.1109/TRPMS.2026.3734275"
TITLE = (
    "CT-Insight VLM: Multimodel Vision-Language Framework for Accurate, "
    "Explainable, and Label-Efficient Lung CT Analysis"
)

PUBLICATION_URLS = [
    "https://research.aston.ac.uk/en/publications/ct-insight-vlm-multimodel-vision-language-framework-for-accurate-/",
    "https://ieeexplore.ieee.org/document/11693975/",
    "https://d197for5662m48.cloudfront.net/documents/publicationstatus/278736/preprint_pdf/9ecd296031833c3305803e632a99bbe5.pdf",
]

SEARCH_QUERIES = [
    '"CT-Insight VLM"',
    '"SegCT-CLIP"',
    '"TRPMS.2026.3734275"',
    '"11693975" "CT-Insight"',
]

AUTHOR_USERS = [
    "Owais-CodeHub",
]

GITHUB_API = "https://api.github.com"

DISCLOSED_COMPONENTS = {
    "task": "Lesion segmentation plus caption alignment/retrieval for lung CT slices.",
    "backbone": "CLIP ViT-L/14-336 is reported as the strongest SegCT-CLIP backbone.",
    "visual_branch": (
        "Dual-level low/high transformer visual features are aggregated and passed "
        "to a lightweight visual decoder for lesion-mask prediction."
    ),
    "text_branch": (
        "A predefined clinically curated caption repository is encoded by the CLIP "
        "text encoder; captions are retrieved by cosine similarity to high-level "
        "image embeddings."
    ),
    "training_triplet": "Input slice, retrieved/relevant caption, ground-truth lesion mask.",
    "training_loss": (
        "Weighted contrastive plus Dice objective with lambda_contrastive=0.3 "
        "and lambda_dice=0.7."
    ),
    "evaluation_protocol": (
        "Bidirectional SegDB-1/SegDB-2 cross-dataset evaluation without target "
        "fine-tuning is described."
    ),
    "published_backbone_result_note": (
        "Published table values are literature values only and are not treated as rerun results."
    ),
}

UNRESOLVED_CHOICES = [
    "Exact public source-code repository URL and released revision/commit.",
    "Exact released checkpoint corresponding to the reported SegCT-CLIP results.",
    "Exact 367-slice SegDB-2 selection used by the paper.",
    "Exact SegDB-1 patient/group mapping and source partition.",
    "Caption-bank contents, caption-to-slice mapping, and caption preprocessing.",
    "Exact low-level and high-level CLIP transformer layer indices.",
    "Exact dual-level aggregation function Fagg.",
    "Exact lightweight decoder topology and normalization/activation choices.",
    "Whether and how much of the CLIP image/text encoders were fine-tuned.",
    "Optimizer, learning rate, weight decay, batch size, number of epochs/updates.",
    "Augmentation and image normalization/resolution preprocessing details.",
    "Early-stopping/model-selection rule and random seeds.",
    "Exact metric aggregation convention used for the paper tables.",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_request(method: str, url: str, *, headers=None, params=None, timeout=45):
    try:
        r = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            allow_redirects=True,
        )
        return {
            "ok": True,
            "status_code": r.status_code,
            "url": r.url,
            "headers": dict(r.headers),
            "text": r.text[:2_000_000],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "headers": {},
            "text": "",
            "error": repr(exc),
        }


def github_headers(token: str | None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "EviCT-reproduction-audit",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_repo_url(url: str) -> str | None:
    if not url:
        return None
    m = re.search(r"https?://github\.com/([^/\s]+)/([^/#?\s]+)", url, flags=re.I)
    if not m:
        return None
    owner = m.group(1)
    repo = m.group(2)
    repo = repo.removesuffix(".git")
    if not owner or not repo:
        return None
    return f"https://github.com/{owner}/{repo}"


def extract_github_urls(text: str) -> list[str]:
    found = []
    for raw in re.findall(
        r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        text or "",
        flags=re.I,
    ):
        normalized = normalize_repo_url(raw)
        if normalized and normalized not in found:
            found.append(normalized)
    return found


def repo_score(repo: dict) -> int:
    blob = " ".join(
        str(repo.get(k, ""))
        for k in ["name", "full_name", "description", "html_url"]
    ).lower()
    score = 0
    for term, weight in [
        ("ct-insight", 8),
        ("ct_insight", 8),
        ("ctinsight", 8),
        ("segct", 8),
        ("vision-language", 2),
        ("lung", 1),
        ("ct", 1),
    ]:
        if term in blob:
            score += weight
    owner = str(repo.get("owner", {}).get("login", "")).lower()
    if owner == "owais-codehub":
        score += 4
    return score


def discover_github_repositories(token: str | None) -> dict:
    headers = github_headers(token)
    candidates = []
    query_logs = []

    for query in SEARCH_QUERIES:
        result = safe_request(
            "GET",
            f"{GITHUB_API}/search/repositories",
            headers=headers,
            params={"q": query, "per_page": 50},
        )
        query_logs.append({
            "query": query,
            "ok": result["ok"],
            "status_code": result["status_code"],
        })
        if result["ok"] and result["status_code"] == 200:
            try:
                payload = json.loads(result["text"])
            except Exception:
                payload = {}
            for repo in payload.get("items", []):
                candidates.append(repo)

    for username in AUTHOR_USERS:
        result = safe_request(
            "GET",
            f"{GITHUB_API}/users/{username}/repos",
            headers=headers,
            params={"per_page": 100, "sort": "updated"},
        )
        query_logs.append({
            "query": f"user:{username}",
            "ok": result["ok"],
            "status_code": result["status_code"],
        })
        if result["ok"] and result["status_code"] == 200:
            try:
                repos = json.loads(result["text"])
            except Exception:
                repos = []
            candidates.extend(repos)

    dedup = {}
    for repo in candidates:
        full_name = repo.get("full_name")
        if full_name:
            dedup[full_name] = repo

    ranked = sorted(
        dedup.values(),
        key=lambda x: (repo_score(x), x.get("updated_at") or ""),
        reverse=True,
    )

    compact = []
    for repo in ranked[:100]:
        compact.append({
            "full_name": repo.get("full_name"),
            "html_url": repo.get("html_url"),
            "description": repo.get("description"),
            "updated_at": repo.get("updated_at"),
            "default_branch": repo.get("default_branch"),
            "fork": repo.get("fork"),
            "score": repo_score(repo),
        })

    return {
        "query_logs": query_logs,
        "ranked_candidates": compact,
    }


def discover_publication_links() -> dict:
    records = []
    github_urls = []

    for url in PUBLICATION_URLS:
        result = safe_request("GET", url, timeout=60)
        links = extract_github_urls(result.get("text", ""))
        records.append({
            "requested_url": url,
            "resolved_url": result.get("url"),
            "ok": result.get("ok"),
            "status_code": result.get("status_code"),
            "github_urls": links,
        })
        for link in links:
            if link not in github_urls:
                github_urls.append(link)

    return {
        "records": records,
        "github_urls": github_urls,
    }


def inspect_candidate_repo(repo_url: str, token: str | None) -> dict:
    normalized = normalize_repo_url(repo_url)
    assert normalized
    owner_repo = normalized.replace("https://github.com/", "", 1)
    owner, repo = owner_repo.split("/", 1)

    headers = github_headers(token)

    meta_resp = safe_request(
        "GET",
        f"{GITHUB_API}/repos/{owner}/{repo}",
        headers=headers,
    )

    if not (meta_resp["ok"] and meta_resp["status_code"] == 200):
        return {
            "repo_url": normalized,
            "accessible": False,
            "status_code": meta_resp["status_code"],
        }

    meta = json.loads(meta_resp["text"])
    branch = meta.get("default_branch", "main")

    readme_resp = safe_request(
        "GET",
        f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md",
        timeout=60,
    )
    readme_text = readme_resp["text"] if readme_resp["status_code"] == 200 else ""

    terms = {
        "segct_clip": bool(re.search(r"segct[-_\s]?clip", readme_text, flags=re.I)),
        "caption": "caption" in readme_text.lower(),
        "checkpoint": any(x in readme_text.lower() for x in ["checkpoint", "pretrained", "weights"]),
        "segdb1": "segdb-1" in readme_text.lower() or "segdb1" in readme_text.lower(),
        "segdb2": "segdb-2" in readme_text.lower() or "segdb2" in readme_text.lower(),
        "vit_l_14_336": any(
            x in readme_text.lower()
            for x in ["vit-l/14-336", "vit_l_14_336", "vit-l-14-336"]
        ),
    }

    return {
        "repo_url": normalized,
        "accessible": True,
        "full_name": meta.get("full_name"),
        "description": meta.get("description"),
        "default_branch": branch,
        "updated_at": meta.get("updated_at"),
        "pushed_at": meta.get("pushed_at"),
        "archived": meta.get("archived"),
        "fork": meta.get("fork"),
        "readme_available": bool(readme_text),
        "readme_term_presence": terms,
    }


def choose_plausible_repo(discovery: dict, publication: dict, token: str | None):
    urls = []
    for url in publication.get("github_urls", []):
        if url not in urls:
            urls.append(url)

    for item in discovery.get("ranked_candidates", []):
        if item.get("score", 0) >= 8 and item.get("html_url"):
            normalized = normalize_repo_url(item["html_url"])
            if normalized and normalized not in urls:
                urls.append(normalized)

    inspected = []
    for url in urls[:20]:
        inspected.append(inspect_candidate_repo(url, token))

    plausible = []
    for item in inspected:
        if not item.get("accessible"):
            continue
        terms = item.get("readme_term_presence", {})
        fullname = str(item.get("full_name", "")).lower()
        desc = str(item.get("description", "") or "").lower()
        if (
            terms.get("segct_clip")
            or "ct-insight" in fullname
            or "ct_insight" in fullname
            or "ct-insight" in desc
            or "segct" in fullname
        ):
            plausible.append(item)

    return inspected, plausible


def clone_and_inventory(repo_url: str) -> dict:
    if AUTHOR_CLONE_DIR.exists():
        shutil.rmtree(AUTHOR_CLONE_DIR)

    clone = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url + ".git", str(AUTHOR_CLONE_DIR)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if clone.returncode != 0:
        return {
            "clone_ok": False,
            "repo_url": repo_url,
            "stderr_tail": clone.stderr[-4000:],
        }

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(AUTHOR_CLONE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    files = []
    interesting = []
    size_total = 0

    for path in AUTHOR_CLONE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        rel = path.relative_to(AUTHOR_CLONE_DIR).as_posix()
        try:
            size = path.stat().st_size
        except Exception:
            size = None
        if size:
            size_total += size

        files.append({
            "path": rel,
            "size_bytes": size,
        })

        lower = rel.lower()
        if any(
            term in lower
            for term in [
                "segct",
                "caption",
                "prompt",
                "clip",
                "checkpoint",
                "weight",
                "segdb",
                "train",
                "config",
            ]
        ):
            interesting.append(rel)

    text_findings = []
    for item in files:
        rel = item["path"]
        path = AUTHOR_CLONE_DIR / rel
        if path.suffix.lower() not in {
            ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini"
        }:
            continue
        if item["size_bytes"] and item["size_bytes"] > 2_000_000:
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        matches = {}
        for term in [
            "SegCT-CLIP",
            "ViT-L/14-336",
            "caption",
            "lambda1",
            "lambda2",
            "0.3",
            "0.7",
            "SegDB-1",
            "SegDB-2",
        ]:
            if term.lower() in txt.lower():
                matches[term] = True
        if matches:
            text_findings.append({
                "path": rel,
                "matches": matches,
            })

    inventory = {
        "repo_url": repo_url,
        "clone_ok": True,
        "head": head,
        "n_files": len(files),
        "total_file_bytes_excluding_git": size_total,
        "interesting_paths": interesting[:500],
        "text_findings": text_findings[:500],
        "files": files[:2000],
    }

    return inventory


def git_sync(message: str) -> None:
    subprocess.run(
        ["git", "config", "user.name", "itsCodeBakery"],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "itsCodeBakery@users.noreply.github.com"],
        cwd=str(ROOT),
        check=True,
    )

    result = subprocess.run(
        [sys.executable, str(GIT_SYNC), message],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Git synchronization failed.")


def main():
    print("=" * 110)
    print("EVICT NOTEBOOK 05D — SEGCT-CLIP REPRODUCTION AUDIT")
    print("=" * 110)

    assert ROOT.exists() and (ROOT / ".git").exists()
    assert STATE_PATH.exists()
    assert GIT_SYNC.exists()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    HANDOFF_STATE.parent.mkdir(parents=True, exist_ok=True)

    state = STATE_PATH.read_text(encoding="utf-8")
    assert "NOTEBOOK_05_UNET_THREE_SEED_SOURCE_BASELINE_FROZEN" in state
    assert "Calibration accessed:\n\nNO" in state
    assert "Target / MedSeg accessed:\n\nNO" in state
    assert "Target lock:\n\nACTIVE" in state

    print("✓ Three-seed U-Net baseline : FROZEN")
    print("✓ Calibration accessed       : NO")
    print("✓ Target / MedSeg accessed   : NO")
    print("✓ Target lock                : ACTIVE")

    try:
        token = UserSecretsClient().get_secret("pushEviCT")
    except Exception:
        token = None

    publication = discover_publication_links()
    github_discovery = discover_github_repositories(token)
    inspected, plausible = choose_plausible_repo(
        github_discovery,
        publication,
        token,
    )

    inventory = None
    chosen_repo = None

    if plausible:
        chosen_repo = plausible[0]["repo_url"]
        print(f"✓ Plausible author repository discovered: {chosen_repo}")
        inventory = clone_and_inventory(chosen_repo)
        atomic_text(
            INVENTORY_PATH,
            json.dumps(inventory, indent=2),
        )
    else:
        print("⚠ No exact SegCT-CLIP author repository could be verified automatically.")

    complete_author_resources = False

    if inventory and inventory.get("clone_ok"):
        all_paths = " ".join(inventory.get("interesting_paths", [])).lower()
        all_text = json.dumps(inventory.get("text_findings", [])).lower()

        has_segct = "segct" in all_paths or "segct-clip" in all_text
        has_caption_assets = any(
            x in all_paths
            for x in [
                "caption",
                "prompt",
            ]
        )
        has_training_code = "train" in all_paths
        has_config = "config" in all_paths

        complete_author_resources = bool(
            has_segct
            and has_caption_assets
            and has_training_code
            and has_config
        )

    if complete_author_resources:
        reproduction_status = "AUTHOR_REPOSITORY_DISCOVERED_REQUIRES_PROTOCOL_MATCH_AUDIT"
        next_action = (
            "Inspect the verified author repository against EviCT data/preprocessing contracts "
            "before executing any author training code."
        )
    else:
        reproduction_status = "EXACT_REPRODUCTION_NOT_CURRENTLY_SPECIFIED"
        next_action = (
            "Do not invent missing SegCT-CLIP parameters. Proceed only with an explicitly "
            "labeled controlled reimplementation/adaptation, or obtain the missing author "
            "repository/checkpoint/caption bank and exact protocol."
        )

    audit = {
        "timestamp_utc": utc_now(),
        "stage": "NOTEBOOK_05D",
        "paper_title": TITLE,
        "doi": DOI,
        "paper_publication_urls_checked": publication,
        "github_discovery": github_discovery,
        "candidate_repositories_inspected": inspected,
        "chosen_repo": chosen_repo,
        "author_repo_inventory_path": (
            str(INVENTORY_PATH.relative_to(ROOT))
            if INVENTORY_PATH.exists()
            else None
        ),
        "disclosed_components_from_supplied_paper": DISCLOSED_COMPONENTS,
        "unresolved_reproduction_choices": UNRESOLVED_CHOICES,
        "complete_author_resources_verified": complete_author_resources,
        "reproduction_status": reproduction_status,
        "historical_paper_metrics_entered_as_rerun_results": False,
        "calibration_accessed": False,
        "target_accessed": False,
        "target_lock_active": True,
        "next_action": next_action,
    }

    atomic_text(
        AUDIT_PATH,
        json.dumps(audit, indent=2),
    )

    report = []
    report.append("# Notebook 05D — SegCT-CLIP Reproduction Audit")
    report.append("")
    report.append(f"Timestamp: {audit['timestamp_utc']}")
    report.append("")
    report.append(f"Paper: {TITLE}")
    report.append("")
    report.append(f"DOI: {DOI}")
    report.append("")
    report.append(f"Status: **{reproduction_status}**")
    report.append("")
    report.append("## Components explicitly disclosed by the paper")
    report.append("")
    for key, value in DISCLOSED_COMPONENTS.items():
        report.append(f"- **{key}**: {value}")
    report.append("")
    report.append("## Unresolved choices blocking an exact reproduction claim")
    report.append("")
    for item in UNRESOLVED_CHOICES:
        report.append(f"- {item}")
    report.append("")
    report.append("## Author repository discovery")
    report.append("")
    if chosen_repo:
        report.append(f"- Candidate repository: {chosen_repo}")
        report.append(
            f"- Complete author resources automatically verified: "
            f"{'YES' if complete_author_resources else 'NO'}"
        )
    else:
        report.append("- No exact SegCT-CLIP author repository was verified automatically.")
    report.append("")
    report.append("## Scientific handling")
    report.append("")
    report.append(
        "- Published SegCT-CLIP table values remain literature values; none are entered as rerun results."
    )
    report.append(
        "- Calibration and MedSeg/target evaluation remain locked."
    )
    report.append(
        "- Missing architecture/training parameters are not silently invented."
    )
    report.append("")
    report.append("## Next")
    report.append("")
    report.append(next_action)

    atomic_text(
        REPORT_PATH,
        "\n".join(report) + "\n",
    )

    state_text = f"""# EviCT Execution State

## Current stage

NOTEBOOK_05D_SEGCT_CLIP_REPRODUCTION_AUDIT_COMPLETE

## Timestamp

{utc_now()}

## Completed source baselines

SegFormer-B1:

FROZEN

Competitive residual 2D U-Net:

FROZEN

## SegCT-CLIP reproduction audit

Paper:

{TITLE}

DOI:

{DOI}

Status:

{reproduction_status}

Verified author repository:

{chosen_repo if chosen_repo else "NOT VERIFIED"}

Complete author resources verified:

{"YES" if complete_author_resources else "NO"}

Historical paper metrics treated as rerun results:

NO

## Isolation

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

{next_action}
"""

    atomic_text(STATE_PATH, state_text)
    atomic_text(HANDOFF_STATE, state_text)

    git_sync("Complete Notebook 05D SegCT-CLIP reproduction audit")

    subprocess.run(
        ["git", "fetch", "origin", "main"],
        cwd=str(ROOT),
        check=True,
    )

    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    remote_head = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    assert local_head == remote_head

    print()
    print("=" * 110)
    print("EVICT NOTEBOOK 05D — SEGCT-CLIP REPRODUCTION AUDIT — DURABLE PASS")
    print("=" * 110)
    print(f"Status                    : {reproduction_status}")
    print(f"Verified author repo      : {chosen_repo if chosen_repo else 'NO'}")
    print(
        f"Complete author resources : "
        f"{'YES' if complete_author_resources else 'NO'}"
    )
    print("Historical metrics rerun  : NO")
    print("Calibration accessed      : NO")
    print("Target / MedSeg accessed  : NO")
    print("Target lock               : ACTIVE")
    print(f"GitHub HEAD               : {local_head}")
    print()
    print("NEXT:")
    print(next_action)


if __name__ == "__main__":
    main()
