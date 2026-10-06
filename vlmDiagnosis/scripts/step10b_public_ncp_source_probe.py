from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG = json.loads((DX / "config" / "step10b_public_ncp_protocol_amendment.json").read_text(encoding="utf-8"))

RUN = DX / "runs" / "DX_step10b_public_ncp_source_probe_v1"
AUDIT = DX / "artifacts" / "audit"
TABLES = DX / "tables"
for p in [RUN, AUDIT, TABLES]:
    p.mkdir(parents=True, exist_ok=True)

OUT_JSON = AUDIT / "step10b_public_ncp_source_probe.json"
OUT_LINKS = TABLES / "step10b_public_ncp_candidate_links.csv"
OUT_MOUNTS = TABLES / "step10b_public_ncp_mounted_candidates.csv"
STATE = RUN / "STATE.json"

OFFICIAL = CFG["amended_primary_subtype_training_resource"]["official_source"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, df: pd.DataFrame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sync_git():
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run(
        [sys.executable, str(helper), "Complete EViCT-Dx Step10B public NCP source probe"],
        cwd=str(ROOT),
        check=False,
    )
    return r.returncode == 0


def probe_official():
    rows = []
    info = {
        "url": OFFICIAL,
        "reachable": False,
        "status_code": None,
        "final_url": None,
        "content_type": None,
        "error": None,
    }
    try:
        r = requests.get(
            OFFICIAL,
            timeout=30,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 EViCT-Dx reproducibility audit"},
        )
        info["status_code"] = int(r.status_code)
        info["final_url"] = str(r.url)
        info["content_type"] = r.headers.get("content-type")
        info["reachable"] = bool(r.ok)

        text = r.text if "text" in (r.headers.get("content-type") or "").lower() or r.text else ""
        hrefs = re.findall(r"""href\s*=\s*["']([^"']+)["']""", text, flags=re.I)
        for href in hrefs:
            full = urljoin(r.url, href)
            low = full.lower()
            score = 0
            for token in ["download","seg","ncp","ccii","cc-ccii","zip","rar","7z","mask","label","data"]:
                if token in low:
                    score += 1
            if score:
                rows.append({
                    "source_page": OFFICIAL,
                    "candidate_url": full,
                    "keyword_score": score,
                })
    except Exception as exc:
        info["error"] = repr(exc)

    df = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(
        columns=["source_page","candidate_url","keyword_score"]
    )
    if not df.empty:
        df = df.sort_values(["keyword_score","candidate_url"], ascending=[False, True])
    return info, df


def scan_mounted():
    base = Path("/kaggle/input")
    rows = []
    if not base.exists():
        return pd.DataFrame(columns=["root","files","name_score","reason_tokens"])
    for root in base.iterdir():
        if not root.is_dir():
            continue
        name = str(root).lower()
        tokens = [t for t in ["ccii","cc-ccii","cc_ccii","ncp","covid"] if t in name]
        if not tokens:
            # one shallow level
            try:
                subs = " ".join(p.name.lower() for p in root.iterdir() if p.is_dir())
            except Exception:
                subs = ""
            tokens = [t for t in ["ccii","cc-ccii","cc_ccii","ncp"] if t in subs]
        if tokens:
            try:
                count = sum(1 for p in root.rglob("*") if p.is_file())
            except Exception:
                count = -1
            rows.append({
                "root": str(root),
                "files": count,
                "name_score": len(tokens),
                "reason_tokens": ",".join(tokens),
            })
    return pd.DataFrame(rows)


def main():
    print("="*108)
    print("EViCT-Dx STEP 10B — PUBLIC CC-CCII/NCP SOURCE PROBE")
    print("NO TRAINING — NO DATASET SUBSTITUTION — NO EXTERNAL TEST REPURPOSING")
    print("="*108)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    info, links = probe_official()
    mounts = scan_mounted()

    atomic_csv(OUT_LINKS, links)
    atomic_csv(OUT_MOUNTS, mounts)

    if not mounts.empty:
        status = "MOUNTED_CANDIDATE_FOUND_REQUIRES_EXACT_CONTENT_AUDIT"
        next_action = "Run exact 750-slice image/mask/grouping audit on the mounted candidate before any training."
    elif info["reachable"] and not links.empty:
        status = "OFFICIAL_SOURCE_REACHABLE_CANDIDATE_DOWNLOAD_LINKS_FOUND"
        next_action = "Review candidate links; acquire only the documented 750-slice NCP lesion-segmentation resource, then run exact content audit."
    elif info["reachable"]:
        status = "OFFICIAL_SOURCE_REACHABLE_BUT_NO_MACHINE_DISCOVERABLE_DOWNLOAD_LINK"
        next_action = "Open the official source page manually and attach/download the 750-slice NCP segmentation resource; then rerun the content audit."
    else:
        status = "OFFICIAL_SOURCE_NOT_REACHABLE_FROM_KAGGLE"
        next_action = "Use a verified mirror/manual copy of the public 750-slice NCP resource; do not use MedSeg or LongCIU for training."

    audit = {
        "project":"EViCT-Dx",
        "stage":"STEP_10B_PUBLIC_NCP_PROTOCOL_AMENDMENT_AND_SOURCE_PROBE",
        "completed_utc":now(),
        "status":status,
        "training_performed":False,
        "official_source_probe":info,
        "candidate_link_count":int(len(links)),
        "mounted_candidate_count":int(len(mounts)),
        "public_resource_contract":{
            "expected_slices":750,
            "expected_patients_or_scans":150,
            "expected_classes":["background","lung_field","ground_glass_opacity","consolidation"],
            "pleural_effusion_available":False,
        },
        "external_test_resources_preserved":{
            "LongCIU":"EXTERNAL_TEST_ONLY",
            "MedSeg":"SECONDARY_EXTERNAL_TEST_ONLY",
        },
        "next_action":next_action,
    }
    atomic_json(OUT_JSON, audit)
    atomic_json(STATE, {
        "run_id":"DX_step10b_public_ncp_source_probe_v1",
        "status":"COMPLETE",
        "updated_utc":now(),
        "audit_status":status,
        "next_action":next_action,
    })

    synced = sync_git()

    print("\n" + "="*108)
    print("✅ STEP 10B COMPLETE — SOURCE PROBE")
    print("Official source reachable       :", info["reachable"])
    print("HTTP status                     :", info["status_code"])
    print("Candidate download links        :", len(links))
    print("Mounted CC-CCII/NCP candidates  :", len(mounts))
    print("Pleural effusion training       : NOT AVAILABLE IN PUBLIC 750-SLICE RESOURCE")
    print("LongCIU / MedSeg                : PRESERVED AS EXTERNAL TEST")
    print("GitHub metadata sync            :", "SUCCESS" if synced else "CHECK REQUIRED")
    print("NEXT                            :", next_action)
    print("="*108)

    if not links.empty:
        print("\nTop candidate links:")
        print(links.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
