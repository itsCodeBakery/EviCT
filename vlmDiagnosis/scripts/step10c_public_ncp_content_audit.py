from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DX = ROOT / "vlmDiagnosis"
CFG = json.loads((DX / "config" / "step10c_public_ncp_content_audit.json").read_text(encoding="utf-8"))

WORK = Path("/kaggle/working/evict_dx_ncp_source")
ARCHIVE = WORK / CFG["source"]["expected_archive_name"]
EXTRACT = WORK / "ct_lesion_seg_extracted"

TABLES = DX / "tables"
AUDIT = DX / "artifacts" / "audit"
RUN = DX / "runs" / "DX_step10c_public_ncp_content_audit_v1"
for p in [WORK, TABLES, AUDIT, RUN]:
    p.mkdir(parents=True, exist_ok=True)

MEMBER_CSV = TABLES / "step10c_public_ncp_archive_members.csv"
IMAGE_CSV = TABLES / "step10c_public_ncp_image_inventory.csv"
PAIR_CSV = TABLES / "step10c_public_ncp_pairing_summary.csv"
AUDIT_JSON = AUDIT / "step10c_public_ncp_content_audit.json"
STATE_JSON = RUN / "STATE.json"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_WORDS = ("mask", "label", "seg", "annotation", "annot", "gt", "groundtruth")


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


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm(s: str):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def download():
    """Resumable downloader tolerant of temporary CNCB connection refusals.

    The CNCB server may close a long transfer and then refuse immediate reconnects.
    We therefore keep the existing partial file and launch multiple independent
    wget -c rounds with increasing cool-downs instead of treating one wget exit
    code as a fatal scientific-pipeline error.
    """
    url = CFG["source"]["direct_https"]
    print(f"Official archive: {url}")
    print(f"Destination     : {ARCHIVE}")

    expected_bytes = 884253723  # HTTP Content-Length observed from official server
    max_rounds = 30

    for round_idx in range(1, max_rounds + 1):
        existing = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0

        if existing == expected_bytes:
            print(f"✓ Archive already complete ({existing:,} bytes).")
            break
        if existing > expected_bytes:
            raise RuntimeError(
                f"Partial archive is larger than expected: {existing:,} > {expected_bytes:,}. "
                "Do not delete it; send this output for inspection."
            )

        pct = 100.0 * existing / expected_bytes
        print(
            f"\nDownload round {round_idx}/{max_rounds} — "
            f"already have {existing:,}/{expected_bytes:,} bytes ({pct:.2f}%)."
        )

        cmd = [
            "wget",
            "-c",
            "--tries=4",
            "--retry-connrefused",
            "--waitretry=10",
            "--timeout=60",
            "--read-timeout=60",
            "--progress=bar:force:noscroll",
            "-O",
            str(ARCHIVE),
            url,
        ]
        result = subprocess.run(cmd, check=False)

        existing_after = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
        if existing_after == expected_bytes:
            print(f"✓ Download complete ({existing_after:,} bytes).")
            break

        if result.returncode == 0 and existing_after != expected_bytes:
            print(
                f"⚠ wget returned success but archive size is {existing_after:,}; "
                f"expected {expected_bytes:,}. Continuing resume loop."
            )
        else:
            print(
                f"⚠ wget round {round_idx} ended with code {result.returncode}. "
                f"Partial archive preserved at {existing_after:,} bytes."
            )

        if round_idx == max_rounds:
            raise RuntimeError(
                "Official server remained unavailable after all resume rounds. "
                f"Partial archive is preserved at {ARCHIVE} ({existing_after:,} bytes). "
                "Rerun Step10C later; wget -c will continue from this exact byte."
            )

        # Give the remote server time to reopen the connection. Cap wait at 60 s.
        wait_s = min(10 + round_idx * 5, 60)
        print(f"Waiting {wait_s}s before the next resume attempt...")
        time.sleep(wait_s)

    if not ARCHIVE.exists() or ARCHIVE.stat().st_size == 0:
        raise RuntimeError("Archive download did not produce a non-empty file.")

    if ARCHIVE.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Archive is incomplete: {ARCHIVE.stat().st_size:,}/{expected_bytes:,} bytes. "
            "Rerun Step10C; the partial file is intentionally preserved."
        )


def extract_archive():
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at member: {bad}")
        members = zf.infolist()
        rows = []
        for m in members:
            rows.append({
                "member": m.filename,
                "bytes_uncompressed": m.file_size,
                "bytes_compressed": m.compress_size,
                "is_dir": m.is_dir(),
            })
        atomic_csv(MEMBER_CSV, pd.DataFrame(rows))

        if not EXTRACT.exists():
            EXTRACT.mkdir(parents=True, exist_ok=True)
        zf.extractall(EXTRACT)
    return members


def unique_signature(arr):
    if arr.ndim == 2:
        flat = arr.reshape(-1)
        vals = np.unique(flat)
        if len(vals) <= 32:
            return len(vals), ",".join(str(v) for v in vals.tolist())
        return len(vals), ""
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].reshape(-1, 3)
        # exact color count can be expensive; use unique on full 512^2 safely enough
        vals = np.unique(rgb, axis=0)
        if len(vals) <= 32:
            sig = ";".join(",".join(str(int(x)) for x in v) for v in vals.tolist())
            return len(vals), sig
        return len(vals), ""
    return -1, ""


def probable_group_id(path: Path):
    rel = path.relative_to(EXTRACT)
    parts = rel.parts
    # Prefer the immediate parent folder if it is informative.
    parent = norm(path.parent.name)
    generic = {"image","images","img","mask","masks","label","labels","annotation","annotations","ct","data"}
    if parent and parent not in generic:
        return f"parent:{parent}"

    s = norm(path.stem)
    # Remove common mask/image suffixes.
    for token in ["mask","label","seg","segmentation","annotation","annot","gt","image","img","ct"]:
        s = re.sub(rf"(^|_){token}($|_)", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    # If filename ends in a likely slice index, strip exactly one trailing integer.
    m = re.match(r"^(.*?)[_-]?(\d{1,4})$", s)
    if m and m.group(1):
        return f"prefix:{m.group(1)}"
    return f"stem:{s}"


def inspect_images():
    rows = []
    image_paths = sorted(
        p for p in EXTRACT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    print(f"Image-like files discovered: {len(image_paths)}")

    for p in image_paths:
        rel = str(p.relative_to(EXTRACT)).replace("\\", "/")
        low = rel.lower()
        word_mask = any(w in low for w in MASK_WORDS)
        try:
            with Image.open(p) as im:
                arr = np.array(im)
                width, height = im.size
                mode = im.mode
            n_unique, sig = unique_signature(arr)
            low_cardinality = (0 <= n_unique <= 16)
            probable_mask = bool(word_mask or low_cardinality)
            minv = float(np.min(arr))
            maxv = float(np.max(arr))
        except Exception as exc:
            width = height = -1
            mode = "READ_ERROR"
            n_unique = -1
            sig = ""
            low_cardinality = False
            probable_mask = word_mask
            minv = maxv = np.nan
            print(f"⚠ Could not inspect {rel}: {exc}")

        rows.append({
            "relative_path": rel,
            "filename": p.name,
            "stem": p.stem,
            "parent": p.parent.name,
            "suffix": p.suffix.lower(),
            "width": width,
            "height": height,
            "mode": mode,
            "min_value": minv,
            "max_value": maxv,
            "unique_value_or_color_count": n_unique,
            "small_unique_signature": sig,
            "mask_word_in_path": word_mask,
            "low_cardinality_candidate": low_cardinality,
            "probable_mask": probable_mask,
            "probable_group_id": probable_group_id(p),
            "bytes": p.stat().st_size,
        })

    df = pd.DataFrame(rows)
    atomic_csv(IMAGE_CSV, df)
    return df


def pairing_audit(df):
    if df.empty:
        out = pd.DataFrame(columns=["key", "image_count", "mask_count", "total_count", "status"])
        atomic_csv(PAIR_CSV, out)
        return out

    x = df.copy()
    # Canonical pairing key: normalized basename after removing common mask/image markers.
    def key(row):
        s = norm(row["stem"])
        for token in [
            "mask","label","seg","segmentation","annotation","annot","gt","groundtruth",
            "image","img","ct"
        ]:
            s = re.sub(rf"(^|_){token}($|_)", "_", s)
        return re.sub(r"_+", "_", s).strip("_")

    x["pair_key"] = x.apply(key, axis=1)
    records = []
    for k, g in x.groupby("pair_key"):
        masks = int(g.probable_mask.sum())
        imgs = int((~g.probable_mask).sum())
        status = "ONE_IMAGE_ONE_MASK" if imgs == 1 and masks == 1 else "REVIEW"
        records.append({
            "key": k,
            "image_count": imgs,
            "mask_count": masks,
            "total_count": len(g),
            "status": status,
        })
    out = pd.DataFrame(records).sort_values(["status", "key"])
    atomic_csv(PAIR_CSV, out)
    return out


def grouping_summary(df):
    if df.empty:
        return {}
    counts = df.groupby("probable_group_id").size()
    dist = Counter(counts.tolist())
    return {
        "probable_group_count": int(len(counts)),
        "files_per_probable_group_distribution": {str(k): int(v) for k, v in sorted(dist.items())},
        "groups_with_exactly_5_files": int((counts == 5).sum()),
        "groups_with_exactly_10_files": int((counts == 10).sum()),
    }


def sync_git():
    helper = DX / "scripts" / "git_sync_dx.py"
    r = subprocess.run(
        [sys.executable, str(helper), "Complete EViCT-Dx Step10C public NCP content audit"],
        cwd=str(ROOT),
        check=False,
    )
    return r.returncode == 0


def main():
    print("=" * 108)
    print("EViCT-Dx STEP 10C — PUBLIC NCP DOWNLOAD + EXACT CONTENT AUDIT")
    print("NO TRAINING")
    print("=" * 108)

    subprocess.run(
        [sys.executable, str(DX / "scripts" / "verify_core_lock.py")],
        cwd=str(ROOT),
        check=True,
    )

    download()
    print("✓ Download complete")
    print(f"Archive size: {ARCHIVE.stat().st_size / (1024**2):.2f} MiB")

    archive_sha = sha256(ARCHIVE)
    print("Archive SHA256:", archive_sha)

    members = extract_archive()
    print(f"✓ ZIP integrity passed; members={len(members)}")

    df = inspect_images()
    pairs = pairing_audit(df)
    groups = grouping_summary(df)

    expected = CFG["documented_contract"]
    image_like_count = int(len(df))
    mask_count = int(df.probable_mask.sum()) if not df.empty else 0
    nonmask_count = int((~df.probable_mask).sum()) if not df.empty else 0
    exact_512_count = int(((df.width == 512) & (df.height == 512)).sum()) if not df.empty else 0
    exact_pair_count = int((pairs.status == "ONE_IMAGE_ONE_MASK").sum()) if not pairs.empty else 0

    signatures = {}
    if not df.empty:
        mdf = df[df.probable_mask]
        for sig, n in mdf.small_unique_signature.fillna("").value_counts().items():
            if sig:
                signatures[str(sig)] = int(n)

    # Conservative decision: the archive can only be called content-consistent here.
    # Stable patient/scan grouping still requires explicit verification before split freeze.
    content_consistent = bool(
        ARCHIVE.exists()
        and len(members) > 0
        and image_like_count >= expected["expected_slices"]
        and mask_count > 0
        and nonmask_count > 0
    )

    status = (
        "OFFICIAL_ARCHIVE_DOWNLOADED_CONTENT_AUDIT_COMPLETE_GROUPING_REVIEW_REQUIRED"
        if content_consistent
        else "OFFICIAL_ARCHIVE_DOWNLOADED_CONTENT_STRUCTURE_NOT_YET_CONSISTENT_WITH_CONTRACT"
    )

    audit = {
        "project": "EViCT-Dx",
        "stage": "STEP_10C_PUBLIC_NCP_DOWNLOAD_AND_CONTENT_AUDIT",
        "completed_utc": now(),
        "status": status,
        "training_performed": False,
        "source_url": CFG["source"]["direct_https"],
        "archive_path_ephemeral_kaggle": str(ARCHIVE),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_sha,
        "zip_integrity_pass": True,
        "archive_member_count": len(members),
        "image_like_file_count": image_like_count,
        "probable_mask_file_count": mask_count,
        "probable_nonmask_image_file_count": nonmask_count,
        "files_512x512": exact_512_count,
        "one_image_one_mask_pair_count_by_conservative_heuristic": exact_pair_count,
        "mask_small_unique_signatures": signatures,
        "grouping_probe": groups,
        "documented_contract": expected,
        "content_consistent_with_public_750_slice_resource": content_consistent,
        "stable_patient_or_scan_grouping_verified": False,
        "annotation_class_mapping_verified": False,
        "important_note": (
            "Image/mask classification and grouping are conservative discovery heuristics. "
            "Step11 training remains blocked until the actual archive structure and mask-label semantics "
            "are reviewed and stable patient/scan grouping is frozen."
        ),
        "next_action": (
            "Review Step10C image inventory, pairing summary, directory structure and mask signatures; "
            "then freeze exact class mapping and patient/scan-disjoint split before Step11 training."
        ),
    }
    atomic_json(AUDIT_JSON, audit)
    atomic_json(STATE_JSON, {
        "run_id": "DX_step10c_public_ncp_content_audit_v1",
        "status": "COMPLETE",
        "audit_status": status,
        "updated_utc": now(),
        "archive_sha256": archive_sha,
        "training_allowed": False,
        "next_action": audit["next_action"],
    })

    synced = sync_git()

    print("\n" + "=" * 108)
    print("✅ STEP 10C COMPLETE — OFFICIAL NCP CONTENT AUDIT")
    print(f"Archive SHA256                 : {archive_sha}")
    print(f"ZIP members                    : {len(members)}")
    print(f"Image-like files               : {image_like_count}")
    print(f"Probable masks                 : {mask_count}")
    print(f"Probable CT/non-mask images    : {nonmask_count}")
    print(f"512×512 files                  : {exact_512_count}")
    print(f"1:1 pair keys                  : {exact_pair_count}")
    print(f"Probable groups                : {groups.get('probable_group_count', 0)}")
    print(f"Audit status                   : {status}")
    print("Patient/scan grouping verified : NO")
    print("Class mapping verified         : NO")
    print("Training                       : BLOCKED UNTIL STEP10D SPLIT/LABEL FREEZE")
    print(f"GitHub metadata sync           : {'SUCCESS' if synced else 'CHECK REQUIRED'}")
    print("=" * 108)

    if signatures:
        print("\nObserved small mask signatures (top 10):")
        for sig, n in list(signatures.items())[:10]:
            print(f"  {n:4d} files -> {sig}")


if __name__ == "__main__":
    main()
