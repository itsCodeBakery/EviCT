from __future__ import annotations

import base64
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from kaggle_secrets import UserSecretsClient


ROOT = Path("/kaggle/working/EviCT")
WORK = Path("/kaggle/working")

CFG = ROOT / "config/notebook07_teacher_student.json"
IMPL = ROOT / "config/notebook07_b050_seed17_branch_implementation.json"

STATE = ROOT / "artifacts/audit/notebook07_b050_seed42_training_state.md"

LABELED = (
    ROOT
    / "manifests/slice_loaders/seed_42/b050/labeled_slices.csv"
)

UNLABELED = (
    ROOT
    / "manifests/slice_loaders/seed_42/b050/unlabeled_slices.csv"
)

SELECTION = ROOT / "manifests/source_selection_slices.csv"

WARMUP_AUDIT = (
    ROOT
    / "artifacts/audit/notebook07_b050_seed42_warmup_final_durable.json"
)

BRANCH = (
    ROOT
    / "artifacts/large/notebook07/b050/seed_42/warmup/branch_step1000.pt"
)

PROTO = ROOT / "artifacts/audit/notebook06_text_prototypes.pt"

MODEL_CODE = ROOT / "src/evict/models/semantic_segformer.py"
METRICS_CODE = ROOT / "src/evict/metrics.py"

AUD = ROOT / "artifacts/audit"
TABLES = ROOT / "tables"
REPORTS = ROOT / "reports"

AUD.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
BUDGET = "b050"

METHODS = [
    "supervised",
    "confidence_only_ema",
    "agreement_filtered_ema",
]

START_STEP = 1000
MAX_STEP = 5000

VALIDATE_EVERY = 250
PATIENCE = 8
RECOVERY_EVERY = 50

ROLLING_STEPS = {2000, 3000, 4000}

MICRO = 4
LABELED_MICROS = 2
UNLABELED_MICROS = 2

LABELED_PER_UPDATE = 8
UNLABELED_PER_EMA_UPDATE = 8

EMA_DECAY = 0.99

ENC_LR = 1e-4
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.01

TEXT_WEIGHT = 0.1

CONFIDENCE_THRESHOLD = 0.95
DISAGREEMENT_THRESHOLD = 0.10
PSEUDO_THRESHOLD = 0.5

EXPECTED_BRANCH_SHA = (
    "c4b2c3450e73f30a6c10f1337cdebd2aea082ab6f528866d6bdab2bf893334e0"
)

REPO_OWNER = "itsCodeBakery"
REPO_NAME = "EViCT"


# ==========================================================================================
# Generic helpers
# ==========================================================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def stable_hash(obj):
    return hashlib.sha256(
        json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def load_module(path, name):

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod


def atomic_json(path, obj):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(str(path) + ".tmp")

    tmp.write_text(
        json.dumps(obj, indent=2) + "\n"
    )

    os.replace(tmp, path)


# ==========================================================================================
# Notebook-06 frozen utilities
# ==========================================================================================

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

nb06 = load_module(
    ROOT / "scripts/nb06_semantic_branch.py",
    "nb06_for_nb07_full",
)

base = nb06.base

from evict.models.semantic_segformer import (
    masked_supervised_loss,
    semantic_patch_auxiliary_loss,
)

from evict.metrics import (
    CaseMetricAccumulator,
    binary_segmentation_metrics,
    macro_case_summary,
    metrics_from_confusion,
)


# ==========================================================================================
# Secure Git synchronization
#
# Unlike the historical git_sync.py helper, this ALSO pushes a clean local
# branch that is ahead of origin/main.
# ==========================================================================================

def git_run(args, check=True):

    r = subprocess.run(
        ["git"] + list(args),
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and r.returncode != 0:

        print(r.stdout[-3000:])
        print(r.stderr[-3000:])

        raise RuntimeError(
            "Git command failed: "
            + " ".join(args)
        )

    return r


def sync_small(paths, message, token):

    paths = [
        Path(x)
        for x in paths
        if Path(x).exists()
    ]

    if paths:

        rels = [
            str(x.relative_to(ROOT))
            for x in paths
        ]

        git_run(
            ["add", "-f", "--"] + rels
        )

        staged = git_run(
            [
                "diff",
                "--cached",
                "--quiet",
            ],
            check=False,
        )

        if staged.returncode != 0:

            git_run([
                "commit",
                "-m",
                message,
            ])

    # Always attempt to push even if working tree is clean.
    git_run(
        ["fetch", "origin", "main"]
    )

    ancestor = git_run(
        [
            "merge-base",
            "--is-ancestor",
            "origin/main",
            "HEAD",
        ],
        check=False,
    )

    if ancestor.returncode != 0:

        # Only small metadata is tracked here.
        pull = git_run(
            [
                "pull",
                "--rebase",
                "origin",
                "main",
            ],
            check=False,
        )

        if pull.returncode != 0:

            print(
                "⚠ Metadata rebase deferred; "
                "large checkpoint durability remains in GitHub Releases."
            )

            git_run(
                ["rebase", "--abort"],
                check=False,
            )

            return False

    auth = base64.b64encode(
        f"itsCodeBakery:{token}".encode()
    ).decode()

    push = subprocess.run(
        [
            "git",
            "-c",
            f"http.extraHeader=AUTHORIZATION: basic {auth}",
            "push",
            "origin",
            "main",
        ],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if push.returncode != 0:

        print(
            "⚠ Metadata push deferred:",
            push.stderr[-2000:],
        )

        return False

    print("✓ Small metadata synchronized to GitHub")
    return True


# ==========================================================================================
# Protocol checks
# ==========================================================================================

def verify_protocol():

    required = [
        CFG,
        IMPL,
        STATE,
        LABELED,
        UNLABELED,
        SELECTION,
        WARMUP_AUDIT,
        BRANCH,
        PROTO,
    ]

    for path in required:
        assert path.exists(), path

    state = STATE.read_text(errors="replace")

    allowed = [
        "NOTEBOOK_07_B050_SEED42_WARMUP_FROZEN",
        "NOTEBOOK_07_B050_SEED42_",
        "NOTEBOOK_07_B050_SEED42_THREE_WAY_COMPLETE",
    ]

    assert any(x in state for x in allowed)

    assert "Calibration accessed:\n\nNO" in state
    assert "Target / MedSeg accessed:\n\nNO" in state
    assert "Target lock:\n\nACTIVE" in state

    cfg = json.loads(CFG.read_text())

    cfg_hash = cfg["config_hash"]

    check = dict(cfg)
    check.pop("config_hash", None)

    assert stable_hash(check) == cfg_hash

    impl = json.loads(IMPL.read_text())

    impl_hash = impl["implementation_hash"]

    check_impl = dict(impl)
    check_impl.pop("implementation_hash", None)

    assert stable_hash(check_impl) == impl_hash

    warmup = json.loads(WARMUP_AUDIT.read_text())

    assert warmup["status"] == "DURABLE_COMPLETE"
    assert warmup["branch_checkpoint_sha256"] == EXPECTED_BRANCH_SHA
    assert sha(BRANCH) == EXPECTED_BRANCH_SHA

    labeled = pd.read_csv(LABELED)
    unlabeled = pd.read_csv(UNLABELED)

    assert labeled["case_id"].astype(str).nunique() == 6
    assert unlabeled["case_id"].astype(str).nunique() == 6

    visible = (
        labeled["label_visible"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    hidden_visible = (
        unlabeled["label_visible"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    assert visible.all()
    assert not hidden_visible.any()

    forbidden = [
        "infection_cache_path",
        "lung_cache_path",
        "ground_truth_path",
        "gt_path",
        "lesion_area",
        "lesion_presence",
    ]

    for col in forbidden:
        assert col not in unlabeled.columns

    for col in unlabeled.columns:

        if unlabeled[col].dtype == object:

            strings = (
                unlabeled[col]
                .fillna("")
                .astype(str)
                .str.lower()
            )

            assert not strings.str.contains(
                "gt_vault",
                regex=False,
            ).any()

    lc = set(labeled["case_id"].astype(str))
    uc = set(unlabeled["case_id"].astype(str))

    assert lc.isdisjoint(uc)

    print("✓ Config hash               :", cfg_hash)
    print("✓ Implementation hash       :", impl_hash)
    print("✓ Frozen branch SHA         :", EXPECTED_BRANCH_SHA)
    print("✓ Visible / hidden patients : 6 / 6")
    print("✓ Hidden mask leakage       : NONE")
    print("✓ Calibration accessed      : NO")
    print("✓ Target / MedSeg accessed  : NO")

    return (
        cfg,
        impl,
        warmup,
        labeled,
        unlabeled,
    )


# ==========================================================================================
# Scheduler — exact continuation of 0 -> 5000 schedule
# ==========================================================================================

def scheduler_multiplier(step):

    warm = 200
    total = 5000

    if step < warm:
        return float(step + 1) / float(warm)

    progress = (
        (step - warm)
        / float(total - warm)
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return 0.5 * (
        1.0
        + math.cos(math.pi * progress)
    )


# ==========================================================================================
# Unlabeled image-only sampler
# ==========================================================================================

class UnlabeledCaseStore:

    def __init__(self, df):

        self.case_ids = sorted(
            df["case_id"]
            .astype(str)
            .unique()
            .tolist()
        )

        self.cases = {}

        for case_id in self.case_ids:

            rows = (
                df[
                    df["case_id"].astype(str)
                    == case_id
                ]
                .sort_values("image_array_index")
            )

            first = rows.iloc[0]

            self.cases[case_id] = {
                "image": np.load(
                    str(first["image_path"]),
                    mmap_mode="r",
                ),

                "valid": np.load(
                    str(first["valid_mask_path"]),
                    mmap_mode="r",
                ),

                "all_z": (
                    rows["image_array_index"]
                    .astype(int)
                    .to_numpy()
                ),
            }


class UnlabeledSampler:

    def __init__(self, store, seed):

        self.store = store
        self.case_ids = list(store.case_ids)
        self.rng = np.random.default_rng(seed)
        self.samples_seen = 0

    def sample(self, batch_size):

        images = []
        valids = []

        for _ in range(batch_size):

            cid = str(
                self.rng.choice(
                    self.case_ids
                )
            )

            case = self.store.cases[cid]

            z = int(
                self.rng.choice(
                    case["all_z"]
                )
            )

            images.append(
                np.asarray(
                    case["image"][z],
                    dtype=np.float32,
                )[None]
            )

            valids.append(
                np.asarray(
                    case["valid"],
                    dtype=np.float32,
                )[None]
            )

        self.samples_seen += batch_size

        return (
            np.stack(images),
            np.stack(valids),
        )

    def state_dict(self):

        return {
            "bit_generator_state":
                self.rng.bit_generator.state,

            "samples_seen":
                self.samples_seen,
        }

    def load_state_dict(self, state):

        self.rng.bit_generator.state = (
            state["bit_generator_state"]
        )

        self.samples_seen = int(
            state["samples_seen"]
        )


# ==========================================================================================
# Unlabeled views
# ==========================================================================================

def shared_geometry(images, valids):

    out_i = []
    out_v = []

    for i in range(images.shape[0]):

        image = images[i]
        valid = valids[i]

        angle = random.uniform(
            -5.0,
            5.0,
        )

        scale = random.uniform(
            0.95,
            1.05,
        )

        tx = int(
            round(
                random.uniform(
                    -0.025,
                    0.025,
                )
                * 336
            )
        )

        ty = int(
            round(
                random.uniform(
                    -0.025,
                    0.025,
                )
                * 336
            )
        )

        image = TVF.affine(
            image,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        valid = TVF.affine(
            valid,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

        valid = (
            valid > 0.5
        ).float()

        image = (
            torch.clamp(
                image,
                0.0,
                1.0,
            )
            * valid
        )

        out_i.append(image)
        out_v.append(valid)

    return (
        torch.stack(out_i),
        torch.stack(out_v),
    )


def weak_view(x, valid):

    out = []

    for i in range(x.shape[0]):

        z = x[i].clone()

        z = TVF.adjust_brightness(
            z,
            random.uniform(
                0.95,
                1.05,
            ),
        )

        z = TVF.adjust_contrast(
            z,
            random.uniform(
                0.95,
                1.05,
            ),
        )

        z = torch.clamp(
            z,
            0.0,
            1.0,
        )

        z = TVF.adjust_gamma(
            z,
            gamma=random.uniform(
                0.95,
                1.05,
            ),
        )

        out.append(
            torch.clamp(
                z,
                0.0,
                1.0,
            )
            * valid[i]
        )

    return torch.stack(out)


def strong_view(x, valid):

    out = []

    for i in range(x.shape[0]):

        z = x[i].clone()

        z = TVF.adjust_brightness(
            z,
            random.uniform(
                0.85,
                1.15,
            ),
        )

        z = TVF.adjust_contrast(
            z,
            random.uniform(
                0.85,
                1.15,
            ),
        )

        z = torch.clamp(
            z,
            0.0,
            1.0,
        )

        z = TVF.adjust_gamma(
            z,
            gamma=random.uniform(
                0.80,
                1.20,
            ),
        )

        if random.random() < 0.50:

            z = (
                z
                + torch.randn_like(z)
                * random.uniform(
                    0.0,
                    0.03,
                )
            )

        if random.random() < 0.30:

            z = TVF.gaussian_blur(
                z,
                kernel_size=[3, 3],
                sigma=[0.1, 1.0],
            )

        out.append(
            torch.clamp(
                z,
                0.0,
                1.0,
            )
            * valid[i]
        )

    return torch.stack(out)


# ==========================================================================================
# Pseudo-label loss
# ==========================================================================================

def balanced_pseudo_bce(
    logits,
    pseudo,
    accepted,
    valid,
):

    mask = (
        accepted
        & (valid > 0.5)
    )

    positive = (
        mask
        & (pseudo > 0.5)
    )

    negative = (
        mask
        & (pseudo <= 0.5)
    )

    loss_map = (
        F.binary_cross_entropy_with_logits(
            logits.float(),
            pseudo.float(),
            reduction="none",
        )
    )

    terms = []

    if positive.any():
        terms.append(
            loss_map[positive].mean()
        )

    if negative.any():
        terms.append(
            loss_map[negative].mean()
        )

    if not terms:

        return logits.sum() * 0.0

    return torch.stack(terms).mean()


# ==========================================================================================
# EMA
# ==========================================================================================

@torch.no_grad()
def ema_update(teacher, student):

    student_parameters = dict(
        student.named_parameters()
    )

    for name, teacher_parameter in (
        teacher.named_parameters()
    ):

        teacher_parameter.mul_(
            EMA_DECAY
        ).add_(
            student_parameters[name].detach(),
            alpha=1.0 - EMA_DECAY,
        )

    student_buffers = dict(
        student.named_buffers()
    )

    for name, teacher_buffer in (
        teacher.named_buffers()
    ):

        student_buffer = (
            student_buffers[name].detach()
        )

        if torch.is_floating_point(
            teacher_buffer
        ):

            teacher_buffer.mul_(
                EMA_DECAY
            ).add_(
                student_buffer,
                alpha=1.0 - EMA_DECAY,
            )

        else:

            teacher_buffer.copy_(
                student_buffer
            )


# ==========================================================================================
# RNG isolation
# ==========================================================================================

def new_unlabeled_rng_state():

    current = base.capture_rng_state()

    base.reset_rng(
        420017
    )

    state = base.capture_rng_state()

    base.restore_rng_state(
        current
    )

    return state


# ==========================================================================================
# Validation
# ==========================================================================================

def evaluate(model, device, selection):

    result = base.validate(
        model=model,
        device=device,
        selection_df=selection,
        masked_supervised_loss=masked_supervised_loss,
        CaseMetricAccumulator=CaseMetricAccumulator,
        binary_segmentation_metrics=binary_segmentation_metrics,
        metrics_from_confusion=metrics_from_confusion,
        macro_case_summary=macro_case_summary,
    )

    return result


# ==========================================================================================
# Paths
# ==========================================================================================

def method_paths(method):

    run = (
        ROOT
        / "artifacts/large/notebook07"
        / BUDGET
        / "seed_42"
        / method
    )

    run.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"notebook07_b050_seed42_{method}"
    )

    return {
        "run": run,

        "last":
            run / "last.pt",

        "recovery":
            run / "recovery.pt",

        "best_student":
            run / "best_student.pt",

        "best_teacher":
            run / "best_teacher.pt",

        "selected":
            run / "selected_best.pt",

        "summary":
            run / "final_summary.json",

        "train":
            AUD / f"{prefix}_train_log.csv",

        "selection":
            AUD / f"{prefix}_selection_metrics.csv",

        "cases":
            AUD / f"{prefix}_selection_case_metrics.csv",

        "rolling_audit":
            AUD / f"{prefix}_rolling_durable.json",

        "final_audit":
            AUD / f"{prefix}_final_durable.json",
    }


# ==========================================================================================
# Hash bundle
# ==========================================================================================

def hash_bundle(cfg, impl):

    return {
        "config_hash":
            cfg["config_hash"],

        "implementation_hash":
            impl["implementation_hash"],

        "warmup_branch_sha256":
            EXPECTED_BRANCH_SHA,

        "model_code_sha256":
            sha(MODEL_CODE),

        "metrics_code_sha256":
            sha(METRICS_CODE),

        "prototype_sha256":
            sha(PROTO),

        "labeled_manifest_sha256":
            sha(LABELED),

        "unlabeled_manifest_sha256":
            sha(UNLABELED),

        "selection_manifest_sha256":
            sha(SELECTION),

        "runner_sha256":
            sha(Path(__file__)),
    }


# ==========================================================================================
# Best-model save
# ==========================================================================================

def save_best(
    path,
    model,
    method,
    model_type,
    step,
    score,
    hashes,
):

    base.atomic_torch_save(
        {
            "stage":
                "NOTEBOOK_07_BRANCH",

            "method":
                method,

            "model_type":
                model_type,

            "budget_id":
                BUDGET,

            "seed":
                SEED,

            "global_step":
                int(step),

            "macro_case_dice":
                float(score),

            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },

            **hashes,

            "calibration_accessed":
                False,

            "target_accessed":
                False,
        },
        path,
    )


# ==========================================================================================
# Recovery state
# ==========================================================================================

def save_recovery(
    path,
    *,
    method,
    student,
    teacher,
    optimizer,
    scheduler,
    labeled_sampler,
    unlabeled_sampler,
    unlabeled_rng_state,
    step,
    student_best_score,
    student_best_step,
    teacher_best_score,
    teacher_best_step,
    branch_best_score,
    branch_best_type,
    patience_count,
    last_validation_step,
    labeled_images_seen,
    unlabeled_images_seen,
    hashes,
):

    payload = {
        "stage":
            "NOTEBOOK_07_BRANCH",

        "method":
            method,

        "budget_id":
            BUDGET,

        "seed":
            SEED,

        "global_step":
            int(step),

        "student_state_dict": {
            k: v.detach().cpu()
            for k, v in student.state_dict().items()
        },

        "teacher_state_dict": (
            None
            if teacher is None
            else {
                k: v.detach().cpu()
                for k, v in teacher.state_dict().items()
            }
        ),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "labeled_sampler_state":
            labeled_sampler.state_dict(),

        "unlabeled_sampler_state": (
            None
            if unlabeled_sampler is None
            else unlabeled_sampler.state_dict()
        ),

        # Global RNG here is the labeled-stream RNG.
        "labeled_rng_state":
            base.capture_rng_state(),

        "unlabeled_rng_state":
            unlabeled_rng_state,

        "student_best_score":
            float(student_best_score),

        "student_best_step":
            int(student_best_step),

        "teacher_best_score": (
            None
            if teacher_best_score is None
            else float(teacher_best_score)
        ),

        "teacher_best_step": (
            None
            if teacher_best_step is None
            else int(teacher_best_step)
        ),

        "branch_best_score":
            float(branch_best_score),

        "branch_best_type":
            branch_best_type,

        "patience_count":
            int(patience_count),

        "last_validation_step":
            int(last_validation_step),

        "labeled_images_seen":
            int(labeled_images_seen),

        "unlabeled_images_seen":
            int(unlabeled_images_seen),

        "amp_enabled":
            False,

        "scaler_state_dict":
            None,

        **hashes,

        "calibration_accessed":
            False,

        "target_accessed":
            False,

        "timestamp_utc":
            now(),
    }

    base.atomic_torch_save(
        payload,
        path,
    )


def check_recovery(q, method, hashes):

    assert q["stage"] == "NOTEBOOK_07_BRANCH"
    assert q["method"] == method
    assert q["budget_id"] == BUDGET
    assert int(q["seed"]) == SEED

    for key, value in hashes.items():

        assert q[key] == value, (
            f"Recovery hash mismatch: {key}"
        )

    assert q["calibration_accessed"] is False
    assert q["target_accessed"] is False


# ==========================================================================================
# GitHub Release restore
# ==========================================================================================

def headers(token):

    return {
        "Authorization":
            f"Bearer {token}",

        "Accept":
            "application/vnd.github+json",

        "X-GitHub-Api-Version":
            "2022-11-28",
    }


def release_by_tag(token, tag):

    r = requests.get(
        (
            f"https://api.github.com/repos/"
            f"{REPO_OWNER}/{REPO_NAME}/"
            f"releases/tags/{tag}"
        ),
        headers=headers(token),
        timeout=60,
    )

    if r.status_code == 404:
        return None

    r.raise_for_status()

    return r.json()


def find_final_release(token, method):

    prefix = (
        f"evict-nb07-b050-seed42-"
        f"{method}-final-step"
    )

    matches = []

    for page in range(1, 5):

        r = requests.get(
            (
                f"https://api.github.com/repos/"
                f"{REPO_OWNER}/{REPO_NAME}/releases"
            ),
            headers=headers(token),
            params={
                "per_page": 100,
                "page": page,
            },
            timeout=60,
        )

        r.raise_for_status()

        data = r.json()

        if not data:
            break

        for rel in data:

            tag = str(
                rel.get("tag_name", "")
            )

            if tag.startswith(prefix):

                m = re.fullmatch(
                    re.escape(prefix) + r"(\d+)",
                    tag,
                )

                if m:

                    matches.append(
                        (
                            str(
                                rel.get(
                                    "published_at",
                                    "",
                                )
                            ),
                            int(m.group(1)),
                            rel,
                        )
                    )

    if not matches:
        return None

    # First published final is canonical.
    return sorted(
        matches,
        key=lambda x: (
            x[0],
            x[1],
        ),
    )[0][2]


def restore_release_archive(
    token,
    release,
):

    assets = {
        a["name"]: a
        for a in release.get(
            "assets",
            []
        )
    }

    tar_name = next(
        (
            n
            for n in assets
            if n.endswith(
                "_Recovery.tar"
            )
        ),
        None,
    )

    sha_name = next(
        (
            n
            for n in assets
            if n.endswith(
                ".tar.sha256"
            )
        ),
        None,
    )

    if not tar_name or not sha_name:

        raise RuntimeError(
            "Incomplete release assets."
        )

    s = requests.get(
        assets[sha_name][
            "browser_download_url"
        ],
        headers=headers(token),
        timeout=60,
    )

    s.raise_for_status()

    expected = s.text.strip()

    assert re.fullmatch(
        r"[0-9a-fA-F]{64}",
        expected,
    )

    destination = (
        WORK / tar_name
    )

    if (
        destination.exists()
        and sha(destination) != expected
    ):
        destination.unlink()

    if not destination.exists():

        print(
            "Downloading recovery      :",
            tar_name,
        )

        with requests.get(
            assets[tar_name][
                "browser_download_url"
            ],
            headers=headers(token),
            stream=True,
            timeout=(60, 3600),
        ) as r:

            r.raise_for_status()

            part = Path(
                str(destination)
                + ".part"
            )

            part.unlink(
                missing_ok=True
            )

            with part.open("wb") as f:

                for chunk in r.iter_content(
                    chunk_size=(
                        8 * 1024 * 1024
                    )
                ):

                    if chunk:
                        f.write(chunk)

            assert sha(part) == expected

            os.replace(
                part,
                destination,
            )

    base.safe_extract_tar(
        destination,
        WORK,
    )

    destination.unlink(
        missing_ok=True
    )

    return expected


# ==========================================================================================
# Recovery archive creation / publication
# ==========================================================================================

def create_archive(
    method,
    p,
    step,
    final,
):

    if final:

        stem = (
            f"EViCT_Notebook07_b050_seed42_"
            f"{method}_final_step{step}_Recovery"
        )

    else:

        stem = (
            f"EViCT_Notebook07_b050_seed42_"
            f"{method}_Rolling_Recovery"
        )

    tar_path = (
        WORK / f"{stem}.tar"
    )

    sha_path = Path(
        str(tar_path)
        + ".sha256"
    )

    tar_path.unlink(
        missing_ok=True
    )

    sha_path.unlink(
        missing_ok=True
    )

    files = [
        p["recovery"],
        p["best_student"],
        p["train"],
        p["selection"],
        p["cases"],
    ]

    if p["best_teacher"].exists():
        files.append(
            p["best_teacher"]
        )

    if p["selected"].exists():
        files.append(
            p["selected"]
        )

    if p["summary"].exists():
        files.append(
            p["summary"]
        )

    with tarfile.open(
        tar_path,
        "w",
    ) as tar:

        for source in files:

            if source.exists():

                tar.add(
                    source,
                    arcname=str(
                        Path("EViCT")
                        / source.relative_to(
                            ROOT
                        )
                    ),
                )

    digest = sha(
        tar_path
    )

    base.atomic_text(
        sha_path,
        digest + "\n",
    )

    return (
        tar_path,
        sha_path,
        digest,
    )


def publish_release(
    token,
    method,
    p,
    step,
    final,
    best_score,
):

    tar_path, sha_path, digest = (
        create_archive(
            method,
            p,
            step,
            final,
        )
    )

    if final:

        tag = (
            f"evict-nb07-b050-seed42-"
            f"{method}-final-step{step}"
        )

        title = (
            f"EviCT Notebook 07 — b050 "
            f"seed42 {method} — final step {step}"
        )

    else:

        tag = (
            f"evict-nb07-b050-seed42-"
            f"{method}-rolling"
        )

        title = (
            f"EviCT Notebook 07 — b050 "
            f"seed42 {method} — rolling"
        )

    body = (
        f"Notebook 07 b050 seed42 {method}. "
        f"Global step={step}. "
        f"Best source-selection macro-case Dice="
        f"{best_score:.8f}. "
        f"Hidden fitting masks, calibration, "
        f"and target/MedSeg were not accessed."
    )

    release, gh_headers, api = (
        base.create_or_get_release(
            token,
            tag,
            title,
            body,
        )
    )

    base.upload_asset(
        release=release,
        headers=gh_headers,
        api=api,
        file_path=tar_path,
        content_type="application/x-tar",
    )

    base.upload_asset(
        release=release,
        headers=gh_headers,
        api=api,
        file_path=sha_path,
        content_type="text/plain",
    )

    tar_path.unlink()
    sha_path.unlink()

    return (
        release,
        digest,
    )


# ==========================================================================================
# State
# ==========================================================================================

def write_state(
    method,
    step,
    best_score,
    best_type,
    patience_count,
):

    text = f"""# EviCT Execution State

## Current stage

NOTEBOOK_07_B050_SEED42_{method.upper()}_RUNNING

## Notebook 07

Budget:

50 percent visible fitting masks

Seed:

17

Method:

{method}

Global optimizer step:

{step}

Best source-selection macro-case Dice:

{best_score:.8f}

Best candidate:

{best_type}

Patience:

{patience_count} / {PATIENCE}

## Common initialization

Warm-up terminal step:

1000

Warm-up branch SHA-256:

{EXPECTED_BRANCH_SHA}

## Isolation

Hidden fitting masks used:

NO

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Continue the frozen Notebook-07 b050 seed42 comparison without changing
pseudo-label thresholds, patient split, text prototypes, or evaluation protocol.
"""

    base.atomic_text(
        STATE,
        text,
    )


# ==========================================================================================
# Build model / optimizer
# ==========================================================================================

def build_student(snapshot, proto, warm, device):

    model = nb06.build(
        snapshot,
        proto,
        "real_text",
        device,
    )

    model.load_state_dict(
        warm["student_state_dict"]
    )

    return model


def build_optimizer_scheduler(
    model,
    warm,
    device,
):

    encoder = list(
        model.encoder.parameters()
    )

    head = [
        p
        for name, p in model.named_parameters()
        if not name.startswith(
            "encoder."
        )
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder,
                "lr": ENC_LR,
            },
            {
                "params": head,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=scheduler_multiplier,
        )
    )

    optimizer.load_state_dict(
        warm["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        warm["scheduler_state_dict"]
    )

    for state in optimizer.state.values():

        for key, value in list(
            state.items()
        ):

            if torch.is_tensor(value):
                state[key] = value.to(
                    device
                )

    return (
        optimizer,
        scheduler,
    )


# ==========================================================================================
# One complete method
# ==========================================================================================

def run_method(
    *,
    method,
    cfg,
    impl,
    warmup_audit,
    labeled_df,
    unlabeled_df,
    selection_df,
    snapshot,
    proto,
    device,
    token,
    hashes,
):

    p = method_paths(
        method
    )

    print()
    print("=" * 118)
    print(
        f"NOTEBOOK 07 — b050 seed42 — {method}"
    )
    print("=" * 118)

    # ----------------------------------------------------------------------
    # Already complete in small Git metadata
    # ----------------------------------------------------------------------

    if p["final_audit"].exists():

        final = json.loads(
            p["final_audit"].read_text()
        )

        if (
            final.get("status")
            == "DURABLE_COMPLETE"
        ):

            print(
                f"✓ {method:<26}: SKIP COMPLETE"
            )

            return final

    # ----------------------------------------------------------------------
    # Restore final release if Git metadata was lost
    # ----------------------------------------------------------------------

    final_release = (
        find_final_release(
            token,
            method,
        )
    )

    if final_release is not None:

        digest = restore_release_archive(
            token,
            final_release,
        )

        if p["summary"].exists():

            summary = json.loads(
                p["summary"].read_text()
            )

            final = {
                **summary,

                "status":
                    "DURABLE_COMPLETE",

                "release_url":
                    final_release[
                        "html_url"
                    ],

                "archive_sha256":
                    digest,
            }

            atomic_json(
                p["final_audit"],
                final,
            )

            sync_small(
                [
                    p["final_audit"],
                    p["train"],
                    p["selection"],
                    p["cases"],
                    STATE,
                ],
                (
                    f"Restore Notebook 07 "
                    f"b050 seed42 {method} final metadata"
                ),
                token,
            )

            print(
                f"✓ {method:<26}: RESTORED FINAL"
            )

            return final

    # ----------------------------------------------------------------------
    # Restore rolling state if local recovery is gone
    # ----------------------------------------------------------------------

    if not p["recovery"].exists():

        rolling_tag = (
            f"evict-nb07-b050-seed42-"
            f"{method}-rolling"
        )

        rolling_release = release_by_tag(
            token,
            rolling_tag,
        )

        if rolling_release is not None:

            restore_release_archive(
                token,
                rolling_release,
            )

            print(
                f"✓ {method:<26}: rolling recovery restored"
            )

    # ----------------------------------------------------------------------
    # Every branch starts from exact frozen warm-up.
    # ----------------------------------------------------------------------

    warm = torch.load(
        BRANCH,
        map_location="cpu",
        weights_only=False,
    )

    assert int(
        warm["global_step"]
    ) == 1000

    assert (
        sha(BRANCH)
        == EXPECTED_BRANCH_SHA
    )

    student = build_student(
        snapshot,
        proto,
        warm,
        device,
    )

    optimizer, scheduler = (
        build_optimizer_scheduler(
            student,
            warm,
            device,
        )
    )

    labeled_store = (
        base.FittingCaseStore(
            labeled_df
        )
    )

    labeled_sampler = (
        base.PatientUniformSampler(
            labeled_store,
            SEED,
        )
    )

    labeled_sampler.load_state_dict(
        copy.deepcopy(
            warm[
                "labeled_sampler_state"
            ]
        )
    )

    teacher = None
    unlabeled_store = None
    unlabeled_sampler = None

    is_ema = (
        method != "supervised"
    )

    if is_ema:

        teacher = copy.deepcopy(
            student
        )

        teacher.eval()

        for parameter in teacher.parameters():
            parameter.requires_grad_(
                False
            )

        unlabeled_store = (
            UnlabeledCaseStore(
                unlabeled_df
            )
        )

        unlabeled_sampler = (
            UnlabeledSampler(
                unlabeled_store,
                42017,
            )
        )

    # ----------------------------------------------------------------------
    # Defaults at branch point
    # ----------------------------------------------------------------------

    step = 1000

    warm_score = float(
        warmup_audit[
            "best_source_selection_macro_case_dice"
        ]
    )

    student_best_score = (
        warm_score
    )

    student_best_step = 1000

    teacher_best_score = (
        warm_score
        if is_ema
        else None
    )

    teacher_best_step = (
        1000
        if is_ema
        else None
    )

    branch_best_score = (
        warm_score
    )

    branch_best_type = (
        "student"
    )

    patience_count = 0
    last_validation = 1000

    labeled_images_seen = int(
        warm[
            "labeled_images_seen"
        ]
    )

    unlabeled_images_seen = 0

    # Exact continuation of labeled global RNG.
    base.restore_rng_state(
        copy.deepcopy(
            warm["rng_state"]
        )
    )

    unlabeled_rng_state = (
        new_unlabeled_rng_state()
        if is_ema
        else None
    )

    train_rows = []
    selection_rows = []
    case_rows = []

    # ----------------------------------------------------------------------
    # Resume branch recovery
    # ----------------------------------------------------------------------

    if p["recovery"].exists():

        q = torch.load(
            p["recovery"],
            map_location="cpu",
            weights_only=False,
        )

        check_recovery(
            q,
            method,
            hashes,
        )

        student.load_state_dict(
            q["student_state_dict"]
        )

        if is_ema:

            assert (
                q["teacher_state_dict"]
                is not None
            )

            teacher.load_state_dict(
                q["teacher_state_dict"]
            )

            teacher.eval()

        optimizer.load_state_dict(
            q["optimizer_state_dict"]
        )

        for state in optimizer.state.values():

            for key, value in list(
                state.items()
            ):

                if torch.is_tensor(value):
                    state[key] = value.to(
                        device
                    )

        scheduler.load_state_dict(
            q["scheduler_state_dict"]
        )

        labeled_sampler.load_state_dict(
            q[
                "labeled_sampler_state"
            ]
        )

        if is_ema:

            unlabeled_sampler.load_state_dict(
                q[
                    "unlabeled_sampler_state"
                ]
            )

            unlabeled_rng_state = (
                q["unlabeled_rng_state"]
            )

        base.restore_rng_state(
            q["labeled_rng_state"]
        )

        step = int(
            q["global_step"]
        )

        student_best_score = float(
            q["student_best_score"]
        )

        student_best_step = int(
            q["student_best_step"]
        )

        teacher_best_score = (
            None
            if q["teacher_best_score"]
            is None
            else float(
                q["teacher_best_score"]
            )
        )

        teacher_best_step = (
            None
            if q["teacher_best_step"]
            is None
            else int(
                q["teacher_best_step"]
            )
        )

        branch_best_score = float(
            q["branch_best_score"]
        )

        branch_best_type = str(
            q["branch_best_type"]
        )

        patience_count = int(
            q["patience_count"]
        )

        last_validation = int(
            q["last_validation_step"]
        )

        labeled_images_seen = int(
            q["labeled_images_seen"]
        )

        unlabeled_images_seen = int(
            q["unlabeled_images_seen"]
        )

        if p["train"].exists():

            train_rows = (
                pd.read_csv(
                    p["train"]
                )
                .query(
                    "global_step <= @step"
                )
                .to_dict("records")
            )

        if p["selection"].exists():

            selection_rows = (
                pd.read_csv(
                    p["selection"]
                )
                .query(
                    "global_step <= @last_validation"
                )
                .to_dict("records")
            )

        if p["cases"].exists():

            case_rows = (
                pd.read_csv(
                    p["cases"]
                )
                .query(
                    "global_step <= @last_validation"
                )
                .to_dict("records")
            )

        print(
            f"✓ Resumed {method:<18}: "
            f"step {step}"
        )

        del q

    else:

        # Clean initialization of best candidates at the common branch point.
        save_best(
            p["best_student"],
            student,
            method,
            "student",
            1000,
            warm_score,
            hashes,
        )

        if is_ema:

            save_best(
                p["best_teacher"],
                teacher,
                method,
                "teacher",
                1000,
                warm_score,
                hashes,
            )

        print(
            f"✓ New {method:<22}: "
            f"step 1000"
        )

    # ----------------------------------------------------------------------
    # Recovery save helper
    # ----------------------------------------------------------------------

    def save_state(path):

        save_recovery(
            path,
            method=method,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            scheduler=scheduler,
            labeled_sampler=labeled_sampler,
            unlabeled_sampler=unlabeled_sampler,
            unlabeled_rng_state=unlabeled_rng_state,
            step=step,
            student_best_score=student_best_score,
            student_best_step=student_best_step,
            teacher_best_score=teacher_best_score,
            teacher_best_step=teacher_best_step,
            branch_best_score=branch_best_score,
            branch_best_type=branch_best_type,
            patience_count=patience_count,
            last_validation_step=last_validation,
            labeled_images_seen=labeled_images_seen,
            unlabeled_images_seen=unlabeled_images_seen,
            hashes=hashes,
        )

    stop_reason = (
        "MAX_UPDATES"
    )

    bar = tqdm(
        total=MAX_STEP - step,
        desc=f"NB07 {method}",
        unit="update",
    )

    # ======================================================================================
    # Training loop
    # ======================================================================================

    while step < MAX_STEP:

        student.train()

        if teacher is not None:
            teacher.eval()

        optimizer.zero_grad(
            set_to_none=True
        )

        supervised_loss_value = 0.0
        text_loss_value = 0.0

        # ------------------------------------------------------------------
        # Labeled stream — exactly 8 images/update.
        # ------------------------------------------------------------------

        for _ in range(
            LABELED_MICROS
        ):

            xn, yn, vn = (
                labeled_sampler.sample(
                    MICRO
                )
            )

            x = torch.from_numpy(
                xn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            y = torch.from_numpy(
                yn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            valid = torch.from_numpy(
                vn
            ).to(
                device=device,
                dtype=torch.float32,
            )

            x, y, valid = base.augment(
                x,
                y,
                valid,
            )

            output = student(
                base.normalize_batch(
                    x,
                    device,
                )
            )

            supervised = (
                masked_supervised_loss(
                    output["logits"],
                    y,
                    valid,
                )
            )

            text_aux = (
                semantic_patch_auxiliary_loss(
                    output[
                        "semantic_quarter_logits"
                    ],
                    y,
                    valid,
                )
            )

            labeled_loss = (
                supervised["loss"]
                + TEXT_WEIGHT
                * text_aux["loss"]
            )

            (
                labeled_loss
                / LABELED_MICROS
            ).backward()

            supervised_loss_value += (
                float(
                    supervised[
                        "loss"
                    ]
                    .detach()
                    .cpu()
                )
                / LABELED_MICROS
            )

            text_loss_value += (
                float(
                    text_aux[
                        "loss"
                    ]
                    .detach()
                    .cpu()
                )
                / LABELED_MICROS
            )

            del (
                x,
                y,
                valid,
                output,
                supervised,
                text_aux,
                labeled_loss,
            )

        # ------------------------------------------------------------------
        # Isolate labeled RNG before unlabeled stochasticity.
        # ------------------------------------------------------------------

        labeled_rng_after = (
            base.capture_rng_state()
        )

        unsupervised_loss_value = 0.0

        accepted_fg = 0
        accepted_bg = 0
        valid_unlabeled_pixels = 0

        confidence_accept_pixels = 0
        agreement_accept_pixels = 0

        next_step = step + 1
        post_warmup_step = (
            next_step - 1000
        )

        lambda_u = (
            0.0
            if not is_ema
            else min(
                1.0,
                post_warmup_step
                / 1000.0,
            )
        )

        # ------------------------------------------------------------------
        # EMA unlabeled stream — exactly 8 images/update.
        # ------------------------------------------------------------------

        if is_ema:

            base.restore_rng_state(
                unlabeled_rng_state
            )

            for _ in range(
                UNLABELED_MICROS
            ):

                ux_np, uv_np = (
                    unlabeled_sampler.sample(
                        MICRO
                    )
                )

                ux = torch.from_numpy(
                    ux_np
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                uv = torch.from_numpy(
                    uv_np
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                geo_x, geo_v = (
                    shared_geometry(
                        ux,
                        uv,
                    )
                )

                weak1 = weak_view(
                    geo_x,
                    geo_v,
                )

                weak2 = weak_view(
                    geo_x,
                    geo_v,
                )

                strong = strong_view(
                    geo_x,
                    geo_v,
                )

                with torch.no_grad():

                    q1 = torch.sigmoid(
                        teacher(
                            base.normalize_batch(
                                weak1,
                                device,
                            )
                        )["logits"]
                    )

                    q2 = torch.sigmoid(
                        teacher(
                            base.normalize_batch(
                                weak2,
                                device,
                            )
                        )["logits"]
                    )

                    qbar = (
                        0.5
                        * (q1 + q2)
                    )

                    disagreement = (
                        torch.abs(
                            q1 - q2
                        )
                    )

                    confidence = (
                        torch.maximum(
                            qbar,
                            1.0 - qbar,
                        )
                    )

                    confidence_accept = (
                        (
                            confidence
                            >= CONFIDENCE_THRESHOLD
                        )
                        & (geo_v > 0.5)
                    )

                    binary_agree = (
                        (
                            q1
                            >= PSEUDO_THRESHOLD
                        )
                        ==
                        (
                            q2
                            >= PSEUDO_THRESHOLD
                        )
                    )

                    agreement_accept = (
                        confidence_accept
                        & (
                            disagreement
                            <= DISAGREEMENT_THRESHOLD
                        )
                        & binary_agree
                    )

                    pseudo = (
                        qbar
                        >= PSEUDO_THRESHOLD
                    ).float()

                assert (
                    int(
                        agreement_accept
                        .sum()
                        .item()
                    )
                    <=
                    int(
                        confidence_accept
                        .sum()
                        .item()
                    )
                )

                if (
                    method
                    == "confidence_only_ema"
                ):

                    accepted = (
                        confidence_accept
                    )

                elif (
                    method
                    == "agreement_filtered_ema"
                ):

                    accepted = (
                        agreement_accept
                    )

                else:

                    raise RuntimeError(
                        method
                    )

                strong_logits = student(
                    base.normalize_batch(
                        strong,
                        device,
                    )
                )["logits"]

                unsupervised = (
                    balanced_pseudo_bce(
                        strong_logits,
                        pseudo,
                        accepted,
                        geo_v,
                    )
                )

                (
                    lambda_u
                    * unsupervised
                    / UNLABELED_MICROS
                ).backward()

                unsupervised_loss_value += (
                    float(
                        unsupervised
                        .detach()
                        .cpu()
                    )
                    / UNLABELED_MICROS
                )

                accepted_fg += int(
                    (
                        accepted
                        & (pseudo > 0.5)
                    )
                    .sum()
                    .item()
                )

                accepted_bg += int(
                    (
                        accepted
                        & (pseudo <= 0.5)
                    )
                    .sum()
                    .item()
                )

                valid_unlabeled_pixels += int(
                    (
                        geo_v > 0.5
                    )
                    .sum()
                    .item()
                )

                confidence_accept_pixels += int(
                    confidence_accept
                    .sum()
                    .item()
                )

                agreement_accept_pixels += int(
                    agreement_accept
                    .sum()
                    .item()
                )

                del (
                    ux,
                    uv,
                    geo_x,
                    geo_v,
                    weak1,
                    weak2,
                    strong,
                    q1,
                    q2,
                    qbar,
                    disagreement,
                    confidence,
                    confidence_accept,
                    binary_agree,
                    agreement_accept,
                    pseudo,
                    accepted,
                    strong_logits,
                    unsupervised,
                )

            # Save independent unlabeled random stream state.
            unlabeled_rng_state = (
                base.capture_rng_state()
            )

            # Restore labeled stream so unlabeled random augmentation
            # cannot alter the next step's labeled trajectory.
            base.restore_rng_state(
                labeled_rng_after
            )

        # ------------------------------------------------------------------
        # Finite gradients
        # ------------------------------------------------------------------

        assert all(
            torch.isfinite(
                parameter.grad
            ).all().item()

            for parameter in student.parameters()

            if parameter.grad is not None
        )

        encoder_lr = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        head_lr = float(
            optimizer.param_groups[1][
                "lr"
            ]
        )

        optimizer.step()
        scheduler.step()

        step = next_step

        labeled_images_seen += (
            LABELED_PER_UPDATE
        )

        if is_ema:

            unlabeled_images_seen += (
                UNLABELED_PER_EMA_UPDATE
            )

            ema_update(
                teacher,
                student,
            )

            assert not any(
                p.grad is not None
                for p in teacher.parameters()
            )

        alpha = float(
            torch.sigmoid(
                student.alpha_logit.detach()
            ).cpu()
        )

        total_accepted = (
            accepted_fg
            + accepted_bg
        )

        acceptance_fraction = (
            total_accepted
            / valid_unlabeled_pixels
            if valid_unlabeled_pixels
            else 0.0
        )

        fg_fraction = (
            accepted_fg
            / valid_unlabeled_pixels
            if valid_unlabeled_pixels
            else 0.0
        )

        bg_fraction = (
            accepted_bg
            / valid_unlabeled_pixels
            if valid_unlabeled_pixels
            else 0.0
        )

        train_rows.append(
            {
                "global_step":
                    step,

                "post_warmup_step":
                    post_warmup_step,

                "method":
                    method,

                "supervised_loss":
                    supervised_loss_value,

                "semantic_auxiliary_loss":
                    text_loss_value,

                "unsupervised_loss":
                    unsupervised_loss_value,

                "lambda_u":
                    lambda_u,

                "accepted_foreground_pixels":
                    accepted_fg,

                "accepted_background_pixels":
                    accepted_bg,

                "accepted_total_pixels":
                    total_accepted,

                "valid_unlabeled_pixels":
                    valid_unlabeled_pixels,

                "accepted_fraction":
                    acceptance_fraction,

                "accepted_foreground_fraction":
                    fg_fraction,

                "accepted_background_fraction":
                    bg_fraction,

                "confidence_accept_pixels":
                    confidence_accept_pixels,

                "agreement_accept_pixels":
                    agreement_accept_pixels,

                "zero_acceptance_event":
                    bool(
                        is_ema
                        and total_accepted == 0
                    ),

                "alpha":
                    alpha,

                "encoder_lr":
                    encoder_lr,

                "head_lr":
                    head_lr,

                "labeled_images_seen":
                    labeled_images_seen,

                "unlabeled_images_seen":
                    unlabeled_images_seen,
            }
        )

        # ------------------------------------------------------------------
        # Local recovery every 50 updates
        # ------------------------------------------------------------------

        if (
            step
            % RECOVERY_EVERY
            == 0
        ):

            save_state(
                p["last"]
            )

            save_state(
                p["recovery"]
            )

            pd.DataFrame(
                train_rows
            ).to_csv(
                p["train"],
                index=False,
            )

        bar.update(1)

        bar.set_postfix(
            step=step,
            best=(
                f"{branch_best_score:.4f}"
            ),
            pat=(
                f"{patience_count}/{PATIENCE}"
            ),
            cov=(
                f"{acceptance_fraction:.3f}"
                if is_ema
                else "-"
            ),
        )

        # ==================================================================================
        # Validation
        # ==================================================================================

        if (
            step
            % VALIDATE_EVERY
            == 0
        ):

            previous_best = (
                branch_best_score
            )

            student_metrics, student_cases, _ = (
                evaluate(
                    student,
                    device,
                    selection_df,
                )
            )

            student_score = float(
                student_metrics[
                    "macro_case_dice"
                ]
            )

            if (
                student_score
                > student_best_score
            ):

                student_best_score = (
                    student_score
                )

                student_best_step = (
                    step
                )

                save_best(
                    p["best_student"],
                    student,
                    method,
                    "student",
                    step,
                    student_score,
                    hashes,
                )

            teacher_metrics = None
            teacher_score = None

            if is_ema:

                teacher.eval()

                teacher_metrics, teacher_cases, _ = (
                    evaluate(
                        teacher,
                        device,
                        selection_df,
                    )
                )

                teacher.eval()

                teacher_score = float(
                    teacher_metrics[
                        "macro_case_dice"
                    ]
                )

                if (
                    teacher_score
                    > teacher_best_score
                ):

                    teacher_best_score = (
                        teacher_score
                    )

                    teacher_best_step = (
                        step
                    )

                    save_best(
                        p["best_teacher"],
                        teacher,
                        method,
                        "teacher",
                        step,
                        teacher_score,
                        hashes,
                    )

            # --------------------------------------------------------------
            # Source-selection choice across student / EMA.
            # Ties prefer student.
            # --------------------------------------------------------------

            branch_best_score = (
                student_best_score
            )

            branch_best_type = (
                "student"
            )

            if (
                is_ema
                and teacher_best_score
                > branch_best_score
            ):

                branch_best_score = (
                    teacher_best_score
                )

                branch_best_type = (
                    "teacher"
                )

            improved = (
                branch_best_score
                > previous_best
                + 1e-12
            )

            if improved:
                patience_count = 0
            else:
                patience_count += 1

            last_validation = (
                step
            )

            # --------------------------------------------------------------
            # Monitor accepted foreground over most recent validation window.
            # Do not silently tolerate all-background collapse.
            # --------------------------------------------------------------

            if is_ema:

                recent = (
                    train_rows[
                        -VALIDATE_EVERY:
                    ]
                )

                window_fg = sum(
                    int(
                        row[
                            "accepted_foreground_pixels"
                        ]
                    )
                    for row in recent
                )

                window_total = sum(
                    int(
                        row[
                            "accepted_total_pixels"
                        ]
                    )
                    for row in recent
                )

                if (
                    window_total > 0
                    and window_fg == 0
                ):

                    raise RuntimeError(
                        "Pseudo-label foreground collapsed "
                        "to zero over an entire validation window."
                    )

            selection_rows.append(
                {
                    "global_step":
                        step,

                    "method":
                        method,

                    "student_macro_case_dice":
                        student_score,

                    "teacher_macro_case_dice":
                        (
                            np.nan
                            if teacher_score is None
                            else teacher_score
                        ),

                    "student_best_score":
                        student_best_score,

                    "student_best_step":
                        student_best_step,

                    "teacher_best_score":
                        (
                            np.nan
                            if teacher_best_score is None
                            else teacher_best_score
                        ),

                    "teacher_best_step":
                        (
                            np.nan
                            if teacher_best_step is None
                            else teacher_best_step
                        ),

                    "branch_best_score":
                        branch_best_score,

                    "branch_best_type":
                        branch_best_type,

                    "patience":
                        patience_count,

                    "student_macro_case_iou":
                        student_metrics[
                            "macro_case_iou"
                        ],

                    "teacher_macro_case_iou":
                        (
                            np.nan
                            if teacher_metrics is None
                            else teacher_metrics[
                                "macro_case_iou"
                            ]
                        ),

                    "student_macro_case_sensitivity":
                        student_metrics[
                            "macro_case_sensitivity"
                        ],

                    "teacher_macro_case_sensitivity":
                        (
                            np.nan
                            if teacher_metrics is None
                            else teacher_metrics[
                                "macro_case_sensitivity"
                            ]
                        ),

                    "student_macro_case_specificity":
                        student_metrics[
                            "macro_case_specificity"
                        ],

                    "teacher_macro_case_specificity":
                        (
                            np.nan
                            if teacher_metrics is None
                            else teacher_metrics[
                                "macro_case_specificity"
                            ]
                        ),
                }
            )

            for row in student_cases:

                case_rows.append(
                    {
                        "global_step":
                            step,

                        "candidate":
                            "student",

                        **row,
                    }
                )

            if is_ema:

                for row in teacher_cases:

                    case_rows.append(
                        {
                            "global_step":
                                step,

                            "candidate":
                                "teacher",

                            **row,
                        }
                    )

            pd.DataFrame(
                train_rows
            ).to_csv(
                p["train"],
                index=False,
            )

            pd.DataFrame(
                selection_rows
            ).to_csv(
                p["selection"],
                index=False,
            )

            pd.DataFrame(
                case_rows
            ).to_csv(
                p["cases"],
                index=False,
            )

            save_state(
                p["last"]
            )

            save_state(
                p["recovery"]
            )

            write_state(
                method,
                step,
                branch_best_score,
                branch_best_type,
                patience_count,
            )

            print()
            print(
                f"[{method}] VAL {step}: "
                f"student={student_score:.8f}"
                + (
                    ""
                    if teacher_score is None
                    else (
                        f", teacher="
                        f"{teacher_score:.8f}"
                    )
                )
                + (
                    f", best="
                    f"{branch_best_score:.8f}"
                    f" ({branch_best_type}), "
                    f"patience="
                    f"{patience_count}/{PATIENCE}"
                )
            )

            stopping = (
                patience_count
                >= PATIENCE
            )

            # --------------------------------------------------------------
            # Rolling durable backup.
            # Avoid duplicate rolling+final upload at stopping validation.
            # --------------------------------------------------------------

            if (
                step in ROLLING_STEPS
                and not stopping
                and step < MAX_STEP
            ):

                release, digest = (
                    publish_release(
                        token,
                        method,
                        p,
                        step,
                        False,
                        branch_best_score,
                    )
                )

                rolling = {
                    "timestamp_utc":
                        now(),

                    "stage":
                        "NOTEBOOK_07_BRANCH",

                    "status":
                        "DURABLE_ROLLING",

                    "method":
                        method,

                    "budget_id":
                        BUDGET,

                    "seed":
                        SEED,

                    "global_step":
                        step,

                    "branch_best_score":
                        branch_best_score,

                    "branch_best_type":
                        branch_best_type,

                    "release_url":
                        release["html_url"],

                    "archive_sha256":
                        digest,

                    "calibration_accessed":
                        False,

                    "target_accessed":
                        False,
                }

                atomic_json(
                    p["rolling_audit"],
                    rolling,
                )

                sync_small(
                    [
                        p["rolling_audit"],
                        p["train"],
                        p["selection"],
                        p["cases"],
                        STATE,
                    ],
                    (
                        f"Record Notebook 07 "
                        f"b050 seed42 {method} "
                        f"rolling step {step}"
                    ),
                    token,
                )

            if stopping:

                stop_reason = (
                    f"EARLY_STOPPING_PATIENCE_{PATIENCE}"
                )

                break

    bar.close()

    # ======================================================================================
    # Finalize branch
    # ======================================================================================

    save_state(
        p["last"]
    )

    save_state(
        p["recovery"]
    )

    # Select source-best candidate.
    if branch_best_type == "teacher":

        chosen_path = (
            p["best_teacher"]
        )

    else:

        chosen_path = (
            p["best_student"]
        )

    chosen = torch.load(
        chosen_path,
        map_location="cpu",
        weights_only=False,
    )

    base.atomic_torch_save(
        chosen,
        p["selected"],
    )

    final_model = nb06.build(
        snapshot,
        proto,
        "real_text",
        device,
    )

    final_model.load_state_dict(
        chosen["model_state_dict"]
    )

    final_metrics, final_cases, _ = (
        evaluate(
            final_model,
            device,
            selection_df,
        )
    )

    assert abs(
        float(
            final_metrics[
                "macro_case_dice"
            ]
        )
        - float(
            branch_best_score
        )
    ) < 1e-6

    train_df_log = pd.DataFrame(
        train_rows
    )

    if is_ema and len(train_df_log):

        mean_acceptance = float(
            train_df_log[
                "accepted_fraction"
            ].mean()
        )

        mean_fg_acceptance = float(
            train_df_log[
                "accepted_foreground_fraction"
            ].mean()
        )

        mean_bg_acceptance = float(
            train_df_log[
                "accepted_background_fraction"
            ].mean()
        )

        zero_events = int(
            train_df_log[
                "zero_acceptance_event"
            ].sum()
        )

    else:

        mean_acceptance = None
        mean_fg_acceptance = None
        mean_bg_acceptance = None
        zero_events = 0

    summary = {
        "timestamp_utc":
            now(),

        "stage":
            "NOTEBOOK_07_BRANCH",

        "status":
            "COMPUTE_COMPLETE",

        "method":
            method,

        "budget_id":
            BUDGET,

        "training_mask_fraction":
            0.50,

        "seed":
            SEED,

        "warmup_terminal_step":
            1000,

        "final_global_step":
            int(step),

        "post_warmup_updates":
            int(step - 1000),

        "stop_reason":
            stop_reason,

        "selected_model_type":
            branch_best_type,

        "selected_best_step":
            (
                teacher_best_step
                if branch_best_type
                == "teacher"
                else student_best_step
            ),

        "selected_macro_case_dice":
            float(
                final_metrics[
                    "macro_case_dice"
                ]
            ),

        "selected_macro_case_iou":
            float(
                final_metrics[
                    "macro_case_iou"
                ]
            ),

        "selected_macro_case_sensitivity":
            float(
                final_metrics[
                    "macro_case_sensitivity"
                ]
            ),

        "selected_macro_case_specificity":
            float(
                final_metrics[
                    "macro_case_specificity"
                ]
            ),

        "selected_macro_slice_dice":
            float(
                final_metrics[
                    "macro_slice_dice"
                ]
            ),

        "selected_pooled_dice":
            float(
                final_metrics[
                    "pooled_dice"
                ]
            ),

        "selected_pooled_iou":
            float(
                final_metrics[
                    "pooled_iou"
                ]
            ),

        "student_best_score":
            float(
                student_best_score
            ),

        "student_best_step":
            int(
                student_best_step
            ),

        "teacher_best_score":
            (
                None
                if teacher_best_score is None
                else float(
                    teacher_best_score
                )
            ),

        "teacher_best_step":
            (
                None
                if teacher_best_step is None
                else int(
                    teacher_best_step
                )
            ),

        "labeled_images_seen":
            int(
                labeled_images_seen
            ),

        "unlabeled_images_seen":
            int(
                unlabeled_images_seen
            ),

        "mean_pseudo_label_acceptance":
            mean_acceptance,

        "mean_accepted_foreground_fraction":
            mean_fg_acceptance,

        "mean_accepted_background_fraction":
            mean_bg_acceptance,

        "zero_acceptance_events":
            zero_events,

        "selected_checkpoint_sha256":
            sha(
                p["selected"]
            ),

        **hashes,

        "hidden_fitting_masks_accessed":
            False,

        "calibration_accessed":
            False,

        "target_accessed":
            False,
    }

    atomic_json(
        p["summary"],
        summary,
    )

    # ----------------------------------------------------------------------
    # Final durable release
    # ----------------------------------------------------------------------

    release, digest = (
        publish_release(
            token,
            method,
            p,
            step,
            True,
            branch_best_score,
        )
    )

    final = {
        **summary,

        "status":
            "DURABLE_COMPLETE",

        "release_url":
            release[
                "html_url"
            ],

        "archive_sha256":
            digest,
    }

    atomic_json(
        p["final_audit"],
        final,
    )

    pd.DataFrame(
        train_rows
    ).to_csv(
        p["train"],
        index=False,
    )

    pd.DataFrame(
        selection_rows
    ).to_csv(
        p["selection"],
        index=False,
    )

    pd.DataFrame(
        case_rows
    ).to_csv(
        p["cases"],
        index=False,
    )

    sync_small(
        [
            p["final_audit"],
            p["rolling_audit"],
            p["train"],
            p["selection"],
            p["cases"],
            STATE,
        ],
        (
            f"Complete Notebook 07 "
            f"b050 seed42 {method}"
        ),
        token,
    )

    print()
    print(
        f"✅ {method} COMPLETE"
    )

    print(
        f"   selected : "
        f"{branch_best_type}"
    )

    print(
        f"   Dice     : "
        f"{final['selected_macro_case_dice']:.8f}"
    )

    print(
        f"   best step: "
        f"{final['selected_best_step']}"
    )

    print(
        f"   final    : "
        f"{step}"
    )

    print(
        f"   release  : "
        f"{release['html_url']}"
    )

    del (
        final_model,
        chosen,
        student,
        optimizer,
        scheduler,
        labeled_sampler,
        labeled_store,
    )

    if teacher is not None:
        del teacher

    if unlabeled_sampler is not None:
        del unlabeled_sampler

    if unlabeled_store is not None:
        del unlabeled_store

    gc.collect()
    torch.cuda.empty_cache()

    return final


# ==========================================================================================
# Aggregate three-way pilot
# ==========================================================================================

def aggregate(results, token):

    rows = []

    for result in results:

        rows.append(
            {
                "method":
                    result["method"],

                "seed":
                    result["seed"],

                "training_mask_fraction":
                    result[
                        "training_mask_fraction"
                    ],

                "selected_model_type":
                    result[
                        "selected_model_type"
                    ],

                "best_step":
                    result[
                        "selected_best_step"
                    ],

                "final_step":
                    result[
                        "final_global_step"
                    ],

                "macro_case_dice":
                    result[
                        "selected_macro_case_dice"
                    ],

                "macro_case_iou":
                    result[
                        "selected_macro_case_iou"
                    ],

                "macro_case_sensitivity":
                    result[
                        "selected_macro_case_sensitivity"
                    ],

                "macro_case_specificity":
                    result[
                        "selected_macro_case_specificity"
                    ],

                "mean_pseudo_label_acceptance":
                    result[
                        "mean_pseudo_label_acceptance"
                    ],

                "mean_accepted_foreground_fraction":
                    result[
                        "mean_accepted_foreground_fraction"
                    ],

                "mean_accepted_background_fraction":
                    result[
                        "mean_accepted_background_fraction"
                    ],

                "zero_acceptance_events":
                    result[
                        "zero_acceptance_events"
                    ],

                "labeled_images_seen":
                    result[
                        "labeled_images_seen"
                    ],

                "unlabeled_images_seen":
                    result[
                        "unlabeled_images_seen"
                    ],
            }
        )

    table_path = (
        TABLES
        / "notebook07_b050_seed42_three_way.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        table_path,
        index=False,
    )

    aggregate_json = {
        "timestamp_utc":
            now(),

        "stage":
            "NOTEBOOK_07_B050_SEED42",

        "status":
            "THREE_WAY_PILOT_COMPLETE",

        "warmup_branch_sha256":
            EXPECTED_BRANCH_SHA,

        "results":
            rows,

        "calibration_accessed":
            False,

        "target_accessed":
            False,
    }

    aggregate_path = (
        AUD
        / "notebook07_b050_seed42_three_way.json"
    )

    atomic_json(
        aggregate_path,
        aggregate_json,
    )

    report_path = (
        REPORTS
        / "notebook07_b050_seed42_three_way.md"
    )

    lines = [
        "# Notebook 07 — b050 seed42 three-way pilot",
        "",
        "| Method | Selected | Dice | IoU | Pseudo-label coverage |",
        "|---|---|---:|---:|---:|",
    ]

    for row in rows:

        coverage = (
            "N/A"
            if row[
                "mean_pseudo_label_acceptance"
            ] is None
            else (
                f"{100 * row['mean_pseudo_label_acceptance']:.2f}%"
            )
        )

        lines.append(
            "| {} | {} | {:.6f} | {:.6f} | {} |".format(
                row["method"],
                row["selected_model_type"],
                row["macro_case_dice"],
                row["macro_case_iou"],
                coverage,
            )
        )

    lines += [
        "",
        "Hidden fitting masks accessed: NO.",
        "Calibration accessed: NO.",
        "Target / MedSeg accessed: NO.",
        "",
        "This is the 50% / seed42 source-only pilot. "
        "No multi-seed conclusion is made from this table.",
    ]

    report_path.write_text(
        "\n".join(lines) + "\n"
    )

    state_text = f"""# EviCT Execution State

## Current stage

NOTEBOOK_07_B050_SEED42_THREE_WAY_COMPLETE

## Notebook 07

Budget:

50 percent visible fitting masks

Seed:

17

Three-way comparison:

COMPLETE

Methods:

1. supervised
2. confidence-only EMA
3. agreement-filtered EMA

Common warm-up branch SHA-256:

{EXPECTED_BRANCH_SHA}

## Isolation

Hidden fitting masks used:

NO

Calibration accessed:

NO

Target / MedSeg accessed:

NO

Target lock:

ACTIVE

## Next

Audit pseudo-label quality on visible source labels and inspect the b050 seed42
pilot before expanding to seeds 42 / 2026 and the 25 percent budget.
"""

    base.atomic_text(
        STATE,
        state_text,
    )

    sync_small(
        [
            table_path,
            aggregate_path,
            report_path,
            STATE,
        ],
        (
            "Complete Notebook 07 "
            "b050 seed42 three-way pilot"
        ),
        token,
    )

    return (
        pd.DataFrame(rows),
        aggregate_path,
    )


# ==========================================================================================
# Locked confirmatory metadata sync override
# ==========================================================================================

sync_small = lambda *args, **kwargs: True


# ==========================================================================================
# Main
# ==========================================================================================

def main():

    print("=" * 118)
    print(
        "EViCT NOTEBOOK 07D — FULL b050 / SEED42 THREE-WAY"
    )
    print("=" * 118)

    assert torch.cuda.is_available()

    device = torch.device(
        "cuda:0"
    )

    print(
        "✓ GPU                       :",
        torch.cuda.get_device_name(0),
    )

    token = UserSecretsClient().get_secret(
        "pushEviCT"
    )

    assert token, (
        "Kaggle secret pushEviCT unavailable."
    )

    (
        cfg,
        impl,
        warmup_audit,
        labeled_df,
        unlabeled_df,
    ) = verify_protocol()

    nb06.ensure_cache()

    selection_df = pd.read_csv(
        SELECTION
    )

    assert (
        selection_df[
            "case_id"
        ]
        .astype(str)
        .nunique()
        == 4
    )

    snapshot = nb06.mit_snapshot()
    proto = nb06.text_prototypes()

    hashes = hash_bundle(
        cfg,
        impl,
    )

    results = []

    for method in METHODS:

        result = run_method(
            method=method,
            cfg=cfg,
            impl=impl,
            warmup_audit=warmup_audit,
            labeled_df=labeled_df,
            unlabeled_df=unlabeled_df,
            selection_df=selection_df,
            snapshot=snapshot,
            proto=proto,
            device=device,
            token=token,
            hashes=hashes,
        )

        results.append(
            result
        )

    table, aggregate_path = (
        aggregate(
            results,
            token,
        )
    )

    print()
    print("=" * 118)
    print("✅ NOTEBOOK 07 b050 / SEED42 THREE-WAY PILOT COMPLETE")
    print("=" * 118)

    print(
        table[
            [
                "method",
                "selected_model_type",
                "best_step",
                "final_step",
                "macro_case_dice",
                "macro_case_iou",
                "mean_pseudo_label_acceptance",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "✓ Aggregate audit           :",
        aggregate_path.relative_to(ROOT),
    )

    print("✓ Hidden fitting masks      : NOT ACCESSED")
    print("✓ Calibration               : NOT ACCESSED")
    print("✓ Target / MedSeg           : NOT ACCESSED")

    print()
    print("NEXT:")
    print(
        "  audit pseudo-label precision on visible source labels;"
    )
    print(
        "  inspect pilot before expanding to seeds 42 / 2026."
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
