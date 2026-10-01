"""
EViCT Notebook-09 corrected qualitative figure regeneration.

Purpose
-------
Regenerate publication qualitative figures from already-saved corrected
MedSeg predictions. This script performs NO model inference and NO training.

Expected repository artifacts
-----------------------------
artifacts/large/notebook09_corrected/
tables/notebook09_corrected_target_case_metrics.csv
tables/notebook09_corrected_prediction_manifest.csv
config/notebook09_qualitative_figure_plan.json

Target Kaggle data
------------------
/kaggle/input/competitions/covid-segmentation/
    images_medseg.npy
    masks_medseg.npy

Scientific status
-----------------
The corrected target preprocessing used the source-consistent CT window:
    clip [-1250,250]
    scale to [0,1]

Changing titles, fonts, layouts, DPI, legends, or panel arrangement here
is a presentation-only change.

Changing case-selection rules must be described as post-hoc visualization.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------

ROOT = None

for p in [
    Path("/kaggle/working/EviCT"),
    Path("/kaggle/working/EViCT"),
    Path.cwd(),
]:

    if (
        p.exists()
        and (
            p
            / "tables/notebook09_corrected_target_case_metrics.csv"
        ).exists()
    ):

        ROOT = p
        break


if ROOT is None:
    raise RuntimeError(
        "Could not locate EViCT repository."
    )


TARGET_ROOT = Path(
    "/kaggle/input/competitions/covid-segmentation"
)

IMAGES_PATH = (
    TARGET_ROOT
    / "images_medseg.npy"
)

MASKS_PATH = (
    TARGET_ROOT
    / "masks_medseg.npy"
)

METRICS_PATH = (
    ROOT
    / "tables/notebook09_corrected_target_case_metrics.csv"
)

MANIFEST_PATH = (
    ROOT
    / "tables/notebook09_corrected_prediction_manifest.csv"
)

QUAL_PLAN_PATH = (
    ROOT
    / "config/notebook09_qualitative_figure_plan.json"
)

PRED_ROOT = (
    ROOT
    / "artifacts/large/notebook09_corrected"
)

FIG_DIR = (
    ROOT
    / "figures/notebook09_corrected"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Scientific constants
# ---------------------------------------------------------------------

DISPLAY_SEED = 17

ANCHOR_METHOD = (
    "confidence_only_ema"
)

VIS_METHOD = (
    "agreement_filtered_ema"
)

WINDOW_MIN = -1250.0
WINDOW_MAX = 250.0
WINDOW_WIDTH = WINDOW_MAX - WINDOW_MIN


# ---------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------

images = np.load(
    IMAGES_PATH,
    mmap_mode="r",
)

masks = np.load(
    MASKS_PATH,
    mmap_mode="r",
)

metrics = pd.read_csv(
    METRICS_PATH
)

manifest = pd.read_csv(
    MANIFEST_PATH
)


gt_all = np.logical_or(
    masks[..., 0] > 0.5,
    masks[..., 1] > 0.5,
).astype(
    np.uint8
)


if QUAL_PLAN_PATH.exists():

    plan = json.loads(
        QUAL_PLAN_PATH.read_text()
    )

    assert (
        plan["status"]
        == "FROZEN_BEFORE_TARGET_ACCESS"
    )


# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------

plt.rcParams.update({

    "font.size":
        11,

    "axes.titlesize":
        11,

    "axes.labelsize":
        11,

    "xtick.labelsize":
        10,

    "ytick.labelsize":
        10,

    "legend.fontsize":
        10,

    "figure.titlesize":
        16,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,
})


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def case_index(case_id):

    return int(
        str(
            case_id
        ).split(
            "_"
        )[-1]
    )


def ct01(case_id):

    idx = case_index(
        case_id
    )

    x = np.asarray(
        images[
            idx,
            :,
            :,
            0,
        ],
        dtype=np.float32,
    )

    x = np.clip(
        x,
        WINDOW_MIN,
        WINDOW_MAX,
    )

    x = (
        x
        - WINDOW_MIN
    ) / WINDOW_WIDTH

    return np.clip(
        x,
        0,
        1,
    )


def gt(case_id):

    return gt_all[
        case_index(
            case_id
        )
    ]


def prediction_path(
    seed,
    method,
    case_id,
):

    return (
        PRED_ROOT
        / f"seed_{seed}"
        / method
        / f"{case_id}.npz"
    )


def load_prediction(
    seed,
    method,
    case_id,
):

    p = prediction_path(
        seed,
        method,
        case_id,
    )

    if not p.exists():

        raise FileNotFoundError(
            p
        )

    return np.load(
        p
    )


def error_overlay(
    reference,
    prediction,
):

    reference = reference.astype(
        bool
    )

    prediction = prediction.astype(
        bool
    )

    rgb = np.zeros(
        (
            reference.shape[0],
            reference.shape[1],
            3,
        ),
        dtype=np.float32,
    )

    tp = reference & prediction
    fp = (~reference) & prediction
    fn = reference & (~prediction)

    rgb[
        tp,
        1
    ] = 1.0

    rgb[
        fp,
        0
    ] = 1.0

    rgb[
        fn,
        2
    ] = 1.0

    return rgb


def seed_metric(
    case_id,
    method,
    metric="dice",
    seed=DISPLAY_SEED,
    view="single",
):

    d = metrics[
        (
            metrics.case_id
            == case_id
        )
        &
        (
            metrics.method
            == method
        )
        &
        (
            metrics.seed
            == seed
        )
        &
        (
            metrics.view
            == view
        )
    ]

    if len(d) != 1:

        raise RuntimeError(
            (
                f"Unexpected metric rows: "
                f"{case_id}, {method}, {seed}, {view}"
            )
        )

    return float(
        d.iloc[
            0
        ][
            metric
        ]
    )


def mean_method_case_dice(
    case_id,
    method,
):

    d = metrics[
        (
            metrics.case_id
            == case_id
        )
        &
        (
            metrics.method
            == method
        )
        &
        (
            metrics.view
            == "single"
        )
    ]

    return float(
        d.dice.mean()
    )


def mean_uncertainty(
    case_id,
    method,
):

    d = manifest[
        (
            manifest.case_id
            == case_id
        )
        &
        (
            manifest.method
            == method
        )
    ]

    return float(
        d.case_uncertainty_score.mean()
    )


def legend(fig):

    handles = [

        Patch(
            facecolor=(0,1,0),
            label="True Positive (TP)",
        ),

        Patch(
            facecolor=(1,0,0),
            label="False Positive (FP)",
        ),

        Patch(
            facecolor=(0,0,1),
            label="False Negative (FN)",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
    )


def save(
    fig,
    stem,
):

    fig.tight_layout(
        rect=[
            0,
            0.03,
            1,
            0.96,
        ]
    )

    fig.savefig(
        FIG_DIR
        / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        / f"{stem}.svg",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_DIR
        / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        stem
    )


# ---------------------------------------------------------------------
# Case ranking tables
# ---------------------------------------------------------------------

anchor = metrics[
    (
        metrics.method
        == ANCHOR_METHOD
    )
    &
    (
        metrics.view
        == "single"
    )
]


anchor_rank = (
    anchor
    .groupby(
        "case_id",
        as_index=False,
    )
    .agg(
        mean_dice=(
            "dice",
            "mean",
        ),
        mean_fp=(
            "fp",
            "mean",
        ),
        mean_fn=(
            "fn",
            "mean",
        ),
        lesion_pixels=(
            "lesion_pixels",
            "first",
        ),
    )
)


unc = (
    manifest[
        manifest.method
        == ANCHOR_METHOD
    ]
    .groupby(
        "case_id",
        as_index=False,
    )
    .agg(
        mean_uncertainty=(
            "case_uncertainty_score",
            "mean",
        )
    )
)


anchor_rank = anchor_rank.merge(
    unc,
    on="case_id",
    how="left",
)


performance = (
    anchor_rank
    .sort_values(
        [
            "mean_dice",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)


lesion = (
    anchor_rank[
        anchor_rank.lesion_pixels
        > 0
    ]
    .sort_values(
        [
            "lesion_pixels",
            "case_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ---------------------------------------------------------------------
# Figure: cross-method atlas
# ---------------------------------------------------------------------

best_case = str(
    performance.iloc[
        0
    ].case_id
)

median_case = str(
    performance.iloc[
        len(performance)//2
    ].case_id
)

worst_case = str(
    performance.iloc[
        -1
    ].case_id
)


small_case = None

for cid in lesion.case_id.astype(
    str
):

    if cid not in {
        best_case,
        median_case,
        worst_case,
    }:

        small_case = cid
        break


cross_cases = [

    (
        "Best external-target case",
        best_case,
    ),

    (
        "Median-rank external-target case",
        median_case,
    ),

    (
        "Worst external-target case",
        worst_case,
    ),

    (
        "Small nonempty-lesion case",
        small_case,
    ),
]


fig, axes = plt.subplots(
    len(
        cross_cases
    ),
    7,
    figsize=(
        20,
        4
        * len(
            cross_cases
        ),
    ),
)


for r, (
    label,
    cid,
) in enumerate(
    cross_cases
):

    x = ct01(
        cid
    )

    g = gt(
        cid
    )

    sup = load_prediction(
        DISPLAY_SEED,
        "supervised",
        cid,
    )

    conf = load_prediction(
        DISPLAY_SEED,
        "confidence_only_ema",
        cid,
    )

    agr = load_prediction(
        DISPLAY_SEED,
        "agreement_filtered_ema",
        cid,
    )

    ds = seed_metric(
        cid,
        "supervised",
    )

    dc = seed_metric(
        cid,
        "confidence_only_ema",
    )

    da = seed_metric(
        cid,
        "agreement_filtered_ema",
    )


    panels = [

        (
            x,
            "CT",
            "gray",
        ),

        (
            g,
            "Reference",
            "gray",
        ),

        (
            sup["single_mask"],
            f"Supervised\nDice={ds:.3f}",
            "gray",
        ),

        (
            conf["single_mask"],
            f"Confidence EMA\nDice={dc:.3f}",
            "gray",
        ),

        (
            agr["single_mask"],
            f"Agreement EMA\nDice={da:.3f}",
            "gray",
        ),

        (
            error_overlay(
                g,
                conf[
                    "single_mask"
                ],
            ),
            "Confidence\nTP / FP / FN",
            None,
        ),

        (
            error_overlay(
                g,
                agr[
                    "single_mask"
                ],
            ),
            "Agreement\nTP / FP / FN",
            None,
        ),
    ]


    for c, (
        arr,
        title,
        cmap,
    ) in enumerate(
        panels
    ):

        axes[
            r,
            c
        ].imshow(
            arr,
            cmap=cmap,
        )

        axes[
            r,
            c
        ].set_title(
            title,
            fontweight="bold",
        )

        axes[
            r,
            c
        ].axis(
            "off"
        )


    axes[
        r,
        0
    ].set_ylabel(
        (
            f"{label}\n"
            f"{cid}\n"
            f"Lesion pixels={int(g.sum()):,}"
        ),
        fontweight="bold",
    )


fig.suptitle(
    (
        "External-Target Segmentation Comparison Across Training "
        "Strategies on Predeclared MedSeg Cases"
    ),
    fontweight="bold",
)

legend(
    fig
)

save(
    fig,
    "main02_cross_method_atlas",
)


# ---------------------------------------------------------------------
# Figure: failure taxonomy
# ---------------------------------------------------------------------

top = performance.iloc[
    :25
]

bottom = performance.iloc[
    -25:
]


failure_cases = [

    (
        "High false-negative burden",
        str(
            anchor_rank
            .sort_values(
                [
                    "mean_fn",
                    "case_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[
                0
            ]
            .case_id
        ),
    ),

    (
        "High false-positive burden",
        str(
            anchor_rank
            .sort_values(
                [
                    "mean_fp",
                    "case_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[
                0
            ]
            .case_id
        ),
    ),

    (
        "Lowest Dice among nonempty lesions",
        str(
            anchor_rank[
                anchor_rank.lesion_pixels
                > 0
            ]
            .sort_values(
                [
                    "mean_dice",
                    "case_id",
                ]
            )
            .iloc[
                0
            ]
            .case_id
        ),
    ),

    (
        "High uncertainty despite strong Dice",
        str(
            top
            .sort_values(
                [
                    "mean_uncertainty",
                    "case_id",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .iloc[
                0
            ]
            .case_id
        ),
    ),

    (
        "Low uncertainty despite poor Dice / empty-reference case",
        str(
            bottom
            .sort_values(
                [
                    "mean_uncertainty",
                    "case_id",
                ]
            )
            .iloc[
                0
            ]
            .case_id
        ),
    ),
]


fig, axes = plt.subplots(
    len(
        failure_cases
    ),
    7,
    figsize=(
        20,
        4
        * len(
            failure_cases
        ),
    ),
)


for r, (
    label,
    cid,
) in enumerate(
    failure_cases
):

    x = ct01(
        cid
    )

    g = gt(
        cid
    )

    p = load_prediction(
        DISPLAY_SEED,
        ANCHOR_METHOD,
        cid,
    )

    d = seed_metric(
        cid,
        ANCHOR_METHOD,
    )

    u = mean_uncertainty(
        cid,
        ANCHOR_METHOD,
    )


    panels = [

        (
            x,
            "CT",
            "gray",
        ),

        (
            g,
            "Reference",
            "gray",
        ),

        (
            p[
                "single_probability"
            ],
            "Probability",
            "viridis",
        ),

        (
            p[
                "single_mask"
            ],
            "Prediction",
            "gray",
        ),

        (
            error_overlay(
                g,
                p[
                    "single_mask"
                ],
            ),
            "TP / FP / FN",
            None,
        ),

        (
            p[
                "uncertainty"
            ],
            "Uncertainty",
            "magma",
        ),

        (
            p[
                "variance"
            ],
            "View variance",
            "inferno",
        ),
    ]


    for c, (
        arr,
        title,
        cmap,
    ) in enumerate(
        panels
    ):

        axes[
            r,
            c
        ].imshow(
            arr,
            cmap=cmap,
        )

        axes[
            r,
            c
        ].set_title(
            title,
            fontweight="bold",
        )

        axes[
            r,
            c
        ].axis(
            "off"
        )


    axes[
        r,
        0
    ].set_ylabel(
        (
            f"{label}\n"
            f"{cid}\n"
            f"Seed-17 Dice={d:.3f} | "
            f"Mean U={u:.3f}\n"
            f"Lesion pixels={int(g.sum()):,}"
        ),
        fontweight="bold",
    )


fig.suptitle(
    (
        "Predeclared External-Target Failure Taxonomy "
        "Under the Frozen Evaluation Protocol"
    ),
    fontweight="bold",
)

legend(
    fig
)

save(
    fig,
    "main05_failure_taxonomy",
)


# ---------------------------------------------------------------------
# Generic high-Dice atlas generator
# ---------------------------------------------------------------------

def high_dice_atlas(
    cases,
    title,
    stem,
):

    fig, axes = plt.subplots(
        len(
            cases
        ),
        7,
        figsize=(
            20,
            3.8
            * len(
                cases
            ),
        ),
    )


    for r, cid in enumerate(
        cases
    ):

        x = ct01(
            cid
        )

        g = gt(
            cid
        )

        p = load_prediction(
            DISPLAY_SEED,
            VIS_METHOD,
            cid,
        )

        mean_d = mean_method_case_dice(
            cid,
            VIS_METHOD,
        )

        seed_d = seed_metric(
            cid,
            VIS_METHOD,
        )


        panels = [

            (
                x,
                "CT",
                "gray",
            ),

            (
                g,
                "Reference",
                "gray",
            ),

            (
                p[
                    "single_probability"
                ],
                "Probability",
                "viridis",
            ),

            (
                p[
                    "single_mask"
                ],
                "Prediction",
                "gray",
            ),

            (
                error_overlay(
                    g,
                    p[
                        "single_mask"
                    ],
                ),
                "TP / FP / FN",
                None,
            ),

            (
                p[
                    "uncertainty"
                ],
                "Uncertainty",
                "magma",
            ),

            (
                p[
                    "variance"
                ],
                "View variance",
                "inferno",
            ),
        ]


        for c, (
            arr,
            subtitle,
            cmap,
        ) in enumerate(
            panels
        ):

            axes[
                r,
                c
            ].imshow(
                arr,
                cmap=cmap,
            )

            axes[
                r,
                c
            ].set_title(
                subtitle,
                fontweight="bold",
            )

            axes[
                r,
                c
            ].axis(
                "off"
            )


        axes[
            r,
            0
        ].set_ylabel(
            (
                f"Rank {r+1}\n"
                f"{cid}\n"
                f"Mean Dice={mean_d:.3f}\n"
                f"Seed-17 Dice={seed_d:.3f}"
            ),
            fontweight="bold",
        )


    fig.suptitle(
        title,
        fontweight="bold",
    )

    legend(
        fig
    )

    save(
        fig,
        stem,
    )


# ---------------------------------------------------------------------
# High-Dice method-specific atlas
# ---------------------------------------------------------------------

agree_rank = (
    metrics[
        (
            metrics.method
            == VIS_METHOD
        )
        &
        (
            metrics.view
            == "single"
        )
    ]
    .groupby(
        "case_id",
        as_index=False,
    )
    .agg(
        mean_dice=(
            "dice",
            "mean",
        )
    )
    .sort_values(
        [
            "mean_dice",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


top_agreement = (
    agree_rank.case_id.astype(
        str
    )
    .tolist()[
        :8
    ]
)


high_dice_atlas(

    top_agreement,

    (
        "Qualitative Analysis of High-Performing External-Target Cases "
        "Under Agreement-Filtered Semi-Supervised Segmentation"
    ),

    "main07_high_dice_agreement_filtered_atlas",
)


# ---------------------------------------------------------------------
# Consensus high-Dice atlas
# ---------------------------------------------------------------------

consensus_rank = (
    metrics[
        metrics.view
        == "single"
    ]
    .groupby(
        "case_id",
        as_index=False,
    )
    .agg(
        mean_dice=(
            "dice",
            "mean",
        )
    )
    .sort_values(
        [
            "mean_dice",
            "case_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
)


top_consensus = (
    consensus_rank.case_id.astype(
        str
    )
    .tolist()[
        :8
    ]
)


high_dice_atlas(

    top_consensus,

    (
        "Qualitative Analysis of Consensus High-Performing "
        "External-Target Cases Across Training Strategies and Random Seeds"
    ),

    "supp07_consensus_high_dice_atlas",
)


print(
    "\n✅ Corrected qualitative figures regenerated."
)
