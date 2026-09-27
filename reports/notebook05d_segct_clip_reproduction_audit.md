# Notebook 05D — SegCT-CLIP Reproduction Audit

Timestamp: 2026-09-27T21:18:39.799099+00:00

Paper: CT-Insight VLM: Multimodel Vision-Language Framework for Accurate, Explainable, and Label-Efficient Lung CT Analysis

DOI: 10.1109/TRPMS.2026.3734275

Status: **EXACT_REPRODUCTION_NOT_CURRENTLY_SPECIFIED**

## Components explicitly disclosed by the paper

- **task**: Lesion segmentation plus caption alignment/retrieval for lung CT slices.
- **backbone**: CLIP ViT-L/14-336 is reported as the strongest SegCT-CLIP backbone.
- **visual_branch**: Dual-level low/high transformer visual features are aggregated and passed to a lightweight visual decoder for lesion-mask prediction.
- **text_branch**: A predefined clinically curated caption repository is encoded by the CLIP text encoder; captions are retrieved by cosine similarity to high-level image embeddings.
- **training_triplet**: Input slice, retrieved/relevant caption, ground-truth lesion mask.
- **training_loss**: Weighted contrastive plus Dice objective with lambda_contrastive=0.3 and lambda_dice=0.7.
- **evaluation_protocol**: Bidirectional SegDB-1/SegDB-2 cross-dataset evaluation without target fine-tuning is described.
- **published_backbone_result_note**: Published table values are literature values only and are not treated as rerun results.

## Unresolved choices blocking an exact reproduction claim

- Exact public source-code repository URL and released revision/commit.
- Exact released checkpoint corresponding to the reported SegCT-CLIP results.
- Exact 367-slice SegDB-2 selection used by the paper.
- Exact SegDB-1 patient/group mapping and source partition.
- Caption-bank contents, caption-to-slice mapping, and caption preprocessing.
- Exact low-level and high-level CLIP transformer layer indices.
- Exact dual-level aggregation function Fagg.
- Exact lightweight decoder topology and normalization/activation choices.
- Whether and how much of the CLIP image/text encoders were fine-tuned.
- Optimizer, learning rate, weight decay, batch size, number of epochs/updates.
- Augmentation and image normalization/resolution preprocessing details.
- Early-stopping/model-selection rule and random seeds.
- Exact metric aggregation convention used for the paper tables.

## Author repository discovery

- No exact SegCT-CLIP author repository was verified automatically.

## Scientific handling

- Published SegCT-CLIP table values remain literature values; none are entered as rerun results.
- Calibration and MedSeg/target evaluation remain locked.
- Missing architecture/training parameters are not silently invented.

## Next

Do not invent missing SegCT-CLIP parameters. Proceed only with an explicitly labeled controlled reimplementation/adaptation, or obtain the missing author repository/checkpoint/caption bank and exact protocol.
