
#The 2nd place solution for AI of GOD - 5 🎉 2k26
### Team Members - Mayank Mishra , Satyroop Chatterjee , Sourish Kumar Dutta

## 👥 Team Members

| Name | GitHub Profile |
|------|----------------|
| Mayank Mishra | [@Mayank2305Mishra](https://github.com/Mayank2305Mishra) |
| Satyaroop Chatterjee | [@CodeitupwithSatya](https://github.com/CodeitupwithSatya) |
| Sourish Kumar Dutta | [@sourishkumardutta](https://github.com/sourishkumardutta) |


# AoG-ML4GOD — Animal Classification (AI of GOD 5)

72-class animal image classification with colourised segmentation overlays, optimised for **macro F1**. Each model is trained with stratified 5-fold CV, checkpointing on val macro-F1, and combined in a fold-averaged ensemble.

## Models

| Folder | Notebook(s) | Backbone | Input | Approach |
|---|---|---|---|---|
| `BEiT/` | `08_beitv2_base.ipynb` | `beitv2_base_patch16_224` (timm) | 224 | 5-fold CV, Optuna HPO, frozen→finetune, LLRD, CLS+mean-patch head |
| `ConVNEXT/` | `animal_classification_convnext.ipynb` | ConvNeXt-Base (pretrained) | 224 | Optuna LR search, cosine annealing, colour-robust aug, per-class F1 + confusion matrix |
| `DINO/` | `dinov2-last-stint.ipynb` | `facebook/dinov2-large` ViT-L/14 | 518 | 5-fold CV, high-res, `fold{i}_best.pt` checkpoints |
| `DINO/` | `dinov3-final-stint.ipynb`, `dinov3-timm.ipynb` | DINOv3 ViT (timm/HF) | 224–256 | Multi-layer CLS+patch fusion, CE + SupCon on dual augmented views, LLRD |
| `EVA2/` | `07_eva02_large_448.ipynb` | `eva02_large_patch14_448` (timm) | 448 | 5-fold CV, Optuna HPO, mean+max pool head (no CLS) |
| `EVA2/` | `aog-eva02.ipynb` | EVA02-Base | 224 | Colour-robust aug (channel permute, hue jitter, grayscale, mixup/cutmix), seed-42 15% split |
| `EfficientNet/` | `animal_classification_efficientnet.ipynb` | EfficientNet-B4 (pretrained) | 380 | Optuna LR search, cosine annealing, colour-robust aug |
| `MaxViT/` | `09_maxvit_base_256.ipynb`, `aog-maxvit-model.ipynb` | `maxvit_base_tf_256` / `maxvit_base_tf_224.in1k` (timm) | 224/256 | Frozen backbone (head-only linear probe), dual avg+max pool head |
| `Swin/` | `swin-final-stint.ipynb` | `swinv2_base_window12to16_192to256` (timm) | 256 | 5-fold CV, 5-fold ensemble + 2× TTA (fits 15 GB T4) |
| root | `ensamble-1.ipynb` | ConvNeXt + DINO + Swin | — | Loads 5-fold `.pt` checkpoints, averages fold probs on untouched `holdout.npz`, equal-weight family ensemble |

## Common training recipe

- **Splits:** stratified 5-fold, seed 42 (`SPLIT_SEED = 42`)
- **Metric:** val macro-F1 for checkpointing, early stopping, and model selection
- **Optimiser:** AdamW + warmup + cosine schedule, layer-wise LR decay (LLRD) for finetuned transformers
- **Regularisation:** label smoothing, MixUp/CutMix, dropout, gradient clipping
- **Augmentation:** colour-robust (ColorJitter, hue shift, grayscale, channel permutation) so models learn shape/texture, not overlay colour
- **Precision:** fp16 AMP on Kaggle GPUs
- **Outputs:** best/last `.pt` checkpoints, val+test predictions, `submission_{f1}.csv`, training curves, per-class F1 + confusion matrix

## Ensemble (`ensamble-1.ipynb`)

1. Point `BASE`/`SPLITS_DIR` at the saved splits and checkpoint zips.
2. Set backbone names + image sizes to match the training notebooks exactly.
3. Runs every fold checkpoint on `holdout.npz`, averages probabilities across folds, then averages families with equal weights.
4. Reports accuracy, macro-F1, classification report, and confusion matrix. No weight tuning on holdout.

## Reproduce

Each notebook is self-contained for Kaggle:

```bash
pip install -q optuna timm transformers accelerate safetensors
```

Then run top-to-bottom on a GPU runtime (T4 or better; DINOv2-L/518 and EVA02-L/448 benefit from more VRAM — reduce batch size if needed).

## Repo layout

```
BEiT/          BEiT-v2 base, 5-fold + Optuna
ConVNEXT/      ConvNeXt-Base baseline upgrade
DINO/          DINOv2-L high-res + DINOv3 fusion/SupCon runs
EVA2/          EVA-02 large@448 + base colour-robust run
EfficientNet/  EfficientNet-B4 baseline upgrade
MaxViT/        MaxViT head-only probes (224 + 256)
Swin/          Swin-V2 5-fold + TTA
ensamble-1.ipynb  cross-family checkpoint ensemble + holdout eval
```

## Notes

- Notebooks reference Kaggle paths like `/kaggle/working/splits` — adjust `SPLITS_DIR`/`BASE` for local runs.
- `00_shared_splits` (referenced by several notebooks) is not included; any stratified seed-42 5-fold split of the 72-class CSV works.
