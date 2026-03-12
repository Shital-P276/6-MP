# FloorPlan ML — Phase 1: SegFormer Preprocessor

Replaces fragile Hough detection in `raster_parser.py` with a SegFormer model
that produces clean wall/door/window masks. The existing pipeline is unchanged —
only the input image to raster_parser gets cleaned up first.

---

## Quick Setup

### 1. Create virtual environment

```bash
python -m venv floorplan_ml
source floorplan_ml/bin/activate        # Linux/Mac
# floorplan_ml\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install albumentations opencv-python pillow numpy matplotlib
pip install tensorboard huggingface_hub scikit-image
```

**No GPU locally?** Use Google Colab (recommended) or Kaggle free tier.
- Colab: Runtime → Change runtime type → A100 (Pro) or T4
- Kaggle: Notebook → Settings → Accelerator → GPU T4 x2

### 3. Download CubiCasa5k

```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git data/raw/cubicasa5k
```

Then run the data prep script:

```bash
python tools/prepare_cubicasa.py
```

This converts SVG annotations → PNG masks and writes train/val/test split JSONs
into `data/processed/splits/`.

### 4. Train SegFormer

```bash
python src/train_segformer.py
```

Checkpoint saved to `checkpoints/segformer/best/` when val mIoU improves.

### 5. Test on a single image

```bash
python src/ml_preprocessor.py --image path/to/floorplan.png --show
```

### 6. Integrate into existing backend

In `app/core/raster_parser.py`, set `USE_ML = True` at the top of `parse()`.
See `src/ml_preprocessor.py` for the 4-line patch.

---

## Directory Structure

```
floorplan-ml/
  data/
    raw/cubicasa5k/           git clone here
    processed/
      images/                 resized PNGs
      masks/                  4-class PNG masks (0=bg,1=wall,2=door,3=win)
      splits/                 train.json, val.json, test.json
    test_images/              fixed evaluation set (never augmented)
  checkpoints/
    segformer/best/           Phase 1 checkpoint (save_pretrained format)
  src/
    augmentations.py          Indian-plan augmentation pipeline
    datasets.py               FloorPlanDataset torch Dataset
    train_segformer.py        training loop
    ml_preprocessor.py        inference + pipeline integration
  tools/
    prepare_cubicasa.py       SVG → mask conversion + split generation
  tests/
    test_ml_preprocessor.py   unit tests for the preprocessor
  notebooks/
    01_data_exploration.ipynb
    02_augmentation_preview.ipynb
```

---

## Target Metrics (Phase 1)

| Metric       | Target |
|-------------|--------|
| Wall IoU    | ≥ 75%  |
| Door AP50   | ≥ 70%  |
| Window AP50 | ≥ 65%  |
| Inference   | < 2s   |

Once these are met, proceed to Phase 2 (MuraNet full replacement).
