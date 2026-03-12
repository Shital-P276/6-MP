# FloorViz — Deployment Plan (MuraNet / Phase 2)
# Full raster_parser replacement. All free tier.
# Last Updated: March 2026

---

## Why MuraNet Changes the Deployment Picture

SegFormer (Phase 1) was a preprocessor — it cleaned the image, then handed off
to the existing raster_parser. MuraNet (Phase 2) is a full replacement. It takes
a raw floor plan image and outputs wall segments + door/window bounding boxes
directly as `ParsedGeometry`. raster_parser is deleted entirely.

This matters for deployment because:

| | SegFormer Phase 1 | MuraNet Phase 2 |
|--|--|--|
| Role | Image preprocessor | Full parser replacement |
| Output | Cleaned wall mask | `ParsedGeometry` directly |
| Backbone | MiT-B2 (25M params) | MiT-B2 (25M params) — same |
| Extra heads | Segmentation only | Segmentation + Detection head |
| Checkpoint size | ~400MB | ~500–600MB |
| Inference (CPU) | ~300ms | ~500–700ms |
| Inference (GPU) | ~90ms | ~120–150ms |
| RAM at runtime | ~1.5GB | ~2–2.5GB |
| Training time (T4) | ~12hr / 50 epochs | ~20–30hr / 80 epochs |

The checkpoint is larger, RAM usage is slightly higher, and inference is a
bit slower — but still well within free tier limits. Nothing about the
deployment stack fundamentally changes. You still don't need a paid GPU server.

---

## The Stack (All Free)

```
User (Browser)
      │
      ▼
┌──────────────────────────────────┐
│  Frontend — Netlify Free         │  Static HTML + Three.js viewer
│  your-app.netlify.app            │  Auto-deploy from GitHub, global CDN
└───────────────┬──────────────────┘
                │ HTTPS REST calls
                ▼
┌──────────────────────────────────┐
│  Backend — HuggingFace Spaces    │  FastAPI + MLParser (MuraNet)
│  Docker, 16GB RAM, always-on     │  raster_parser.py deleted in Phase 2
│  your-name-floorviz.hf.space     │
└───────────────┬──────────────────┘
                │ from_pretrained() on startup
                ▼
┌──────────────────────────────────┐
│  Model — HuggingFace Hub         │  MuraNet checkpoint ~500–600MB
│  Private repo, free forever      │  Versioned: v1=SegFormer, v2=MuraNet
└──────────────────────────────────┘

                ┌─────────────────────────────┐
                │  Training — Kaggle Free      │  T4 GPU, 30hrs/week
                │  ~20–30hrs for MuraNet       │  Saves to HF Hub on completion
                └─────────────────────────────┘

Monthly cost: $0
```

---

## Component 1 — Training MuraNet on Kaggle

MuraNet takes longer than SegFormer (~20–30 hours for 80 epochs vs ~12 for 50).
Kaggle gives you 30 GPU hours/week so you will need to train across 2 weeks,
or use checkpoint resuming to continue across sessions.

### Backbone Choice (Important for Free Tier)

From `ClaudeGuideExtra.md`:

| Backbone | Params | GPU Inference | CPU Inference | RAM | Recommended |
|----------|--------|--------------|--------------|-----|-------------|
| MiT-B0 | 3.7M | ~50ms | ~200ms | ~0.8GB | CPU-only / edge |
| **MiT-B2** | **25M** | **~120ms** | **~600ms** | **~2GB** | **Best balance** |
| MiT-B4 | 64M | ~200ms | ~1200ms | ~4GB | Max accuracy, overkill |

**Use MiT-B2.** It fits in HF Spaces' 16GB easily and gives ~78% wall IoU.
MiT-B4 would also fit in 16GB RAM but inference on CPU becomes sluggish (~1.2s)
and training on Kaggle T4 takes ~50+ hours.

### Kaggle Setup for MuraNet Training

Same as SegFormer setup but with a longer run. Key differences:

**1 — Enable checkpoint resuming** (critical for 30hr/week limit)

Add this at the top of your training config in `train_muranet.py`:
```python
CONFIG = {
    ...
    'resume_from': '/kaggle/working/checkpoints/muranet/latest',  # resumes if exists
    'save_latest': True,   # save latest checkpoint every epoch (not just best)
    ...
}
```

This way if Kaggle's 12-hour session limit hits mid-training, you resume
from the latest checkpoint in the next session rather than starting over.

**2 — Training time estimate on Kaggle T4**
- 80 epochs on CubiCasa5k (~4200 train samples): ~20–25 hours
- Split across 2 Kaggle sessions (12hr + 10hr) across 2 days

**3 — Push to HF Hub at the end** (same as SegFormer):
```python
from huggingface_hub import HfApi
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID  = "YOUR_HF_USERNAME/floorviz-muranet-phase2"  # separate repo from SegFormer

api = HfApi()
api.create_repo(REPO_ID, private=True, exist_ok=True, token=HF_TOKEN)
api.upload_folder(
    folder_path="/kaggle/working/checkpoints/muranet/best",
    repo_id=REPO_ID,
    token=HF_TOKEN,
)
print(f"✅ MuraNet checkpoint → https://huggingface.co/{REPO_ID}")
```

**4 — Keep your SegFormer checkpoint too**

Even after Phase 2, keep the SegFormer repo on HF Hub. MuraNet has a
confidence fallback — if `wall_ratio` is outside 3–25%, it falls back.
The fallback now calls `RasterParser` (if you kept it) or simply returns
an error. Keeping SegFormer around means you could use it as an intermediate
fallback if MuraNet fails on an unusual image.

---

## Component 2 — Model Storage: HuggingFace Hub

Same as Phase 1, just a new repo for the MuraNet checkpoint.

| Repo | Size | Purpose |
|------|------|---------|
| `your-name/floorviz-segformer-phase1` | ~400MB | Phase 1 (keep for reference) |
| `your-name/floorviz-muranet-phase2` | ~500–600MB | Phase 2 (active) |
| `your-name/floorviz-muranet-indian` | ~600MB | Phase 3 (future) |

All free, all private, all versioned.

The backend loads whichever checkpoint is set in the environment variable:
```bash
ML_CHECKPOINT_PATH=your-name/floorviz-muranet-phase2
```

Swapping from SegFormer to MuraNet in production = change one env variable + restart.
No code change, no redeploy, no rebuild.

---

## Component 3 — Backend: HuggingFace Spaces

MuraNet's `MLParser` replaces `RasterParser` entirely. The FastAPI API surface
doesn't change — same endpoints, same request/response format — so the frontend
needs zero updates.

### What Changes in the Backend Code

In `app/core/pipeline.py`, Phase 2 replaces the parser:
```python
# Phase 1 (SegFormer preprocessor — remove this):
# from app.core.raster_parser import RasterParser
# parser = RasterParser()

# Phase 2 (MuraNet full replacement):
from src.ml_parser import MLParser
from app.core.raster_parser import RasterParser   # keep as fallback only

parser = MLParser(
    checkpoint_path=os.environ["ML_CHECKPOINT_PATH"],
    fallback_parser=RasterParser(),   # safety net for unusual inputs
)
# Call signature is identical — nothing else in pipeline.py changes:
geometry = parser.parse(image_path)
```

### RAM Usage on HF Spaces

| Component | RAM |
|-----------|-----|
| MuraNet model weights (MiT-B2) | ~2.0GB |
| FastAPI + Python runtime | ~0.3GB |
| OpenCV / processing | ~0.2GB |
| Per-request peak (inference) | ~0.5GB |
| **Total peak** | **~3.0GB** |

HF Spaces free tier gives 16GB. You're using ~3GB peak. Comfortable.

### Dockerfile (Phase 2)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
# raster_parser.py is still here as a fallback — do NOT delete it yet

ENV HF_TOKEN=""
ENV ML_CHECKPOINT_PATH="YOUR_HF_USERNAME/floorviz-muranet-phase2"
ENV USE_ML_PARSER="true"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app.core.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Note: keep `--workers 1`. MuraNet (like SegFormer) is loaded once per process.
Multiple workers = multiple copies of a 2GB model = OOM on any free tier.
Single worker with async FastAPI handles concurrent requests fine.

### HF Spaces Setup (Same as Phase 1)

1. huggingface.co/new-space → SDK: Docker → Hardware: CPU basic (free)
2. `app_port: 8000` in README.md front matter
3. Connect your GitHub repo (auto-redeploys on push)
4. Add secrets: `HF_TOKEN`
5. Update env var: `ML_CHECKPOINT_PATH` → `your-name/floorviz-muranet-phase2`

Your backend URL: `https://YOUR_HF_USERNAME-floorviz-backend.hf.space`

### Cold Start Warning

When you redeploy (push new code), HF Spaces rebuilds and restarts the container.
On restart, the backend downloads ~600MB from HF Hub before it can serve requests.
This takes ~60–90 seconds. After that first boot, it stays warm indefinitely.

To reduce this: pre-bake the weights into the Docker image.
```dockerfile
# Add this to Dockerfile (after pip install):
RUN python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'YOUR_HF_USERNAME/floorviz-muranet-phase2',
    local_dir='/app/checkpoints/muranet/best',
    token=os.environ.get('HF_TOKEN','')
)
"
# Then set: ENV ML_CHECKPOINT_PATH=/app/checkpoints/muranet/best
```
This makes the image larger (~600MB heavier) but startup is instant.
Trade-off is your call — both work fine on free tier.

---

## Component 4 — Frontend: Netlify (Unchanged)

The frontend doesn't change between Phase 1 and Phase 2 at all.
Same Netlify deployment, same API calls, same viewer. Nothing to update here.

---

## Inference Speed on Free Tier (CPU)

MuraNet on CPU (HF Spaces free tier has no GPU):

| Input size | Inference time (MiT-B2, CPU) |
|-----------|------------------------------|
| 512×512 | ~500–700ms |
| 1024×1024 | ~1.5–2s |

Full pipeline end-to-end (upload → parse → 3D model):
- DXF files: ~200ms (no ML, unchanged)
- Image/PDF files: ~700ms–2s depending on size

This is acceptable for a web app. Users are uploading architectural drawings
and expecting some processing time. Sub-2 seconds is fine.

If it becomes too slow for your use case, the only free upgrade is:
- HF Spaces **T4 GPU Space** is $0.60/hr pay-per-use (not free, but cheap)
- Kaggle inference notebooks (free but not a real server)

For Phase 2, CPU is fine. Plan for GPU only if you're getting consistent traffic.

---

## Rollback Strategy

Because checkpoint path is an env variable, rollback between phases is instant:

```
Phase 1 active:  ML_CHECKPOINT_PATH=your-name/floorviz-segformer-phase1
                 USE_ML_PREPROCESSOR=true  (wraps raster_parser)

Phase 2 active:  ML_CHECKPOINT_PATH=your-name/floorviz-muranet-phase2
                 USE_ML_PARSER=true  (replaces raster_parser)

Rollback to P1:  Change env var back → restart Space → done in 2 minutes
```

Keep both HF Hub repos. Never delete the Phase 1 checkpoint.

---

## Complete Checklist

```
PHASE 1 TRAINING (SegFormer — if not done yet)
  □ Kaggle: import notebook, enable T4 GPU, add HF_TOKEN secret
  □ Apply 3 notebook changes (CHECKPOINT_DIR, no Drive cell, HF push cell)
  □ Run All → wait ~12hr → verify checkpoint at huggingface.co/your-name/floorviz-segformer-phase1
  □ Test: val mIoU ≥ 75% (check TensorBoard logs in /kaggle/working/runs/)

PHASE 2 TRAINING (MuraNet)
  □ Add resume_from + save_latest to MuraNet training config
  □ New Kaggle notebook for train_muranet.py
  □ Enable T4 GPU, add HF_TOKEN, set CHECKPOINT_DIR
  □ Session 1: run ~12 hours (epochs 1–~40)
  □ Session 2 (next day): resume from latest checkpoint, run remaining epochs
  □ Verify checkpoint at huggingface.co/your-name/floorviz-muranet-phase2
  □ Check metrics: Wall IoU ≥ 78%, Door AP50 ≥ 75%, Window AP50 ≥ 70%

BACKEND DEPLOYMENT
  □ Push backend code to GitHub
  □ Update pipeline.py to use MLParser (MuraNet) instead of MLPreprocessor
  □ Create HF Space (Docker, CPU basic, public)
  □ Add HF_TOKEN secret to Space
  □ Set ML_CHECKPOINT_PATH=your-name/floorviz-muranet-phase2
  □ Connect GitHub → auto-build triggers
  □ Wait for cold start (~90s) → test /health endpoint
  □ Test with a real floor plan image

FRONTEND DEPLOYMENT
  □ Netlify: connect GitHub repo or drag viewer/ folder
  □ Set API_BASE to your HF Space URL
  □ Update CORS origins in main.py
  □ End-to-end test: upload floor plan → get 3D model back

ONGOING
  □ Phase 3 fine-tune on Indian plans → push to floorviz-muranet-indian repo
  □ Swap env var to activate Indian-tuned model — no code change needed
```

---

## Free Tier Limits Summary

| Service | Free Limit | Your Usage | OK? |
|---------|-----------|------------|-----|
| Kaggle GPU | 30hr/week | ~25hr over 2 weeks for MuraNet | ✅ |
| HF Hub storage | 50GB/repo | ~600MB per checkpoint | ✅ |
| HF Spaces RAM | 16GB | ~3GB peak | ✅ |
| HF Spaces CPU | 2 vCPU | Fine for async single-worker | ✅ |
| Netlify bandwidth | 100GB/month | Negligible for floor plan app | ✅ |
| Netlify builds | 300 min/month | ~2 min per deploy | ✅ |

No free tier limit is close to being hit. This stack handles Phase 2 MuraNet
comfortably without spending anything.
