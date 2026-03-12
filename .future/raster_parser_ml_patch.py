# raster_parser_ml_patch.py
#
# This is NOT a standalone file.
# It shows exactly where and how to add ML preprocessing to raster_parser.py.
#
# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# In your existing file: app/core/raster_parser.py
#
# Find the parse() method. It likely starts with something like:
#
#     def parse(self, filepath: str, pixels_per_meter: float = 60.25, ...) -> ParsedGeometry:
#         """..."""
#         # Convert PDF to image if needed
#         image_path = ...
#
# ADD the following block right after the image_path is established
# (after PDF→PNG conversion, before any cv2.imread or Hough detection):
#
# ─────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ PHASE 1 ML PATCH — add this block to raster_parser.py parse() method   │
# └─────────────────────────────────────────────────────────────────────────┘

USE_ML_PREPROCESSOR = True   # Set False to instantly revert to original behaviour

if USE_ML_PREPROCESSOR:
    try:
        import tempfile
        import cv2 as _cv2
        # Adjust path if your project structure differs
        import sys as _sys
        _sys.path.insert(0, './src')
        from ml_preprocessor import MLPreprocessorWithFallback as _MLPre

        # Cache the preprocessor instance (avoid reloading model per request)
        # In production, initialise this once at module level instead
        if not hasattr(self, '_ml_pre') or self._ml_pre is None:
            self._ml_pre = _MLPre(
                checkpoint_path='./checkpoints/segformer/best',
                wall_ratio_min=0.03,
                wall_ratio_max=0.25,
            )

        clean_path, used_ml = self._ml_pre.get_clean_image_path(image_path)
        if used_ml:
            image_path = clean_path   # raster_parser processes the ML-cleaned image
            # Optionally log which images triggered ML vs fallback:
            # print(f"[RasterParser] ML preprocessing: {image_path}")

    except Exception as _e:
        # If anything goes wrong with ML, silently fall through to original
        print(f"[RasterParser] ML preprocessing failed, using original: {_e}")

# ┌─────────────────────────────────────────────────────────────────────────┐
# │ END OF PATCH — everything below is original raster_parser.py unchanged │
# └─────────────────────────────────────────────────────────────────────────┘

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT NOTES
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. The MLPreprocessorWithFallback automatically falls back to the original
#    image if the ML prediction doesn't look like a floor plan (wall_ratio
#    outside 3-25%). So this is safe to deploy even before the model is great.
#
# 2. To cache the preprocessor at module level (recommended for production),
#    move the instantiation outside the parse() method:
#
#    class RasterParser:
#        _ml_pre = None   # class-level cache
#
#        def __init__(self, ...):
#            # existing init code
#
#            # Load ML preprocessor if checkpoint exists
#            if os.path.exists('./checkpoints/segformer/best'):
#                from ml_preprocessor import MLPreprocessorWithFallback
#                RasterParser._ml_pre = MLPreprocessorWithFallback(
#                    checkpoint_path='./checkpoints/segformer/best'
#                )
#
# 3. The cleaned image written to /tmp is auto-managed by MLPreprocessorWithFallback.
#    Call self._ml_pre.cleanup() at app shutdown to remove temp files.
#
# 4. DXF files bypass raster_parser entirely (processed by dxf_parser.py).
#    This patch only affects raster (PNG/JPG/PDF) inputs.
