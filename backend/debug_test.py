import sys
sys.path.insert(0, '.')

from app.core.raster_parser import RasterParser

RasterParser().save_debug_image(
    "D:\\Projects\\floorviz\\sample_data\\img\\fp7.png",
    "debug.png"
)