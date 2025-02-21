from pathlib import Path
from brainglobe_utils.IO.image import load_any,save_any
from brainglobe_template_builder.preproc.mirroring_wingdisc import mirroring
import numpy as np

path = Path('D:/UCL/Postgraduate_programme/wing_disc_imaging_data/2025.2.13_tests/wd4_500ms_25_nobin_XY1739448853_Z000_T0_C0_0/wd4_500ms_25_nobin_XY1739448853_Z000_T0_C0_0.tif')
image = load_any(path)
mirrored_image = np.flip(image, axis=2)
save_any(mirrored_image, path.parent / 'mirrored_npflip_axis2.tif')