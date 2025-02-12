from pathlib import Path
from brainglobe_utils.IO.image import load_any,save_any
from brainglobe_template_builder.preproc.mirroring_wingdisc import mirroring

path = Path('D:/UCL/Postgraduate_programme/downsampled_imaging_data/rawdata/sub-wd2_res-6um_channel-membrane.tif')
image = load_any(path)
mirrored_image = mirroring(image,axis=1)
save_any(mirrored_image, path.parent / 'mirrored.tif')