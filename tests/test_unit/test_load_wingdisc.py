from brainglobe_template_builder.preproc.load_wingdisc import load_images,load_channel_data
from brainglobe_template_builder.preproc.mirroring_wingdisc import mirroring
from pathlib import Path
import tifffile as tiff

def test_load_images_czi():
    path_czi = Path(
        'D:/UCL/Postgraduate_programme/wing_disc_imaging_data/wd2_overview-Airyscan-Processing-02/wd2_overview-Airyscan-Processing-02.czi')
    assert path_czi.exists()
    path_tiff = Path(
        'D:/UCL/Postgraduate_programme/wing_disc_imaging_data/wd2_overview-Airyscan-Processing-02/wd2_overview-Airyscan-Processing-02_C1.tif')
    assert path_tiff.exists()
    czi_array = load_channel_data(path_czi, image_format='czi', channel=1)
    tiff_array = tiff.imread(path_tiff)
    assert (czi_array == tiff_array).all()