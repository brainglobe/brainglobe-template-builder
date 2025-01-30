from bioio import BioImage
import bioio_czi
from pathlib import Path
import bioio_sldy

def load_images(path:Path,file_type:str,channel:int):
    if file_type == 'czi':
        array = BioImage(path, reader=bioio_czi.Reader)
        channel_array = array.get_image_data("ZYX", T=0, C=channel) # will only read channel 1 into memory
    elif file_type == 'sldy':
        array = BioImage(path, reader=bioio_sldy.Reader)
        channel_array = array.get_image_data("ZYX", T=0, C=channel)
    else:
        print('File format not supported!')
    return channel_array


# Check the image dimensions
original_file_path = Path('D:/UCL/Postgraduate_programme/wing_disc_imaging_data/wd2_overview-Airyscan-Processing-02/wd2_overview-Airyscan-Processing-02.czi')
assert original_file_path.exists()
czi_array = load_images(original_file_path,file_type='czi',channel=1)
print(czi_array.shape)
