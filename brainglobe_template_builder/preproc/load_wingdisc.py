from bioio import BioImage
import bioio_czi
from pathlib import Path
import bioio_sldy
import numpy as np
import tifffile as tiff

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


def load_channel_data(
        image_path: Path,
        image_format: str,
        channel: int
) -> np.ndarray:
    """load the images for certain formats

    Args:
        image_path: path to Path
        image_format: image format ('czi' or 'sldy', can add more simplely
        channel: the image channel that need to be loaded

    Returns:
        np.ndarray:

    Raises:
        ValueError:
    """
    # make a dictionary
    readers = {
        "czi": bioio_czi.Reader,
        "sldy": bioio_sldy.Reader
    }

    # verify if the specified image format is supported
    if image_format not in readers:
        raise ValueError(f"Unsupported format: {image_format}. Supported: {list(readers.keys())}")

    # load images
    bio_image = BioImage(image_path, reader=readers[image_format])
    return bio_image.get_image_data("ZYX", T=0, C=channel)



# Check the image dimensions
original_file_path = Path('D:/UCL/Postgraduate_programme/wing_disc_imaging_data/wd2_overview-Airyscan-Processing-02/wd2_overview-Airyscan-Processing-02.czi')
assert original_file_path.exists()
czi_array = load_channel_data(original_file_path,image_format='czi',channel=1)
print(czi_array.shape)