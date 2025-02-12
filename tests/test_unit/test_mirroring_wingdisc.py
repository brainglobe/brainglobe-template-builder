from brainglobe_template_builder.preproc.mirroring_wingdisc import mirroring
import numpy as np


def test_mirroring():
    shape = np.random.randint(1, 10, size=3)
    random_image_array = np.random.randint(0, 100, size=shape)
    mirrored_image = mirroring(random_image_array)
    assert (mirrored_image == np.flip(random_image_array, 1)).all()
