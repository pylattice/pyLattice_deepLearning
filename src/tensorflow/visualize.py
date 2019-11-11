from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import numpy as np

def display_slice_from_batch(image_batch, image_n=0, z=8):
    plt.figure()
    one_slice = image_batch[image_n, z, :, :, :]
    #TODO Change 100, 100 hardcord to size, size
    img = Image.fromarray(one_slice.reshape(one_slice.shape[0],one_slice.shape[1]))
    imshow(np.asarray(img))
