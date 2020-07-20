import numpy as np
import skimage

def predict_mask(model, image_path, image_size, patch_size):
    latticeMovieImage = skimage.external.tifffile.imread(image_path)
    latticeMovieImage = latticeMovieImage[:image_size, :image_size, :image_size]
    result = np.zeros((image_size, image_size, image_size, 1))

    for x in range(image_size//patch_size):
        for y in range(image_size//patch_size):
            for z in range(image_size//patch_size):
                x_index = x*patch_size
                y_index = y*patch_size
                z_index = z*patch_size
            
                current_lattice_patch = latticeMovieImage[x_index:x_index+patch_size, y_index:y_index+patch_size, z_index:z_index+patch_size]
                current_lattice_patch = current_lattice_patch.reshape(1, patch_size, patch_size, patch_size, 1)
            
                result_patch = model.predict(current_lattice_patch)
                for i in range(patch_size):
                    for j in range(patch_size):
                        for k in range(patch_size):
                            result_pixel = result_patch[0, i, j, k, 0]
                            result[x_index+i, y_index+j, z_index+k, 0] = result_pixel
                        
    return result.reshape(1, image_size, image_size, image_size, 1)

#Automatically trim image of rectangular shape and generate patches
def predict_mask(model, image_path, patch_size, offset=np.zeros((3,), dtype=int)):
    assert offset.size is 3, "Offset array needs to have a size of 3."
    
    latticeMovieImage = skimage.external.tifffile.imread(image_path)
    x_extra = latticeMovieImage.shape[0]%patch_size
    x_size = latticeMovieImage.shape[0] - x_extra
    if offset[0] > x_extra:
        print("1st dim offset exceeds image dim")
        offset[0] = 0
    
    y_extra = latticeMovieImage.shape[1]%patch_size
    y_size = latticeMovieImage.shape[1] - y_extra
    if offset[1] > y_extra:
        print("2st dim offset exceeds image dim")
        offset[1] = 0
    
    z_extra = latticeMovieImage.shape[2]%patch_size
    z_size = latticeMovieImage.shape[2] - z_extra
    if offset[2] > z_extra:
        print("3rd dim offset exceeds image dim")
        offset[2] = 0
    
    latticeMovieImage = latticeMovieImage[offset[0]:x_size+offset[0], offset[1]:y_size+offset[1], offset[2]:z_size+offset[2]]
    print("Image cropped to: " + str(x_size) + ", " + str(y_size) + ", " + str(z_size))
    result = np.zeros((x_size, y_size, z_size, 1))

    for x in range(x_size // patch_size):
        for y in range(y_size // patch_size):
            for z in range(z_size // patch_size):
                x_index = x * patch_size
                y_index = y * patch_size
                z_index = z * patch_size

                current_lattice_patch = latticeMovieImage[x_index:x_index + patch_size, y_index:y_index + patch_size,
                                        z_index:z_index + patch_size]
                current_lattice_patch = current_lattice_patch.reshape(1, patch_size, patch_size, patch_size, 1)

                result_patch = model.predict(current_lattice_patch)
                for i in range(patch_size):
                    for j in range(patch_size):
                        for k in range(patch_size):
                            result_pixel = result_patch[0, i, j, k, 0]
                            result[x_index + i, y_index + j, z_index + k, 0] = result_pixel

    return result.reshape(1, x_size, y_size, z_size, 1)