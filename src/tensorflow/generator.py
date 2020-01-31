import os

import numpy as np
import skimage
from skimage.util.shape import view_as_blocks

from tensorflow import keras


class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=100, patch_size=32, percent_covered=1e-10):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.percent_covered = percent_covered
        self.on_epoch_end()
        
    def __load__(self, id_name):
        image_path = os.path.join(self.path, id_name, "lattice_light_sheet") + ".tif"
        mask_path = os.path.join(self.path, id_name, "truth") + ".tif"
        
        latticeMovieImage = skimage.external.tifffile.imread(image_path)
        latticeMovieImage = latticeMovieImage[:self.image_size, :self.image_size, :self.image_size]
        
        #Standardizing globally
        #image_mean = np.mean(latticeMovieImage, axis=(0, 1, 2), keepdims=True)
        #image_std = np.std(latticeMovieImage,  axis=(0, 1, 2), keepdims=True)
        #latticeMovieImage = (latticeMovieImage - image_mean) / image_std
        
        lattice_patches = view_as_blocks(latticeMovieImage, block_shape=(self.patch_size, self.patch_size, self.patch_size))
        lattice_patches = lattice_patches.reshape(int((self.image_size/self.patch_size)**3), self.patch_size, self.patch_size, self.patch_size)
        lattice_patches = np.expand_dims(lattice_patches, axis=-1)
        
        mask_image = skimage.external.tifffile.imread(mask_path)
        mask_image = mask_image[:self.image_size, :self.image_size, :self.image_size]
        #mask_image = np.expand_dims(mask_image, axis=-1)
        
        mask = np.zeros((self.image_size, self.image_size, self.image_size))
        mask = np.maximum(mask, mask_image)
        
        #TODO Check if view_as_blocks gives all possible blocks
        mask_patches = view_as_blocks(mask, block_shape=(self.patch_size, self.patch_size, self.patch_size))
        mask_patches = mask_patches.reshape(int((self.image_size/self.patch_size)**3), self.patch_size, self.patch_size, self.patch_size)
        mask_patches = np.expand_dims(mask_patches, axis=-1)

        #lattice_patches = lattice_patches/(2/3*65535.0)
        #BOTTOM LINE COMMENTED FOR PSNR5 Data. UNCOMMENT FOR FUTURE USE and use 255.0
        mask_patches = mask_patches/255.0
        
        weight_patches = np.zeros((self.patch_size ** 3, 2))
        weight_patches[:, 0] = 0.005
        weight_patches[:, 1] = 0.995
        weight_patches = np.squeeze(np.sum(weight_patches, axis=-1))
        
        #print(np.count_nonzero(mask == 1.0)/1000000.0)

        return lattice_patches, mask_patches, weight_patches
    
    def __filter_patches__(self, lattice_patches, mask_patches, weight_patches):
        zero_mask_ids = []
        
        for patch_index in range (0, mask_patches.shape[0]):
            patch = mask_patches[patch_index]
            if(np.count_nonzero(patch == 1.0)/(mask_patches.shape[1]**3) < self.percent_covered): #Means that the mask has all 0s
                zero_mask_ids.append(patch_index)
        
        lattice_patches = np.delete(lattice_patches, zero_mask_ids, axis=0)
        mask_patches = np.delete(mask_patches, zero_mask_ids, axis=0)
        weight_patches = np.delete(weight_patches, zero_mask_ids, axis=0)
            
        return lattice_patches, mask_patches, weight_patches
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
           self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask = []
        weight = []
        
        for id_name in files_batch:
            _img, _mask, _weight = self.__load__(id_name)
            
            image.append(_img)
            mask.append(_mask)
            weight.append(_weight)
            
        image = np.array(image)
        mask = np.array(mask)
        image = image.reshape(self.batch_size*image.shape[1], image.shape[2], image.shape[3], image.shape[4], image.shape[5])
        mask = mask.reshape(self.batch_size*mask.shape[1], mask.shape[2], mask.shape[3], mask.shape[4], mask.shape[5])
        
        image, mask, weight = self.__filter_patches__(image, mask, weight)
        
        #Standardizing locally
        image_mean = np.mean(image, axis=(1,2,3), keepdims=True)
        image_std = np.std(image,  axis=(1,2,3), keepdims=True)
        image = (image - image_mean) / image_std
        
        return image, mask#, weight
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
