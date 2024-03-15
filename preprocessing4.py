import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.transform import resize

img_path = "C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Images/*"
label_path = "C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Labels/*"
save_path_img = "C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Images/"
save_path_label = "C:/Users/user/Desktop/data/HECTOR2021/crops/(1,1,1)/Labels/"

imgs = glob.glob(img_path)
labels = glob.glob(label_path)

# for i in range(len(imgs)):

#     img = nib.load(imgs[i]).get_fdata()
#     resized_img = resize(img, (144, 144, 144))

#     nii_img = nib.Nifti1Image(resized_img, affine = np.eye(4))
    
#     nib.save(nii_img, save_path_img + "{}.nii.gz".format(i))
    

for i in range(len(labels)):
    
    label = nib.load(labels[i]).get_fdata()
    resized_label = resize(label, (144, 144, 144))
    resized_label[label >= 0.2] = 1
    resized_label[label < 0.2] = 0

    nii_label = nib.Nifti1Image(resized_label, affine = np.eye(4))

    nib.save(nii_label, save_path_label + "{}.nii.gz".format(i))


