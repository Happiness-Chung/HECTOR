import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.transform import resize
import matplotlib.cm as cm

ct_path = 'C:/Users/user/Desktop/data/HECTOR2021/(2,2,2)/imgs_resampled/*CT*'
pt_path = 'C:/Users/user/Desktop/data/HECTOR2021/(2,2,2)/imgs_resampled/*PT*'
path_label = 'C:/Users/user/Desktop/data/HECTOR2021/(2,2,2)/labels_resampled/*'
save_path = 'C:/Users/user/Desktop/data/HECTOR2021/(2,2,2)/Images/'
save_path_label = 'C:/Users/user/Desktop/data/HECTOR2021/(2,2,2)/Labels/'

ct_imgs = glob.glob(ct_path)
print(len(ct_imgs))
pt_imgs = glob.glob(pt_path)
print(len(pt_imgs))
labels = glob.glob(path_label)[:224]
print(len(labels))

for i in range(len(ct_imgs)):
    ct = nib.load(ct_imgs[i]).get_fdata()
    pt = nib.load(pt_imgs[i]).get_fdata()

    label = nib.load(labels[i])

    mix = 0.5 * ct + 0.5 + pt
    nii_mix = nib.Nifti1Image(mix, affine = np.eye(4))
    nii_ct = nib.Nifti1Image(ct, affine = np.eye(4))
    nii_pt = nib.Nifti1Image(pt, affine = np.eye(4))
    nib.save(nii_ct, save_path + "ct{}.nii.gz".format(i))
    nib.save(nii_pt, save_path + "pt{}.nii.gz".format(i))
    nib.save(nii_mix, save_path + "mix{}.nii.gz".format(i))
    nib.save(label, save_path_label + "ct{}.nii.gz".format(i))
    nib.save(label, save_path_label + "pt{}.nii.gz".format(i))
    nib.save(label, save_path_label + "mix{}.nii.gz".format(i))