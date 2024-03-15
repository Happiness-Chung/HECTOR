import numpy as np
import nibabel as nib
import glob
from tqdm import tqdm

npys = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/npy_imgs/*")
save_npys = "C:/Users/user/Desktop/data/HECTOR2021/nnUNet_raw/Dataset001_HECTOR/imagesTr/"
labels = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/npy_labels/*")
save_labels = "C:/Users/user/Desktop/data/HECTOR2021/nnUNet_raw/Dataset001_HECTOR/labelsTr/"

for i in tqdm(range(len(labels))):
    image_index = str(1 + i).zfill(3)
    label = np.load(labels[i])
    label[label >= 0.2] = 1
    label[label < 0.2] = 0
    ni_img = nib.Nifti1Image(label, affine = np.eye(4))
    # print(save_labels + "HECTOR_" + image_index + '.nii.gz')
    nib.save(ni_img, save_labels + "HECTOR_" + image_index + '.nii.gz')

for i in tqdm(range(len(npys))):
    image_index = str(1 + i // 2).zfill(3)
    modality_index = str(i % 2).zfill(4)

    ni_img = nib.Nifti1Image(label, affine = np.eye(4))
    # print(save_npys + "HECTOR_" + str(image_index) + "_" + str(modality_index))
    nib.save(ni_img, save_npys + "HECTOR_" + image_index + "_" + modality_index + '.nii.gz')
