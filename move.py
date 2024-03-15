from tqdm import tqdm
import os
import glob
import shutil

imgs = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/imgs_resampled_oropharynx/*")
labels = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/labels_resampled_oropharynx/*")

dst_folder_img = "C:/Users/user/Desktop/data/HECTOR2021/nnUNet_raw/Dataset001_HECTOR/imagesTr/"
dst_folder_lb = "C:/Users/user/Desktop/data/HECTOR2021/nnUNet_raw/Dataset001_HECTOR/labelsTr/"

for i in tqdm(range(len(imgs))):
    image_index = str(1 + i // 2).zfill(3)
    modality_index = str(i % 2).zfill(4)
    src = imgs[i]
    dst = dst_folder_img + "HECTOR_" + image_index + "_" + modality_index + '.nii.gz'
    shutil.copy(src, dst)

for i in tqdm(range(len(labels))):
    image_index = str(1 + i).zfill(3)
    src = labels[i]
    dst = dst_folder_lb + "HECTOR_" + image_index + '.nii.gz'
    shutil.copy(src, dst)


