import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from skimage.transform import resize

folder_path = "C:/Users/user/Desktop/data/HECTOR2021/" ### 
hospitals = ["CHGJ", "CHMR", "CHUM", "CHUP", "CHUS"]

total = 0
cnt = 0

for h in range(1,6):
    save_path_img = "C:/Users/user/Desktop/data/HECTOR2021/npy_imgs/"
    save_path_label = "C:/Users/user/Desktop/data/HECTOR2021/npy_labels/"
    img_paths = glob.glob(folder_path + 'imgs_resampled/' + hospitals[h-1] + '*')
    label_paths = glob.glob(folder_path + 'labels_resampled/' + hospitals[h-1] + '*')

    pass_cnt = 0
    data_idx = 0

    for i in tqdm(range(len(label_paths)), desc= "CT Mean, Hospital {}: ".format(h)): 
        
        cnt += 1

        ct = nib.load(img_paths[2*i]).get_fdata()
        ct = resize(ct, (200, 200, 200))
        ct = ct.astype(np.float32)

        # Hounsfild normalization
        min_hu_value = -1000  # Typical lower bound for HU
        max_hu_value = 1000   # Typical upper bound for HU
        clipped_ct = np.clip(ct, min_hu_value , max_hu_value)

        total += clipped_ct.mean()

ct_mean = total / cnt
print("Total CT mean: ", ct_mean)

total = 0
cnt = 0

for h in range(1,6):
    save_path_img = "C:/Users/user/Desktop/data/HECTOR2021/npy_imgs/"
    save_path_label = ":/Users/user/Desktop/data/HECTOR2021/npy_labels/"
    img_paths = glob.glob(folder_path + 'imgs_resampled/' + hospitals[h-1] + '*')
    label_paths = glob.glob(folder_path + 'labels_resampled/' + hospitals[h-1] + '*')

    pass_cnt = 0
    data_idx = 0

    for i in tqdm(range(len(label_paths)), desc= "CT SD, Hospital {}: ".format(h)): 
        
        cnt += 1

        ct = nib.load(img_paths[2*i]).get_fdata()
        ct = resize(ct, (200, 200, 200))
        ct = ct.astype(np.float32)

        # Hounsfild normalization
        min_hu_value = -1000  # Typical lower bound for HU
        max_hu_value = 1000   # Typical upper bound for HU
        clipped_ct = np.clip(ct, min_hu_value , max_hu_value)

        total += ((clipped_ct - ct_mean) * (clipped_ct - ct_mean)).mean()

ct_sd = np.sqrt(total/cnt)
print("Total CT SD: ", ct_sd)

for h in range(1,6):
    save_path_img = "C:/Users/user/Desktop/data/HECTOR2021/npy_imgs/"
    save_path_label = "C:/Users/user/Desktop/data/HECTOR2021/npy_labels/"
    img_paths = glob.glob(folder_path + 'imgs_resampled/' + hospitals[h-1] + '*')
    label_paths = glob.glob(folder_path + 'labels_resampled/' + hospitals[h-1] + '*')

    pass_cnt = 0
    data_idx = 0

    for i in tqdm(range(len(label_paths)), desc= "Hospital {}: ".format(h)): 
        
        ct = nib.load(img_paths[2*i]).get_fdata()
        ct = resize(ct, (200, 200, 200))
        ct = ct.astype(np.float32)

        # Hounsfild normalization
        min_hu_value = -1000  # Typical lower bound for HU
        max_hu_value = 1000   # Typical upper bound for HU
        clipped_ct = np.clip(ct, min_hu_value , max_hu_value)
        # normalized_ct_hu_image = (clipped_ct - min_hu_value) / (max_hu_value - min_hu_value) * 2 - 1

        # standardization
        ct = (clipped_ct - ct_mean) / ct_sd

        pt = nib.load(img_paths[2*i+1]).get_fdata()
        pt = resize(pt, (200, 200, 200))
        pt = pt.astype(np.float32)

        # standardization
        pt = (pt - 0) / 5.16

        label = nib.load(label_paths[i]).get_fdata()
        label = resize(label, (200, 200, 200))
        label = label.astype(np.float32)
        
     
        np.save(save_path_img + hospitals[h-1] + str(i) + '_ct', pt)
        np.save(save_path_img + hospitals[h-1] + str(i) + '_pt', ct)
        np.save(save_path_label + hospitals[h-1] + str(i), label)