import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import glob

imgs_path = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/imgs_resampled/*")
labels_path = glob.glob("C:/Users/user/Desktop/data/HECTOR2021/labels_resampled/*")

ct = nib.load(imgs_path[0]).get_fdata()
pt = nib.load(imgs_path[1]).get_fdata()
label = nib.load(labels_path[0]).get_fdata()

for i in range(len(ct)):

    if label[:,:,i].sum() > 0 :

        print(label[:,:,i].sum())

        ct[:,:,i] = (ct[:,:,i] - ct[:,:,i].min()) / (ct.max() - ct.min())
        pt[:,:,i] = (pt[:,:,i] - pt[:,:,i].min()) / (pt.max() - pt.min())

        fig, axs = plt.subplots(1,2, figsize=(12,8))
        axs[0].imshow(ct[:,:,i] * 0.5 + label[:,:,i] * 0.5)
        axs[1].imshow(pt[:,:,i] * 0.5 + label[:,:,i] * 0.5)
        plt.show()
