import nibabel as nib
import glob
import numpy as np
import matplotlib.pyplot as plt

# folder_path = "C:/Users/user/Desktop/data/HECTOR2021/"
# hospital = "CHUS"
# imgs = glob.glob(folder_path + 'imgs_resampled/'+ hospital + '*')
# labels = glob.glob(folder_path + 'labels_resampled/'+ hospital + '*')

# for i in range(len(imgs)):
#     img = nib.load(imgs[i]).get_fdata()
#     label = nib.load(labels[i]).get_fdata()
#     print(img.shape)

file_path = "C:/Users/user/Desktop/data/HECTOR2021/nnUNet_preprocessed/Dataset001_HECTOR/nnUNetPlans_3d_fullres/HECTOR_155.npy"
npy = np.load(file_path)

print(npy.shape)
plt.imshow(npy[0][150])
plt.show()