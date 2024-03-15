import skimage, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
from skimage.util import montage as montage2d
import gif_your_nifti.core as gif2nif

ct_path = 'C:/Users/user/Desktop/data/HECTOR2021/imgs_resampled_oropharynx/CHGJ007_CT.nii.gz'
pt_path = 'C:/Users/user/Desktop/data/HECTOR2021/imgs_resampled_oropharynx/CHGJ007_PT.nii.gz'
label_path = 'C:/Users/user/Desktop/data/HECTOR2021/labels_resampled_oropharynx/CHGJ007_gtvt.nii.gz'

ct_image = nib.load(ct_path).get_fdata()
pt_image = nib.load(pt_path).get_fdata()
test_mask = nib.load(label_path).get_fdata()

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
ax1.imshow(ct_image[:, :, ct_image.shape[1]//2], cmap = cm.hot)
ax1.set_title('CT')
ax2.imshow(pt_image[:, :, pt_image.shape[1]//2], cmap = cm.hot)
ax2.set_title('PT')
ax3.imshow(test_mask[:, :, test_mask.shape[1]//2], cmap = cm.hot)
ax3.set_title('Mask')
plt.hot()
fig.savefig('Axial.png')

fig, ax1 = plt.subplots(1, 1, figsize = (20, 20))
ax1.imshow(montage2d(test_mask), cmap ='bone')
fig.savefig('label_scan.png')

gif2nif.write_gif_normal(pt_path)