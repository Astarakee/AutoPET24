from tools.sitk_stuff import read_nifti
from copy import deepcopy
import SimpleITK as itk
import os
import numpy as np
from tools.writer import write_nifti_from_vol

data_path = '/mnt/workspace/data/00_junks/test_ct_crop/fdg_a41d59682f_09-29-2006-NA-PET-CTGanzkoerperprimaermitKM-76863_0000.nii.gz'
write_path = '/mnt/workspace/data/00_junks/test_ct_crop'
xy_margin = 5

img_array, img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(data_path)

img_binary = deepcopy(img_array)
img_binary[img_binary>=0] = 1
img_binary[img_binary<0] = 0
img_binary = img_binary.astype(np.uint8)


binary_itk = itk.GetImageFromArray(img_binary)
binary_itk.SetSpacing(img_spacing)
binary_itk.SetOrigin(img_origin)
binary_itk.SetDirection(img_direction)

# largets connect component contains only body
component_image = itk.ConnectedComponent(binary_itk)
sorted_component_image = itk.RelabelComponent(component_image, sortByObjectSize=True)
largest_component_binary_image = sorted_component_image == 1
itk_array = itk.GetArrayFromImage(largest_component_binary_image)

# bounding box coordinates
lsif = itk.LabelShapeStatisticsImageFilter()
lsif.Execute(largest_component_binary_image)
boundingBox = np.array(lsif.GetBoundingBox(1))
x_start, y_start, z_start, x_size, y_size, z_size = boundingBox

z_dim = itk_array.shape[0]

x_end = x_start + x_size
y_end = y_start + y_size
z_end = z_dim  # get all axial slices
z_start = 0
# add margins for sanity check
x_start -= xy_margin
x_end += xy_margin
y_start -= xy_margin
y_end += xy_margin

# creating masks for orthogonal views
new_array1 = np.zeros_like(itk_array)
new_array2 = np.zeros_like(itk_array)
new_array3 = np.zeros_like(itk_array)
new_array1[z_start:z_end, :, :] = 10
new_array2[:, :, x_start:x_end] = 10
new_array3[:, y_start:y_end, :] = 10
temp_new_array = new_array1 + new_array2 + new_array3
# binarizing the mask
temp_new_array[temp_new_array == 30] = 100
temp_new_array[temp_new_array != 100] = 0
temp_new_array[temp_new_array == 100] = 1
temp_new_array = temp_new_array.astype('uint8')

new_itk = itk.GetImageFromArray(temp_new_array)
new_itk.SetSpacing(img_spacing)
new_itk.SetOrigin(img_origin)
new_itk.SetDirection(img_direction)

# bounding box coordinates
lsif = itk.LabelShapeStatisticsImageFilter()
lsif.Execute(new_itk)
boundingBox = np.array(lsif.GetBoundingBox(1))
x_start, y_start, z_start, x_size, y_size, z_size = boundingBox

x_end = x_start + x_size
y_end = y_start + y_size
z_end = z_start + z_size

masked_img1 = temp_new_array * img_array
cropped_array1 = np.zeros((z_size, y_size, x_size))
cropped_array1 = masked_img1[z_start:z_end, y_start:y_end, x_start:x_end]

cropped_path1 = os.path.join(write_path, 'cropped1.nii.gz')


write_nifti_from_vol(cropped_array1, img_origin, img_spacing, img_direction, cropped_path1)










