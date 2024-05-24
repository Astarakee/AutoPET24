import numpy as np
import SimpleITK as itk

def bbox_coordinate(binary_itk, xy_margin):
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
    # add margins for sanity
    x_start -= xy_margin
    x_end += xy_margin
    y_start -= xy_margin
    y_end += xy_margin

    return itk_array, x_start, x_end, y_start, y_end, z_start, z_end

def creat_bbox(itk_array, x_start, x_end, y_start, y_end, z_start, z_end, ct_spacing, ct_origin, ct_direction):
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
    new_itk.SetSpacing(ct_spacing)
    new_itk.SetOrigin(ct_origin)
    new_itk.SetDirection(ct_direction)

    # bounding box coordinates
    lsif = itk.LabelShapeStatisticsImageFilter()
    lsif.Execute(new_itk)
    boundingBox = np.array(lsif.GetBoundingBox(1))
    x_start, y_start, z_start, x_size, y_size, z_size = boundingBox
    x_end = x_start+x_size
    y_end = y_start+y_size
    z_end = z_start+z_size

    return temp_new_array, x_size, y_size, z_size, x_end, y_end, z_end
