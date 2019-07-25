import os
import shutil
import numpy as np
import nibabel as nib
from PIL import Image

def load_image_data(filename, dirname=''):
  full_path = os.path.join('./niftii', dirname, filename)
  img = nib.load(full_path)
  img_data = img.get_fdata(dtype=np.float32)

  return img_data

def labels_count_by_axis(seg_data, labels, axis):
  slice_count = seg_data.shape[axis]
  slice_found = 0
  slices = np.moveaxis(seg_data, axis, 0)

  for i in range(slice_count):
    slice = slices[i, :, :]
          
    if any(any(x in row for x in labels) for row in slice):
      slice_found += 1
      
  print('axis={}, {:.0%}'.format(axis, slice_found / slice_count))
  
def labels_count(seg_data, labels):
  for axis in range(3):
    labels_count_by_axis(seg_data, labels, axis)

def convert_to_binary_mask(slice_data, labels):
  binary_slice = np.array([[255.0 if pixel in labels else 0.0 for pixel in row] for row in slice_data]).astype(np.float32)
  return binary_slice

def resize_image(image):
  width, height = image.size

  if width == height:
    new_im = image.resize((256, 256))
  elif width > height:
    new_im = image.resize((256, 176))
  else:
    new_im = image.resize((176, 256))

  return new_im

def norm_to_uint8(data):
  max_value = data.max()
  if not max_value == 0:
    data = data / max_value

  data = 255 * data
  img = data.astype(np.uint8)
  return img

def save_slice(slice_data, filename):
  slice_data = norm_to_uint8(slice_data)
  
  im = Image.fromarray(np.rot90(slice_data))
  im = resize_image(im)
  im.save(filename)

def create_training_dataset(images, masks, labels, out_file, axis=0, out_dir='z_train'):
  slice_count = images.shape[axis]

  # break if number of source scan slices are different than mask slices count
  if not images.shape[axis] == masks.shape[axis]:
    print('Mask and images slice count is not the same.')
    exit()

  # prepare directory structure
  mask_fullpath = os.path.join('.', out_dir, 'mask/axis{}'.format(axis))
  img_fullpath = os.path.join('.', out_dir, 'img/axis{}'.format(axis))

  if not os.path.exists(mask_fullpath):
    os.makedirs(mask_fullpath)
  if not os.path.exists(img_fullpath):
    os.makedirs(img_fullpath)

  # depending of chosen axis we have to move it to the beginning to easily extract 2d slice
  image_slices = np.moveaxis(images, axis, 0)
  mask_slices = np.moveaxis(masks, axis, 0)

  for i in range(slice_count):
    image_slice = image_slices[i, :, :]
    mask_slice = mask_slices[i, :, :]

    # if any(any(x in row for x in labels) for row in mask_slice):
      # if only images with present mask

    img_fullname = '{}/{}_slice_{:03d}.png'.format(img_fullpath, out_file, i)
    mask_fullname = '{}/{}_slice_{:03d}.png'.format(mask_fullpath, out_file, i)

    binary_mask_slice = convert_to_binary_mask(mask_slice, labels)
    
    save_slice(image_slice, img_fullname)
    save_slice(binary_mask_slice, mask_fullname)

labels = [4.0, 43.0]
axis = 0
dataset_dir = 'z_train'

input_mask_niftii = 'aseg-in-t1weighted_2std.nii.gz'
input_img_niftii = 't1weighted_2std.nii.gz'
output_dir = '{}/{}_axis{}'.format('datasets', dataset_dir, axis)

if os.path.exists(output_dir):
  shutil.rmtree(output_dir)

# Set 1
input_dir = 'Colin27-1'
output_filename = input_dir

mask_data = load_image_data(filename=input_mask_niftii, dirname=input_dir)
image_data = load_image_data(filename=input_img_niftii, dirname=input_dir)
labels_count(mask_data, labels)
create_training_dataset(image_data, mask_data, labels, output_filename, axis=axis, out_dir=output_dir)

# # Set 2
# input_dir = 'NKI-RS-22-20'
# output_filename = input_dir

# mask_data = load_image_data(filename=input_mask_niftii, dirname=input_dir)
# image_data = load_image_data(filename=input_img_niftii, dirname=input_dir)
# labels_count(mask_data, labels)
# create_training_dataset(image_data, mask_data, labels, output_filename, axis=axis, out_dir=output_dir)

# # Set 3
# input_dir = 'NKI-TRT-20-1'
# output_filename = input_dir

# mask_data = load_image_data(filename=input_mask_niftii, dirname=input_dir)
# image_data = load_image_data(filename=input_img_niftii, dirname=input_dir)
# labels_count(mask_data, labels)
# create_training_dataset(image_data, mask_data, labels, output_filename, axis=axis, out_dir=output_dir)
