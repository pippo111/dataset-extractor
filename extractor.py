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

def save_dataset(images, masks, labels, out_file, axis=0, out_dir='z_train'):
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

    if any(any(x in row for x in labels) for row in mask_slice):
      img_fullname = '{}/{}_slice_{:03d}.png'.format(img_fullpath, out_file, i)
      mask_fullname = '{}/{}_slice_{:03d}.png'.format(mask_fullpath, out_file, i)

      binary_mask_slice = convert_to_binary_mask(mask_slice, labels)
      
      save_slice(image_slice, img_fullname)
      save_slice(binary_mask_slice, mask_fullname)

def create_dataset(input_scan, niftii_mask_filename, niftii_img_filename, labels, axis, output_dir):
  print('Processing {}...'.format(input_scan))
  output_filename = input_scan

  mask_data = load_image_data(filename=niftii_mask_filename, dirname=input_scan)
  image_data = load_image_data(filename=niftii_img_filename, dirname=input_scan)
  labels_count(mask_data, labels)
  save_dataset(image_data, mask_data, labels, output_filename, axis=axis, out_dir=output_dir)

labels = [4.0, 43.0]
input_mask_niftii = 'aseg-in-t1weighted_2std.nii.gz'
input_img_niftii = 't1weighted_2std.nii.gz'
train_valid_ratio = 0.75

input_scans = [
  'NKI-RS-22-1',
  'NKI-RS-22-2',
  'NKI-RS-22-3',
  'NKI-RS-22-4',
  'NKI-RS-22-5',
  'NKI-RS-22-6',
  'NKI-RS-22-7',
  'NKI-RS-22-8',
  'NKI-RS-22-9',
  'NKI-RS-22-10',
  'NKI-RS-22-11',
  'NKI-RS-22-12',
  'NKI-RS-22-13',
  'NKI-RS-22-14',
  'NKI-RS-22-15',
  'NKI-RS-22-16',
  'NKI-RS-22-17',
  'NKI-RS-22-18',
  'NKI-RS-22-19',
  'NKI-RS-22-20'
]

train_valid_split = int(len(input_scans) * 0.85)

axis = 1
train_dir = 'z_train'
validation_dir = 'z_validation'
output_train = os.path.join('datasets', '{}_axis{}'.format(train_dir, axis))
output_validation = os.path.join('datasets', '{}_axis{}'.format(validation_dir, axis))

if os.path.exists(train_dir):
  shutil.rmtree(train_dir)
if os.path.exists(validation_dir):
  shutil.rmtree(validation_dir)

print('Creating training set...')
for input_scan in input_scans[:train_valid_split]:
  create_dataset(input_scan, input_mask_niftii, input_img_niftii, labels, axis, output_train)

print('Creating validation set...')
for input_scan in input_scans[train_valid_split:]:
  create_dataset(input_scan, input_mask_niftii, input_img_niftii, labels, axis, output_validation)
