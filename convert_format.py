__author__ = 'gatenbcd'
import sys
sys.path.append("/share/apps/opencv/3.0.0/lib/python3")
from skimage import io, filters, exposure
import numpy as np
from PIL import Image
# import tifffile
import matplotlib.pyplot as plt
import os
#import cv2
import pandas as pd
def rescale_grey_intensity(image):
    image_min = image.min()
    image_max = image.max()
    rescaled = ((image - image_min) / (image_max - image_min) * 255)
    return rescaled

#
# img_file = '../Moffitt/Moffitt Ova TMA Pilot 3 Jan 26_Pano 01_I10.tiff'
# #
# with tifffile.TiffFile(img_file) as tif:
#     images = tif.asarray()
#     for i, page in enumerate(tif):
#         pg_img = page.asarray()
#         pg_img = rescale_grey_intensity(pg_img )
#         cv2.imwrite(''.join(['./', str(i), '.png']), pg_img)

# b = cv2.imread('./28.png', cv2.IMREAD_GRAYSCALE)
# g = cv2.imread('./15.png', cv2.IMREAD_GRAYSCALE)
# r = cv2.imread('./25.png', cv2.IMREAD_GRAYSCALE)
#
#
# fake = np.dstack((b, g, r))
# cv2.imwrite('./false_fish2.png', fake)

# img_list = [f for f in os.listdir('./') if f.endswith('.png')]
# for img_file in img_list:
#     print(img_file)
#     img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
#     # img = cv2.medianBlur(img, 3)
#     # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
#     ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     print(thresh.shape, thresh.dtype)
#
#     img_prefix = img_file.split('.')[0]
#
#     file_out = ''.join(['./thresh/', img_prefix, '.png'])
#     cv2.imwrite(filename=file_out, img=thresh )
# #
# img_txt_file = '../Moffitt/Moffitt Ova TMA Pilot 3 Jan 26_Pano 01_D10.txt'
img_txt_file = '../Moffitt/Ovarian cancer TMA T10 175 ov ca.txt'
img_txt_df = pd.read_table(img_txt_file)
marker_list = [f for f in list(img_txt_df) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z']]
c_pos = img_txt_df['X']
r_pos = img_txt_df['Y']
max_r = r_pos.max()
max_c = c_pos.max()
out_dir = './full_range/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

thresh_dir = './thresh2/'
if not os.path.exists(thresh_dir):
    os.makedirs(thresh_dir)
# img_txt_file = '../Moffitt/Ovarian cancer TMA T10 175 ov ca.txt'
# marker_list = ['169Tm-CollagenI(Tm169Di)', '176Yb-HistoneH3(Yb176Di)', '153Eu-CD44s(Eu153Di)', '158Gd-E-cadherin(Gd158Di)', '143Nd-Vimentin(Nd143Di)']
# marker_list = ['169Tm-CollagenI(Tm169Di)', '143Nd-Vimentin(Nd143Di)', '158Gd-E-cadherin(Gd158Di)']
img_stack = []
for m in marker_list:
    print(m)
    marker_intensities = img_txt_df[m]
    img = np.reshape(marker_intensities , (max_r+1, max_c+1))
    img = exposure.rescale_intensity(img)
    img = exposure.equalize_adapthist(img, 5)
    img = exposure.rescale_intensity(img)
    img_f_out = ''.join([m, '.png'])
    io.imsave(img_f_out, img)
    # img_flat = img[img != 0]
    #
    # t = filters.threshold_otsu(img_flat)
    #
    # mask = np.zeros_like(img)
    # mask[img>=t] = 1
    # f_out = ''.join([thresh_dir, m, '_otsu_mask.png'])
    # io.imsave(f_out, mask)
    #
    # img[img <t] = 0
    #

    # img = exposure.equalize_hist(img, mask=mask)
    # # img = exposure.adjust_gamma(img, 0.5)
    # img[mask == 0] = 0
    # f_out = ''.join([out_dir, m, '.png'])
    # io.imsave(f_out, img)
    #
    #
    # # img = exposure.equalize_adapthist(img, 5)
    # # img = exposure.equalize_hist(img)
    # # img = exposure.adjust_gamma(img, 0.75)
    # img_stack.append(img)
    # img_stack.append(mask)


# fish = np.dstack(img_stack[:-2])
# # yellow_channel = np.dstack([img_stack[-1], np.zeros_like(img), img_stack[-1]])
# yellow_channel = np.dstack([img_stack[-1], img_stack[-1], np.zeros_like(img)])
# mag_channel = np.dstack([img_stack[-2], np.zeros_like(img), img_stack[-2]])
# fish += yellow_channel
# fish += mag_channel
# fish = exposure.rescale_intensity(fish)
# # fish[np.where(np.all(fish==0, axis=2)==True)] = 1
# f_out = ''.join(['fish2.png'])
# io.imsave(f_out, fish)
#

# img = np.reshape(intensity_array, (max_r+1, max_c+1))
# plt.matshow(img)
# plt.colorbar()
# plt.show()
# print(img.dtype)
# img = exposure.rescale_intensity(img)
# print(img.dtype)
# t = filters.threshold_otsu(img)

# mask = np.zeros_like(img)
# mask[img>=t] = 1
# # print(img.min(), img.max())
# io.imsave('./sk_full_dr.png', img)
# io.imsave('./full_dr_mask.png', mask)
#
# plt.matshow(img, cmap='afmhot')
# plt.colorbar()
# plt.savefig('./full_range.pdf')
# plt.close()
# plt.show()
#
#
#
#