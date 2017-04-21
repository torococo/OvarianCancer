import numpy as np
import pandas as pd
#import cv2

def slice_images(img, window_size, dst_dir):

    n_row, n_col = img.shape[0:2]
    q_id = 0
    q_id_list = []
    ppc_list = []
    q_id_list_append = q_id_list.append
    ppc_list_append = ppc_list.append


    for min_row in np.arange(0, n_row, window_size):
        max_row = min_row + window_size

        if max_row >= n_row:
            max_row = n_row - 1
        for min_col in np.arange(0, n_col, window_size):

            max_col = min_col + window_size
            if max_col >= n_col:
                max_col = n_col - 1


            region_slice = img[min_row:max_row, min_col:max_col]
            n_ppc = len(np.where(region_slice != 0)[0])
            ppc_list_append(n_ppc)
            q_id_list_append(q_id)

            q_id += 1

    return ppc_list, q_id_list





# img_txt_file = '../Moffitt/Moffitt Ova TMA Pilot 3 Jan 26_Pano 01_D10.txt'
img_txt_file = '../Moffitt/Ovarian cancer TMA T10 175 ov ca.txt'
img_txt_df = pd.read_table(img_txt_file)
marker_list = [f for f in list(img_txt_df) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z']]
c_pos = img_txt_df['X']
r_pos = img_txt_df['Y']
max_r = r_pos.max()
max_c = c_pos.max()


all_marker_df = None
for m in marker_list:

    marker_name = m.split('(')[0]
    marker_name = '_'.join(marker_name.split('-')[1:])
    print(m, marker_name)
    marker_intensities = img_txt_df[m]
    img = marker_intensities.values.reshape(max_r + 1, max_c + 1)
    ppc_list, q_id_list = slice_images(img, 50, '')

    marker_df = pd.DataFrame({'Quadrat':q_id_list,
                              marker_name: ppc_list
    })

    if all_marker_df is None:
        all_marker_df = marker_df
    else:
        all_marker_df = pd.merge(all_marker_df, marker_df, on='Quadrat')


f_out = '../stats/quadrat_counts.csv'
all_marker_df.to_csv(f_out, index=False)