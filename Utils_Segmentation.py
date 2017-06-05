__author__ = 'gatenbcd'
from skimage import io, filters, exposure, morphology, color, measure, segmentation, draw, restoration
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import spatial
import scipy.ndimage as ndi
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
import numpy as np
import pickle


def create_mask(img, min_px_size, fill=True, adaptive=True):
    '''
    Threshold image. Will be used to find initial estimate of cell centroids
    :param img: image to threshold
    :param min_px_size: remove all contours that have an area below this value
    :param fill: bool. Fill holes in mask?
    :param adaptive: bool. Use adaptive thresholding? If False, Otsu thresholding is used 
    :return: thresholded image
    '''
    eq_img = exposure.equalize_adapthist(img)
    if adaptive:
        local_thresh = filters.threshold_local(img, 21)
        mask = np.zeros_like(img)
        mask[img > local_thresh] = 255

    else:
        img_flat = eq_img[eq_img != 0]
        t = filters.threshold_otsu(img_flat)
        mask = np.zeros_like(eq_img)
        mask[eq_img >= t] = 255

    if fill:
        mask = ndi.binary_fill_holes(mask).astype(np.int)
        mask[mask > 0] = 255

    final_mask = morphology.remove_small_objects(mask.astype(np.bool), min_size=min_px_size, connectivity=2)
    final_mask = final_mask.astype(np.int)
    return final_mask


def get_centroids_from_region_props(labeled_img):
    '''
    Find centroids of labeled contours
    :param labeled_img:  
    :return: image with labeled centroids, list of centroids [(all_rows), (all_cols)]
    '''
    contour_props = measure.regionprops(labeled_img)

    centroid_list = []
    label_list = []
    centroid_list_append = centroid_list.append
    label_list_append = label_list.append

    for p in contour_props:
        centroid_list_append(p.centroid)
        label_list_append(p.label)

    centroid_img = np.zeros_like(labeled_img)
    zipped_centroids = list(zip(*centroid_list))
    centroids_r, centroids_c = zipped_centroids
    centroids_r = np.round(centroids_r).astype(np.int)
    centroids_c = np.round(centroids_c).astype(np.int)
    centroid_img[centroids_r, centroids_c] = label_list

    return centroid_img, zipped_centroids


def get_initial_cell_centroids(thresh, extrema_h=0.05, erosion_selem=None, dilation_selem=None):
    '''
    
    :param thresh: thresholded image a.k.a. the mask 
    :param extrema_h: 
    :param erosion_selem: structuring element to be used in erosion. Can have different shapes and dimensions. See http://scikit-image.org/docs/dev/api/skimage.morphology.html 
    :param dilation_selem: structuring element to be used in dilation
    :return:  image with labeled centroids, list of centroids [(all_rows), (all_cols)]
    '''
    if erosion_selem is not None:
        mask = morphology.binary_erosion(thresh, erosion_selem)
    else:
        mask = thresh.copy()

    ### Conduct distance transform ###
    D = ndi.distance_transform_bf(mask)
    D = exposure.rescale_intensity(D)

    ### Find local maxima of distance transform ###
    h_maxima = morphology.extrema.h_maxima(D, extrema_h)
    h_maxima[h_maxima != 0] = 1

    ### Merge local maxima that are close in space using dilation. ###
    if dilation_selem is not None:
        h_maxima = morphology.binary_dilation(h_maxima, dilation_selem)

    ### Label centroids ###
    labeled_contour_img = ndi.label(h_maxima)[0]
    centroid_img, zipped_centroids = get_centroids_from_region_props(labeled_contour_img)

    return centroid_img, zipped_centroids


def check_out_of_bounds(x, max_x, min_x=0):
    '''
    Determine if value falls outside of limits. If True, set value to min or max
    :param x: 
    :param max_x: 
    :param min_x: 
    :return: 
    '''
    new_x = x
    if x < min_x:
        new_x = min_x
    elif x > max_x:
        new_x = max_x

    return new_x


def voronoi_seg(cell_centroid_img, cell_centroid_list):
    '''
    Use Voronoi tessellation to estimate cell borders
    :param cell_centroid_img: 
    :param cell_centroid_list: points to be used in Voronoi  
    :return: 
    '''

    centroid_y, centroid_x = cell_centroid_list
    points = list(zip(*(centroid_x, centroid_y)))
    vor = spatial.Voronoi(points)

    ### Values of 0 in mask reflect cell borders
    voroni_mask = np.zeros_like(cell_centroid_img) + 255
    max_r, max_c = voroni_mask.shape
    max_r -= 1
    max_c -= 1

    ### Draw Voroni edges on mask ###
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            xy_pair = vor.vertices[simplex]
            xy_pair = xy_pair.astype(np.int)

            c0, r0 = xy_pair[0]
            c1, r1 = xy_pair[1]

            r0 = check_out_of_bounds(r0, max_r)
            r1 = check_out_of_bounds(r1, max_r)
            c0 = check_out_of_bounds(c0, max_c)
            c1 = check_out_of_bounds(c1, max_c)

            rr, cc = draw.line(r0, c0, r1, c1)
            voroni_mask[rr, cc] = 0

    ### Increase distance between regions for labeling ###
    voroni_mask = morphology.binary_erosion(voroni_mask, morphology.square(2))
    v_mask_label_img, n_segs = ndi.label(voroni_mask)

    ### Return labeled regions to original size ###
    v_mask_label_img = morphology.dilation(v_mask_label_img, morphology.square(2))

    return v_mask_label_img, n_segs


def watershed_seg(img, thresh, cell_centroid_img, erosion_selem=None, method='edges'):
    '''
    Use watershed segmentation to separate cell nuclei
    :param img: greyscale image to segment
    :param thresh: thresholded image aka mask
    :param cell_centroid_img: 
    :param erosion_selem: 
    :param method: edges or gradient. 
        If edges, uses edges of mask for watershed. If gradient, the gradient of greyscale values in img are used 
    :return: labels, centroid_img, zipped_centroids. 
        labels = image where each segmented nuclei has a unique ID > 1
        centroid_img = image labeling the center of each nuclei
        zipped_centroids = list of centroids [(all_rows), (all_cols)]
    '''
    if erosion_selem is not None:
        mask = morphology.binary_erosion(thresh, erosion_selem)
    else:
        mask = thresh.copy()

    markers = ndi.label(cell_centroid_img)[0]
    if method == 'edges':
        edges = filters.sobel(mask)
        labels = segmentation.watershed(edges, markers, mask=mask, compactness=0.01)

    elif method == 'gradient':

        gradient = filters.rank.gradient(img, morphology.disk(2))
        labels = segmentation.watershed(gradient, markers, mask=mask, compactness=0.01)

    centroid_img, zipped_centroids = get_centroids_from_region_props(labels)

    return labels, centroid_img, zipped_centroids


def spectral_clustering_seg(img, mask, centroid_img, beta=1, eps=0.0, assign_labels='kmeans'):
    '''
    Use spectral clustering to segment image. See http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html
    and http://scikit-learn.org/stable/auto_examples/cluster/plot_face_segmentation.html.
    Here, each labeled region is segmented individually. The number of cell centroids found within that region is used
    to determine the number of clusters in the image.
    :param img: 
    :param mask: 
    :param centroid_img: 
    :param beta: For beta=1, the segmentation is close to a voronoi 
    :param eps: 
    :param assign_labels: kmeans or discretize
    :return: spectral_label_img, spectral_centroid_img, zipped_centroids. 
        spectral_label_img = image where each segmented nuclei has a unique ID > 1
        spectral_centroid_img = image labeling the center of each nuclei
        zipped_centroids = list of centroids [(all_rows), (all_cols)]
    '''

    labeled_mask = ndi.label(mask)[0]
    region_prop_list = measure.regionprops(labeled_mask)

    spectral_label_img = np.zeros_like(mask)
    current_label = 1

    ### Go through each region and segment touching cells ###
    for i, p in enumerate(region_prop_list):
        ### Slice region from image, mask, and centroid image ###
        min_row, min_col, max_row, max_col = p.bbox

        mask_slice = labeled_mask[min_row:max_row, min_col:max_col]
        mask_slice[mask_slice != p.label] = 0
        mask_slice = mask_slice.astype(np.bool)
        region_slice = img[min_row:max_row, min_col:max_col]
        centroid_slice = centroid_img[min_row:max_row, min_col:max_col]

        ### Mask region, so only segmenting labeled contour ###
        region_slice[mask_slice == False] = 0
        centroid_slice[mask_slice == False] = 0
        n_cells_in_slice = len(np.where(centroid_slice != 0)[0])

        if region_slice.size <= 1 or n_cells_in_slice == 0:
            ### Noise or empty part of image###
            continue

        ### Perform clustering on image ###
        graph = image.img_to_graph(region_slice, mask=mask_slice)
        graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

        labels = spectral_clustering(graph, n_clusters=n_cells_in_slice, eigen_solver='arpack',
                                     assign_labels=assign_labels)

        ### Label the segmented nuclei in the slice ###
        label_im = np.zeros(mask_slice.shape)
        label_im[mask_slice] = labels + current_label  ##labeling starts at 0

        ### Copy labeled nuclei to full size labeled image ###
        current_label += len(np.unique(labels))
        label_pos_px_row, label_pos_px_col = np.where(label_im != 0)
        spectral_label_img[min_row + label_pos_px_row, min_col + label_pos_px_col] = label_im[
            label_pos_px_row, label_pos_px_col]

    spectral_centroid_img, zipped_centroids = get_centroids_from_region_props(spectral_label_img)
    return spectral_label_img, spectral_centroid_img, zipped_centroids

def create_cell_mask(voronoi_mask, nuclear_mask, max_cell_area=400):
    '''
    Use voronoi_mask and nuclear_mask to create final mask. This is accomplished by dilating the nuclear mask until
    it reaches a max_cell_area, or fills up the voronoi tile.
    :param voronoi_mask: 
    :param nuclear_mask: 
    :param max_cell_area: 
    :return: image of labeled cells
    '''
    region_prop_list = measure.regionprops(voronoi_mask)

    nuclear_dilation_selem = morphology.disk(3)
    cell_mask = np.zeros_like(voronoi_mask)
    for i, p in enumerate(region_prop_list):
        ### Slice region masks ###
        region_label = p.label
        min_row, min_col, max_row, max_col = p.bbox
        region_slice = nuclear_mask[min_row:max_row, min_col:max_col]
        tess_mask = voronoi_mask[min_row:max_row, min_col:max_col]

        ### Mask parts of slice that do not belong to voronoi tile ###
        negative_mask_pos = np.where(tess_mask != region_label)

        ### Determine current size of nucleus ###
        n_pos_px = len(np.where(region_slice != 0)[0])
        if region_slice.size <= 1 or n_pos_px == 0:
            continue

        ### Repeat dilating nucleus and masking regions outside of Voronoi tile until
        ### max_cell_area is reached or  Voronoi tile is filled up
        while n_pos_px < max_cell_area:

            region_slice = morphology.binary_dilation(region_slice, nuclear_dilation_selem)
            region_slice[negative_mask_pos] = 0
            new_n_pos_px = len(np.where(region_slice != 0)[0])
            if new_n_pos_px == n_pos_px:
                # region is filled up, so won't reach cell max
                break
            else:
                n_pos_px = new_n_pos_px

        ### Copy labeled nuclei to full size labeled image ###
        region_slice[negative_mask_pos] = 0
        region_slice_pos_px_row, region_slice_pos_px_col = np.where(region_slice != 0)
        cell_mask[min_row + region_slice_pos_px_row, min_col + region_slice_pos_px_col] = region_label

    return cell_mask




def clean_image(img, box_tl, box_width, method="max"):
    '''
    Use a corner of the image find the amount of antibody that did not wash of slide.
    Use that value o threshold image, removing background noise 
    :param img: 
    :param box_tl: (row, col) of box top-left corner
    :param box_width: 
    :return: 
    '''
    box_tl_r, box_tl_c = box_tl

    box = img[box_tl_r:box_tl_r + box_width, box_tl_c:box_tl_c + box_width]
    if (method=="max"): thresh_i = np.max(box)
    if (method=="mean"): thresh_i = np.mean(box)
    img[img <= thresh_i] = 0
    return img

def label_image(img, nuclear_mask, cell_mask):
    '''
    label original image with outline of cell bounderies, and colored nuclei
    :param img: 
    :param nuclear_mask: 
    :param cell_mask: 
    :return: 
    '''
    img_eq = exposure.equalize_adapthist(img)
    img_eq = color.grey2rgb(img_eq)
    neg_pos_idx = np.where(nuclear_mask == 0)

    overlay = color.label2rgb(nuclear_mask, img_eq)
    overlay[neg_pos_idx] = img_eq[neg_pos_idx]
    label_edges = segmentation.find_boundaries(cell_mask, mode='inner')

    overlay[label_edges] = [1.0, -1.0, -1.0]
    return overlay


def label_all_stains_in_txt(cell_mask, img_text_file, dst_dir, file_out_prefix):
    '''
    Adds column for cell ID in txt file. Also draws cell bounderies on the other stains.
    
    :param cell_mask: image containing labeled cells 
    :param img_text_file:  original txt file containing Fluidigm data
    :param dst_dir: where to save images and new file
    :param file_out_prefix: prefix for labeled txt file.
    :return: None
    '''
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


    img_txt_df = pd.read_table(img_text_file)

    ### ADD CELL LABELS TO DF
    flattened_labels = cell_mask.flatten()
    img_txt_df['cell_id'] = flattened_labels

    img_txt_df.to_csv(''.join([dst_dir, file_out_prefix, '.txt']), sep='\t')

    marker_list = [f for f in list(img_txt_df) if f not in ['Start_push', 'End_push', 'Pushes_duration', 'X', 'Y', 'Z', 'cell_id']]
    c_pos = img_txt_df['X']
    r_pos = img_txt_df['Y']
    max_r = r_pos.max()
    max_c = c_pos.max()



    ### DRAW CELL BOUNDERIES ON EACH IMAGE, USING RECONSTRUCTED MASK FROM DF ###
    cell_mask_flat = img_txt_df['cell_id']
    cell_mask_recon = np.reshape(cell_mask_flat, (max_r + 1, max_c + 1))
    labeled_edges = segmentation.find_boundaries(cell_mask_recon, mode='inner')
    for m in marker_list:
        m_name = m.split('-')[1:]
        m_name = ''.join(m_name)
        m_name = m_name.split('(')[0]
        print('Drawing mask on', m_name)

        marker_intensities = img_txt_df[m]
        img = np.reshape(marker_intensities, (max_r + 1, max_c + 1))
        img = clean_image(img, (10, 10), 100)
        img = exposure.rescale_intensity(img)

        img_eq = exposure.equalize_adapthist(img)

        img_rgb = color.grey2rgb(img_eq)
        img_rgb[labeled_edges != 0] = [np.max(img_rgb), 0, 0]

        f_out = ''.join([dst_dir, m_name, '.png'])
        io.imsave(f_out, img_rgb)




if __name__ == '__main__':


    ### OPTIONS
    use_nuclear_centroids_for_voronoi = True
    nuclear_detection_method = 'watershed' #'watershed'  # or  watershed or spectral
    watershed_method = 'edges' # 'edges'  # edges or gradient
    spectral_clustering_label_assignment = 'discretize' # discretize or kmeans
    img_f = './HistoneH3.png'

    ### READ IMAGE, CLEAN UP, AND CREATE MASK
    print('Opening and cleaning image')

    img = io.imread(img_f, True)
    img_clean = clean_image(img, (10, 10), 100)

    print('Creating initial mask')
    img_mask = create_mask(img, min_px_size=10)

    #### FIND CELL CENTROIDS ####
    print('Finding cell centroids')
    erosion_selem = morphology.square(3)
    dilation_selem = morphology.square(3)
    initial_centroid_img, initial_centroid_list = get_initial_cell_centroids(img_mask, erosion_selem=erosion_selem,
                                                                             dilation_selem=dilation_selem)

    #### WATERSHED OR SPECTRAL CLUSTERING TO GET NUCLEAR BOUNDARIES ####
    print('Segmenting nuclei using', nuclear_detection_method)
    if nuclear_detection_method == 'watershed':
        nuclear_mask, nuclear_centroid_img, nuclear_centroid_list = watershed_seg(img, img_mask, initial_centroid_img,
                                                                                  method=watershed_method)

    elif nuclear_detection_method == 'spectral_clustering':
        nuclear_mask, nuclear_centroid_img, nuclear_centroid_list = spectral_clustering_seg(img, img_mask,
                                                                                            initial_centroid_img,
                                                                                            beta=4, eps=1e-6,
                                                                                            assign_labels='discretize')

    #### CREATE VORONOI MASK TO GET MAX CELL BOUNDARIES ####
    print('Determining cell bounderies')
    if use_nuclear_centroids_for_voronoi:
        vor_mask, n_segs = voronoi_seg(nuclear_centroid_img, nuclear_centroid_list)
    else:
        vor_mask, n_segs = voronoi_seg(initial_centroid_img, initial_centroid_list)
    # view_vmask = color.label2rgb(vor_mask)
    # io.imsave('./test/v_mask_initial_watershed_grad.png', view_vmask)

    #### USE BOTH MASKS TO CREATE A CELL MASK ####
    print('Creating final mask')
    cell_mask = create_cell_mask(vor_mask, nuclear_mask)
    # pickle.dump(cell_mask, open('./test/cell_mask.pickle', 'wb'))

    ##LABEL IMAGE ###
    print('Labeling image')
    labeled_image = label_image(img, nuclear_mask, cell_mask)
    io.imsave('./segmented.png', labeled_image)


    ### APPLY MASK TO ALL STAINS IN IMAGE ###
    print('Adding cell ID labels to txt file and drawing cell borders on each stain for visualization')
    img_txt_file = '../Moffitt/Moffitt Ova TMA Pilot 3 Jan 26_Pano 01_D10.txt'
    dst_dir = './segmented_images/'
    file_prefix_for_new_txt_file = '1_labeled'
    label_all_stains_in_txt(cell_mask, img_txt_file, dst_dir, file_prefix_for_new_txt_file)
