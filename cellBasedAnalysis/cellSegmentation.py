# ========================================================================
# Script to perform cell segmentation on the fluidigm images using Chandler's
# segmentation functions.
# ========================================================================
# When running via ipython
import os
# os.chdir("dataProcessing")
# ========================== Load Libraries ==============================
import sys
sys.path.insert(0, '..')
import Utils_Segmentation as cSeg
import cnnUtils
import numpy as np
from skimage import io,exposure,morphology,color
import shutil
# ============================== Options ==================================
use_nuclear_centroids_for_voronoi = True
nuclear_detection_method = 'watershed'  # 'watershed'  # or  watershed or spectral
watershed_method = 'edges'  # 'edges'  # edges or gradient
spectral_clustering_label_assignment = 'discretize'  # discretize or kmeans
fileDir = "../data/patientsWithOutcomes/npArraysRaw"
dst_dir = './segmented_images_max/'
# ============================= Main Code ================================
coreVec = cnnUtils.GetAllCoreIds(fileDir)

# Set up environment
if not os.path.exists(dst_dir + "segmentations"):
    os.mkdir(dst_dir)
    os.mkdir(dst_dir + "segmentations")
    os.mkdir(dst_dir + "segmentations_coloured")

for coreId in coreVec:
    img_txt_file = '../data/patientsWithOutcomes/txtFiles_reordered/core_'+str(coreId)+".txt"
    file_prefix_for_new_txt_file = "core_"+str(coreId)+'_labeled'

    print("================================================")
    print("Segmenting Core: "+str(coreId))
    print("================================================")
    # Load the histone stain and clean it
    print('Opening and cleaning image')
    image = np.load(fileDir+"/core_"+str(coreId) + ".npy", "r")
    histoneSlice = image[:,:,34] # Extract the histone layer
    histoneSlice = cSeg.clean_image(np.copy(histoneSlice),(10, 10),100,method="max")
    histoneSlice  = exposure.rescale_intensity(histoneSlice ) # Normalise

    # Create initial mask by local thresholding thresholding
    print('Creating initial mask')
    img_mask = cSeg.create_mask(histoneSlice, min_px_size=10)

    # Find the cell centroids
    print('Finding cell centroids')
    erosion_selem = morphology.square(3)
    dilation_selem = morphology.square(3)
    initial_centroid_img, initial_centroid_list = cSeg.get_initial_cell_centroids(img_mask,erosion_selem=erosion_selem,dilation_selem=dilation_selem)

    # Do watershed or spectral clutering to get nuclear boundaries
    print('Segmenting nuclei using', nuclear_detection_method)
    if nuclear_detection_method == 'watershed':
        nuclear_mask, nuclear_centroid_img, nuclear_centroid_list = cSeg.watershed_seg(histoneSlice,img_mask,initial_centroid_img,method=watershed_method)

    elif nuclear_detection_method == 'spectral_clustering':
        nuclear_mask, nuclear_centroid_img, nuclear_centroid_list = cSeg.spectral_clustering_seg(histoneSlice, img_mask,
                                                                                            initial_centroid_img,
                                                                                            beta=4, eps=1e-6,
                                                                                            assign_labels='discretize')

    # Optionally, use a voronoi tesselation to get a measure of the maximum size for each cell
    # and use it to increase the cell sizes in the previous, conservative masks. This is done
    # by increasing the cells in the previous masks through diffusion until they touch the voronoi
    # boundaries.
    if use_nuclear_centroids_for_voronoi:
        print('Determining cell bounderies using Voronoi Tesselation')
        vor_mask, n_segs = cSeg.voronoi_seg(nuclear_centroid_img, nuclear_centroid_list)

        # Use both masks to generate a cell mask
        print('Creating final mask')
        cell_mask = cSeg.create_cell_mask(vor_mask, nuclear_mask)
    else:
        cell_mask = nuclear_mask

    # Label Image
    print('Labeling image')
    labeled_image = cSeg.label_image(histoneSlice, nuclear_mask, cell_mask)

    # Save image of mask
    maskFName = dst_dir+"segmentations/core_"+str(coreId)+".png"
    view_mask = color.label2rgb(cell_mask)
    io.imsave(maskFName,labeled_image)
    maskFName_colour = dst_dir+"segmentations_coloured/core_"+str(coreId)+".png"
    view_mask = color.label2rgb(cell_mask)
    io.imsave(maskFName_colour,view_mask)

    # Apply mask to all stains in image
    print('Adding cell ID labels to txt file and drawing cell borders on each stain for visualization')
    cSeg.label_all_stains_in_txt(cell_mask, img_txt_file, dst_dir, file_prefix_for_new_txt_file)

    # Move pngs to new directory
    source = os.listdir(dst_dir)
    destination = dst_dir+"core_"+str(coreId)+"/"
    if not os.path.exists(destination):
        os.mkdir(destination)
    for files in source:
        if files.endswith(".png"):
            if not os.path.exists(destination+files):
                shutil.move(dst_dir+files, destination)