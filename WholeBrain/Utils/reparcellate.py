# ========================================================================
# Reparcellation function
#     Used to move a vector of data from one parcellation to other
#     Needs Siibra and nilearn for the conversion...
#
#  Code by Ignacio Martín, refactored by Gustavo Patow
# ========================================================================
import sys

import siibra
from nilearn import plotting
import numpy as np
import scipy.io as sio
from nibabel import Nifti1Image
import nibabel.processing


def reparcellate(ratio, inParcellation, outParcellation, doPlotting=False):
    atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS

    # Here we keep the original code names of desikan for in and jullich for out... ;-)
    desikan_parcellation = atlas.get_parcellation(inParcellation)
    desikan_map = desikan_parcellation.get_map("mni152")
    desikan_label_2_index = {(r.map, int(r.label)): i for i, r in enumerate(desikan_map.regions)}

    julich_parcellation = atlas.get_parcellation(outParcellation)
    julich_map = julich_parcellation.get_map(space="mni152", maptype="labelled")
    julich_label_2_index = {(r.map, int(r.label)): i for i, r in enumerate(julich_map.regions)}

    # Input (Desikan) parcellation has a single image, no need to iterate
    img_d = desikan_map.fetch()

    data_img = np.squeeze(img_d.get_fdata())
    data_ratio_d = np.zeros(data_img.shape)

    # Fill the voxels with gene expression data from vector ratio
    for i in range(data_img.shape[0]):
        for j in range(data_img.shape[1]):
            for k in range(data_img.shape[2]):
                index = desikan_label_2_index.get((0, int(data_img[i, j, k])), -1)
                if index >= 0:
                    data_ratio_d[i, j, k] = ratio[index]

    ratio_img_d = Nifti1Image(data_ratio_d, img_d.affine)
    if doPlotting:
        plotting.plot_stat_map(ratio_img_d, output_file=inParcellation+'_ratio.png')

    # Initialize new ratio vectors for the Julich parcellation
    ratio_j = np.zeros(len(julich_label_2_index))
    ratio_j_counter = np.zeros(len(julich_label_2_index), dtype=np.uint32)

    for mapindex, img_j in enumerate(julich_map.fetch_iter()):
        # Resample Desikan ratio data into Julich parcellation
        ratio_img_j = nibabel.processing.resample_from_to(ratio_img_d, (img_j.shape, img_j.affine))
        if doPlotting:
            plotting.plot_stat_map(ratio_img_d, output_file=outParcellation + f'_ratio_{mapindex}.png')

        data_orig_j = img_j.get_fdata()
        data_ratio_j = ratio_img_j.get_fdata()

        # Traverse voxels to update ratio data
        for i in range(data_ratio_j.shape[0]):
            for j in range(data_ratio_j.shape[1]):
                for k in range(data_ratio_j.shape[2]):
                    index = julich_label_2_index.get((mapindex, int(data_orig_j[i, j, k])), -1)
                    if index >= 0:
                        ratio_j[index] += data_ratio_j[i, j, k]
                        ratio_j_counter[index] += 1

    # Compute the final average
    ratio_j /= ratio_j_counter
    return ratio_j


# Convenience function to "translate" from Desikan to JuBrain, just a wrapper around the funciton above, plus
# some pre-processing that was originally done in Deco et al.'s 2021 paper...
def reparcellateDesikanToJuBrain(matrix_input_file, matrix_output_file):
    # TODO: right now this is not generic, it only works for Aquino's matrix ...
    print(f"Loading {matrix_input_file}")
    DKGenes = sio.loadmat(matrix_input_file)
    expMeasures = DKGenes['expMeasures']

    N = 68

    coefe = np.sum(expMeasures[:, 17:25], 1)  # / np.sum(expMeasures[:,1:6],1)  # ampa+nmda/gaba
    ratioE = np.zeros(N)
    ratioI = np.zeros(N)
    ratioE[0:34] = coefe / (np.max(coefe))
    ratioE[34:68] = ratioE[0:34]

    coefrange = np.union1d(np.arange(1, 9), np.arange(11, 14))
    coefi = np.sum(expMeasures[:, coefrange], 1)  # 18:21 ampa+ 22:25 nmda/gaba
    ratioI[0:34] = coefi / (np.max(coefi))
    ratioI[34:68] = ratioI[0:34]
    ratio = ratioE / ratioI
    ratio = ratio / (np.max(ratio) - np.min(ratio))
    ratio = ratio - np.max(ratio) + 1

    ratio_j = reparcellate(ratio,
                           inParcellation='desikan',
                           outParcellation='julich',
                           doPlotting=True)
    print(f"saving: {matrix_output_file}")
    sio.savemat(matrix_output_file, {'expMeasures': ratio_j})


if __name__ == '__main__':
    # matrix_file = sys.argv[1] # Ex.: https://github.com/KevinAquino/HNM/blob/main/InputData/DKcortex_selectedGenes.mat
    matrix_input_file = '../../Data_Raw/DecoEtAl2020/DKcortex_selectedGenes.mat'
    matrix_output_file = '../../Data_Produced/DecoEtAl2020/JUBrainCortex_selectedGenes.mat'
    reparcellateDesikanToJuBrain(matrix_input_file, matrix_output_file)
