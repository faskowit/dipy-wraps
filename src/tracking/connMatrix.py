__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University

inspiration taken from the dipy website

"""

import sys
import ntpath
import nibabel as nib
import numpy as np
import csv
# dipy
from dipy.tracking.utils import connectivity_matrix
# local
from src.dw_utils.basics import flprint
from src.dw_utils.basics import load_streamlines_from_file


def main():
    # expose structural conn matrix tools to command line
    from dipy.io import read_bvals_bvecs
    from dipy.core.gradients import gradient_table

    arguments = len(sys.argv) - 1
    flprint("the script is called with %i arguments" % arguments)
    if len(sys.argv) < 5:
        flprint("usage: {} trk_file  mask_img  (parc_img or .txt of parc paths) "
                " base_name [ dwi_img  bvec  bval ] or [ info_img ]".format(sys.argv[0]))
        exit(0)
    trk_path = sys.argv[1]
    ref_path = sys.argv[2]
    mask_img = nib.load(ref_path)
    parc_path = sys.argv[3]
    # parc_img = nib.load(parc_path)
    base_name = sys.argv[4]
    dwi_img = None
    info_img = None
    gtab = None
    info_path = ''
    if (len(sys.argv) > 5) and (len(sys.argv) == 8):
        # then we getting dwi and gradient table too
        dwi_path = sys.argv[5]
        bvecs_path = sys.argv[6]
        bvals_path = sys.argv[7]
        bval, bvec = read_bvals_bvecs(bvals_path, bvecs_path)
        gtab = gradient_table(bval, bvec, b0_threshold=50)
        dwi_img = nib.load(dwi_path)
    if (len(sys.argv) > 5) and (len(sys.argv) == 6):
        info_path = sys.argv[5]
        info_img = nib.load(info_path)
    elif len(sys.argv) > 5:
        flprint("if requesting tensor fit, need dwi, bvecs, bvals paths")
        exit(1)

    streamlines = load_streamlines_from_file(trk_path, mask_img)
    flprint("making streamline matrix")

    parc_img = []
    parc_name_mod = []
    # check if provided parc is list of files, or just one nifti, via file ending
    parc_split = parc_path.split(".")
    if ''.join(parc_split[-2:]) == "niigz":
        flprint("found parc image")
        parc_img.append(nib.load(parc_path))
        parc_name_mod.append('')
    elif parc_split[-1] == "txt":
        flprint("found multiple parcs in txt file")
        parc_list_read = open(parc_path, "r")
        for line in parc_list_read:
            parc_img.append(nib.load(line.split()[0])) # to avoid new line char in text file
            parc_name_mod.append(ntpath.basename(ntpath.splitext(ntpath.splitext(line)[0])[0]))
    else:
        flprint("cannot determine if parc input is valid. exiting")
        exit(1)

    # now iterate over parc_img
    for parc, parc_n in zip(parc_img, parc_name_mod):
        parc_base_name = ''.join([base_name, '_', parc_n])
        struct_mat, struct_groups = \
            streams_to_matrix(streamlines, parc, mask_img, parc_base_name)

        if (dwi_img is not None) or (info_img is not None):
            if dwi_img is not None:
                # first fit the tensor
                from src.dw_utils.basics import make_fa_map
                flprint("fitting tensor")
                along_tract_data, _ = make_fa_map(dwi_img.get_fdata(), mask_img.get_fdata(), gtab)
                flprint("measuring fa along tracts")
                name_mod = 'fa'
            else:
                name_mod = 'info'
                along_tract_data = info_img.get_fdata()
                flprint("measuring provided image ({}) map along tracts".format(info_path))
            info_base_name = ''.join([parc_base_name, name_mod])
            info_to_matrix(struct_mat, struct_groups, mask_img.affine, along_tract_data, info_base_name)


def streams_to_matrix(streamlines, parc_img, mask_img, out_base_name):
    # first recast the parc_img data to make sure it is int
    parc_data = parc_img.get_fdata().astype(np.int16)
    stream_counts, stream_grouping = connectivity_matrix(streamlines, mask_img.affine,
                                                         parc_data,
                                                         symmetric=True,
                                                         return_mapping=True,
                                                         mapping_as_streamlines=True)
    # get rid of the first row because these are connections to '0'
    stream_counts[:1, :] = 0
    stream_counts[:, :1] = 0

    # use the default np.eye affine here... to not apply affine twice
    stream_lengths = sl_len_matrix(stream_counts, stream_grouping)
    vol_norm = get_parc_volume(parc_img, True)
    vol_norm_counts = np.divide(stream_counts, vol_norm,
                                out=np.zeros_like(vol_norm), where=vol_norm != 0)

    # save the files
    ex_csv(''.join([out_base_name, 'slcounts.csv']), stream_counts)
    ex_csv(''.join([out_base_name, 'sllens.csv']), stream_lengths)
    ex_csv(''.join([out_base_name, 'slnormcounts.csv']), vol_norm_counts)

    return stream_counts, stream_grouping


def info_to_matrix(mat, group, aff, info_data, out_base_name):

    mean_mat, median_mat = info_along_streams(mat, group, info_data, aff)

    # write it
    ex_csv(''.join([out_base_name, 'mean.csv']), mean_mat)
    ex_csv(''.join([out_base_name, 'mean.csv']), median_mat)


def info_along_streams(mat, group, info_volume, aff=np.eye(4), stdstreamlen=50):

    from dipy.tracking.streamlinespeed import set_number_of_points
    from dipy.tracking.streamline import values_from_volume

    mean_matrix = np.zeros(mat.shape, dtype=np.float)
    median_matrix = np.zeros(mat.shape, dtype=np.float)

    # will trim the lower %5 and top %5 of streamline
    trim_length = np.int(np.floor(stdstreamlen / 20))

    # by row
    for x in range(mat.shape[0]):
        # by column
        for y in range(mat.shape[1]):
            # check if entry in dict, if so, do more stuff
            if (x, y) in group:
                # first lets re-sample all streamlines to const length
                stand_stream_group = set_number_of_points(group[x, y], stdstreamlen)
                stream_vals = values_from_volume(info_volume, stand_stream_group, aff)
                vals = np.zeros(len(stream_vals))
                for ind, sv in enumerate(stream_vals):
                    vals[ind] = np.mean(sv[trim_length:-trim_length])
                mean_matrix[x, y] = np.around(np.mean(vals), 6)
                median_matrix[x, y] = np.around(np.median(vals), 6)
    return mean_matrix, median_matrix


def sl_len_matrix(mat, group, aff=np.eye(4)):
    length_mat = np.zeros(mat.shape, dtype=np.float)
    # by row
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            # check if entry in dict, if so, do more stuff
            if (x, y) in group:
                from dipy.tracking.utils import length
                # now we record these values
                stream_group_len = length(group[x, y])
                length_mat[x, y] = np.around(np.nanmean(list(stream_group_len)), 2)
    # get rid of the connections to 0
    length_mat[:1, :] = 0
    length_mat[:, :1] = 0
    return length_mat


def get_parc_volume(parc_img, geonorm_matrix=False):
    # get multiplier, assuming isotropic voxel here
    vox_2_mm = np.prod(parc_img.header.get_zooms())
    # get num vox
    vox_counts = np.bincount(parc_img.get_fdata().astype(np.int16).flatten())
    mm_vec = np.multiply(vox_counts, vox_2_mm)
    if not geonorm_matrix:
        return mm_vec
    else:
        geonorm_mat = np.zeros((len(mm_vec), len(mm_vec)))
        for idx in range(len(mm_vec)):
            for jdx in range(len(mm_vec)):
                if idx <= jdx:
                    continue
                geonorm_mat[idx, jdx] = np.sqrt(mm_vec[idx] * mm_vec[jdx])
        return geonorm_mat + geonorm_mat.T


def ex_csv(filename, data):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    main()
