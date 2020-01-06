
import nibabel as nib
import numpy as np
import ntpath
import csv
# dipy
from dipy.tracking.utils import connectivity_matrix
# local
from src.dw_utils.basics import flprint


def main():
    # expose making structural connectivity matrix
    flprint("todo")


def streams_to_matrix(streamlines, parc_data, mask_img, out_base_name):

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

    # save the files
    ex_csv(''.join([out_base_name, 'slcounts.csv']), stream_counts)
    ex_csv(''.join([out_base_name, 'sllens.csv']), stream_lengths)

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
    return length_mat


def ex_csv(filename, data):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == "__main__":
    main()
