__author__ = 'jfaskowitz'

import os
import numpy as np
import nibabel as nib
import h5py
from src.dw_utils.basics import flprint


def savePAM(map_peaks, imgAffine, outNameStr, orderStr):

    # name
    outNameBase = ''.join([outNameStr, '_order', str(orderStr)])

    # organize the PAM
    fod_coeff = map_peaks.shm_coeff.astype(np.float32)

    # add other elements of csd_peaks to npz file
    fod_gfa = map_peaks.gfa
    fod_qa = map_peaks.qa
    fod_peak_dir = map_peaks.peak_dirs
    fod_peak_val = map_peaks.peak_values
    fod_peak_ind = map_peaks.peak_indices

    flprint('writing to the file the coefficients for order order of: {0}'.format(str(orderStr)))

    # lets write this to the disk yo
    fullOutput = ''.join([outNameBase, '_mapPAM.h5'])

    with h5py.File(fullOutput, 'w') as hf:
        group1 = hf.create_group('PAM')
        group1.create_dataset('coeff', data=fod_coeff, compression="gzip")
        group1.create_dataset('gfa', data=fod_gfa, compression="gzip")
        group1.create_dataset('qa', data=fod_qa, compression="gzip")
        group1.create_dataset('peak_dir', data=fod_peak_dir, compression="gzip")
        group1.create_dataset('peak_val', data=fod_peak_val, compression="gzip")
        group1.create_dataset('peak_ind', data=fod_peak_ind, compression="gzip")

    # =======================================================================

    # lets also write out a gfa image yo, just for fun
    gfaImg = nib.Nifti1Image(map_peaks.gfa.astype(np.float32), imgAffine)

    # make the output name yo
    gfaOutputName = ''.join([outNameBase, '_gfa.nii.gz'])

    # same this FA
    nib.save(gfaImg, gfaOutputName)
