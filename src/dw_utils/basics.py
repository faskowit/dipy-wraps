__author__ = 'jfaskowitz'

import os
import sys
import nibabel as nib
import numpy as np
import h5py
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def flprint(instr):
    print(str(instr))
    sys.stdout.flush()


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def checkisfile(fname):
    if not os.path.isfile(fname):
        print('files does not exist: {}\nexiting'.format(fname))
        exit(1)


def chunker(seq, size, fudgefactor=10):
    # https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    chunks_list = list((seq[pos:pos + size] for pos in range(0, len(seq), size)))

    # handle some fudge
    if len(chunks_list[-1]) <= fudgefactor:
        flprint("fudging the chunk size")
        chunks_list[-2] = chunks_list[-2] + chunks_list[-1]
        del chunks_list[-1]

    return chunks_list


def loaddwibasics(dwipath, maskpath, bvalpath, bvecpath):

    # convert the strings into images we can do stuff with in dipy yo!
    if dwipath:
        checkisfile(dwipath)
        dwi_image = nib.load(dwipath)
    else:
        dwi_image = None
    if maskpath:
        checkisfile(maskpath)
        mask_image = nib.load(maskpath)
    else:
        mask_image = None

    # ~~~~~~~~~~ Bvals and Bvecs stuff ~~~~~~~~~~~~~~~

    # read the bvals and bvecs into data architecture that dipy understands
    bval_data, bvec_data = read_bvals_bvecs(bvalpath, bvecpath)

    if not is_normalized_bvecs(bvec_data):
        print("not normalized bvecs")
        exit(1)

    # need to create the gradient table yo
    gtab = gradient_table(bval_data, bvec_data, b0_threshold=25)

    # show user
    if dwipath:
        flprint('Nifti shape:{}\n'.format(dwi_image.shape))
    flprint('\nBVALS look like this:{}\n'.format(bval_data))
    flprint('\nBVECS look like this:{}\n'.format(bvec_data))

    return dwi_image, mask_image, gtab


def is_normalized_bvecs(bvecs):
    # https://github.com/BIG-S2/PSC/blob/master/Scilpy/scilpy/utils/bvec_bval_tools.py
    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3, bvecs_norm == 0))


def write_pam_h5py(peaks, out_base, sh_ord):

    # gather output
    fod_coeff = peaks.shm_coeff.astype(np.float32)
    # add other elements of csd_peaks to npz file
    fod_gfa = peaks.gfa
    fod_qa = peaks.qa
    fod_peak_dir = peaks.peak_dirs
    fod_peak_val = peaks.peak_values
    fod_peak_ind = peaks.peak_indices

    flprint('writing to the file the coefficients for sh order of: {0}'.format(str(sh_ord)))
    full_output = ''.join([out_base, '_csdPAM.h5'])

    with h5py.File(full_output, 'w') as hf:
        group1 = hf.create_group('PAM')
        group1.create_dataset('coeff', data=fod_coeff, compression="gzip")
        group1.create_dataset('gfa', data=fod_gfa, compression="gzip")
        group1.create_dataset('qa', data=fod_qa, compression="gzip")
        group1.create_dataset('peak_dir', data=fod_peak_dir, compression="gzip")
        group1.create_dataset('peak_val', data=fod_peak_val, compression="gzip")
        group1.create_dataset('peak_ind', data=fod_peak_ind, compression="gzip")