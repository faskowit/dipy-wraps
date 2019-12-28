__author__ = 'jfaskowitz'

import os
import sys
import nibabel as nib
import numpy as np
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
