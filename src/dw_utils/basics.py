__author__ = 'jfaskowitz'

import sys
import os
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
    dwiImage = nib.load(dwipath)
    maskImage = nib.load(maskpath)

    # ~~~~~~~~~~ Bvals and Bvecs stuff ~~~~~~~~~~~~~~~

    # read the bvals and bvecs into data artictecture that dipy understands
    bvalData, bvecData = read_bvals_bvecs(bvalpath, bvecpath)

    if not is_normalized_bvecs(bvecData):
        print("not normalized bvecs")
        exit(1)

    # need to create the gradient table yo
    gtab = gradient_table(bvalData, bvecData, b0_threshold=25)

    # show user
    flprint('Nifti shape:{}\n'.format(dwiImage.shape))
    flprint('\nBVALS look like this:{}\n'.format(bvalData))
    flprint('\nBVECS look like this:{}\n'.format(bvecData))

    return dwiImage, maskImage, gtab

def is_normalized_bvecs(bvecs):
    # https://github.com/BIG-S2/PSC/blob/master/Scilpy/scilpy/utils/bvec_bval_tools.py

    bvecs_norm = np.linalg.norm(bvecs, axis=1)
    return np.all(np.logical_or(np.abs(bvecs_norm - 1) < 1e-3, bvecs_norm == 0))
