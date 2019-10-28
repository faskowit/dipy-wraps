__author__ = 'jfaskowitz'

import os
import sys
import numpy as np
import nibabel as nib
import h5py
from src.modeling.args_fitFORECAST import CmdLineFitFORECAST
from src.modeling.modeling_utils import savePAM
from src.dw_utils.basics import flprint, loaddwibasics

from dipy.reconst.forecast import ForecastModel
from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst import mapmri
from dipy.direction import peaks_from_model
from dipy.core.gradients import gradient_table

# ignore some numpy errors
np.seterr(all='ignore')

def main():
    # variables that we will need

    print(''.join(sys.path))

    cmdLine = CmdLineFitFORECAST("fit the FORECAST yo")
    cmdLine.get_args()
    cmdLine.check_args()

    dwiImg, maskImg, gtab = loaddwibasics(cmdLine.dwi_,
                                          cmdLine.mask_,
                                          cmdLine.bval_,
                                          cmdLine.bvec_)

    # get the data from all the images yo
    dwiData = dwiImg.get_data()
    maskData = maskImg.get_data()

    # mask the dwiData
    dwiData = applymask(dwiData, maskData)

    sh_order = cmdLine.shOrder_ 
    peaksSphereObj = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ MAPMRI FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    fm = ForecastModel(gtab, sh_order=sh_order, dec_alg='CSD')

    flprint("making peaks from model")

    map_peaks = peaks_from_model(model=fm,
                                 data=dwiData,
                                 sphere=peaksSphereObj,
                                 relative_peak_threshold=0.5,
                                 min_separation_angle=20,
                                 mask=maskData.astype(np.bool),
                                 return_sh=True,
                                 normalize_peaks=True,
                                 parallel=False,
                                 nbr_processes=1)

    flprint("done making peaks from model")

    savePAM(map_peaks,dwiImg.affine,cmdLine.output_,cmdLine.shOrder_)


if __name__ == '__main__':
    main()
