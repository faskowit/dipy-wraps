__author__ = 'jfaskowitz'

import os
import sys
import numpy as np
import nibabel as nib
import h5py
from src.modeling.args_fitMAPMRI import CmdLineFitMAPMRI
from src.modeling.modeling_utils import savePAM
from src.dw_utils.basics import flprint, loaddwibasics

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

    cmdLine = CmdLineFitMAPMRI("fit the MAPMRI yo")
    cmdLine.get_args()
    cmdLine.check_args()

    dwiImg, maskImg, gtab1 = loaddwibasics(cmdLine.dwi_,
                                          cmdLine.mask_,
                                          cmdLine.bval_,
                                          cmdLine.bvec_)

    # load a more detailed gtab
    gtab = gradient_table(bvals=gtab1.bvals, bvecs=gtab1.bvecs,
                          big_delta=cmdLine.bigDelta_,
                          small_delta=cmdLine.smallDelta_)

    # get the data from all the images yo
    dwiData = dwiImg.get_data()
    maskData = maskImg.get_data()

    # mask the dwiData
    dwiData = applymask(dwiData, maskData)

    radial_order = cmdLine.radialOrder_ 
    peaksSphereObj = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ MAPMRI FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    map_model_both_iso = mapmri.MapmriModel(gtab, radial_order=radial_order,
                                            laplacian_regularization=True,
                                            laplacian_weighting=.2,
                                            positivity_constraint=True,
                                            anisotropic_scaling=False,
                                            dti_scale_estimation=False)

    flprint("making peaks from model")

    map_peaks = peaks_from_model(model=map_model_both_iso,
                                 data=dwiData,
                                 sphere=peaksSphereObj,
                                 relative_peak_threshold=0.5,
                                 min_separation_angle=25,
                                 mask=maskData.astype(np.bool),
                                 return_sh=True,
                                 normalize_peaks=True,
                                 parallel=False,
                                 nbr_processes=1)

    flprint("done making peaks from model")

    savePAM(map_peaks,dwiImg.affine,cmdLine.output_,cmdLine.radialOrder_)


if __name__ == '__main__':
    main()
