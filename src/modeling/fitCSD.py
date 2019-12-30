__author__ = 'jfaskowitz'

import os
import numpy as np
import nibabel as nib
import h5py
from src.modeling.args_fitCSD import CmdLineFitCSD
from src.dw_utils.basics import flprint, loaddwibasics

from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model

# ignore some numpy errors
np.seterr(all='ignore')

def main():
    # variables that we will need

    cmdLine = CmdLineFitCSD("fit the CSD yo")
    cmdLine.get_args()
    cmdLine.check_args()

    if cmdLine.recurResp_:
        flprint("Using recursive response for CSD fit")

    dwiImg, maskImg, gtab = loaddwibasics(cmdLine.dwi_,
                                          cmdLine.mask_,
                                          cmdLine.bval_,
                                          cmdLine.bvec_)

    # get the data from all the images yo
    dwiData = dwiImg.get_data()
    maskData = maskImg.get_data()

    # mask the dwiData
    dwiData = applymask(dwiData, maskData)

    regSphereObj = get_sphere('symmetric362')
    peaksSphereObj = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RESPONSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    outNameBase = ''.join([cmdLine.output_, '_recurRes_sh', str(cmdLine.shOrder_)])

    if cmdLine.recurResp_ == 1:

        if os.path.isfile(outNameBase + '.npz'):

            flprint("already found the recursive stuffs, will load it up")

            # load up the np file
            responseTemp = np.load(outNameBase + '.npz')
            response = responseTemp['arr_0']
            response = response.item()

        else:

            # check if wm mask provided, if not, make one
            if cmdLine.wmMask_:

                flprint("using wm mask provided")

                wmMaskImg = nib.load(cmdLine.wmMask_)
                wmMaskData = wmMaskImg.get_data()

                flprint(wmMaskImg.shape)

                # check that is wm mask is same dim as dwi
                if not np.array_equal(dwiImg.shape[0:3], wmMaskImg.shape[0:3]):
                    flprint("wm mask wrong shape")
                    exit(1)

            else:

                import dipy.reconst.dti as dti
                # least weighted is standard
                tenmodel = dti.TensorModel(gtab)

                flprint("fitting tensor")
                tenfit = tenmodel.fit(dwiData, mask=maskData)
                flprint("done fitting tensor")

                from dipy.reconst.dti import fractional_anisotropy
                FA = fractional_anisotropy(tenfit.evals)
                MD = dti.mean_diffusivity(tenfit.evals)
                wmMaskData = (np.logical_or(FA >= cmdLine.faThr_,
                                            (np.logical_and(FA >= 0.15, MD >= 0.0011))))

            flprint("using dipy function recursive_response to generate response")
            from dipy.reconst.csdeconv import recursive_response

            flprint("doing recusive stuffs...")
            response = recursive_response(gtab,
                                          dwiData,
                                          mask=wmMaskData.astype(np.bool), sh_order=cmdLine.shOrder_,
                                          sphere=peaksSphereObj, parallel=False,
                                          nbr_processes=1)
            flprint("finish doing recusive stuffs")
            flprint(str(response))

            # lets save the response as an np object to the disk
            np.savez_compressed(outNameBase + '.npz', response, 'arr_0')

    else:

        flprint("using dipy function auto_response to generate response")

        from dipy.reconst.csdeconv import auto_response
        response, ratio = auto_response(gtab,
                                        dwiData,
                                        roi_radius=10, fa_thr=cmdLine.faThr_)

        # lets save the response as an np object to the disk
        np.savez_compressed(outNameBase + '.npz', response, 'arr_0')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if len(cmdLine.actClasses_) > 0:

        # load images
        csf_image = nib.load(cmdLine.actClasses_[0])
        csf_data = csf_image.get_data()
        gm_image = nib.load(cmdLine.actClasses_[1])
        gm_data = gm_image.get_data()

        # fit dti model
        import dipy.reconst.dti as dti
        tensor_model = dti.TensorModel(gtab)
        flprint("fitting tensor to get MD")
        tensor_fit = tensor_model.fit(dwiData)
        flprint("finished fitting tensor")
        md_data = tensor_fit.md

        # get needed inds
        inds_csf = np.where(csf_data > 0)
        inds_gm = np.where(gm_data > 0)
        selected_csf = np.zeros(md_data.shape, dtype='bool')
        selected_gm = np.zeros(md_data.shape, dtype='bool')
        selected_csf[inds_csf] = True
        selected_gm[inds_gm] = True
        csf_md = np.mean(md_data[selected_csf])
        gm_md = np.mean(md_data[selected_gm])

        from dipy.sims.voxel import multi_shell_fiber_response
        response_mcsd = multi_shell_fiber_response(sh_order=cmdLine.shOrder_,
                                                   bvals=gtab.bvals,
                                                   evals=response[0],
                                                   csf_md=np.float(csf_md),
                                                   gm_md=np.float(gm_md))

        from dipy.reconst.mcsd import MultiShellDeconvModel
        csd_model = MultiShellDeconvModel(gtab, response_mcsd, reg_sphere=regSphereObj)

    else:
        csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                    sh_order=cmdLine.shOrder_,
                                                    reg_sphere=regSphereObj)

    flprint("making peaks from model")
    csd_peaks = peaks_from_model(model=csd_model,
                                 data=dwiData,
                                 sphere=peaksSphereObj,
                                 relative_peak_threshold=0.5,
                                 min_separation_angle=25,
                                 mask=maskData.astype(np.bool),
                                 return_sh=True,
                                 normalize_peaks=True,
                                 parallel=False)
    flprint("done making peaks from model")

    # gather output
    fod_coeff = csd_peaks.shm_coeff.astype(np.float32)
    # add other elements of csd_peaks to npz file
    fod_gfa = csd_peaks.gfa
    fod_qa = csd_peaks.qa
    fod_peak_dir = csd_peaks.peak_dirs
    fod_peak_val = csd_peaks.peak_values
    fod_peak_ind = csd_peaks.peak_indices

    flprint('writing to the file the coefficients for sh order of: {0}'.format(str(cmdLine.shOrder_)))
    # lets write this to the disk yo
    fullOutput = ''.join([outNameBase, '_csdPAM.h5'])

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
    gfaImg = nib.Nifti1Image(csd_peaks.gfa.astype(np.float32), dwiImg.get_affine())
    # make the output name yo
    gfaOutputName = ''.join([outNameBase, '_gfa.nii.gz'])
    # same this FA
    nib.save(gfaImg, gfaOutputName)


if __name__ == '__main__':
    main()
