__author__ = 'jfaskowitz'

import os
import numpy as np
import nibabel as nib
import h5py
from src.modeling.args_fitCSD import CmdLineFitCSD
from src.dw_utils.basics import flprint, loaddwibasics, write_pam_h5py

from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model

# ignore some numpy errors
np.seterr(all='ignore')


def main():
    # variables that we will need

    command_line = CmdLineFitCSD("fit the CSD yo")
    command_line.get_args()
    command_line.check_args()

    if command_line.recurResp_:
        flprint("Using recursive response for CSD fit")

    dwi_img, mask_img, gradient_tab = loaddwibasics(command_line.dwi_,
                                                    command_line.mask_,
                                                    command_line.bval_,
                                                    command_line.bvec_)

    # get the data from all the images yo
    dwi_data = dwi_img.get_fdata()
    mask_data = mask_img.get_fdata()

    # mask the dwi_data
    dwi_data = applymask(dwi_data, mask_data)

    reg_sphere_obj = get_sphere('symmetric362')
    peaks_sphere_obj = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RESPONSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    out_name_base = ''

    if command_line.recurResp_ == 1:

        out_name_base = ''.join([command_line.output_, '_recurresp_sh', str(command_line.shOrder_)])

        if os.path.isfile(out_name_base + '.npz'):

            flprint("already found the recursive stuffs, will load it up")

            # load up the np file
            response_temp = np.load(out_name_base + '.npz')
            response = response_temp['arr_0']
            response = response.item()

        else:

            # check if wm mask provided, if not, make one
            if command_line.wmMask_:

                flprint("using wm mask provided")

                wm_mask_img = nib.load(command_line.wmMask_)
                wm_mask_data = wm_mask_img.get_fdata()

                flprint(wm_mask_img.shape)

                # check that is wm mask is same dim as dwi
                if not np.array_equal(dwi_img.shape[0:3], wm_mask_img.shape[0:3]):
                    flprint("wm mask wrong shape")
                    exit(1)

            else:

                import dipy.reconst.dti as dti
                # least weighted is standard
                tensor_model = dti.TensorModel(gradient_tab)

                flprint("fitting tensor")
                tensor_fit = tensor_model.fit(dwi_data, mask=mask_data)
                flprint("done fitting tensor")

                from dipy.reconst.dti import fractional_anisotropy
                FA = fractional_anisotropy(tensor_fit.evals)
                MD = dti.mean_diffusivity(tensor_fit.evals)
                wm_mask_data = (np.logical_or(FA >= command_line.faThr_,
                                            (np.logical_and(FA >= 0.15, MD >= 0.0011))))

            flprint("using dipy function recursive_response to generate response")
            from dipy.reconst.csdeconv import recursive_response
            response = recursive_response(gradient_tab,
                                          dwi_data,
                                          mask=wm_mask_data.astype(np.bool), sh_order=command_line.shOrder_,
                                          sphere=peaks_sphere_obj, parallel=False,
                                          nbr_processes=1)
            flprint("finish doing recusive stuffs")
            flprint(str(response))

            # lets save the response as an np object to the disk
            np.savez_compressed(out_name_base + '.npz', response, 'arr_0')
    else:
        out_name_base = ''.join([command_line.output_, '_response_sh', str(command_line.shOrder_)])

        flprint("using dipy function auto_response to generate response")
        from dipy.reconst.csdeconv import auto_response
        response, ratio = auto_response(gradient_tab,
                                        dwi_data,
                                        roi_radius=10, fa_thr=command_line.faThr_)

        # lets save the response as an np object to the disk
        np.savez_compressed(out_name_base + '.npz', response, 'arr_0')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if len(command_line.actClasses_) > 0:

        # load images
        csf_image = nib.load(command_line.actClasses_[0])
        csf_data = csf_image.get_fdata()
        gm_image = nib.load(command_line.actClasses_[1])
        gm_data = gm_image.get_fdata()

        # fit dti model
        import dipy.reconst.dti as dti
        tensor_model = dti.TensorModel(gradient_tab)
        flprint("fitting tensor to get MD")
        tensor_fit = tensor_model.fit(dwi_data)
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
        response_mcsd = multi_shell_fiber_response(sh_order=command_line.shOrder_,
                                                   bvals=gradient_tab.bvals,
                                                   evals=response[0],
                                                   csf_md=np.float(csf_md),
                                                   gm_md=np.float(gm_md))

        from dipy.reconst.mcsd import MultiShellDeconvModel
        csd_model = MultiShellDeconvModel(gradient_tab, response_mcsd, reg_sphere=reg_sphere_obj)

    else:
        csd_model = ConstrainedSphericalDeconvModel(gradient_tab, response,
                                                    sh_order=command_line.shOrder_,
                                                    reg_sphere=reg_sphere_obj)

    flprint("making peaks from model")
    csd_peaks = peaks_from_model(model=csd_model,
                                 data=dwi_data,
                                 sphere=peaks_sphere_obj,
                                 relative_peak_threshold=0.5,
                                 min_separation_angle=25,
                                 mask=mask_data.astype(np.bool),
                                 return_sh=True,
                                 normalize_peaks=True,
                                 parallel=False)
    flprint("done making peaks from model")

    # write peaks out to file, to use for later tracking
    write_pam_h5py(csd_peaks, out_name_base, command_line.shOrder_)

    # =======================================================================

    # lets also write out a gfa image yo, just for fun
    gfa_img = nib.Nifti1Image(csd_peaks.gfa.astype(np.float32), dwi_img.affine)
    # make the output name yo
    gfa_output_name = ''.join([out_name_base, '_gfa.nii.gz'])
    # same this FA
    nib.save(gfa_img, gfa_output_name)


if __name__ == '__main__':
    main()
