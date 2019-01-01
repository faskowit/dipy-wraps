__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University
University of Southern California

inspiration taken from the dipy website

"""

import sys
import numpy as np
import nibabel as nib
from src.tracking.args_runTracking import CmdLineHandler
from src.dw_utils.basics import flprint, loaddwibasics

from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, TensorModel , quantize_evecs


def main():
    
    cmdLine = CmdLineHandler()
    cmdLine.get_args()
    cmdLine.check_args()
    
    flprint("command line args read")
    for attr, value in cmdLine.__dict__.iteritems():
        print attr, value

    dwiImg, maskImg, gtab = loaddwibasics(cmdLine.dwi_,
                                          cmdLine.mask_,
                                          cmdLine.bval_,
                                          cmdLine.bvec_)

    # get the data from all the images yo
    dwiData = dwiImg.get_data()
    maskData = maskImg.get_data()

    # mask the dwiData
    dwiData = applymask(dwiData, maskData)

    wmMaskData = None

    if cmdLine.wmMask_:
        
        flprint("using wm mask provided")

        wmMaskImg = nib.load(cmdLine.wmMask_)
        wmMaskData = wmMaskImg.get_data()

        flprint(wmMaskImg.shape)

    sphereData = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RANDOM SEED and DENSITY ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO give user option to save seed points as npz for future usage

    seed_points = None

    if not cmdLine.seedPointsFile_:

        if cmdLine.wmMask_:

            # this is the common usage
            if cmdLine.randSeed_:
                # make the seeds randomly yo
                print("using a wm mask for random seeds with density of {}\n".format(str(cmdLine.seedDensity_)))

                from dipy.tracking.utils import random_seeds_from_mask
                seed_points = random_seeds_from_mask(wmMaskData,
                                                     affine=maskImg.get_affine())
            else:
                # make the seeds yo
                print(
                    "using a wm mask for NON-random seeds with a density of {}\n".format(str(cmdLine.seedDensity_)))

                from dipy.tracking.utils import seeds_from_mask
                seed_points = seeds_from_mask(wmMaskData,
                                              density=int(cmdLine.seedDensity_),
                                              affine=maskImg.get_affine())
        else:

            from dipy.tracking.utils import seeds_from_mask
            seed_points = seeds_from_mask(maskData,
                                          density=1,
                                          affine=maskImg.get_affine())

    else:

        # TODO when writing seeds, write the affine, so we can compare when loading in

        print("loading seed points from file: {}".format(str(cmdLine.seedPointsFile_)))
        sys.stdout.flush()

        seedFile = np.load(cmdLine.seedPointsFile_)
        seed_points = seedFile['arr_0']

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ LIMIT TOT SEEDS? ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # now limit the amount of seeds if specificed by tot_seeds with replacement
    if cmdLine.limitTotSeeds_ > 0:

        totSeeds = int(round(float(cmdLine.limitTotSeeds_)))

        print("limiting total seeds to {}".format(totSeeds))

        seedSize = seed_points.shape[0]

        if totSeeds > seedSize:
            # need this because of determinisitc,
            # if you pick the same seed, it will just
            # be the same fiber yo
            totSeeds = seedSize

        # indicies = random.sample(xrange(seedSize), totSeeds)
        # print(indicies)
        indicies = np.random.choice(xrange(seedSize),
                                    size=totSeeds,
                                    replace=True)

        new_seeds = [seed_points[i] for i in indicies]

        seed_points = np.asarray(new_seeds)

        flprint("new shape of seed points is : {}".format(seed_points.shape))

    # ~~~~~~~~~ saving / loading seed points? ~~~~~~~~~~~~~~~~~~~~~~

    if cmdLine.saveSeedPoints_:

        if cmdLine.randSeed_:
            seedPointsNpz_name = ''.join([cmdLine.output_, '_',
                                          '_randinvox',
                                          '_den',
                                          str(cmdLine.seedDensity_),
                                          '_seedPoints',
                                          '.npz'])
        else:
            seedPointsNpz_name = ''.join([cmdLine.output_, '_',
                                          '_nonrandinvox',
                                          '_den',
                                          str(cmdLine.seedDensity_),
                                          '_seedPoints',
                                          '.npz'])

        np.savez_compressed(seedPointsNpz_name, seed_points, 'arr_0')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CLASSIFIER YO ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    classifier = None

    if cmdLine.actClasses_:

        classifier = act_classifier(cmdLine)

    else:

        faData = make_fa_map(dwiData, maskData, gtab)

        from dipy.tracking.local import ThresholdTissueClassifier
        classifier = ThresholdTissueClassifier(faData, float(cmdLine.faClassify_))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIFFUSION / FIBER MODELS ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    shcoeffs = None
    peakDirections = None

    # must make this a bit if elif else statement
    if cmdLine.tractModel_ == 'csd':

        print('using the csd model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if not cmdLine.coeffFile_:

            flprint("making the CSD ODF model wish sh of {}\n".format(cmdLine.shOrder_))

            from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                               auto_response)

            # get the response yo.
            response, ratio = auto_response(gtab, dwiData, roi_radius=10, fa_thr=0.7)

            flprint("the response for the csd is:\n{0}\nwith a ratio of:\n{1}\n".format(response, ratio))

            csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=int(cmdLine.shOrder_))

            flprint("making the CSD fit yo")

            csd_fit = csd_model.fit(dwiData, mask=maskData)
            shcoeffs = csd_fit.shm_coeff

        else:  # i.e. we already have generated the coeffs

            print("loading coeffs from file: {}".format(str(cmdLine.coeffFile_)))
            sys.stdout.flush()

            coeffsFile = np.load(cmdLine.coeffFile_)
            shcoeffs = coeffsFile['arr_0']

    elif cmdLine.tractModel_ == 'csa':

        print('using the csa model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSA FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model

        print("generating csa model. the sh order is {}".format(cmdLine.shOrder_))

        csa_model = CsaOdfModel(gtab, sh_order=cmdLine.shOrder_)
        csa_peaks = peaks_from_model(model=csa_model,
                                     data=dwiData,
                                     sphere=sphereData,
                                     relative_peak_threshold=0.5,
                                     min_separation_angle=25,
                                     mask=maskData,
                                     return_sh=True,
                                     parallel=False)

        shcoeffs = csa_peaks.shm_coeff

    elif cmdLine.tractModel_ == 'sparse':

        # TODO gotta perhaps implement this one
        pass

    else:  # this is for DTI model yo.

        print('using the dti model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ TENSOR FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        tensor_model = TensorModel(gtab)
        faData = make_fa_map(dwiData, maskData, gtab)

        faImg = nib.Nifti1Image(faData.astype(np.float32), maskImg.get_affine())

        # make the output name yo
        fa_outputname = ''.join([cmdLine.output_, '_fa.nii.gz'])
        print('The output FA name is: {}'.format(fa_outputname))
        sys.stdout.flush()

        # same this FA
        nib.save(faImg, fa_outputname)

        if cmdLine.dirGttr_ != 'eudx':

            from dipy.reconst.peaks import peaks_from_model
            peakDirections = peaks_from_model(tensor_model,
                                              data=dwiData.astype(np.float_),
                                              sphere=sphereData,
                                              relative_peak_threshold=0.5,
                                              min_separation_angle=25,
                                              mask=maskData,
                                              parallel=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIRECTION GETTR ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    directionGetter = None

    print("generating the direction gettr\n")
    sys.stdout.flush()

    if (cmdLine.tractModel_ == 'csa') or (cmdLine.tractModel_ == 'csd'):

        if cmdLine.dirGttr_ == 'deterministic':

            from dipy.direction import DeterministicMaximumDirectionGetter

            print("determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=30.0,
                                                                               sphere=sphereData)
        elif cmdLine.dirGttr_ == 'probabilistic':

            from dipy.direction import ProbabilisticDirectionGetter

            print("prob direction gettr\n")
            directionGetter = ProbabilisticDirectionGetter.from_shcoeff(shcoeffs,
                                                                        max_angle=30.0,
                                                                        sphere=sphereData)

        else:

            from dipy.direction import DeterministicMaximumDirectionGetter

            print("you forog to to specify a dir gtter, so we will use determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=30.0,
                                                                               sphere=sphereData)

    else:
        # dont need a deter or prob dir getter

        directionGetter = peakDirections

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ STREAMLINES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    streamlines = None

    print("making streamlines yo\n")
    sys.stdout.flush()

    if cmdLine.dirGttr_ != 'eudx':

        from dipy.tracking.local import LocalTracking

        streamlines = LocalTracking(directionGetter,
                                    classifier,
                                    seed_points,
                                    maskImg.get_affine(),
                                    step_size=cmdLine.stepSize_,
                                    max_cross=cmdLine.maxCross_,
                                    return_all=cmdLine.returnAll_)

        # Compute streamlines and store as a list.
        streamlines = list(streamlines)

        # the length to trim is 5. but this can be changed
        # TODO make this user option
        from dipy.tracking.metrics import length
        streamlines = [s for s in streamlines if length(s) > cmdLine.lenThresh_]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ TRK IO.TODO--> change it ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tracks_outputname = None

    if (cmdLine.tractModel_ == 'csa') or (cmdLine.tractModel_ == 'csd'):

        tracks_outputname = ''.join([cmdLine.output_, '_',
                                     cmdLine.dirGttr_,
                                     '_',
                                     cmdLine.tractModel_,
                                     '_sh',
                                     str(cmdLine.shOrder_),
                                     '.trk'])
    else:
        tracks_outputname = ''.join([cmdLine.output_, '_',
                                     cmdLine.dirGttr_,
                                     '_',
                                     cmdLine.tractModel_,
                                     '.trk'])

    print('The output tracks name is: {}'.format(tracks_outputname))

    from dipy.io.trackvis import save_trk
    save_trk(tracks_outputname, streamlines, maskImg.get_affine(), maskData.shape)

    streamlines_np = np.array(streamlines, dtype=np.object)
    np.savez_compressed('streamlines.npz', streamlines_np)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CONNECTIVITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # the number of streamlines, we will use later to normalize
    numStreams = len(streamlines)
    print('THE NUMBER OF STREAMS IS {0}'.format(numStreams))
    sys.stdout.flush()

    if cmdLine.fsSegs_:
        print('now making the connectivity matricies')
        sys.stdout.flush()

        fsSegs_img = nib.load(cmdLine.fsSegs_)
        fsSegs_data = fsSegs_img.get_data()

        from dipy.tracking.utils import connectivity_matrix
        M, grouping = connectivity_matrix(streamlines, fsSegs_data,
                                          affine=maskImg.get_affine(),
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
        # get rid of the first row yo...
        M[:1, :] = 0
        M[:, :1] = 0

        print('here is the connectivity')
        print (M)

        # need to normalize the matrix yo
        norm_M = np.divide(np.array(M, dtype=float), numStreams)

        import csv

        with open(''.join([cmdLine.output_, '_connectivity_norm_segs.csv']), "wb") as f:
            writer = csv.writer(f)
            writer.writerows(norm_M)

        with open(''.join([cmdLine.output_, '_connectivity_count_segs.csv']), "wb") as f:
            writer = csv.writer(f)
            writer.writerows(M)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DENISTY MAP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    from dipy.tracking.utils import density_map

    dimX = maskData.shape[0]
    dimY = maskData.shape[1]
    dimZ = maskData.shape[2]

    densityData = density_map(streamlines=streamlines,
                              vol_dims=(dimX, dimY, dimZ),
                              affine=maskImg.get_affine())
    # save the image
    print("saving the new image yo")
    sys.stdout.flush()

    density_image = nib.Nifti1Image(densityData.astype(np.float32), maskImg.get_affine())

    density_outputname = ''.join([cmdLine.output_,
                                  '_',
                                  cmdLine.dirGttr_,
                                  '_',
                                  cmdLine.tractModel_,
                                  '_density.nii.gz'])

    nib.save(density_image, density_outputname)

def make_fa_map(dwidata, maskdata, gtabdata):

    tensor_model = TensorModel(gtabdata)

    tenfit = tensor_model.fit(dwidata, mask=maskdata)
    fadata = fractional_anisotropy(tenfit.evals)

    # just saving the FA image yo
    fadata[np.isnan(fadata)] = 0

    # also we can clip values outside of 0 and 1
    fadata = np.clip(fadata, 0, 1)

    return fadata

def act_classifier(cmdlineobj):

    from dipy.tracking.local import ActTissueClassifier

    flprint("making the classifier from your segs yo\n")

    # csf, gm, wm
    csfImage = nib.load(cmdlineobj.actClasses_[0])
    csfData = csfImage.get_data()
    gmImage = nib.load(cmdlineobj.actClasses_[1])
    gmData = gmImage.get_data()
    wmImage = nib.load(cmdlineobj.actClasses_[2])
    wmData = wmImage.get_data()

    # make a background
    background = np.ones(csfImage.shape)
    background[(gmData + wmData + csfData) > 0] = 0

    include_map = gmData
    include_map[background > 0] = 1

    exclude_map = csfData

    return ActTissueClassifier(include_map, exclude_map)


if __name__ == '__main__':
    main()
