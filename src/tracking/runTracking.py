__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University
University of Southern California

inspiration taken from the dipy website

"""

import h5py
import ntpath
import csv
import numpy as np
import nibabel as nib
import time
from datetime import timedelta

from src.tracking.args_runTracking import CmdLineRunTracking
from src.dw_utils.basics import flprint, loaddwibasics

from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, TensorModel
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask


def main():
    
    cmdLine = CmdLineRunTracking("dipy streamline tracking yo")
    cmdLine.get_args()
    cmdLine.check_args()
    
    flprint("command line args read")
    # for attr, value in cmdLine.__dict__.iteritems():
    #    print(str(attr), str(value))

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

    seed_points = None

    if not cmdLine.seedPointsFile_:

        if cmdLine.wmMask_:

            # this is the common usage
            if cmdLine.randSeed_:

                # make the seeds randomly yo
                flprint("using a wm mask for random seeds with density of {}\n".format(str(cmdLine.seedDensity_)))

                seed_points = random_seeds_from_mask(wmMaskData,
                                                     seeds_count=int(cmdLine.seedDensity_),
                                                     seed_count_per_voxel=True,
                                                     affine=maskImg.affine)
            else:

                # make the seeds yo
                flprint("using a wm mask for NON-random seeds with a density of {}\n".format(str(cmdLine.seedDensity_)))

                seed_points = seeds_from_mask(wmMaskData,
                                              density=int(cmdLine.seedDensity_),
                                              affine=maskImg.affine)
        else:

            seed_points = seeds_from_mask(maskData,
                                          density=1,
                                          affine=maskImg.affine)

    else:

        flprint("loading seed points from file: {}".format(str(cmdLine.seedPointsFile_)))

        seedFile = np.load(cmdLine.seedPointsFile_)
        seed_points = seedFile['seeds']

        if not np.array_equal(seedFile['affine'], maskImg.affine):
            flprint("affine read from seeds file does not look good")
            exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ LIMIT TOT SEEDS? ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # now limit the amount of seeds if specificed by tot_seeds with replacement
    if cmdLine.limitTotSeeds_ > 0:

        totSeeds = int(round(float(cmdLine.limitTotSeeds_)))

        flprint("limiting total seeds to {}".format(totSeeds))

        seedSize = seed_points.shape[0]

        if totSeeds > seedSize:
            # need this because of determinisitc,
            # if you pick the same seed, it will just
            # be the same fiber yo
            totSeeds = seedSize

        indicies = np.random.choice(range(seedSize),
                                    size=totSeeds,
                                    replace=True)

        new_seeds = [seed_points[i] for i in indicies]
        seed_points = np.asarray(new_seeds)

        flprint("new shape of seed points is : {}".format(seed_points.shape))

    # ~~~~~~~~~ saving / loading seed points? ~~~~~~~~~~~~~~~~~~~~~~

    if cmdLine.saveSeedPoints_:

        if cmdLine.randSeed_:
            seedPointsNpz_name = ''.join([cmdLine.output_,
                                          'rand_den',
                                          str(cmdLine.seedDensity_),
                                          '_seeds.npz'])
        else:
            seedPointsNpz_name = ''.join([cmdLine.output_,
                                          'nonrand_den',
                                          str(cmdLine.seedDensity_),
                                          '_seeds.npz'])

        # save mask array, also save the affine corresponding to this space
        np.savez_compressed(seedPointsNpz_name,
                            seeds=seed_points,
                            affine=maskImg.affine)

    # echo to the user how many seeds there are
    flprint("Seed points with shape: {}\n".format(str(seed_points.shape)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CLASSIFIER YO ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    classifier = None

    if cmdLine.actClasses_:

        classifier = act_classifier(cmdLine)

    else:

        faData, _ = make_fa_map(dwiData, maskData, gtab)

        from dipy.tracking.local import ThresholdTissueClassifier
        classifier = ThresholdTissueClassifier(faData, float(cmdLine.faThr_))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIFFUSION / FIBER MODELS ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    shcoeffs = None
    peakDirections = None

    # must make this a bit if elif else statement
    if cmdLine.tractModel_ == 'csd':

        flprint('using the csd model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSD FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if not cmdLine.coeffFile_:

            flprint("making the CSD ODF model wish sh of {}\n".format(cmdLine.shOrder_))

            from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                               auto_response)

            # get the response yo.
            response, ratio = auto_response(gtab, dwiData, roi_radius=10, fa_thr=cmdLine.faThr_)

            flprint("the response for the csd is:\n{0}\nwith a ratio of:\n{1}\n".format(response, ratio))

            csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=int(cmdLine.shOrder_))

            flprint("making the CSD fit yo")

            csd_fit = csd_model.fit(dwiData, mask=maskData)
            shcoeffs = csd_fit.shm_coeff

        else:  # i.e. we already have generated the coeffs

            flprint("loading coeffs from file: {}".format(str(cmdLine.coeffFile_)))

            # the new format yo
            coeffsFile = h5py.File(cmdLine.coeffFile_, 'r')
            shcoeffs = np.array(coeffsFile['PAM/coeff'])

    elif cmdLine.tractModel_ == 'csa':

        flprint('using the csa model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ CSA FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model

        flprint("generating csa model. the sh order is {}".format(cmdLine.shOrder_))

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

        # TODO gotta implement this one
        pass

    else:  # this is for DTI model yo.

        flprint('using the dti model yo')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~ TENSOR FIT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        tensor_model = TensorModel(gtab)

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

    flprint("generating the direction gettr\n")

    if (cmdLine.tractModel_ == 'csa') or (cmdLine.tractModel_ == 'csd'):

        if cmdLine.dirGttr_ == 'deterministic':

            from dipy.direction import DeterministicMaximumDirectionGetter

            flprint("determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=np.float(cmdLine.maxAngle_),
                                                                               sphere=sphereData)
        elif cmdLine.dirGttr_ == 'probabilistic':

            from dipy.direction import ProbabilisticDirectionGetter

            flprint("prob direction gettr\n")
            directionGetter = ProbabilisticDirectionGetter.from_shcoeff(shcoeffs,
                                                                        max_angle=np.float(cmdLine.maxAngle_),
                                                                        sphere=sphereData)

        else:

            # dti tracking must be deterministic

            from dipy.direction import DeterministicMaximumDirectionGetter

            flprint("you forgot to to specify a dir gtter, so we will use determinisitc direction gettr\n")
            directionGetter = DeterministicMaximumDirectionGetter.from_shcoeff(shcoeffs,
                                                                               max_angle=np.float(cmdLine.maxAngle_),
                                                                               sphere=sphereData)

    else:
        # dont need a deter or prob dir getter

        directionGetter = peakDirections

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ STREAMLINES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    streamlines = None

    flprint("making streamlines yo\n")

    if cmdLine.dirGttr_ != 'eudx':

        from dipy.tracking.local import LocalTracking

        streamlines = LocalTracking(directionGetter,
                                    classifier,
                                    seed_points,
                                    maskImg.affine,
                                    step_size=np.float(cmdLine.stepSize_),
                                    max_cross=cmdLine.maxCross_,
                                    return_all=False)

        start_time = time.time()

        # Compute streamlines
        streamlines = list(streamlines)

        end_time = time.time()

        flprint("finished generating streams, took {}".format(str(timedelta(seconds=end_time - start_time))))

        # this is the length function that acts on lists
        from dipy.tracking.metrics import length
        streamlines = [s for s in streamlines if length(s) > np.int(cmdLine.lenThresh_)]

        from dipy.tracking.streamline import Streamlines
        streamlines = Streamlines(streamlines)

        flprint("initially generated {} streamlines".format(str(len(streamlines))))

    else:

        flprint("this script no longer supports eudx")
        exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ cluster con ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if cmdLine.runCCI_:

        numcciReps = 5
        cciIterResults = np.zeros([len(streamlines), numcciReps])

        for i in range(numcciReps):
            cciIterResults[:, i] = cci_iterative(streamlines, 1000, 10)

        cci = np.nanmax(cciIterResults, axis=1)

        from dipy.tracking.streamline import Streamlines

        ccistreamlines = Streamlines()
        numRemoved = 0

        for i, sl in enumerate(streamlines):
            if cci[i] >= 0.25:
                ccistreamlines.append(sl)
            else:
                numRemoved += 1

        flprint("number of streamlines removed with cci: {}".format(str(numRemoved)))

        # old_stream = streamlines
        streamlines = ccistreamlines

        cciResults_name = ''.join([cmdLine.output_, 'cciresults.npz'])

        # save mask array, also save the affine corresponding to this space
        np.savez_compressed(cciResults_name, cciIterResults)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ TRK IO.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO incorporate new streamline io

    tracks_outputname = None

    if (cmdLine.tractModel_ == 'csa') or (cmdLine.tractModel_ == 'csd'):

        tracks_outputname = ''.join([cmdLine.output_,
                                     cmdLine.dirGttr_,
                                     '_',
                                     cmdLine.tractModel_,
                                     '_sh',
                                     str(cmdLine.shOrder_),
                                     '.trk'])

        # tracks_outputname2 = ''.join([cmdLine.output_,
        #                              cmdLine.dirGttr_,
        #                              '_',
        #                              cmdLine.tractModel_,
        #                              '_sh',
        #                              str(cmdLine.shOrder_),
        #                              'old.trk'])

    else:
        tracks_outputname = ''.join([cmdLine.output_,
                                     cmdLine.dirGttr_,
                                     '_',
                                     cmdLine.tractModel_,
                                     '.trk'])

    flprint('The output tracks name is: {}'.format(tracks_outputname))

    # old usage
    from dipy.io.trackvis import save_trk
    save_trk(tracks_outputname, streamlines, maskImg.affine, maskData.shape)

    # from dipy.io.trackvis import save_trk
    # save_trk(tracks_outputname2, streamlines, np.eye(4), maskData.shape)

    # from dipy.io.streamline import save_trk
    # save_trk(tracks_outputname, streamlines, maskImg.affine,
    #         header=maskImg.header,
    #         vox_size=maskImg.header.get_zooms(),
    #         shape=maskImg.shape)

    # streamlines_np = np.array(streamlines, dtype=np.object)
    # np.savez_compressed('streamlines.npz', streamlines_np)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CONNECTIVITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # the number of streamlines, we will use later to normalize
    numStreams = len(streamlines)
    flprint('THE NUMBER OF STREAMS IS {0}'.format(numStreams))

    if cmdLine.parcImgs_:

        flprint("fitting the fa map for conn mat cal")
        _, tenfit = make_fa_map(dwiData, maskData, gtab)

        for i in range(len(cmdLine.parcImgs_)):

            flprint('\n\nnow making the connectivity matrices for: {}'.format(str(cmdLine.parcImgs_[i])))

            fsSegs_img = nib.load(cmdLine.parcImgs_[i])
            fsSegs_data = fsSegs_img.get_data().astype(np.int16)

            # lets get the name of the seg to use
            # in the output writing
            segBaseName = ntpath.basename(
              ntpath.splitext(ntpath.splitext(cmdLine.parcImgs_[i])[0])[0])
            # basename of all the outputs
            conmatBasename = ''.join([cmdLine.output_, segBaseName, '_'])

            from dipy.tracking.utils import connectivity_matrix
            M, grouping = connectivity_matrix(streamlines, fsSegs_data,
                                              symmetric=True,
                                              affine=maskImg.affine,
                                              return_mapping=True,
                                              mapping_as_streamlines=True)

            # get rid of the first row because these are connections to '0'
            M[:1, :] = 0
            M[:, :1] = 0

            fib_lengths = mat_stream_lengths(M, grouping, aff=maskImg.affine)
            mean_fa, _ = mat_indicies_along_streams(M, grouping, tenfit.fa, aff=maskImg.affine)
            mean_md, _ = mat_indicies_along_streams(M, grouping, tenfit.md, aff=maskImg.affine)

            # save the files
            ex_csv(''.join([conmatBasename, 'slcounts.csv']), M)
            ex_csv(''.join([conmatBasename, 'lengths.csv']), fib_lengths)
            ex_csv(''.join([conmatBasename, 'meanfa.csv']), mean_fa)
            ex_csv(''.join([conmatBasename, 'meanmd.csv']), mean_md)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DENSITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    density_outputname = ''.join([cmdLine.output_, cmdLine.dirGttr_,
                                  '_', cmdLine.tractModel_,
                                  '_density.nii.gz'])

    densImg = make_density_img(streamlines, maskImg, 1)
    nib.save(densImg, density_outputname)


def mat_stream_lengths(mat, group, aff=np.eye(4)):

    retmat = np.zeros(mat.shape, dtype=np.float)

    # by row
    for x in range(mat.shape[0]):
        # by column
        for y in range(mat.shape[1]):

            # check if entry in dict, if so, do more stuff
            if (x, y) in group:
                from dipy.tracking.utils import length

                # now we record these values
                stream_group_len = length(group[x, y], affine=aff)
                retmat[x, y] = np.around(np.nanmean(list(stream_group_len)), 2)

    return retmat


def mat_indicies_along_streams(mat, group, infovol, aff=np.eye(4), stdstreamlen=50):

    meanretmat = np.zeros(mat.shape, dtype=np.float)
    medianretmat = np.zeros(mat.shape, dtype=np.float)

    # will trim the lower %5 and top %5 of streamline
    trimlen = np.int(np.floor(stdstreamlen / 20))

    # by row
    for x in range(mat.shape[0]):
        # by column
        for y in range(mat.shape[1]):

            # check if entry in dict, if so, do more stuff
            if (x, y) in group:

                # first lets resample all streamlines to const length
                from dipy.tracking.streamlinespeed import set_number_of_points
                stand_stream_group = set_number_of_points(group[x, y], stdstreamlen)

                from dipy.tracking.streamline import values_from_volume
                stream_vals = values_from_volume(infovol, stand_stream_group, aff)

                vals = np.zeros(len(stream_vals))
                for ind, sv in enumerate(stream_vals):
                    vals[ind] = np.mean(sv[trimlen:-trimlen])

                # flprint(str(np.mean(vals)))
                meanretmat[x, y] = np.around(np.mean(vals), 6)
                medianretmat[x, y] = np.around(np.median(vals), 6)

    return meanretmat, medianretmat


def make_density_img(streams, maskimg, resmultiply):

    from dipy.tracking.utils import density_map

    # initialize a new affine
    new_affine = maskimg.affine

    # divide by the res multiplier
    new_affine[0, 0] = new_affine[0, 0] * 1/resmultiply
    new_affine[1, 1] = new_affine[1, 1] * 1/resmultiply
    new_affine[2, 2] = new_affine[2, 2] * 1/resmultiply

    densityData = density_map(streams,
                              vol_dims=(np.multiply(maskimg.shape, resmultiply)),
                              affine=new_affine)

    return nib.Nifti1Image(densityData, new_affine)

def make_fa_map(dwidata, maskdata, gtabdata):

    tensor_model = TensorModel(gtabdata)

    tenfit = tensor_model.fit(dwidata, mask=maskdata)
    fadata = fractional_anisotropy(tenfit.evals)

    # just saving the FA image yo
    fadata[np.isnan(fadata)] = 0

    # also we can clip values outside of 0 and 1
    fadata = np.clip(fadata, 0, 1)

    return fadata, tenfit


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


def cci_iterative(inputstreams, cciSize=10000, cciSubSmp=12):

    from dipy.tracking.streamline import cluster_confidence

    flprint("running the cci iterative with subsamp: {}".format(str(cciSubSmp)))

    # need to breakup streamlines into manageable chunks
    totalStreams = len(inputstreams)

    if totalStreams < cciSize:
        # you can just run it normally

        start_time = time.time()

        try:
            cciResults = cluster_confidence(inputstreams,
                                            subsample=cciSubSmp,
                                            max_mdf=6)
        except ValueError:
            print("caught rare value error")
            nanvals = np.empty(len(inputstreams))
            nanvals.fill(np.nan)
            cciResults = nanvals

        end_time = time.time()

    else:

        randInd = list(range(totalStreams))
        # shuffle the indices in place
        np.random.shuffle(randInd)

        streamChunkInd = list(chunker(randInd, cciSize))

        # allocate the array
        cciResults = np.zeros(totalStreams)

        start_time = time.time()

        for i in range(len(streamChunkInd)):

            flprint("cci iter: {} of {}".format(str(i+1), str(len(streamChunkInd))))

            try:
                cciResults[streamChunkInd[i]] = cluster_confidence(inputstreams[streamChunkInd[i]],
                                                                   subsample=cciSubSmp,
                                                                   max_mdf=6)

            except ValueError:
                print("caught rare value error")
                nanvals = np.empty(len(inputstreams[streamChunkInd[i]]))
                nanvals.fill(np.nan)
                cciResults[streamChunkInd[i]] = nanvals

        end_time = time.time()

    flprint("finished cci, took {}".format(str(timedelta(seconds=end_time - start_time))))

    return cciResults


def chunker(seq, size):
    # https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def ex_csv(filename, data):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    main()
