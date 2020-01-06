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
# dipy wraps imports
from src.tracking.args_runTracking import CmdLineRunTracking
from src.dw_utils.basics import flprint, loaddwibasics, chunker
# dipy imports
from dipy.segment.mask import applymask
from dipy.data import get_sphere
from dipy.reconst.dti import fractional_anisotropy, TensorModel
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.tracking.streamline import Streamlines
from dipy.tracking.distances import approx_polygon_track


def main():
    
    command_line = CmdLineRunTracking("dipy streamline tracking yo")
    command_line.get_args()
    command_line.check_args()
    
    flprint("command line args read")
    # for attr, value in command_line.__dict__.iteritems():
    #    print(str(attr), str(value))

    dwi_img, mask_img, grad_tab = loaddwibasics(command_line.dwi_,
                                                command_line.mask_,
                                                command_line.bval_,
                                                command_line.bvec_)

    # get the data from all the images yo
    dwi_data = None
    if dwi_img:
        dwi_data = dwi_img.get_fdata()
    mask_data = mask_img.get_fdata()
    # mask the dwi_data
    if dwi_img:
        dwi_data = applymask(dwi_data, mask_data)
    wm_mask_data = None
    if command_line.wmMask_:
        flprint("using wm mask provided")
        wm_mask_img = nib.load(command_line.wmMask_)
        wm_mask_data = wm_mask_img.get_fdata()
        flprint(wm_mask_img.shape)

    sphere_data = get_sphere('repulsion724')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ RANDOM SEED and DENSITY ~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    seed_points = None

    if not command_line.seedPointsFile_:
        if command_line.wmMask_:
            # this is the common usage
            if command_line.randSeed_:
                # make the seeds randomly yo
                flprint("using a wm mask for random seeds with density of {}\n".format(str(command_line.seedDensity_)))
                seed_points = random_seeds_from_mask(wm_mask_data,
                                                     seeds_count=int(command_line.seedDensity_),
                                                     seed_count_per_voxel=True,
                                                     affine=mask_img.affine)
            else:
                # make the seeds yo
                flprint("using a wm mask for NON-random seeds with a density of {}\n".format(
                    str(command_line.seedDensity_)))
                seed_points = seeds_from_mask(wm_mask_data,
                                              density=int(command_line.seedDensity_),
                                              affine=mask_img.affine)
        else:
            seed_points = seeds_from_mask(mask_data,
                                          density=1,
                                          affine=mask_img.affine)
    else:
        flprint("loading seed points from file: {}".format(str(command_line.seedPointsFile_)))
        seed_file = np.load(command_line.seedPointsFile_)
        seed_points = seed_file['seeds']
        if not np.array_equal(seed_file['affine'], mask_img.affine):
            flprint("affine read from seeds file does not look good")
            exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ LIMIT TOT SEEDS? ~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # now limit the amount of seeds if specified by tot_seeds with replacement
    if command_line.limitTotSeeds_ > 0:
        tot_seeds = int(round(float(command_line.limitTotSeeds_)))

        flprint("limiting total seeds to {}".format(tot_seeds))

        seed_size = seed_points.shape[0]

        if tot_seeds > seed_size:
            # need this because of deterministic,
            # if you pick the same seed, it will just
            # be the same fiber yo
            tot_seeds = seed_size

        seed_inds = np.random.choice(range(seed_size),
                                     size=tot_seeds,
                                     replace=True)

        new_seeds = [seed_points[i] for i in seed_inds]
        seed_points = np.asarray(new_seeds)
        flprint("new shape of seed points is : {}".format(seed_points.shape))

    # ~~~~~~~~~ saving / loading seed points? ~~~~~~~~~~~~~~~~~~~~~~

    if command_line.saveSeedPoints_:
        if command_line.randSeed_:
            seed_points_npz_name = ''.join([command_line.output_,
                                            'rand_den',
                                            str(command_line.seedDensity_),
                                            '_seeds.npz'])
        else:
            seed_points_npz_name = ''.join([command_line.output_,
                                            'nonrand_den',
                                            str(command_line.seedDensity_),
                                            '_seeds.npz'])

        # save mask array, also save the affine corresponding to this space
        np.savez_compressed(seed_points_npz_name,
                            seeds=seed_points,
                            affine=mask_img.affine)

    # echo to the user how many seeds there are
    flprint("Seed points with shape: {}\n".format(str(seed_points.shape)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CLASSIFIER YO ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    classifier = None
    if command_line.actClasses_:
        classifier = act_classifier(command_line)
    else:
        if dwi_data is not None:
            fa_data, _ = make_fa_map(dwi_data, mask_data, grad_tab)
        else:
            flprint('need to provide dwi data if no tissues for classification')
            exit(1)

        from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
        classifier = ThresholdStoppingCriterion(fa_data, float(command_line.faThr_))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIFFUSION / FIBER MODELS ~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sphere_harmonic_coeffs = None
    peak_directions = None

    # must make this a bit if elif else statement
    if command_line.tractModel_ == 'csd':
        flprint('using the csd model yo')
        if not command_line.coeffFile_:
            flprint("making the CSD ODF model wish sh of {}\n".format(command_line.shOrder_))
            from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                               auto_response)

            # get the response yo.
            response, ratio = auto_response(grad_tab, dwi_data, roi_radius=10, fa_thr=command_line.faThr_)
            flprint("the response for the csd is:\n{0}\nwith a ratio of:\n{1}\n".format(response, ratio))
            csd_model = ConstrainedSphericalDeconvModel(grad_tab, response, sh_order=int(command_line.shOrder_))
            flprint("making the CSD fit yo")
            csd_fit = csd_model.fit(dwi_data, mask=mask_data)
            sphere_harmonic_coeffs = csd_fit.shm_coeff

        else:  # i.e. we already have generated the coeffs
            flprint("loading coeffs from file: {}".format(str(command_line.coeffFile_)))

            # the new format yo
            coeffs_file = h5py.File(command_line.coeffFile_, 'r')
            sphere_harmonic_coeffs = np.array(coeffs_file['PAM/coeff'])

    elif command_line.tractModel_ == 'csa':
        flprint('using the csa model yo')

        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model

        flprint("generating csa model. the sh order is {}".format(command_line.shOrder_))

        csa_model = CsaOdfModel(grad_tab, sh_order=command_line.shOrder_)
        csa_peaks = peaks_from_model(model=csa_model,
                                     data=dwi_data,
                                     sphere=sphere_data,
                                     relative_peak_threshold=0.5,
                                     min_separation_angle=25,
                                     mask=mask_data,
                                     return_sh=True,
                                     parallel=False)

        sphere_harmonic_coeffs = csa_peaks.shm_coeff

    elif command_line.tractModel_ == 'sparse':
        # TODO gotta implement this one
        pass
    else:  # this is for DTI model yo.
        flprint('using the dti model yo')
        tensor_model = TensorModel(grad_tab)

        if command_line.dirGttr_ != 'eudx':

            from dipy.direction import peaks_from_model
            peak_directions = peaks_from_model(tensor_model,
                                               data=dwi_data.astype(np.float_),
                                               sphere=sphere_data,
                                               relative_peak_threshold=0.5,
                                               min_separation_angle=25,
                                               mask=mask_data,
                                               parallel=False)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DIRECTION GETTR ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dir_getter = None

    flprint("generating the direction gettr\n")

    if (command_line.tractModel_ == 'csa') or (command_line.tractModel_ == 'csd'):
        if command_line.dirGttr_ == 'deterministic':
            from dipy.direction import DeterministicMaximumDirectionGetter
            flprint("determinisitc direction gettr\n")
            dir_getter = DeterministicMaximumDirectionGetter.from_shcoeff(sphere_harmonic_coeffs,
                                                                          max_angle=np.float(command_line.maxAngle_),
                                                                          sphere=sphere_data)
        elif command_line.dirGttr_ == 'probabilistic':
            from dipy.direction import ProbabilisticDirectionGetter
            flprint("prob direction gettr\n")
            dir_getter = ProbabilisticDirectionGetter.from_shcoeff(sphere_harmonic_coeffs,
                                                                   max_angle=np.float(command_line.maxAngle_),
                                                                   sphere=sphere_data)
        # elif command_line.dirGttr_ == 'eudx':
        #    from dipy.reconst.peak_direction_getter import EuDXDirectionGetter
        #    flprint("prob direction gettr\n")
        #    dir_getter = EuDXDirectionGetter.from_shcoeff(sphere_harmonic_coeffs,
        #                                                  max_angle=np.float(command_line.maxAngle_),
        #                                                  sphere=sphere_data)
        else:
            # dti tracking must be deterministic
            from dipy.direction import DeterministicMaximumDirectionGetter

            flprint("you forgot to to specify a dir gtter, so we will use determinisitc direction gettr\n")
            dir_getter = DeterministicMaximumDirectionGetter.from_shcoeff(sphere_harmonic_coeffs,
                                                                          max_angle=np.float(command_line.maxAngle_),
                                                                          sphere=sphere_data)
    else:
        # dont need a deter or prob dir getter
        dir_getter = peak_directions

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ STREAMLINES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    streamlines = None
    flprint("making streamlines yo\n")

    if command_line.dirGttr_ != 'eudx':
        start_time = time.time()
        from dipy.tracking.local_tracking import LocalTracking

        if command_line.chunkTrack_:
            flprint("low mem tracking will track in chunks")
            # chunk the seedpoints
            seed_points_chun_sz = 50000
            seed_points_chunked = chunker(seed_points, seed_points_chun_sz, 5000)
            # initialize streamlines
            streamlines = Streamlines()
            # loop over seeds
            for idx in range(len(seed_points_chunked)):
                flprint("tracking chunk {} of {}".format(str(idx+1), str(len(seed_points_chunked))))
                streamline_generator = LocalTracking(dir_getter,
                                                     classifier,
                                                     seed_points_chunked[idx],
                                                     mask_img.affine,
                                                     step_size=np.float(command_line.stepSize_),
                                                     max_cross=command_line.maxCross_,
                                                     return_all=False)
                # Compute streamlines
                chunk_streamlines = list(streamline_generator)
                from dipy.tracking.metrics import length
                chunk_streamlines = [s for s in chunk_streamlines if length(s) > np.float(command_line.lenThresh_)]
                chunk_streamlines = [approx_polygon_track(s, 0.2) for s in chunk_streamlines]
                flprint("chunk generated {} streamlines".format(str(len(chunk_streamlines))))
                for i, sl in enumerate(chunk_streamlines):
                    streamlines.append(sl, cache_build=True)
                flprint("appending this chunk")
                streamlines.finalize_append()
        else:
            streamline_generator = LocalTracking(dir_getter,
                                                 classifier,
                                                 seed_points,
                                                 mask_img.affine,
                                                 step_size=np.float(command_line.stepSize_),
                                                 max_cross=command_line.maxCross_,
                                                 return_all=False)
            # Compute streamlines
            streamlines = list(streamline_generator)

            # this is the length function that acts on lists
            from dipy.tracking.metrics import length
            streamlines = [s for s in streamlines if length(s) > np.float(command_line.lenThresh_)]
            # use the Streamlines type now
            streamlines = Streamlines(streamlines)

        end_time = time.time()
        flprint("\nfinished generating streams, took {}".format(str(timedelta(seconds=end_time - start_time))))
        flprint("initially generated {} streamlines".format(str(len(streamlines))))
    else:
        flprint("this script no longer supports eudx")
        exit(1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ cluster con ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if command_line.runCCI_ > 0:

        from src.tracking.clustConfidence import cluster_confidence_filter
        cci_streamlines, cci_iter_results, num_removed = \
            cluster_confidence_filter(streamlines, command_line.runCCI_, 50, 1000)

        streamlines = cci_streamlines
        cci_results_name = ''.join([command_line.output_, 'cciresults.npz'])
        np.savez_compressed(cci_results_name, cci_iter_results)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ TRK IO.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tracks_outputname = None

    if (command_line.tractModel_ == 'csa') or (command_line.tractModel_ == 'csd'):
        tracks_outputname = ''.join([command_line.output_,
                                     command_line.dirGttr_,
                                     '_',
                                     command_line.tractModel_,
                                     '_sh',
                                     str(command_line.shOrder_),
                                     '.trk'])
    else:
        tracks_outputname = ''.join([command_line.output_,
                                     command_line.dirGttr_,
                                     '_',
                                     command_line.tractModel_,
                                     '.trk'])

    # downsampling to save disk space
    output_streamlines = Streamlines([approx_polygon_track(s, 0.2) for s in streamlines])

    from src.dw_utils.basics import save_trk_to_file
    save_trk_to_file(output_streamlines, mask_img, tracks_outputname)
    flprint('The output tracks name is: {}'.format(tracks_outputname))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ CONNECTIVITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # the number of streamlines, we will use later to normalize
    num_streams = len(streamlines)
    flprint('THE NUMBER OF STREAMS IS {0}'.format(num_streams))

    if command_line.parcImgs_:

        if dwi_data is not None:
            flprint("fitting the fa map for conn mat cal")
            _, tensor_fit = make_fa_map(dwi_data, mask_data, grad_tab)

        for i in range(len(command_line.parcImgs_)):

            start_time = time.time()

            flprint('\n\nnow making the connectivity matrices for: {}'.format(str(command_line.parcImgs_[i])))

            parcellation_img = nib.load(command_line.parcImgs_[i])
            parcellation_data = parcellation_img.get_fdata().astype(np.int16)

            # lets get the name of the seg to use
            # in the output writing
            seg_base_name = ntpath.basename(
              ntpath.splitext(ntpath.splitext(command_line.parcImgs_[i])[0])[0])
            # basename of all the outputs
            conmat_basename = ''.join([command_line.output_, seg_base_name, '_'])

            from dipy.tracking.utils import connectivity_matrix
            m, grouping = connectivity_matrix(streamlines, mask_img.affine,
                                              parcellation_data,
                                              symmetric=True,
                                              return_mapping=True,
                                              mapping_as_streamlines=True)

            # get rid of the first row because these are connections to '0'
            m[:1, :] = 0
            m[:, :1] = 0

            # use the default np.eye affine here... to not apply affine twice
            fib_lengths = mat_stream_lengths(m, grouping)

            if dwi_data is not None:
                mean_fa, _ = mat_ind_along_streams(m, grouping, tensor_fit.fa,
                                                   aff=mask_img.affine, stdstreamlen=20)
                mean_md, _ = mat_ind_along_streams(m, grouping, tensor_fit.md,
                                                   aff=mask_img.affine, stdstreamlen=20)

            # save the files
            ex_csv(''.join([conmat_basename, 'slcounts.csv']), m)
            ex_csv(''.join([conmat_basename, 'lengths.csv']), fib_lengths)
            if dwi_data is not None:
                ex_csv(''.join([conmat_basename, 'meanfa.csv']), mean_fa)
                ex_csv(''.join([conmat_basename, 'meanmd.csv']), mean_md)

            end_time = time.time()

            flprint("time to make mats: {}".format(str(timedelta(seconds=end_time - start_time))))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~ DENSITY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    density_output_name = ''.join([command_line.output_, command_line.dirGttr_,
                                  '_', command_line.tractModel_,
                                   '_density.nii.gz'])

    density_img = make_density_img(streamlines, mask_img, 1)
    nib.save(density_img, density_output_name)


def mat_stream_lengths(mat, group, aff=np.eye(4)):

    length_mat = np.zeros(mat.shape, dtype=np.float)

    # by row
    for x in range(mat.shape[0]):
        # by column
        for y in range(mat.shape[1]):

            # check if entry in dict, if so, do more stuff
            if (x, y) in group:
                from dipy.tracking.utils import length

                # now we record these values
                stream_group_len = length(group[x, y])
                length_mat[x, y] = np.around(np.nanmean(list(stream_group_len)), 2)

    return length_mat


def mat_ind_along_streams(mat, group, infovol, aff=np.eye(4), stdstreamlen=50):

    mean_retmat = np.zeros(mat.shape, dtype=np.float)
    median_retmat = np.zeros(mat.shape, dtype=np.float)

    # will trim the lower %5 and top %5 of streamline
    trim_length = np.int(np.floor(stdstreamlen / 20))

    # by row
    for x in range(mat.shape[0]):
        # by column
        for y in range(mat.shape[1]):

            # check if entry in dict, if so, do more stuff
            if (x, y) in group:

                # first lets re-sample all streamlines to const length
                from dipy.tracking.streamlinespeed import set_number_of_points
                stand_stream_group = set_number_of_points(group[x, y], stdstreamlen)

                from dipy.tracking.streamline import values_from_volume
                stream_vals = values_from_volume(infovol, stand_stream_group, aff)

                vals = np.zeros(len(stream_vals))
                for ind, sv in enumerate(stream_vals):
                    vals[ind] = np.mean(sv[trim_length:-trim_length])

                mean_retmat[x, y] = np.around(np.mean(vals), 6)
                median_retmat[x, y] = np.around(np.median(vals), 6)

    return mean_retmat, median_retmat


def make_density_img(streams, maskimg, resmultiply):

    from dipy.tracking.utils import density_map

    # initialize a new affine
    new_affine = maskimg.affine

    # divide by the res multiplier
    new_affine[0, 0] = new_affine[0, 0] * 1/resmultiply
    new_affine[1, 1] = new_affine[1, 1] * 1/resmultiply
    new_affine[2, 2] = new_affine[2, 2] * 1/resmultiply

    density_data = density_map(streams,
                               vol_dims=(np.multiply(maskimg.shape, resmultiply)),
                               affine=new_affine)

    return nib.Nifti1Image(density_data, new_affine)


def make_fa_map(dwidata, maskdata, gtabdata):

    tensor_model = TensorModel(gtabdata)
    tensort_fit = tensor_model.fit(dwidata, mask=maskdata)
    fa_data = fractional_anisotropy(tensort_fit.evals)
    # just saving the FA image yo
    fa_data[np.isnan(fa_data)] = 0
    # also we can clip values outside of 0 and 1
    fa_data = np.clip(fa_data, 0, 1)

    return fa_data, tensort_fit


def act_classifier(cmdlineobj):

    from dipy.tracking.stopping_criterion import ActStoppingCriterion
    flprint("making the classifier from your segs yo\n")
    # csf, gm, wm
    csfImage = nib.load(cmdlineobj.actClasses_[0])
    csfData = csfImage.get_fdata()
    gmImage = nib.load(cmdlineobj.actClasses_[1])
    gmData = gmImage.get_fdata()
    wmImage = nib.load(cmdlineobj.actClasses_[2])
    wmData = wmImage.get_fdata()
    # make a background
    background = np.ones(csfImage.shape)
    background[(gmData + wmData + csfData) > 0] = 0
    include_map = gmData
    include_map[background > 0] = 1
    exclude_map = csfData

    return ActStoppingCriterion(include_map, exclude_map)


def ex_csv(filename, data):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    main()
