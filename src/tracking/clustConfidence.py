__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University

inspiration taken from the dipy website

"""
# python imports
import numpy as np
import time
import sys
from datetime import timedelta
from sklearn.cluster import MiniBatchKMeans
from itertools import compress
import nibabel as nib
# this dipy wraps
from src.dw_utils.basics import flprint, chunker
# dipy imports
from dipy.tracking import Streamlines
from dipy.tracking.streamline import cluster_confidence


def main():
    # the main function here will interface with outside world, in case cci is
    # to be run on a completed tractogram on disk
    from src.dw_utils.basics import load_streamlines_from_file, save_trk_to_file

    # default values here
    cci_thr = 0.1
    cci_regions = 50
    cci_chunksz = 2000
    cci_rep_num = 3

    # get some command line args, the lazy way... trusting the user here...
    arguments = len(sys.argv) - 1
    flprint("the script is called with %i arguments" % arguments)
    trk_path = sys.argv[1]
    ref_path = sys.argv[2]
    ref_img = nib.load(ref_path)
    if len(sys.argv) > 3:
        cci_thr = float(sys.argv[3])
    if len(sys.argv) > 4:
        cci_regions = int(sys.argv[4])
    if len(sys.argv) > 5:
        cci_chunksz = int(sys.argv[5])
    if len(sys.argv) > 6:
        cci_rep_num = int(sys.argv[6])

    # load the tracks
    in_streamlines = load_streamlines_from_file(trk_path, ref_img)

    # run it
    flprint("running cci with thr: {}, num regions {}, chunk size {}, repetitions {}".format(
        str(cci_thr), str(cci_regions), str(cci_chunksz), str(cci_rep_num)))
    cci_streamlines, _, _ = \
        cluster_confidence_filter(in_streamlines, cci_thr,
                                  cci_reps=cci_rep_num, kregions=cci_regions, chunksize=cci_chunksz)

    # save the tracks, in the place where this is called
    out_name = ''.join(['ccistreams_thr', str(cci_thr), '.trk'])
    save_trk_to_file(cci_streamlines, ref_img, out_name)


def cluster_conf_endp(inputstreams, kregions, chunksize):
    # function to run cci based on clusters of endpoints

    start_time = time.time()

    end_p1 = [sl[0] for sl in inputstreams]
    end_p2 = [sl[-1] for sl in inputstreams]

    mini_b1 = MiniBatchKMeans(n_clusters=kregions, max_iter=10)
    mini_b2 = MiniBatchKMeans(n_clusters=kregions, max_iter=10)
    lab1 = mini_b1.fit_predict(end_p1)
    lab2 = mini_b2.fit_predict(end_p2)

    cci_res = np.zeros([len(inputstreams), 2])

    # loop through each lab
    for lab_val in range(kregions):

        flprint("\n\ncci region: {} of {}\n".format(str(lab_val + 1), str(kregions)))
        tmp_streams = Streamlines(compress(inputstreams, lab1 == lab_val))
        cci_res[lab1 == lab_val, 0] = cci_chunk(tmp_streams, ccichunksize=chunksize, ccisubsamp=8)
        tmp_streams = Streamlines(compress(inputstreams, lab2 == lab_val))
        cci_res[lab2 == lab_val, 1] = cci_chunk(tmp_streams, ccichunksize=chunksize, ccisubsamp=8)

    end_time = time.time()
    flprint("finished cci, took {}".format(str(timedelta(seconds=end_time - start_time))))

    return np.nanmax(cci_res, axis=1)


def cci_chunk(inputstreams, ccichunksize=5000, ccisubsamp=12):

    flprint("running the cci iterative with subsamp: {}".format(str(ccisubsamp)))
    # need to breakup streamlines into manageable chunks
    total_streams = len(inputstreams)

    if total_streams < ccichunksize:
        # you can just run it normally
        start_time = time.time()
        try:
            cci_results = cluster_confidence(inputstreams,
                                             subsample=ccisubsamp,
                                             max_mdf=5,
                                             override=True)
        except ValueError:
            print("caught rare value error")
            nan_vals = np.empty(len(inputstreams))
            nan_vals.fill(np.nan)
            cci_results = nan_vals

        end_time = time.time()
    else:

        random_inds = list(range(total_streams))
        # shuffle the indices in place
        np.random.shuffle(random_inds)

        stream_chunk_ind = chunker(random_inds, ccichunksize, fudgefactor=np.floor(ccichunksize / 5))

        # allocate the array
        cci_results = np.zeros(total_streams)
        start_time = time.time()

        for i in range(len(stream_chunk_ind)):
            flprint("cci iter: {} of {} with size {}".format(str(i+1),
                                                             str(len(stream_chunk_ind)),
                                                             len(stream_chunk_ind[i])))
            try:
                cci_results[stream_chunk_ind[i]] = cluster_confidence(inputstreams[stream_chunk_ind[i]],
                                                                      subsample=ccisubsamp,
                                                                      max_mdf=5,
                                                                      override=True)
            except ValueError:
                print("caught rare value error")
                nan_vals = np.empty(len(inputstreams[stream_chunk_ind[i]]))
                nan_vals.fill(np.nan)
                cci_results[stream_chunk_ind[i]] = nan_vals

        end_time = time.time()

    flprint("finished cci chunk, took {}".format(str(timedelta(seconds=end_time - start_time))))
    return cci_results


def cluster_confidence_filter(streamlines, cci_threshold, cci_reps=3, kregions=50, chunksize=5000):

    cci_iter_results = np.zeros([len(streamlines), cci_reps])
    for i in range(cci_reps):
        cci_iter_results[:, i] = cluster_conf_endp(streamlines, kregions, chunksize)

    # mean because we using the clustering on endpoints
    cci = np.nanmean(cci_iter_results, axis=1)

    cci_streamlines = Streamlines()
    num_removed = 0

    start_time = time.time()

    for i, sl in enumerate(streamlines):
        if cci[i] >= np.float(cci_threshold):
            cci_streamlines.append(sl, cache_build=True)
        else:
            num_removed += 1

    # finalize the append
    cci_streamlines.finalize_append()

    flprint("number of streamlines removed with cci: {}".format(str(num_removed)))
    end_time = time.time()
    flprint("time to create new streams: {}".format(str(timedelta(seconds=end_time - start_time))))

    return cci_streamlines, cci_iter_results, num_removed


if __name__ == "__main__":
    main()
