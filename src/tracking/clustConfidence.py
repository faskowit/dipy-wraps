__author__ = 'jfaskowitz'

"""

josh faskowitz
Indiana University
University of Southern California

inspiration taken from the dipy website

"""
# python imports
import numpy as np
import time
from datetime import timedelta
from sklearn.cluster import MiniBatchKMeans
# this dipy wraps
from src.dw_utils.basics import flprint, chunker
# dipy imports
from dipy.tracking import Streamlines
from dipy.tracking.streamline import cluster_confidence


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

    from itertools import compress

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


def cluster_confidence_filter(streamlines, cci_threshold, kregions=50, chunksize=5000):

    num_cci_repetitions = 3
    cci_iter_results = np.zeros([len(streamlines), num_cci_repetitions])

    for i in range(num_cci_repetitions):
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

    # # old_stream = streamlines
    # streamlines = cci_streamlines
    #
    # cci_results_name = ''.join([command_line.output_, 'cciresults.npz'])
    #
    # # save mask array, also save the affine corresponding to this space
    # np.savez_compressed(cci_results_name, cci_iter_results)

    return cci_streamlines, cci_iter_results, num_removed
