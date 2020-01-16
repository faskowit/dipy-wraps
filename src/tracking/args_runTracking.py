__author__ = 'jfaskowitz'

import numpy as np

from src.dw_utils.cmdline import CmdLineHandler
from src.dw_utils.basics import isfloat, isint, checkisfile, flprint


class CmdLineRunTracking(CmdLineHandler):

    def __init__(self, parserdesc):

        # initialize the dwi essentials
        CmdLineHandler.__init__(self, parserdesc)

        # add application specific variables
        self.wmMask_ = ''

        # seeds
        self.seedDensity_ = 1
        self.seedPointsFile_ = ''
        self.saveSeedPoints_ = True
        self.randSeed_ = True
        self.limitTotSeeds_ = 0

        # tissue classifier
        self.actClasses_ = []
        self.faThr_ = 0.7

        # related to tracking opts
        self.tractModel_ = ''
        self.dirGttr_ = 'deterministic'
        self.trkEngine_ = 'local'

        # streamline opts
        self.stepSize_ = 0.5
        self.maxCross_ = None
        self.lenThresh_ = 20
        self.maxAngle_ = 20
        self.runCCI_ = 0

        # parcellations for con mats
        self.parcImgs_ = []

        # spherical harmonic order for model fitting
        self.shOrder_ = 6
        self.coeffFile_ = ''

        self.chunkTrack_ = False

    def get_args(self):

        # get the basics: dwi, bvals, bvecs, mask, output with object function
        self.dwibasics()

        # add application specific stuff

        # wm mask

        self.parser.add_argument('-wm_mask', nargs='?',
                                 help="white matter mask")

        # seeds stuff
        self.parser.add_argument('-seed_den', nargs='?', type=int, help="voxelwise seed density")
        self.parser.add_argument('-seed_file', nargs='?', help="seed points npz file")
        self.parser.add_argument('-save_seeds', action='store_true', help="save seed points to file")
        self.parser.add_argument('-rand_seed', action='store_true', help="voxelwise seed density")
        self.parser.add_argument('-limit_seed_count', nargs='?', help="limit total seeds to this number")
        # tissue classifier stuff
        self.parser.add_argument('-act_imgs', nargs=3, help="tissue classifier images")
        self.parser.add_argument('-fa_thr', nargs='?', help="fa threshold to use")
        # tracking stuff
        self.parser.add_argument('-tract_model', nargs='?', help="tracotraphy model you want to use: csa, csd, dti",
                                 choices=['csa', 'csd', 'dti'])
        self.parser.add_argument('-dir_gttr', nargs='?', help="direction getter to use when streamline tracking",
                                 choices=['deterministic', 'probabilistic'])
        self.parser.add_argument('-track_engine', nargs='?', help="method to propogate streamlines",
                                 choices=['local', 'particle'])
        # streamline stuff
        self.parser.add_argument('-step_size', nargs='?', help="step size to proceed in streamline tracking")
        self.parser.add_argument('-max_cross', nargs='?', help="how many possible directions to start out from seed")
        self.parser.add_argument('-len_thr', nargs='?', help="streamline length threshold")
        self.parser.add_argument('-max_angle', nargs='?',
                                 help="maximum angle to use at each procession in streamline tracking")
        self.parser.add_argument('-cci', nargs='?', help="run cluster confidence index")
        # parcellation stuff
        self.parser.add_argument('-segs', nargs='*', help="Parcellation in same space as dwi, can be multiple segs")
        # model fit stuff
        self.parser.add_argument('-sh_ord', nargs='?', type=int, choices=range(2, 14, 2),
                                 help="sh order of the model yo", )
        self.parser.add_argument('-coeff_file', nargs='?', help="sh coeff file, if coeffs are already fit")
        # runnit stuff
        self.parser.add_argument('-chunk_track', action='store_true',
                                 help="track in chunks, resampling streamlines to smaller size between each chunk")

        ################################################################################################################

        # this is a builtin function to parse the arguments of the arg parse module
        args = self.parser.parse_args()

        # get the arguments and store them as variables yo
        self.dwi_ = args.dwi
        self.mask_ = args.mask
        self.bvec_ = args.bvec
        self.bval_ = args.bval
        if args.output:
            self.output_ = args.output

        # options
        if args.wm_mask:
            self.wmMask_ = args.wm_mask

        if args.seed_den:
            self.seedDensity_ = np.int(args.seed_den)
        if args.seed_file:
            self.seedPointsFile_ = args.seed_file
        if args.save_seeds:
            self.saveSeedPoints_ = True
        if args.rand_seed:
            self.randSeed_ = True
        if args.limit_seed_count:
            self.limitTotSeeds_ = args.limit_seed_count

        if args.act_imgs:
            self.actClasses_ = args.act_imgs
        if args.fa_thr:
            self.faThr_ = np.float(args.fa_thr)

        if args.tract_model:
            self.tractModel_ = args.tract_model
        if args.dir_gttr:
            self.dirGttr_ = args.dir_gttr
        if args.track_engine:
            self.trkEngine_ = args.track_engine

        if args.step_size:
            self.stepSize_ = np.float(args.step_size)
        if args.max_cross:
            self.maxCross_ = np.int(args.max_cross)
        if args.len_thr:
            self.lenThresh_ = np.float(args.len_thr)
        if args.max_angle:
            self.maxAngle_ = np.float(args.max_angle)
        if args.cci:
            self.runCCI_ = np.float(args.cci)

        if args.segs:
            self.parcImgs_ = args.segs

        if args.sh_ord:
            self.shOrder_ = np.int(args.sh_ord)
        if args.coeff_file:
            self.coeffFile_ = args.coeff_file

        if args.chunk_track:
            self.chunkTrack_ = True

    def check_args(self):

        self.checkdwibasics()
        # check all the possible files
        if self.wmMask_:
            checkisfile(self.wmMask_)
        if self.seedPointsFile_:
            checkisfile(self.seedPointsFile_)
        if self.coeffFile_:
            checkisfile(self.coeffFile_)
        if self.actClasses_:
            for x in range(0, 3):
                checkisfile(self.actClasses_[x])
        if self.parcImgs_:
            for x in range(len(self.parcImgs_)):
                checkisfile(self.parcImgs_[x])
        # if it is set, because it can be 'None' if not set
        if self.maxCross_:
            self.maxCross_ = int(self.maxCross_)
        if self.seedDensity_ and not isint(self.seedDensity_) or self.seedDensity_ > 10:
            flprint("seed density looks wrong. exiting")
            exit(1)
        if self.limitTotSeeds_ and isint(self.limitTotSeeds_):
            flprint("limit total seeds looks wrong. exiting")
            exit(1)
        if not isfloat(self.faThr_) or (self.faThr_ < 0.5 or self.faThr_ > 1.0):
            print("fa threshold wrong. please fix")
            exit(1)
        if self.trkEngine_ == 'particle':
            if self.actClasses_ is None:
                flprint("need act classifiers for particle filtering tractography")
                exit(1)
            if self.dirGttr_ == 'deterministic':
                flprint("particle filtering needs to be probabilistic")
                exit(1)

