__author__ = 'jfaskowitz'

from src.dw_utils.cmdline import CmdLineHandler
from src.dw_utils.basics import isfloat, checkisfile

class CmdLineFitMAPMRI(CmdLineHandler):

    def __init__(self, parserdesc):

        # initalize the dwi essentials
        CmdLineHandler.__init__(self, parserdesc)

        # add application specific variables
        self.radialOrder_ = 6
        self.bigDelta_ = None
        self.smallDelta_ = None

    def get_args(self):

        # get the basics: dwi, bvals, bvecs, mask, output with object function
        self.dwibasics()

        # add application specific stuff
        self.parser.add_argument('-big_delta', nargs='?',
                                 help="big diffusion time",
                                 type=float)

        self.parser.add_argument('-small_delta', nargs='?',
                                 help="small diffusion time",
                                 type=float)

        self.parser.add_argument('-radial_ord', nargs='?', type=int, 
                                 choices=range(2, 14, 2),
                                 help="sh order of the model yo", )

        # this is a builtin function to parse the arguments of the arg parse module
        args = self.parser.parse_args()

        # get the arguments and store them as variables yo
        self.dwi_ = args.dwi
        self.mask_ = args.mask
        self.bvec_ = args.bvec
        self.bval_ = args.bval

        # optional arguemnts
        if args.output:
            self.output_ = args.output
        if args.radial_ord:
            self.radialOrder_ = args.radial_ord
        if args.big_delta:
            self.bigDelta_ = args.big_delta
        if args.small_delta:
            self.smallDelta_ = args.small_delta

    def check_args(self):

        self.checkdwibasics()
