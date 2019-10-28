__author__ = 'jfaskowitz'

from src.dw_utils.cmdline import CmdLineHandler
from src.dw_utils.basics import isfloat, checkisfile

class CmdLineFitFORECAST(CmdLineHandler):

    def __init__(self, parserdesc):

        # initalize the dwi essentials
        CmdLineHandler.__init__(self, parserdesc)

        # add application specific variables
        self.shOrder_ = 6
        self.recurResp_ = False
        self.wmMask_ = ''

    def get_args(self):

        # get the basics: dwi, bvals, bvecs, mask, output with object function
        self.dwibasics()

        self.parser.add_argument('-sh_ord', nargs='?', type=int, choices=range(2, 14, 2),
                                 help="sh order of the model yo", )

        self.parser.add_argument('-recur_resp', action='store_true',
                                 help="if you want recursive reponse")

        self.parser.add_argument('-wm_mask', nargs='?',
                                 help="white matter mask to use in response estimation")

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

        if args.sh_ord:
            self.shOrder_ = args.sh_ord
        if args.recur_resp:
            self.recurResp_ = True
        if args.wm_mask:
            self.wmMask_ = args.wm_mask

    def check_args(self):

        self.checkdwibasics()

        if self.wmMask_:
            checkisfile(self.wmMask_)

