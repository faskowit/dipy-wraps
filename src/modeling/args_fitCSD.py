__author__ = 'jfaskowitz'

from src.dw_utils.cmdline import CmdLineHandler
from src.dw_utils.basics import isfloat, checkisfile

class CmdLineFitCSD(CmdLineHandler):

    def __init__(self, parserdesc):

        # initalize the dwi essentials
        CmdLineHandler.__init__(self, parserdesc)

        # add application specific variables
        self.faThr_ = 0.7
        self.shOrder_ = 6
        self.recurResp_ = False
        self.wmMask_ = ''
        self.actClasses_ = ''

    def get_args(self):

        # get the basics: dwi, bvals, bvecs, mask, output with object function
        self.dwibasics()

        # add application specific stuff
        self.parser.add_argument('-fa_thr', nargs='?',
                                 help="fa threshold to use")

        self.parser.add_argument('-sh_ord', nargs='?', type=int, choices=range(2, 14, 2),
                                 help="sh order of the model yo", )

        self.parser.add_argument('-recur_resp', action='store_true',
                                 help="if you want recursive reponse")

        self.parser.add_argument('-wm_mask', nargs='?',
                                 help="white matter mask to use in response estimation")

        self.parser.add_argument('-act_imgs', nargs=3,
                                 help="tissue images for multi-shell fit, implies multitissue fit")

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
        if args.fa_thr:
            self.faThr_ = args.fa_thr
        if args.sh_ord:
            self.shOrder_ = args.sh_ord
        if args.recur_resp:
            self.recurResp_ = True
        if args.wm_mask:
            self.wmMask_ = args.wm_mask
        if args.act_imgs:
            self.actClasses_ = args.act_imgs

    def check_args(self):

        self.checkdwibasics()

        # optional arguments, check, if not handles by arg checker
        if not isfloat(self.faThr_) or (self.faThr_ < 0.5 or self.faThr_ > 1.0):
            print("fa threshold wrong. please fix")
            exit(1)

        if self.wmMask_:
            checkisfile(self.wmMask_)

