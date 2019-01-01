__author__ = 'jfaskowitz'

import argparse
from src.dw_utils.cmdline import dwibasics
from src.dw_utils.basics import isfloat, checkisfile

# TODO make CmdLineHanler a base class that the class here would inheret from

class CmdLineHandler:

    def __init__(self):

        # initialize all the variables to empty strings
        self.dwi_ = ''
        self.mask_ = ''
        self.bvec_ = ''
        self.bval_ = ''
        self.output_ = 'out'

        self.faThr_ = 0.7
        self.shOrder_ = 6
        self.recurResp_ = False
        self.wmMask_ = ''

    def get_args(self):

        parser = argparse.ArgumentParser(description="Going to streamline track yo")

        # get the basics: dwi, bvals, bvecs, mask
        parser = dwibasics(parser)

        # the output should be provided, but I guess it could be optional
        parser.add_argument('-output', nargs='?',
                            help="Add the name of the output prefix you want")

        parser.add_argument('-fa_thr', nargs='?',
                            help="fa threshold to use")

        parser.add_argument('-sh_ord', nargs='?', type=int, choices=range(2, 14, 2),
                            help="sh order of the model yo", )

        parser.add_argument('-recur_resp', action='store_true',
                            help="if you want recursive reponse")

        parser.add_argument('-wm_mask', nargs='?',
                            help="white matter mask to use in response estimation")

        # this is a builtin function to parse the arguments of the arg parse module
        args = parser.parse_args()

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

    def check_args(self):

        checkisfile(self.dwi_)
        checkisfile(self.mask_)
        checkisfile(self.bval_)
        checkisfile(self.bvec_)

        # optional arguments, check, if not handles by arg checker
        if not isfloat(self.faThr_) or (self.faThr_ < 0.5 or self.faThr_ > 1.0):
            print("fa threshold wrong. please fix")
            exit(1)

        if self.wmMask_:
            checkisfile(self.wmMask_)

