__author__ = 'jfaskowitz'

import argparse
from src.dw_utils.basics import checkisfile

# this base class will make sure that we get the basic stuff from command line args
class CmdLineHandler:

    def __init__(self, parserdesc):

        # initialize all the variables to empty strings
        self.dwi_ = ''
        self.mask_ = ''
        self.bvec_ = ''
        self.bval_ = ''
        self.output_ = 'out'

        # initialize the argparser
        self.parser = argparse.ArgumentParser(description=parserdesc)

    def dwibasics(self):
        # this is the whole dwi with all the volumes yo
        self.parser.add_argument('-dwi', nargs='?', required=False,
                                 help="path to dwi yo")
        # this is an easy binary mask that you should aready have
        self.parser.add_argument('-mask', nargs='?', required=True,
                                 help="path to mask yo")
        # this is the bvecs file, which should be eddy and flirt rotated
        self.parser.add_argument('-bvec', nargs='?', required=True,
                                 help="path to bvecs file yo")
        # this is the bval, which sometimes is the same for the whole dataset
        self.parser.add_argument('-bval', nargs='?', required=True,
                                 help="path to bvals file yo")
        # the output should be provided, but I guess it could be optional
        self.parser.add_argument('-output', nargs='?',
                                 help="Add the name of the output prefix you want")

    def checkdwibasics(self):

        if self.dwi_:
            checkisfile(self.dwi_)
        checkisfile(self.mask_)
        checkisfile(self.bval_)
        checkisfile(self.bvec_)
