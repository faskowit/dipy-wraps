#

def dwibasics(parser):

    # this is the whole dwi with all the volumes yo
    parser.add_argument('-dwi', nargs='?', required=True,
                        help="path to dwi yo")

    # this is an easy binary mask that you should aready have
    parser.add_argument('-mask', nargs='?', required=True,
                        help="path to mask yo")

    # this is the bvecs file, which should be eddy and flirt rotated
    parser.add_argument('-bvec', nargs='?', required=True,
                        help="path to bvecs file yo")

    # this is the bval, which sometimes is the same for the whole dataset
    parser.add_argument('-bval', nargs='?', required=True,
                        help="path to bvals file yo")

    return parser
