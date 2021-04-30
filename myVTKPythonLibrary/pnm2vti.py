#!python3
#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2021                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

from builtins import range

import argparse

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

########################################################################

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("pnm_filename", type=str)
    parser.add_argument("--extent", type=int, nargs=6, default=None)
    args = parser.parse_args()

    ext_lst = ["pbm", "pgm", "ppm"]
    assert (any("."+ext in args.pnm_filename for ext in ext_lst))

    image = myvtk.readPNMImage(
        filename=args.pnm_filename,
        extent=args.extent)

    for ext in ext_lst:
        if ("."+ext in args.pnm_filename):
           vti_filename = args.pnm_filename.split("."+ext)[0]+".vti"
    myvtk.writeImage(
        image=image,
        filename=vti_filename)
