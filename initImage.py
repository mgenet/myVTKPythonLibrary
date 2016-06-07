#!/usr/bin/python
#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2016                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import myVTKPythonLibrary as myVTK

########################################################################

def initImage(
        image=None,
        image_filename=None,
        verbose=0):

    assert ((image is not None) or (image_filename is not None)), "Need an image or an image_filename. Aborting."

    if image is None:
        return myVTK.readImage(
            filename=image_filename,
            verbose=verbose)
    else:
        return image
