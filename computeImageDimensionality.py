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

import vtk

import myVTKPythonLibrary as myVTK

########################################################################

def computeImageDimensionality(
        image=None,
        image_filename=None,
        verbose=1):

    myVTK.myPrint(verbose, "*** computeImageDimensionality ***")

    assert ((image is not None) or (image_filename is not None)), "Need an image or an image_filename. Aborting."

    if image is None:
        image = myVTK.readImage(
            filename=image_filename,
            verbose=verbose-1)

    extent = image.GetExtent()
    dimensionality = sum([extent[2*k_dim+1]>extent[2*k_dim] for k_dim in range(3)])

    #print "dimensionality = " + str(dimensionality)

    return dimensionality
