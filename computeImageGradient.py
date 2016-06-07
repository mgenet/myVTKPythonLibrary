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

import argparse
import vtk

import myVTKPythonLibrary as myVTK

########################################################################

def computeImageGradient(
        image=None,
        image_filename=None,
        verbose=0):

    myVTK.myPrint(verbose, "*** computeImageGradient ***")

    image = myVTK.initImage(image, image_filename, verbose-1)

    image_dimensionality = myVTK.computeImageDimensionality(
        image=image,
        verbose=verbose-1)

    image_gradient = vtk.vtkImageGradient()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        image_gradient.SetInputData(image)
    else:
        image_gradient.SetInput(image)
    image_gradient.SetDimensionality(image_dimensionality)
    image_gradient.Update()
    image_w_grad = image_gradient.GetOutput()

    return image_w_grad

########################################################################

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_filename", type=str)
    parser.add_argument("--verbose", "-v", type=int, default=1)
    args = parser.parse_args()

    image = myVTK.readImage(
        filename=args.image_filename,
        verbose=args.verbose)

    image_w_grad = myVTK.computeImageGradient(
        image=image,
        verbose=args.verbose)

    myVTK.writeImage(
        image=image_w_grad,
        filename=args.image_filename.replace(".vti", "-gradient.vti"),
        verbose=args.verbose)
