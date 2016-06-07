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

def createImageInterpolator(
        image,
        mode="linear",
        out_value=None,
        verbose=0):

    myVTK.myPrint(verbose, "*** createImageInterpolator ***")

    interpolator = vtk.vtkImageInterpolator()
    assert (mode in ("nearest", "linear", "cubic"))
    if (mode == "nearest"):
        interpolator.SetInterpolationModeToNearest()
    elif (mode == "linear"):
        interpolator.SetInterpolationModeToLinear()
    elif (mode == "cubic"):
        interpolator.SetInterpolationModeToCubic()
    if (out_value is not None):
        interpolator.SetOutValue(out_value)
    interpolator.Initialize(image)
    interpolator.Update()

    return interpolator

