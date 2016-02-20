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

def computeImageGradient(
        image,
        verbose=1):

    myVTK.myPrint(verbose, "*** computeImageGradient ***")

    image_gradient = vtk.vtkImageGradient()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        image_gradient.SetInputData(image)
    else:
        image_gradient.SetInput(image)
    image_gradient.Update()
    image = image_gradient.GetOutput()

    return image

