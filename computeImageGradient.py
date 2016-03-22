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
    extent = image.GetExtent()
    DX = extent[1]+1-extent[0]
    DY = extent[3]+1-extent[2]
    DZ = extent[5]+1-extent[4]
    if   (DX > 1) and (DY > 1) and (DZ > 1):
        image_gradient.SetDimensionality(3)
    elif (DX > 1) and (DY > 1) and (DZ == 1):
        image_gradient.SetDimensionality(2)
    elif (DX > 1) and (DY == 1) and (DZ == 1):
        image_gradient.SetDimensionality(1)
    else:
        assert (0), "Wrong image dimensionality ("+str(extent)+")"
    image_gradient.Update()
    image = image_gradient.GetOutput()

    return image
