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
from mat_vec_tools import *

########################################################################

def clipPDataUsingPlane(
        pdata_mesh,
        plane_O,
        plane_N,
        verbose=0):

    myVTK.myPrint(verbose, "*** clipPDataUsingPlane ***")

    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_O)
    plane.SetNormal(plane_N)

    #myVTK.myPrint(verbose-1, "pdata_mesh.GetBounds() = "+str(pdata_mesh.GetBounds()))
    #myVTK.myPrint(verbose-1, "plane_O = "+str(plane_O))
    #myVTK.myPrint(verbose-1, "plane_N = "+str(plane_N))

    clip = vtk.vtkClipPolyData()
    clip.SetClipFunction(plane)
    clip.GenerateClippedOutputOn()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        clip.SetInputData(pdata_mesh)
    else:
        clip.SetInput(pdata_mesh)
    clip.Update()
    clipped0 = clip.GetOutput(0)
    clipped1 = clip.GetOutput(1)

    #myVTK.myPrint(verbose-1, "clipped0.GetNumberOfPoints() = "+str(clipped0.GetNumberOfPoints()))
    #myVTK.myPrint(verbose-1, "clipped1.GetNumberOfPoints() = "+str(clipped1.GetNumberOfPoints()))
    #myVTK.myPrint(verbose-1, "clipped0.GetNumberOfCells() = "+str(clipped0.GetNumberOfCells()))
    #myVTK.myPrint(verbose-1, "clipped1.GetNumberOfCells() = "+str(clipped1.GetNumberOfCells()))

    if (clipped0.GetNumberOfCells() > clipped1.GetNumberOfCells()):
        return clipped0, clipped1
    else:
        return clipped1, clipped0
