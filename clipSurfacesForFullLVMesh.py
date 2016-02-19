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

import sys
import vtk

import myVTKPythonLibrary as myVTK
from mat_vec_tools import *

########################################################################

def clipSurfacesForFullLVMesh(
        endo,
        epi,
        verbose=1):

    myVTK.myPrint(verbose, "*** clipSurfacesForFullLVMesh ***")

    endo_implicit_distance = vtk.vtkImplicitPolyDataDistance()
    endo_implicit_distance.SetInput(endo)

    epi_implicit_distance = vtk.vtkImplicitPolyDataDistance()
    epi_implicit_distance.SetInput(epi)

    epi_clip = vtk.vtkClipPolyData()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        epi_clip.SetInputData(epi)
    else:
        epi_clip.SetInput(epi)
    epi_clip.SetClipFunction(endo_implicit_distance)
    epi_clip.GenerateClippedOutputOn()
    epi_clip.Update()
    clipped_epi = epi_clip.GetOutput(0)
    clipped_valve = epi_clip.GetOutput(1)

    endo_clip = vtk.vtkClipPolyData()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        endo_clip.SetInputData(endo)
    else:
        endo_clip.SetInput(endo)
    endo_clip.SetClipFunction(epi_implicit_distance)
    endo_clip.InsideOutOn()
    endo_clip.Update()
    clipped_endo = endo_clip.GetOutput(0)

    return (clipped_endo,
            clipped_epi,
            clipped_valve)










