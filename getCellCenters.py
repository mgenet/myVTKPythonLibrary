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

def getCellCenters(
        mesh,
        verbose=1):

    myVTK.myPrint(verbose, "*** getCellCenters ***")

    filter_cell_centers = vtk.vtkCellCenters()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        filter_cell_centers.SetInputData(mesh)
    else:
        filter_cell_centers.SetInput(mesh)
    filter_cell_centers.Update()

    return filter_cell_centers.GetOutput()
