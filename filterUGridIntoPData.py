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

def filterUGridIntoPData(
        ugrid,
        only_trianlges=False,
        verbose=0):

    myVTK.myPrint(verbose, "*** filterUGridIntoPData ***")

    filter_geometry = vtk.vtkGeometryFilter()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        filter_geometry.SetInputData(ugrid)
    else:
        filter_geometry.SetInput(ugrid)
    filter_geometry.Update()
    pdata = filter_geometry.GetOutput()

    if (only_trianlges):
        filter_triangle = vtk.vtkTriangleFilter()
        if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
            filter_triangle.SetInputData(pdata)
        else:
            filter_triangle.SetInput(pdata)
        filter_triangle.Update()
        pdata = filter_triangle.GetOutput()

    return pdata
