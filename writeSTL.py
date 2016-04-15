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

def writeSTL(
        pdata,
        filename,
        verbose=1):

    myVTK.myPrint(verbose, "*** writeSTL: "+filename+" ***")

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        stl_writer.SetInputData(pdata)
    else:
        stl_writer.SetInput(pdata)
    stl_writer.Update()
    stl_writer.Write()
