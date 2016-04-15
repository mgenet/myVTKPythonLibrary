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

import os
import vtk

import myVTKPythonLibrary as myVTK

########################################################################

def readSTL(
        filename,
        verbose=1):

    myVTK.myPrint(verbose, "*** readSTL: "+filename+" ***")

    assert (os.path.isfile(filename)), "Wrong filename (\""+filename+"\"). Aborting."

    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(filename)
    stl_reader.Update()
    pdata = stl_reader.GetOutput()

    myVTK.myPrint(verbose, "n_points ="+str(pdata.GetNumberOfPoints()))
    myVTK.myPrint(verbose, "n_cells ="+str(pdata.GetNumberOfCells()))

    return pdata
