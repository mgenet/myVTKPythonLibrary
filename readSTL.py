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

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def readSTL(
        filename,
        verbose=0):

    mypy.my_print(verbose, "*** readSTL: "+filename+" ***")

    assert (os.path.isfile(filename)), "Wrong filename (\""+filename+"\"). Aborting."

    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(filename)
    stl_reader.Update()
    pdata = stl_reader.GetOutput()

    mypy.my_print(verbose-1, "n_points = "+str(pdata.GetNumberOfPoints()))
    mypy.my_print(verbose-1, "n_cells = "+str(pdata.GetNumberOfCells()))

    return pdata
