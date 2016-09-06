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

def readUGrid(
        filename,
        verbose=0):

    mypy.my_print(verbose, "*** readUGrid: "+filename+" ***")

    assert (os.path.isfile(filename)), "Wrong filename (\""+filename+"\"). Aborting."

    if ('vtk' in filename):
        ugrid_reader = vtk.vtkUnstructuredGridReader()
    elif ('vtu' in filename):
        ugrid_reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        assert 0, "File must be .vtk or .vtu. Aborting."

    ugrid_reader.SetFileName(filename)
    ugrid_reader.Update()
    ugrid = ugrid_reader.GetOutput()

    mypy.my_print(verbose, "n_points = "+str(ugrid.GetNumberOfPoints()))
    mypy.my_print(verbose, "n_cells = "+str(ugrid.GetNumberOfCells()))

    return ugrid
