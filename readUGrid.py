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

def readUGrid(
        filename,
        verbose=1):

    myVTK.myPrint(verbose, "*** readUGrid: "+filename+" ***")

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

    myVTK.myPrint(verbose-1, "n_points ="+str(ugrid.GetNumberOfPoints()))
    myVTK.myPrint(verbose-1, "n_cells ="+str(ugrid.GetNumberOfCells()))

    return ugrid
