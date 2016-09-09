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

def readPData(
        filename,
        verbose=0):

    mypy.my_print(verbose, "*** readPData: "+filename+" ***")

    assert (os.path.isfile(filename)), "Wrong filename (\""+filename+"\"). Aborting."

    if ('vtk' in filename):
        pdata_reader = vtk.vtkPolyDataReader()
    elif ('vtp' in filename):
        pdata_reader = vtk.vtkXMLPolyDataReader()
    else:
        assert 0, "File must be .vtk or .vtp. Aborting."

    pdata_reader.SetFileName(filename)
    pdata_reader.Update()
    pdata = pdata_reader.GetOutput()

    mypy.my_print(verbose-1, "n_points = "+str(pdata.GetNumberOfPoints()))
    mypy.my_print(verbose-1, "n_verts = "+str(pdata.GetNumberOfVerts()))
    mypy.my_print(verbose-1, "n_lines = "+str(pdata.GetNumberOfLines()))
    mypy.my_print(verbose-1, "n_polys = "+str(pdata.GetNumberOfPolys()))
    mypy.my_print(verbose-1, "n_strips = "+str(pdata.GetNumberOfStrips()))

    return pdata
