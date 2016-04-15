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

def readPData(
        filename,
        verbose=1):

    myVTK.myPrint(verbose, "*** readPData: "+filename+" ***")

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

    myVTK.myPrint(verbose, "n_points = "+str(pdata.GetNumberOfPoints()))
    myVTK.myPrint(verbose, "n_verts = "+str(pdata.GetNumberOfVerts()))
    myVTK.myPrint(verbose, "n_lines = "+str(pdata.GetNumberOfLines()))
    myVTK.myPrint(verbose, "n_polys = "+str(pdata.GetNumberOfPolys()))
    myVTK.myPrint(verbose, "n_strips = "+str(pdata.GetNumberOfStrips()))

    return pdata
