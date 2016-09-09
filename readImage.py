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

def readImage(
        filename,
        verbose=0):

    mypy.my_print(verbose, "*** readImage: "+filename+" ***")

    assert (os.path.isfile(filename)), "Wrong filename (\""+filename+"\"). Aborting."

    if ('vtk' in filename):
        image_reader = vtk.vtkImageDataReader()
    elif ('vti' in filename):
        image_reader = vtk.vtkXMLImageDataReader()
    else:
        assert 0, "File must be .vtk or .vti. Aborting."

    image_reader.SetFileName(filename)
    image_reader.Update()
    image = image_reader.GetOutput()

    mypy.my_print(verbose-1, "n_points = "+str(image.GetNumberOfPoints()))

    return image
