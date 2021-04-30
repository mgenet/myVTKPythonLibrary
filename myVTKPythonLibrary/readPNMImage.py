#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2021                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

from builtins import range

import os
import vtk

import myPythonLibrary    as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def readPNMImage(
        filename,
		extent=None,
        verbose=0):

    mypy.my_print(verbose, "*** readPNMImage: "+filename+" ***")

    if (extent is None):
        assert (os.path.isfile(filename)),\
            "Wrong filename (\""+filename+"\"). Aborting."
    else:
        assert (os.path.isfile(filename+".0")),\
            "Wrong filename (\""+filename+".0"+"\"). Aborting."

    image_reader = vtk.vtkPNMReader()
    image_reader.SetFilePattern(filename)
    if (extent is not None):
	    image_reader.SetDataExtent(extent)
    image_reader.Update()
    image = image_reader.GetOutput()

    mypy.my_print(verbose-1, "n_points = "+str(image.GetNumberOfPoints()))

    return image
