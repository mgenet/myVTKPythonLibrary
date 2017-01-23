#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2017                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import numpy
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def rotatePoints(
        points,
        C,
        R,
        verbose=0):

    mypy.my_print(verbose, "*** rotatePoints ***")

    n_points = points.GetNumberOfPoints()

    point = numpy.empty(3)
    for k_point in xrange(n_points):
        points.GetPoint(k_point, point)
        #print point

        point = C + numpy.dot(R, point - C)
        #print new_point

        points.SetPoint(k_point, point)
