#!/usr/bin/python
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

import numpy

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def moveMeshWithWorldMatrix(
        mesh,
        M,
        verbose=0):

    mypy.my_print(verbose, "*** moveMeshWithWorldMatrix ***")

    n_points = mesh.GetNumberOfPoints()
    points = mesh.GetPoints()
    P = numpy.empty(4)

    for k_point in xrange(n_points):
        P[0:3] = points.GetPoint(k_point)
        P[3] = 1.
        #print P

        P = numpy.dot(M, P)
        #print new_P

        points.SetPoint(k_point, P[0:3])
