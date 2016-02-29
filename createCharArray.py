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

import vtk

import myVTKPythonLibrary as myVTK

########################################################################

def createCharArray(
        name,
        n_components=1,
        n_tuples=0,
        init_to_zero=0,
        verbose=1):

    carray = vtk.vtkCharArray()
    carray.SetName(name)
    carray.SetNumberOfComponents(n_components)
    carray.SetNumberOfTuples(n_tuples)

    if (init_to_zero):
        for k_tuple in xrange(n_tuples):
            iarray.SetTuple(k_tuple, [0]*n_components)

    return carray

def createUnsignedCharArray(
        name,
        n_components=1,
        n_tuples=0,
        init_to_zero=0,
        verbose=1):

    ucarray = vtk.vtkUnsignedCharArray()
    ucarray.SetName(name)
    ucarray.SetNumberOfComponents(n_components)
    ucarray.SetNumberOfTuples(n_tuples)

    if (init_to_zero):
        for k_tuple in xrange(n_tuples):
            iarray.SetTuple(k_tuple, [0]*n_components)

    return ucarray
