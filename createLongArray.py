#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2012-2019                               ###
###                                                                  ###
### University of California at San Francisco (UCSF), USA            ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

def createLongArray(
        name,
        n_components=1,
        n_tuples=0,
        init_to_zero=0,
        verbose=0):

    larray = vtk.vtkLongArray()
    larray.SetName(name)
    larray.SetNumberOfComponents(n_components)
    larray.SetNumberOfTuples(n_tuples)

    if (init_to_zero):
        for k_tuple in xrange(n_tuples):
            iarray.SetTuple(k_tuple, [0]*n_components)

    return larray

def createUnsignedLongArray(
        name,
        n_components=1,
        n_tuples=0,
        init_to_zero=0,
        verbose=0):

    ularray = vtk.vtkUnsignedLongArray()
    ularray.SetName(name)
    ularray.SetNumberOfComponents(n_components)
    ularray.SetNumberOfTuples(n_tuples)

    if (init_to_zero):
        for k_tuple in xrange(n_tuples):
            iarray.SetTuple(k_tuple, [0]*n_components)

    return ularray
