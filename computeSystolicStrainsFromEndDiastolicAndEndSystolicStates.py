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

import myVTKPythonLibrary as myVTK
from mat_vec_tools import *

########################################################################

def computeSystolicStrainsFromEndDiastolicAndEndSystolicStates(
        farray_F_dia,
        farray_F_sys,
        verbose=1):

    myVTK.myPrint(verbose, "*** computeSystolicStrainsFromEndDiastolicAndEndSystolicStates ***")

    n_tuples = farray_F_dia.GetNumberOfTuples()
    assert (farray_F_sys.GetNumberOfTuples() == n_tuples)

    farray_E_dia = myVTK.createFloatArray('E_dia', 6, n_tuples)
    farray_E_sys = myVTK.createFloatArray('E_sys', 6, n_tuples)
    farray_F_num = myVTK.createFloatArray('F_num', 6, n_tuples)
    farray_E_num = myVTK.createFloatArray('E_num', 6, n_tuples)

    for k_tuple in xrange(n_tuples):
        F_dia = numpy.reshape(farray_F_dia.GetTuple(k_tuple), (3,3), order='C')
        F_sys = numpy.reshape(farray_F_sys.GetTuple(k_tuple), (3,3), order='C')
        #print 'F_dia =', F_dia
        #print 'F_sys =', F_sys

        C = numpy.dot(numpy.transpose(F_dia), F_dia)
        E = (C - numpy.eye(3))/2
        farray_E_dia.SetTuple(k_tuple, mat_sym_to_vec_col(E))

        C = numpy.dot(numpy.transpose(F_sys), F_sys)
        E = (C - numpy.eye(3))/2
        farray_E_sys.SetTuple(k_tuple, mat_sym_to_vec_col(E))

        F = numpy.dot(F_sys, numpy.linalg.inv(F_dia))
        farray_F_num.SetTuple(k_tuple, numpy.reshape(F, 9, order='C'))
        #print 'F =', F

        C = numpy.dot(numpy.transpose(F), F)
        E = (C - numpy.eye(3))/2
        farray_E_num.SetTuple(k_tuple, mat_sym_to_vec_col(E))

    return (farray_E_dia,
            farray_E_sys,
            farray_F_num,
            farray_E_num)
