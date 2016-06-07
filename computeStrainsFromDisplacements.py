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
import vtk

import myVTKPythonLibrary as myVTK
from mat_vec_tools import *

########################################################################

def computeStrainsFromDisplacements(
        mesh,
        disp_array_name="displacement",
        ref_mesh=None,
        verbose=0):

    myVTK.myPrint(verbose, "*** computeStrainsFromDisplacements ***")

    myVTK.myPrint(min(verbose,1), "*** Warning: at some point the ordering of vector gradient components has changed, and uses C ordering instead of F. ***")
    if   (vtk.vtkVersion.GetVTKMajorVersion() >= 8):
        ordering = "C"
    elif (vtk.vtkVersion.GetVTKMajorVersion() == 7) and ((vtk.vtkVersion.GetVTKMinorVersion() > 0) or (vtk.vtkVersion.GetVTKBuildVersion() > 0)):
        ordering = "C"
    else:
        ordering = "F"

    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    assert (mesh.GetPointData().HasArray(disp_array_name))
    mesh.GetPointData().SetActiveVectors(disp_array_name)
    cell_derivatives = vtk.vtkCellDerivatives()
    cell_derivatives.SetVectorModeToPassVectors()
    cell_derivatives.SetTensorModeToComputeGradient()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        cell_derivatives.SetInputData(mesh)
    else:
        cell_derivatives.SetInput(mesh)
    cell_derivatives.Update()
    farray_gu = cell_derivatives.GetOutput().GetCellData().GetArray("VectorGradient")

    if (ref_mesh is not None):
        farray_strain = myVTK.createFloatArray(
            name="Strain_CAR",
            n_components=6,
            n_tuples=n_cells)
    else:
        farray_strain = myVTK.createFloatArray(
            name="Strain",
            n_components=6,
            n_tuples=n_cells)
    mesh.GetCellData().AddArray(farray_strain)
    I = numpy.eye(3)
    E_vec = numpy.empty(6)
    #e_vec = numpy.empty(6)
    for k_cell in range(n_cells):
        GU = numpy.reshape(farray_gu.GetTuple(k_cell), (3,3), ordering)
        F = I + GU
        C = numpy.dot(numpy.transpose(F), F)
        E = (C - I)/2
        mat_sym33_to_vec_col6(E, E_vec)
        farray_strain.SetTuple(k_cell, E_vec)
        #if (add_almansi_strain):
            #Finv = numpy.linalg.inv(F)
            #c = numpy.dot(numpy.transpose(Finv), Finv)
            #e = (I - c)/2
            #mat_sym33_to_vec_col6(e, e_vec)
            #farray_almansi.SetTuple(k_cell, e_vec)

    if (ref_mesh is not None) and (ref_mesh.GetCellData().HasArray("eR")) and (ref_mesh.GetCellData().HasArray("eC")) and (ref_mesh.GetCellData().HasArray("eL")):
        farray_strain_cyl = myVTK.rotateMatrix(
            old_array=mesh.GetCellData().GetArray("Strain_CAR"),
            out_vecs=[ref_mesh.GetCellData().GetArray("eR"),
                      ref_mesh.GetCellData().GetArray("eC"),
                      ref_mesh.GetCellData().GetArray("eL")],
            verbose=0)
        farray_strain_cyl.SetName("Strain_CYL")
        mesh.GetCellData().AddArray(farray_strain_cyl)

    if (ref_mesh is not None) and (ref_mesh.GetCellData().HasArray("eRR")) and (ref_mesh.GetCellData().HasArray("eCC")) and (ref_mesh.GetCellData().HasArray("eLL")):
        farray_strain_pps = myVTK.rotateMatrix(
            old_array=mesh.GetCellData().GetArray("Strain_CAR"),
            out_vecs=[ref_mesh.GetCellData().GetArray("eRR"),
                      ref_mesh.GetCellData().GetArray("eCC"),
                      ref_mesh.GetCellData().GetArray("eLL")],
            verbose=0)
        farray_strain_pps.SetName("Strain_PPS")
        mesh.GetCellData().AddArray(farray_strain_pps)
