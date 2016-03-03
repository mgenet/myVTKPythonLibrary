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
        displacement_array_name="displacement",
        ref_mesh=None,
        verbose=1):

    myVTK.myPrint(verbose, "*** computeStrainsFromDisplacements ***")

    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    assert (mesh.GetPointData().HasArray(displacement_array_name))
    mesh.GetPointData().SetActiveVectors(displacement_array_name)
    cell_derivatives = vtk.vtkCellDerivatives()
    cell_derivatives.SetVectorModeToPassVectors()
    cell_derivatives.SetTensorModeToComputeGradient()
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        cell_derivatives.SetInputData(mesh)
    else:
        cell_derivatives.SetInput(mesh)
    cell_derivatives.Update()
    farray_gu = cell_derivatives.GetOutput().GetCellData().GetArray("VectorGradient")

    if (ref_mesh is None):
        farray_strain = myVTK.createFloatArray(
            name="Strain",
            n_components=6,
            n_tuples=n_cells)
    else:
        farray_strain = myVTK.createFloatArray(
            name="Strain_CAR",
            n_components=6,
            n_tuples=n_cells)
    mesh.GetCellData().AddArray(farray_strain)
    for k_cell in range(n_cells):
        GU = numpy.reshape(farray_gu.GetTuple(k_cell), (3,3), "F")
        E = (GU + numpy.transpose(GU) + numpy.dot(numpy.transpose(GU), GU))/2
        farray_strain.SetTuple(k_cell, mat_sym_to_vec_col(E))

    if (ref_mesh is not None):
        assert (ref_mesh.GetCellData().HasArray("eR"))
        assert (ref_mesh.GetCellData().HasArray("eC"))
        assert (ref_mesh.GetCellData().HasArray("eL"))
        assert (ref_mesh.GetCellData().HasArray("eRR"))
        assert (ref_mesh.GetCellData().HasArray("eCC"))
        assert (ref_mesh.GetCellData().HasArray("eLL"))

        farray_strain_cyl = myVTK.rotateSymmetricMatrix(
            old_array=mesh.GetCellData().GetArray("Strain_CAR"),
            out_vecs=[ref_mesh.GetCellData().GetArray("eR"),
                      ref_mesh.GetCellData().GetArray("eC"),
                      ref_mesh.GetCellData().GetArray("eL")],
            verbose=0)
        farray_strain_cyl.SetName("Strain_CYL")
        mesh.GetCellData().AddArray(farray_strain_cyl)

        farray_strain_pps = myVTK.rotateSymmetricMatrix(
            old_array=mesh.GetCellData().GetArray("Strain_CAR"),
            out_vecs=[ref_mesh.GetCellData().GetArray("eRR"),
                      ref_mesh.GetCellData().GetArray("eCC"),
                      ref_mesh.GetCellData().GetArray("eLL")],
            verbose=0)
        farray_strain_pps.SetName("Strain_PPS")
        mesh.GetCellData().AddArray(farray_strain_pps)

