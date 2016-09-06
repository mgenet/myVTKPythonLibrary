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

import argparse
import numpy

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

########################################################################

if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_in_filename', type=str)
    parser.add_argument('image_in', type=str)
    parser.add_argument('image_out', type=str)
    parser.add_argument('--mesh_out_filename', type=str, default=None)
    parser.add_argument('-v', '--verbose', type=int, default=1)
    args = parser.parse_args()

    S_in = numpy.diag(list(numpy.loadtxt("Image_"+args.image_in+"_Scaling.dat"))+[1])
    W_in = numpy.loadtxt("Image_"+args.image_in+"_WorldMatrix.dat")
    I2W_in = numpy.dot(W_in, numpy.linalg.inv(S_in))

    S_out = numpy.diag(list(numpy.loadtxt("Image_"+args.image_out+"_Scaling.dat"))+[1])
    W_out = numpy.loadtxt("Image_"+args.image_out+"_WorldMatrix.dat")
    I2W_out = numpy.dot(W_out, numpy.linalg.inv(S_out))

    I2I = numpy.dot(numpy.linalg.inv(I2W_out), I2W_in)

    if (args.mesh_in_filename.endswith(".vtk") or args.mesh_in_filename.endswith(".vtu")):
        mesh = myvtk.readUGrid(
            filename=args.mesh_in_filename,
            verbose=args.verbose)
    elif (args.mesh_in_filename.endswith(".stl")):
        mesh = myvtk.readSTL(
            filename=args.mesh_in_filename,
            verbose=args.verbose)
    else:
        assert (0), "Input file must be .vtk, .vtu, or .stl. Aborting."

    myvtk.moveMeshWithWorldMatrix(
        mesh=mesh,
        M=I2I,
        verbose=args.verbose)

    if (args.mesh_out_filename == None):
        args.mesh_out_filename = args.mesh_in_filename.replace(args.image_in, args.image_out)

    if (args.mesh_out_filename.endswith(".vtk") or args.mesh_out_filename.endswith(".vtu")):
        myvtk.writeUGrid(
            ugrid=mesh,
            filename=args.mesh_out_filename,
            verbose=args.verbose)
    elif (args.mesh_out_filename.endswith(".stl")):
        myvtk.writeSTL(
            pdata=mesh,
            filename=args.mesh_out_filename,
            verbose=args.verbose)
    else:
        assert (0), "Output file must be .vtk, .vtu, or .stl. Aborting."
