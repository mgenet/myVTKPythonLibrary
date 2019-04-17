#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2012-2016                                       ###
###                                                                          ###
### University of California at San Francisco (UCSF), USA                    ###
### Swiss Federal Institute of Technology (ETH), Zurich, Switzerland         ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import numpy
import vtk

import myPythonLibrary as mypy
import myVTKPythonLibrary as myvtk

################################################################################

def compute_mask_from_mesh(ref_image_folder,
                           ref_image_basename,
                           working_folder,
                           working_basename,
                           ref_frame=0,
                           working_ext="vtu",
                           warp_mesh=1,
                           working_displacement_field_name="U",
                           binary_mask=1,
                           verbose=0):

    ref_image_zfill = len(glob.glob(ref_image_folder+"/"+ref_image_basename+"_*.vti")[0].rsplit("_")[-1].split(".")[0])
    ref_image_filename = ref_image_folder+"/"+ref_image_basename+"_"+str(ref_frame).zfill(ref_image_zfill)+".vti"
    ref_image = myvtk.readImage(
        filename=ref_image_filename)

    working_zfill = len(glob.glob(working_folder+"/"+working_basename+"_*."+working_ext)[0].rsplit("_")[-1].split(".")[0])
    n_frames = len(glob.glob(working_folder+"/"+working_basename+"_"+"[0-9]"*working_zfill+"."+working_ext))

    if binary_mask:
        thres = vtk.vtkImageThreshold()
        thres.SetInputData(ref_image)
        thres.ThresholdByUpper(0.)
        thres.SetInValue(1.0)
        thres.ReplaceInOn()
        thres.Update()
        image = thres.GetOutput()
    else:
        n_images = len(glob.glob(ref_image_folder+"/"+ref_image_basename+"_"+"[0-9]"*ref_image_zfill+".vti"))
        assert n_images == n_frames

    for k_frame in xrange(n_frames):
        mypy.my_print(verbose, "k_frame = "+str(k_frame))

        mesh = myvtk.readUGrid(
            filename=working_folder+"/"+working_basename+"_"+str(k_frame).zfill(working_zfill)+"."+working_ext)
        assert (mesh.GetPointData().HasArray(working_displacement_field_name)), "no array '" + working_displacement_field_name + "' in mesh"
        mesh.GetPointData().SetActiveVectors(working_displacement_field_name)

        if not binary_mask:
            image = myvtk.readImage(
                filename=ref_image_folder+"/"+ref_image_basename+"_"+str(k_frame).zfill(ref_image_zfill)+".vti")

        geom = vtk.vtkGeometryFilter()
        if warp_mesh:
            mesh.GetPointData().SetActiveVectors(working_displacement_field_name)
            warp = vtk.vtkWarpVector()
            warp.SetInputData(mesh)
            warp.Update()
            geom.SetInputConnection(warp.GetOutputPort())
        else:
            geom.SetInputData(mesh)
        geom.Update()

        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputConnection(geom.GetOutputPort())
        pol2stenc.SetOutputOrigin(ref_image.GetOrigin())
        pol2stenc.SetOutputSpacing(ref_image.GetSpacing())
        pol2stenc.SetOutputWholeExtent(ref_image.GetExtent())
        pol2stenc.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        myvtk.writeImage(
            image=imgstenc.GetOutput(),
            filename=working_folder+"/"+working_basename+"-mask_"+str(k_frame).zfill(working_zfill)+".vti")
