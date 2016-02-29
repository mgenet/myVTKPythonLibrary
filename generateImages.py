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

import math
import numpy
import os
import random
import vtk

import myVTKPythonLibrary as myVTK

########################################################################

def I0_structure(X, images, structure):
    if (structure["type"] == "no"):
        return 1.
    elif (structure["type"] == "heart"):
        #print "X = " + str(X)
        X = numpy.array(X[0:2])-images["L"]/2
        #print "X = " + str(X)
        R = numpy.linalg.norm(X)
        #print "R = " + str(R)
        if (R >= structure["Ri"]) and (R <= structure["Re"]):
            return 1.
        else:
            return 0.
    else:
        assert (0), "structure type must be \"no\" or \"cylinder\". Aborting."

def I0_texture(X, images, texture):
    if (texture["type"] == "no"):
        return 1.
    elif (texture["type"] == "sine"):
        if   (images["n_dim"] == 1):
            return math.sin(math.pi*X[0]/images["L"])**2
        elif (images["n_dim"] == 2):
            return math.sin(math.pi*X[0]/images["L"])**2 * math.sin(math.pi*X[1]/images["L"])**2
        elif (images["n_dim"] == 3):
            return math.sin(math.pi*X[0]/images["L"])**2 * math.sin(math.pi*X[1]/images["L"])**2 * math.sin(math.pi*X[2]/images["L"])**2
    elif (texture["type"] == "tagging"):
        if   (images["n_dim"] == 1):
            return math.sin(math.pi*X[0]/texture["s"])**2
        elif (images["n_dim"] == 2):
            return math.sin(math.pi*X[0]/texture["s"])**2 * math.sin(math.pi*X[1]/texture["s"])**2
        elif (images["n_dim"] == 3):
            return math.sin(math.pi*X[0]/texture["s"])**2 * math.sin(math.pi*X[1]/texture["s"])**2 * math.sin(math.pi*X[2]/texture["s"])**2
    else:
        assert (0), "texture type must be \"no\", \"sine\" or \"tagging\". Aborting."

def I0_noise(noise):
    if (noise["type"] == "no"):
        return 0
    elif (noise["type"] == "normal"):
        return random.normalvariate(noise["avg"], noise["std"])
    else:
        assert (0), "noise type must be \"no\" or \"normal\". Aborting."

def I0(X, images, structure, texture, noise):
    return I0_structure(X, images, structure) * I0_texture(X, images, texture) + I0_noise(noise)

########################################################################

def phi(t, evolution):
    if (evolution["type"] == "linear"):
        return t
    elif (evolution["type"] == "sine"):
        return math.sin(math.pi*t/evolution["T"])**2
    else:
        assert (0), "evolution type must be \"linear\" or \"sine\". Aborting."

def X(x, t, images, structure, deformation, evolution):
    if (deformation["type"] == "no"):
        return x
    elif (deformation["type"] == "homogeneous"):
        Exx = deformation["Exx"]*phi(t, evolution) if ("Exx" in deformation.keys()) else 0.
        Eyy = deformation["Eyy"]*phi(t, evolution) if ("Eyy" in deformation.keys()) else 0.
        Ezz = deformation["Ezz"]*phi(t, evolution) if ("Ezz" in deformation.keys()) else 0.
        Exy = deformation["Exy"]*phi(t, evolution) if ("Exy" in deformation.keys()) else 0.
        Eyx = deformation["Eyx"]*phi(t, evolution) if ("Eyx" in deformation.keys()) else 0.
        Exz = deformation["Exz"]*phi(t, evolution) if ("Exz" in deformation.keys()) else 0.
        Ezx = deformation["Ezx"]*phi(t, evolution) if ("Ezx" in deformation.keys()) else 0.
        Eyz = deformation["Eyz"]*phi(t, evolution) if ("Eyz" in deformation.keys()) else 0.
        Ezy = deformation["Ezy"]*phi(t, evolution) if ("Ezy" in deformation.keys()) else 0.
        F = numpy.array([[math.sqrt(1.+Exx),              Exy ,              Exz ],
                         [             Eyx , math.sqrt(1.+Eyy),              Eyz ],
                         [             Ezx ,              Ezy , math.sqrt(1.+Ezz)]])
        Finv = numpy.linalg.inv(F)

        X = numpy.dot(Finv, x)
        return X
    elif (deformation["type"] == "heart"):
        assert (structure["type"] == "heart"), "structure type must be cylinder for heart deformation. Aborting."
        Ri = structure["Ri"]
        Re = structure["Re"]
        dRi = deformation["dRi"]*phi(t, evolution) if ("dRi" in deformation.keys()) else 0.
        dRe = deformation["dRi"]*phi(t, evolution) if ("dRi" in deformation.keys()) else 0.
        dTi = deformation["dTi"]*phi(t, evolution) if ("dTi" in deformation.keys()) else 0.
        dTe = deformation["dTe"]*phi(t, evolution) if ("dTe" in deformation.keys()) else 0.
        A = numpy.array([[1.-(dRi-dRe)/(Re-Ri), 0.],
                         [  -(dTi-dTe)/(Re-Ri), 1.]])
        Ainv = numpy.linalg.inv(A)
        B = numpy.array([[(1.+Ri/(Re-Ri))*dRi-Ri/(Re-Ri)*dRe],
                         [(1.+Ri/(Re-Ri))*dTi-Ri/(Re-Ri)*dTe]])
        #print "A = " + str(A)
        #print "Ainv = " + str(Ainv)
        #print "B = " + str(B)
        x = numpy.array(x[0:2])-images["L"]/2
        #print "x = " + str(x)
        rr = numpy.linalg.norm(x)
        #print "rr = " + str(rr)
        tt = math.atan2(x[1], x[0])
        #print "tt = " + str(tt)
        [RR,
         TT] = numpy.dot(Ainv, numpy.array([[rr],
                                            [tt]])-B)
        #print "RR = " + str(RR)
        #print "TT = " + str(TT)
        if   (images["n_dim"] == 2):
            X = numpy.array([[RR * math.cos(TT)],
                             [RR * math.sin(TT)]])
        elif (images["n_dim"] == 3):
            X = numpy.array([[RR * math.cos(TT)],
                             [RR * math.sin(TT)],
                             [              0. ]])
        else:
            assert (0), "n_dim must be 2 or 3. Aborting."
        #print "X = " + str(X)
        X[0:2] += images["L"]/2
        #print "X = " + str(X)
        return X
    else:
        assert (0), "deformation type must be \"no\", \"homogeneous\" or \"heart\". Aborting."

########################################################################

def generateImages(
        images,
        structure,
        texture,
        noise,
        deformation,
        evolution,
        verbose=1):

    myVTK.myPrint(verbose, "*** generateImages ***")

    image = vtk.vtkImageData()

    if   (images["n_dim"] == 1):
        image.SetExtent([0, images["n_voxels"]-1, 0,                    0, 0,                    0])
    elif (images["n_dim"] == 2):
        image.SetExtent([0, images["n_voxels"]-1, 0, images["n_voxels"]-1, 0,                    0])
    elif (images["n_dim"] == 3):
        image.SetExtent([0, images["n_voxels"]-1, 0, images["n_voxels"]-1, 0, images["n_voxels"]-1])
    else:
        assert (0), "n_dim must be 1, 2 or 3. Aborting."

    spacing = images["L"]/(numpy.array(image.GetExtent()[1::2])+1)
    image.SetSpacing(spacing)

    origin = numpy.array(image.GetSpacing())/2
    if   (images["n_dim"] == 1):
        origin[1] = 0.
        origin[2] = 0.
    elif (images["n_dim"] == 2):
        origin[2] = 0.
    elif (images["n_dim"] == 3):
        pass
    else:
        assert (0), "n_dim must be 1, 2 or 3. Aborting."
    image.SetOrigin(origin)

    array_data = myVTK.createFloatArray(
        name="scalars",
        n_components=1,
        n_tuples=image.GetNumberOfPoints())
    image.GetPointData().AddArray(array_data)
    image.GetPointData().SetActiveScalars("scalars")

    if not os.path.exists(images["images_folder"]):
        os.mkdir(images["images_folder"])

    global_min = float("+Inf")
    global_max = float("-Inf")
    for k_frame in xrange(images["n_frames"]+1):
        t = images["T"]*float(k_frame)/images["n_frames"]
        for k_point in xrange(image.GetNumberOfPoints()):
            x = image.GetPoint(k_point)
            I = I0(X(x, t, images, structure, deformation, evolution), images, structure, texture, noise)
            array_data.SetTuple(k_point, [I])
            if (I < global_min): global_min = I
            if (I > global_max): global_max = I
        myVTK.writeImage(
            image=image,
            filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")

    if (images["data_type"] in ("float")):
        pass
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "uint8", "uint16", "uint32", "uint64")):
        print "global_min = " + str(global_min)
        print "global_max = " + str(global_max)
        shifter = vtk.vtkImageShiftScale()
        shifter.SetShift(-global_min)
        if   (images["data_type"] in ("unsigned char", "uint8")):
            shifter.SetScale(float(2**8-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedChar()
        elif (images["data_type"] in ("unsigned short", "uint16")):
            shifter.SetScale(float(2**16-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedShort()
        elif (images["data_type"] in ("unsigned int", "uint32")):
            shifter.SetScale(float(2**32-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedInt()
        elif (images["data_type"] in ("unsigned long", "uint64")):
            shifter.SetScale(float(2**64-1)/(global_max-global_min))
            shifter.SetOutputScalarTypeToUnsignedLong()
        for k_frame in xrange(images["n_frames"]+1):
            image = myVTK.readImage(filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")
            if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
                shifter.SetInputData(image)
            else:
                shifter.SetInput(image)
            shifter.Update()
            image = shifter.GetOutput()
            myVTK.writeImage(
                image=image,
                filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")
    else:
        assert (0), "Wrong data type. Aborting."

