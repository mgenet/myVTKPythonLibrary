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

class ImageData():
    def __init__(self, images, structure, texture, noise):
        self.images = images

        self.structure = structure
        if (self.structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no
        elif (self.structure["type"] == "heart"):
            assert (self.images["n_dim"] in (2,3)), "n_dim must be \"2\" or \"3 for \"heart\" type structure. Aborting."
            self.X_inplane = numpy.array([float()]*2)
            self.R = float()
            self.I0_structure = self.I0_structure_heart
        else:
            assert (0), "structure type must be \"no\" or \"heart\". Aborting."

        self.texture = texture
        if (self.texture["type"] == "no"):
            self.I0_texture = self.I0_texture_no
        elif (self.texture["type"] == "sine"):
            if   (self.images["n_dim"] == 1):
                self.I0_texture = self.I0_texture_sine_X
            elif (self.images["n_dim"] == 2):
                self.I0_texture = self.I0_texture_sine_XY
            elif (self.images["n_dim"] == 3):
                self.I0_texture = self.I0_texture_sine_XYZ
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
        elif (self.texture["type"] == "sinX"):
            self.I0_texture = self.I0_texture_sine_X
        elif (self.texture["type"] == "sinY"):
            self.I0_texture = self.I0_texture_sine_Y
        elif (self.texture["type"] == "sinZ"):
            self.I0_texture = self.I0_texture_sine_Z
        elif (self.texture["type"] == "tagging"):
            if   (self.images["n_dim"] == 1):
                self.I0_texture = self.I0_texture_tagging_X
            elif (self.images["n_dim"] == 2):
                self.I0_texture = self.I0_texture_tagging_XY
            elif (self.images["n_dim"] == 3):
                self.I0_texture = self.I0_texture_tagging_XYZ
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
        elif (self.texture["type"] == "taggX"):
            self.I0_texture = self.I0_texture_tagging_X
        elif (self.texture["type"] == "taggY"):
            self.I0_texture = self.I0_texture_tagging_Y
        elif (self.texture["type"] == "taggZ"):
            self.I0_texture = self.I0_texture_tagging_Z
        else:
            assert (0), "texture type must be \"no\", \"sine\", \"sinX\", \"sinY\", \"sinZ\", \"tagging\", \"taggX\", \"taggY\" or \"taggZ\". Aborting."

        self.noise = noise
        if (self.noise["type"] == "no"):
            self.I0_noise = self.I0_noise_no
        elif (self.noise["type"] == "normal"):
            self.I0_noise = self.I0_noise_normal
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def I0(self, X):
        return self.I0_structure(X) * self.I0_texture(X) + self.I0_noise()

    def I0_structure_no(self, X):
        return 1.

    def I0_structure_heart(self, X):
        self.X_inplane[0] = X[0]-self.images["L"][0]/2
        self.X_inplane[1] = X[1]-self.images["L"][1]/2
        self.R = numpy.linalg.norm(self.X_inplane)
        if (self.R >= self.structure["Ri"]) and (self.R <= self.structure["Re"]):
            return 1.
        else:
            return 0.

    def I0_texture_no(self, X):
        return 1.

    def I0_texture_sine_X(self, X):
        return math.sin(math.pi*X[0]/self.images["L"][0])**2

    def I0_texture_sine_Y(self, X):
        return math.sin(math.pi*X[1]/self.images["L"][1])**2

    def I0_texture_sine_Z(self, X):
        return math.sin(math.pi*X[2]/self.images["L"][2])**2

    def I0_texture_sine_XY(self, X):
        return self.I0_texture_sine_X(X) * self.I0_texture_sine_X(Y)

    def I0_texture_sine_XYZ(self, X):
        return self.I0_texture_sine_X(X) * self.I0_texture_sine_X(Y) * self.I0_texture_sine_Z(Y)

    def I0_texture_tagging_X(self, X):
        return math.sin(math.pi*X[0]/self.texture["s"])**2

    def I0_texture_tagging_Y(self, X):
        return math.sin(math.pi*X[1]/self.texture["s"])**2

    def I0_texture_tagging_Z(self, X):
        return math.sin(math.pi*X[2]/self.texture["s"])**2

    def I0_texture_tagging_XY(self, X):
        return self.I0_texture_tagging_X(X) * self.I0_texture_tagging_Y(X)

    def I0_texture_tagging_XYZ(self, X):
        return self.I0_texture_tagging_X(X) * self.I0_texture_tagging_Y(X) * self.I0_texture_tagging_Z(X)

    def I0_noise_no(self):
        return 0

    def I0_noise_normal(self):
        return random.normalvariate(self.noise["avg"], self.noise["std"])

########################################################################

class Mapping:
    def __init__(self, images, structure, deformation, evolution):
        self.images = images

        self.structure = structure

        self.deformation = deformation
        if (self.deformation["type"] == "no"):
            self.init_t = self.init_t_no
            self.X = self.X_no
        elif (self.deformation["type"] == "homogeneous"):
            self.init_t = self.init_t_homogeneous
            self.X = self.X_homogeneous
        elif (self.deformation["type"] == "heart"):
            assert (self.structure["type"] == "heart"), "structure type must be \"heart\" for \"heart\" type deformation. Aborting."
            self.init_t = self.init_t_heart
            self.x_inplane = numpy.array([float()]*2)
            self.rt = numpy.array([float()]*2)
            self.RT = numpy.array([float()]*2)
            self.X_full = numpy.array([float()]*3)
            self.X = self.X_heart
        else:
            assert (0), "deformation type must be \"no\", \"homogeneous\" or \"heart\". Aborting."

        self.evolution = evolution
        if (self.evolution["type"] == "linear"):
            self.phi = self.phi_linear
        elif (self.evolution["type"] == "sine"):
            self.phi = self.phi_sine
        else:
            assert (0), "evolution type must be \"linear\" or \"sine\". Aborting."

    def phi_linear(self, t):
        return t

    def phi_sine(self, t):
        return math.sin(math.pi*t/self.evolution["T"])**2

    def init_t_no(self, t):
        pass

    def init_t_homogeneous(self, t):
        Exx = self.deformation["Exx"]*self.phi(t) if ("Exx" in self.deformation.keys()) else 0.
        Eyy = self.deformation["Eyy"]*self.phi(t) if ("Eyy" in self.deformation.keys()) else 0.
        Ezz = self.deformation["Ezz"]*self.phi(t) if ("Ezz" in self.deformation.keys()) else 0.
        Exy = self.deformation["Exy"]*self.phi(t) if ("Exy" in self.deformation.keys()) else 0.
        Eyx = self.deformation["Eyx"]*self.phi(t) if ("Eyx" in self.deformation.keys()) else 0.
        Exz = self.deformation["Exz"]*self.phi(t) if ("Exz" in self.deformation.keys()) else 0.
        Ezx = self.deformation["Ezx"]*self.phi(t) if ("Ezx" in self.deformation.keys()) else 0.
        Eyz = self.deformation["Eyz"]*self.phi(t) if ("Eyz" in self.deformation.keys()) else 0.
        Ezy = self.deformation["Ezy"]*self.phi(t) if ("Ezy" in self.deformation.keys()) else 0.
        self.F = numpy.array([[math.sqrt(1.+Exx),              Exy ,              Exz ],
                              [             Eyx , math.sqrt(1.+Eyy),              Eyz ],
                              [             Ezx ,              Ezy , math.sqrt(1.+Ezz)]])
        self.Finv = numpy.linalg.inv(self.F)

    def init_t_heart(self, t):
        Ri = self.structure["Ri"]
        Re = self.structure["Re"]
        dRi = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation.keys()) else 0.
        dRe = self.deformation["dRi"]*self.phi(t) if ("dRi" in self.deformation.keys()) else 0.
        dTi = self.deformation["dTi"]*self.phi(t) if ("dTi" in self.deformation.keys()) else 0.
        dTe = self.deformation["dTe"]*self.phi(t) if ("dTe" in self.deformation.keys()) else 0.
        self.A = numpy.array([[1.-(dRi-dRe)/(Re-Ri), 0.],
                              [  -(dTi-dTe)/(Re-Ri), 1.]])
        self.Ainv = numpy.linalg.inv(self.A)
        self.B = numpy.array([(1.+Ri/(Re-Ri))*dRi-Ri/(Re-Ri)*dRe, (1.+Ri/(Re-Ri))*dTi-Ri/(Re-Ri)*dTe])

    def X_no(self, x):
        return x

    def X_homogeneous(self, x):
        return numpy.dot(self.Finv, x)

    def X_heart(self, x):
        #print "x = " + str(x)
        self.x_inplane[0] = x[0] - self.images["L"][0]/2
        self.x_inplane[1] = x[1] - self.images["L"][1]/2
        #print "x_inplane = " + str(self.x_inplane)
        self.rt[0] = numpy.linalg.norm(self.x_inplane)
        self.rt[1] = math.atan2(self.x_inplane[1], self.x_inplane[0])
        #print "rt = " + str(self.rt)
        self.RT[:] = numpy.dot(self.Ainv, self.rt-self.B)
        #print "RT = " + str(self.RT)
        self.X_full[0] = self.RT[0] * math.cos(self.RT[1]) + self.images["L"][0]/2
        self.X_full[1] = self.RT[0] * math.sin(self.RT[1]) + self.images["L"][1]/2
        self.X_full[2] = x[2]
        #print "X_full = " + str(self.X_full)
        return self.X_full

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
        image.SetExtent([0, images["n_voxels"][0]-1, 0,                       0, 0,                       0])
    elif (images["n_dim"] == 2):
        image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0,                       0])
    elif (images["n_dim"] == 3):
        image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0, images["n_voxels"][2]-1])
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."

    spacing = numpy.array(images["L"])/numpy.array(images["n_voxels"])
    if (images["n_dim"] == 1):
        spacing = [spacing[0], 1., 1.]
    elif (images["n_dim"] == 2):
        spacing = [spacing[0], spacing[1], 1.]
    image.SetSpacing(spacing)

    origin = numpy.array(image.GetSpacing())/2
    if   (images["n_dim"] == 1):
        origin[1] = 0.
        origin[2] = 0.
    elif (images["n_dim"] == 2):
        origin[2] = 0.
    image.SetOrigin(origin)

    array_data = myVTK.createFloatArray(
        name="scalars",
        n_components=1,
        n_tuples=image.GetNumberOfPoints())
    image.GetPointData().AddArray(array_data)
    image.GetPointData().SetActiveScalars("scalars")

    if not os.path.exists(images["images_folder"]):
        os.mkdir(images["images_folder"])

    x0 = numpy.array([float()]*3)
    x  = numpy.array([float()]*3)
    if   (images["n_dim"] == 1):
        dx = spacing[0]/images["n_int"][0]
    elif (images["n_dim"] == 2):
        dx = spacing[0]/images["n_int"][0]
        dy = spacing[1]/images["n_int"][1]
    elif (images["n_dim"] == 3):
        dx = spacing[0]/images["n_int"][0]
        dy = spacing[1]/images["n_int"][1]
        dz = spacing[2]/images["n_int"][2]
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
    global_min = float("+Inf")
    global_max = float("-Inf")
    image_data = ImageData(images, structure, texture, noise)
    mapping = Mapping(images, structure, deformation, evolution)
    for k_frame in xrange(images["n_frames"]+1):
        t = images["T"]*float(k_frame)/images["n_frames"]
        mapping.init_t(t)
        for k_point in xrange(image.GetNumberOfPoints()):
            image.GetPoint(k_point, x0)
            print "x0 = " + str(x0)
            x[:] = x0[:]
            I = 0.
            if   (images["n_dim"] == 1):
                for k_x in xrange(images["n_int"][0]):
                    x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_int"][0]
                    I += image_data.I0(mapping.X(x))
                I /= images["n_int"][0]
            elif (images["n_dim"] == 2):
                for k_y in xrange(images["n_int"][1]):
                    x[1] = x0[1] - dy/2 + (k_y+1./2)*dy/images["n_int"][1]
                    for k_x in xrange(images["n_int"][0]):
                        x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_int"][0]
                        I += image_data.I0(mapping.X(x))
                I /= images["n_int"][1]*images["n_int"][0]
            elif (images["n_dim"] == 3):
                for k_z in xrange(images["n_int"][2]):
                    x[2] = x0[2] - dz/2 + (k_z+1./2)*dz/images["n_int"][2]
                    for k_y in xrange(images["n_int"][1]):
                        x[1] = x0[1] - dy/2 + (k_y+1./2)*dy/images["n_int"][1]
                        for k_x in xrange(images["n_int"][0]):
                            x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_int"][0]
                            print "x = " + str(x)
                            I += image_data.I0(mapping.X(x))
                            print "I = " + str(I)
                I /= images["n_int"][2]*images["n_int"][1]*images["n_int"][0]
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
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
