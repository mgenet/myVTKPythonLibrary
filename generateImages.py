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

#class ImagesInfo():
    #def __init__(self, n_dim, L, n_voxels, n_integration, T, n_frames, data_type, images_folder, images_basename):
        #assert (n_dim in (1,2,3))
        #self.n_dim = n_dim

        #if (type(L) == float):
            #assert (L>0)
            #self.L = numpy.array([L]*self.n_dim)
        #elif (type(L) == int):
            #assert (L>0)
            #self.L = numpy.array([float(L)]*self.n_dim)
        #else:
            #assert (len(L) == self.n_dim)
            #self.L = numpy.array(L)
            #assert ((self.L>0).all())

        #if (type(n_voxels) == int):
            #assert (n_voxels>0)
            #self.n_voxels = numpy.array([n_voxels]*self.n_dim)
        #else:
            #assert (len(n_voxels) == self.n_dim)
            #self.n_voxels = numpy.array(n_voxels)
            #assert ((self.n_voxels>0).all())

        #if (type(n_integration) == int):
            #assert (n_integration>0)
            #self.n_integration = numpy.array([n_integration]*self.n_dim)
        #else:
            #assert (len(n_integration) == self.n_dim)
            #self.n_integration = numpy.array(n_integration)
            #assert ((self.n_integration>0).all())

        #assert (T>0.)
        #self.T = T

        #assert (n_frames>0)
        #self.n_frames = n_frames

        #assert (data_type in ("int", "float", "unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float" "uint8", "uint16", "uint32", "uint64", "ufloat"))
        #self.data_type = data_type

        #self.images_folder = images_folder
        #self.images_basename = images_basename

#class StructureInfo():
    #def __init__(self, images, type, **kwargs):
        #assert (type in ("no", "heart"))
        #self["type"] = type
        #if (self["type"] == "heart"):
            #self.Ri = kwargs["Ri"]
            #self.Re = kwargs["Re"]
            #if (images.n_dim == 3):
                #self.Zmin = kwargs["Zmin"] if ("Zmin" in kwargs.keys()) else 0.
                #self.Zmax = kwargs["Zmax"] if ("Zmax" in kwargs.keys()) else images.L[2]

#class TextureInfo():
    #def __init__(self, type, **kwargs):
        #assert (type in ("no", "sine", "sinX", "sinY", "sinZ", "taggX", "taggY", "taggZ"))
        #self["type"] = type

#class NoiseInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

#class DeformationInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

#class EvolutionInfo():
    #def __init__(self, type, **kwargs):
        #self["type"] = type

########################################################################

class Image():
    def __init__(self, images, structure, texture, noise):
        if (structure["type"] == "no"):
            self.I0_structure = self.I0_structure_no
        elif (structure["type"] == "heart"):
            if (images["n_dim"] == 2):
                #self.I0_structure = self.I0_structure_no
                self.I0_structure = self.I0_structure_heart_2
                self.L = [images["L"][0], images["L"][1]]
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
            elif (images["n_dim"] == 3):
                self.I0_structure = self.I0_structure_heart_3
                self.L = [images["L"][0], images["L"][1], images["L"][2]]
                self.R = float()
                self.Ri = structure["Ri"]
                self.Re = structure["Re"]
                self.Zmin = structure.Zmin if ("Zmin" in structure.keys()) else 0.
                self.Zmax = structure.Zmax if ("Zmax" in structure.keys()) else images["L"][2]
            else:
                assert (0), "n_dim must be \"2\" or \"3 for \"heart\" type structure. Aborting."
        else:
            assert (0), "structure type must be \"no\" or \"heart\". Aborting."

        texture = texture
        if (texture["type"] == "no"):
            self.I0_texture = self.I0_texture_no
        elif (texture["type"] == "sine"):
            if   (images["n_dim"] == 1):
                self.I0_texture = self.I0_texture_sine_X
                self.L[0] = images["L"][0]
            elif (images["n_dim"] == 2):
                self.I0_texture = self.I0_texture_sine_XY
                self.L[0] = images["L"][0]
                self.L[1] = images["L"][1]
            elif (images["n_dim"] == 3):
                self.I0_texture = self.I0_texture_sine_XYZ
                self.L[0] = images["L"][0]
                self.L[1] = images["L"][1]
                self.L[2] = images["L"][2]
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
        elif (texture["type"] == "sinX"):
            self.I0_texture = self.I0_texture_sine_X
            self.L[0] = images["L"][0]
        elif (texture["type"] == "sinY"):
            self.I0_texture = self.I0_texture_sine_Y
            self.L[1] = images["L"][1]
        elif (texture["type"] == "sinZ"):
            self.I0_texture = self.I0_texture_sine_Z
            self.L[2] = images["L"][2]
        elif (texture["type"] == "tagging"):
            if   (images["n_dim"] == 1):
                self.I0_texture = self.I0_texture_tagging_X
                self.s = texture["s"]
            elif (images["n_dim"] == 2):
                self.I0_texture = self.I0_texture_tagging_XY
                self.s = texture["s"]
            elif (images["n_dim"] == 3):
                self.I0_texture = self.I0_texture_tagging_XYZ
                self.s = texture["s"]
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
        elif (texture["type"] == "taggX"):
            self.I0_texture = self.I0_texture_tagging_X
            self.s = texture["s"]
        elif (texture["type"] == "taggY"):
            self.I0_texture = self.I0_texture_tagging_Y
            self.s = texture["s"]
        elif (texture["type"] == "taggZ"):
            self.I0_texture = self.I0_texture_tagging_Z
            self.s = texture["s"]
        else:
            assert (0), "texture type must be \"no\", \"sine\", \"sinX\", \"sinY\", \"sinZ\", \"tagging\", \"taggX\", \"taggY\" or \"taggZ\". Aborting."

        if (noise["type"] == "no"):
            self.I0_noise = self.I0_noise_no
        elif (noise["type"] == "normal"):
            self.I0_noise = self.I0_noise_normal
            self.avg = noise.avg  if ("avg" in noise.keys()) else 0.
            self.std = noise.std
        else:
            assert (0), "noise type must be \"no\" or \"normal\". Aborting."

    def I0(self, X):
        return self.I0_structure(X) * self.I0_texture(X) + self.I0_noise()

    def I0_structure_no(self, X):
        return 1.

    def I0_structure_heart_2(self, X):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re):
            return 1.
        else:
            return 0.

    def I0_structure_heart_3(self, X):
        self.R = ((X[0]-self.L[0]/2)**2 + (X[1]-self.L[1]/2)**2)**(1./2)
        if (self.R >= self.Ri) and (self.R <= self.Re) and (X[2] >= self.Zmin) and (X[2] <= self.Zmax):
            return 1.
        else:
            return 0.

    def I0_texture_no(self, X):
        return 1.

    def I0_texture_sine_X(self, X):
        return math.sin(math.pi*X[0]/self.L[0])**2

    def I0_texture_sine_Y(self, X):
        return math.sin(math.pi*X[1]/self.L[1])**2

    def I0_texture_sine_Z(self, X):
        return math.sin(math.pi*X[2]/self.L[2])**2

    def I0_texture_sine_XY(self, X):
        return math.sin(math.pi*X[0]/self.L[0])**2 * math.sin(math.pi*X[1]/self.L[1])**2

    def I0_texture_sine_XYZ(self, X):
        return math.sin(math.pi*X[0]/self.L[0])**2 * math.sin(math.pi*X[1]/self.L[1])**2 * math.sin(math.pi*X[2]/self.L[2])**2

    def I0_texture_tagging_X(self, X):
        return math.sin(math.pi*X[0]/self.s)**2

    def I0_texture_tagging_Y(self, X):
        return math.sin(math.pi*X[1]/self.s)**2

    def I0_texture_tagging_Z(self, X):
        return math.sin(math.pi*X[2]/self.s)**2

    def I0_texture_tagging_XY(self, X):
        return math.sin(math.pi*X[0]/self.s)**2 * math.sin(math.pi*X[1]/self.s)**2

    def I0_texture_tagging_XYZ(self, X):
        return math.sin(math.pi*X[0]/self.s)**2 * math.sin(math.pi*X[1]/self.s)**2 * math.sin(math.pi*X[2]/self.s)**2

    def I0_noise_no(self):
        return 0.

    def I0_noise_normal(self):
        return random.normalvariate(self.avg, self.std)

########################################################################

class Mapping:
    def __init__(self, images, structure, deformation, evolution):
        self.deformation = deformation
        if (self.deformation["type"] == "no"):
            self.init_t = self.init_t_no
            self.X = self.X_no
            self.x = self.x_no
        elif (self.deformation["type"] == "homogeneous"):
            self.init_t = self.init_t_homogeneous
            self.X = self.X_homogeneous
            self.x = self.x_homogeneous
        elif (self.deformation["type"] == "heart"):
            assert (structure["type"] == "heart"), "structure type must be \"heart\" for \"heart\" type deformation. Aborting."
            self.init_t = self.init_t_heart
            self.X = self.X_heart
            self.x = self.x_heart
            self.x_inplane = numpy.empty(2)
            self.X_inplane = numpy.empty(2)
            self.rt = numpy.empty(2)
            self.RT = numpy.empty(2)
            self.X_full = numpy.empty(3)
            self.x_full = numpy.empty(3)
            self.L = images["L"]
            self.Ri = structure["Ri"]
            self.Re = structure["Re"]
        else:
            assert (0), "deformation type must be \"no\", \"homogeneous\" or \"heart\". Aborting."

        if (evolution["type"] == "linear"):
            self.phi = self.phi_linear
            self.T = evolution.T
        elif (evolution["type"] == "sine"):
            self.phi = self.phi_sine
            self.T = evolution["T"]
        else:
            assert (0), "evolution type must be \"linear\" or \"sine\". Aborting."

    def phi_linear(self, t):
        return t

    def phi_sine(self, t):
        return math.sin(math.pi*t/self.T)**2

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
        Ri = self.Ri
        Re = self.Re
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
        self.x_inplane[0] = x[0] - self.L[0]/2
        self.x_inplane[1] = x[1] - self.L[1]/2
        #print "x_inplane = " + str(self.x_inplane)
        self.rt[0] = numpy.linalg.norm(self.x_inplane)
        self.rt[1] = math.atan2(self.x_inplane[1], self.x_inplane[0])
        #print "rt = " + str(self.rt)
        self.RT[:] = numpy.dot(self.Ainv, self.rt-self.B)
        #print "RT = " + str(self.RT)
        self.X_full[0] = self.RT[0] * math.cos(self.RT[1]) + self.L[0]/2
        self.X_full[1] = self.RT[0] * math.sin(self.RT[1]) + self.L[1]/2
        self.X_full[2] = x[2]
        #print "X_full = " + str(self.X_full)
        return self.X_full

    def x_no(self, X):
        return X

    def x_homogeneous(self, X):
        return numpy.dot(self.F, X)

    def x_heart(self, X):
        #print "X = " + str(X)
        self.X_inplane[0] = X[0] - self.L[0]/2
        self.X_inplane[1] = X[1] - self.L[1]/2
        #print "X_inplane = " + str(self.X_inplane)
        self.RT[0] = numpy.linalg.norm(self.X_inplane)
        self.RT[1] = math.atan2(self.X_inplane[1], self.X_inplane[0])
        #print "RT = " + str(self.RT)
        self.rt[:] = numpy.dot(self.A, self.RT) + self.B
        #print "rt = " + str(self.rt)
        self.x_full[0] = self.rt[0] * math.cos(self.rt[1]) + self.L[0]/2
        self.x_full[1] = self.rt[0] * math.sin(self.rt[1]) + self.L[1]/2
        self.x_full[2] = X[2]
        #print "x_full = " + str(self.x_full)
        return self.x_full

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

    vtk_image = vtk.vtkImageData()

    if   (images["n_dim"] == 1):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0,                       0, 0,                       0])
    elif (images["n_dim"] == 2):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0,                       0])
    elif (images["n_dim"] == 3):
        vtk_image.SetExtent([0, images["n_voxels"][0]-1, 0, images["n_voxels"][1]-1, 0, images["n_voxels"][2]-1])
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."

    spacing = numpy.array(images["L"])/numpy.array(images["n_voxels"])
    if (images["n_dim"] == 1):
        spacing = [spacing[0], 1., 1.]
    elif (images["n_dim"] == 2):
        spacing = [spacing[0], spacing[1], 1.]
    vtk_image.SetSpacing(spacing)

    origin = numpy.array(vtk_image.GetSpacing())/2
    if   (images["n_dim"] == 1):
        origin[1] = 0.
        origin[2] = 0.
    elif (images["n_dim"] == 2):
        origin[2] = 0.
    vtk_image.SetOrigin(origin)
    if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
        vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
    else:
        vtk_image.SetScalarTypeToFloat();
        vtk_image.SetNumberOfScalarComponents(1);
        vtk_image.AllocateScalars()
    vtk_image_scalars = vtk_image.GetPointData().GetScalars()

    if not os.path.exists(images["images_folder"]):
        os.mkdir(images["images_folder"])

    x0 = numpy.empty(3)
    x  = numpy.empty(3)
    if   (images["n_dim"] == 1):
        dx = spacing[0]/images["n_integration"][0]
    elif (images["n_dim"] == 2):
        dx = spacing[0]/images["n_integration"][0]
        dy = spacing[1]/images["n_integration"][1]
    elif (images["n_dim"] == 3):
        dx = spacing[0]/images["n_integration"][0]
        dy = spacing[1]/images["n_integration"][1]
        dz = spacing[2]/images["n_integration"][2]
    else:
        assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
    global_min = float("+Inf")
    global_max = float("-Inf")
    image = Image(images, structure, texture, noise)
    mapping = Mapping(images, structure, deformation, evolution)
    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mapping.init_t(t)
        for k_point in xrange(vtk_image.GetNumberOfPoints()):
            vtk_image.GetPoint(k_point, x0)
            #print "x0 = " + str(x0)
            x[:] = x0[:]
            I = 0.
            if   (images["n_dim"] == 1):
                for k_x in xrange(images["n_integration"][0]):
                    x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_integration"][0]
                    I += image.I0(mapping.X(x))
                I /= images["n_integration"][0]
            elif (images["n_dim"] == 2):
                for k_y in xrange(images["n_integration"][1]):
                    x[1] = x0[1] - dy/2 + (k_y+1./2)*dy/images["n_integration"][1]
                    for k_x in xrange(images["n_integration"][0]):
                        x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_integration"][0]
                        I += image.I0(mapping.X(x))
                I /= images["n_integration"][1]*images["n_integration"][0]
            elif (images["n_dim"] == 3):
                for k_z in xrange(images["n_integration"][2]):
                    x[2] = x0[2] - dz/2 + (k_z+1./2)*dz/images["n_integration"][2]
                    for k_y in xrange(images["n_integration"][1]):
                        x[1] = x0[1] - dy/2 + (k_y+1./2)*dy/images["n_integration"][1]
                        for k_x in xrange(images["n_integration"][0]):
                            x[0] = x0[0] - dx/2 + (k_x+1./2)*dx/images["n_integration"][0]
                            #print "x = " + str(x)
                            I += image.I0(mapping.X(x))
                            #print "I = " + str(I)
                I /= images["n_integration"][2]*images["n_integration"][1]*images["n_integration"][0]
            else:
                assert (0), "n_dim must be \"1\", \"2\" or \"3\". Aborting."
            vtk_image_scalars.SetTuple1(k_point, I)
            if (I < global_min): global_min = I
            if (I > global_max): global_max = I
        myVTK.writeImage(
            image=vtk_image,
            filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")

    if (images["data_type"] in ("float")):
        pass
    elif (images["data_type"] in ("unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned float" "uint8", "uint16", "uint32", "uint64", "ufloat")):
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
        elif (images["data_type"] in ("unsigned float", "ufloat")):
            shifter.SetScale(1./(global_max-global_min))
            shifter.SetOutputScalarTypeToFloat()
        for k_frame in xrange(images["n_frames"]):
            vtk_image = myVTK.readImage(
                filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")
            if (vtk.vtkVersion.GetVTKMajorVersion() >= 6):
                shifter.SetInputData(vtk_image)
            else:
                shifter.SetInput(vtk_image)
            shifter.Update()
            vtk_image = shifter.GetOutput()
            myVTK.writeImage(
                image=vtk_image,
                filename=images["images_folder"]+"/"+images["images_basename"]+"_"+str(k_frame).zfill(2)+".vti")
    else:
        assert (0), "Wrong data type. Aborting."

########################################################################

def warpMesh(
        mesh_folder,
        mesh_basename,
        images,
        structure,
        deformation,
        evolution,
        verbose=1):

    myVTK.myPrint(verbose, "*** warpMesh ***")

    mesh = myVTK.readUGrid(
        filename=mesh_folder+"/"+mesh_basename+".vtk",
        verbose=0)
    n_points = mesh.GetNumberOfPoints()
    n_cells = mesh.GetNumberOfCells()

    if os.path.exists(mesh_folder+"/"+mesh_basename+"-WithLocalBasis.vtk"):
        ref_mesh = myVTK.readUGrid(
            filename=mesh_folder+"/"+mesh_basename+"-WithLocalBasis.vtk",
            verbose=0)
    else:
        ref_mesh = None

    farray_disp = myVTK.createFloatArray(
        name="displacement",
        n_components=3,
        n_tuples=n_points,
        verbose=0)
    mesh.GetPointData().AddArray(farray_disp)

    mapping = Mapping(images, structure, deformation, evolution)

    X = numpy.empty(3)
    x = numpy.empty(3)
    U = numpy.empty(3)
    for k_frame in xrange(images["n_frames"]):
        t = images["T"]*float(k_frame)/(images["n_frames"]-1) if (images["n_frames"]>1) else 0.
        mapping.init_t(t)

        for k_point in xrange(n_points):
            mesh.GetPoint(k_point, X)
            x[:] = mapping.x(X)
            U[:] = x[:] - X[:]
            farray_disp.SetTuple(k_point, U)

        myVTK.computeStrainsFromDisplacements(
            mesh=mesh,
            disp_array_name="displacement",
            ref_mesh=ref_mesh,
            verbose=0)

        myVTK.writeUGrid(
            ugrid=mesh,
            filename=mesh_folder+"/"+mesh_basename+"_"+str(k_frame).zfill(2)+".vtk",
            verbose=0)





















