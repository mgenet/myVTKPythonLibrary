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

import myVTKPythonLibrary as myVTK

########################################################################

def createArray(
        name,
        n_components=1,
        n_tuples=0,
        array_type="float",
        init_to_zero=0,
        verbose=1):

    assert (type(array_type) in (type, str)), "array_type must be a type or a str. Aborting."

    if (type(array_type) is type):
        assert (array_type in (float, int)), "Wrong array type. Aborting."

        if (array_type == float):
            return myVTK.createFloatArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type == int):
            return myVTK.createIntArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
    elif (type(array_type) is str):
        assert (array_type in ("double", "float", "long", "unsigned long", "int", "unsigned int", "short", "unsigned short", "char", "unsigned char", "int64", "uint64", "int32", "uint32", "int16", "uint16", "int8", "uint8")), "Wrong array type. Aborting."

        if (array_type in ("float", "double")):
            return myVTK.createFloatArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("long", "int64")):
            return myVTK.createLongArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("unsigned long", "uint64")):
            return myVTK.createUnsignedLongArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("int", "int32")):
            return myVTK.createIntArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("unsigned int", "uint32")):
            return myVTK.createUnsignedIntArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("short", "int16")):
            return myVTK.createShortArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("unsigned short", "uint16")):
            return myVTK.createUnsignedShortArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("char", "int8")):
            return myVTK.createCharArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
        elif (array_type in ("unsigned char", "uint8")):
            return myVTK.createUnsignedCharArray(
                       name=name,
                       n_components=n_components,
                       n_tuples=n_tuples,
                       init_to_zero=init_to_zero,
                       verbose=verbose-1)
