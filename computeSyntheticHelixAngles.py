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

def computeSyntheticHelixAngles(
        farray_rr,
        helix_angle_end,
        helix_angle_epi,
        farray_angle_helix=None,
        verbose=0):

    myVTK.myPrint(verbose, "*** computeSyntheticHelixAngles ***")

    n_cells = farray_rr.GetNumberOfTuples()

    if (farray_angle_helix is None):
        farray_angle_helix = myVTK.createFloatArray(
            name="angle_helix",
            n_components=1,
            n_tuples=n_cells)

    for k_cell in xrange(n_cells):
        rr = farray_rr.GetTuple1(k_cell)

        helix_angle_in_degrees = (1.-rr) * helix_angle_end \
                               +     rr  * helix_angle_epi
        farray_angle_helix.SetTuple1(
            k_cell,
            helix_angle_in_degrees)

    return farray_angle_helix

########################################################################

def addSyntheticHelixAngles(
        ugrid,
        helix_angle_end,
        helix_angle_epi,
        type_of_support="cell",
        verbose=0):

    myVTK.myPrint(verbose, "*** addSyntheticHelixAngles ***")

    if (type_of_support == "cell"):
        ugrid_data = ugrid.GetCellData()
    elif (type_of_support == "point"):
        ugrid_data = ugrid.GetPointData()

    farray_rr = ugrid_data.GetArray("rr")
    farray_angle_helix = ugrid_data.GetArray("angle_helix")

    farray_angle_helix = computeSyntheticHelixAngles(
        farray_rr=farray_rr,
        helix_angle_end=helix_angle_end,
        helix_angle_epi=helix_angle_epi,
        farray_angle_helix=farray_angle_helix,
        verbose=verbose-1)

    ugrid_data.AddArray(farray_angle_helix)

    return farray_angle_helix
