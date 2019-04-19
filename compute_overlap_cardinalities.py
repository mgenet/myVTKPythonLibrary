import myVTKPythonLibrary as myvtk
import vtk

def compute_overlap_cardinalities(image0,
                                  image1,
                                  image0_array_name='scalars',
                                  image1_array_name='scalars'):

    assert image0.GetDimensions() == image1.GetDimensions()
    assert image0.GetNumberOfPoints() == image1.GetNumberOfPoints()
    assert image0.GetPointData().HasArray(image0_array_name)
    assert image1.GetPointData().HasArray(image1_array_name)
    assert image0.GetScalarType() == image1.GetScalarType()
    assert image0.GetPointData().GetArray(image0_array_name).GetRange() == (0.0, 1.0)
    assert image1.GetPointData().GetArray(image1_array_name).GetRange() == (0.0, 1.0)

    image0.GetPointData().SetActiveScalars(image0_array_name)
    image1.GetPointData().SetActiveScalars(image1_array_name)

    addMaths = vtk.vtkImageMathematics()
    addMaths.SetInputData(0,image0)
    addMaths.SetInputData(1,image1)
    addMaths.SetOperationToAdd()
    addMaths.Modified()
    addMaths.Update()
    somme = addMaths.GetOutput()

    histo = vtk.vtkImageHistogram()
    histo.SetInputData(somme)
    histo.Update()
    histo_array = histo.GetHistogram()
    TP = histo_array.GetTuple(2)[0]
    TN = histo_array.GetTuple(0)[0]
    assert int(TP + TN + histo_array.GetTuple(1)[0]) == image0.GetNumberOfPoints()

    subMaths = vtk.vtkImageMathematics()
    subMaths.SetInputData(0,image0)
    subMaths.SetInputData(1,image1)
    subMaths.SetOperationToSubtract()
    subMaths.Modified()
    subMaths.Update()
    diff = subMaths.GetOutput()

    histo2 = vtk.vtkImageHistogram()
    histo2.SetInputData(diff)
    histo2.Update()
    histo_array2 = histo2.GetHistogram()
    FN = histo_array2.GetTuple(1)[0]
    FP = histo_array2.GetTuple(255)[0]
    assert int(FN + histo_array2.GetTuple(0)[0] + FP) == image0.GetNumberOfPoints()

    print 'TP =', TP, ', TN =', TN, ', FP =', FP, ', FN =', FN
    assert int(TP + TN + FP + FN) == image0.GetNumberOfPoints()

    return (TP, TN, FP, FN)
