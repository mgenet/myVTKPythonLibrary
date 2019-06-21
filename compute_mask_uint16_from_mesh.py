import vtk
import myVTKPythonLibrary as myvtk

def compute_mask_uint16_from_mesh(image,
                                  mesh,
                                  out_value,
                                  binary_mask=0,
                                  warp_mesh=0,
                                  ):

    assert out_value <= 65535
    cast = vtk.vtkImageCast()
    cast.SetInputData(image)
    cast.SetOutputScalarTypeToUnsignedShort()
    cast.Update()
    image_for_mask = cast.GetOutput()

    mask = myvtk.compute_mask_from_mesh(image_for_mask,mesh,warp_mesh=warp_mesh,binary_mask=binary_mask,out_value=out_value)
    assert mask.GetPointData().GetScalars().GetDataType() == 5

    return mask
