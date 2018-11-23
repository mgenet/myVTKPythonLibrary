# myVTKPythonLibrary
A collection of tools to manipulate meshes and images using vtkpython.
### Requirements
First you need to install [myPythonLibrary](https://gitlab.inria.fr/mgenet/myPythonLibrary).
### Installation
Get the code:
```
git clone https://gitlab.inria.fr/mgenet/myVTKPythonLibrary
```
To load the library within python, the simplest is to add the folder containing `myVTKPythonLibrary` to `PYTHONPATH`:
```
export PYTHONPATH=$PYTHONPATH:/path/to/folder
```
(To make this permanent, add the line to `~/.bashrc`.)
Then you can load the library within python:
```
import myVTKPythonLibrary as myvtk
```
