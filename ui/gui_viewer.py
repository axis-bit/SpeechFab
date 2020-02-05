from PyQt5.QtWidgets import *
from PyQt5.QtCore import QSize
from PyQt5 import Qt
import vtk
import numpy as np
import os

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ui.algo import Eval
class GUIViewer(QVTKRenderWindowInteractor):

    def __init__(self):
        QVTKRenderWindowInteractor.__init__(self)

        self.vtkWidget = QVTKRenderWindowInteractor()

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1,1,1)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)

        # Create points
        self.points = vtk.vtkPoints()

        # Setup scales. This can also be an Int array
        # char is used since it takes the least memory
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetName("colors")
        self.colors.SetNumberOfComponents(4)

        # Combine into a polydata
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(self.points)
        polyData.GetPointData().SetScalars(self.colors)

        # Create anything you want here, we will use a cube for the demo.
        cubeSource = vtk.vtkCubeSource()

        self.glyph3D = vtk.vtkGlyph3D()
        self.glyph3D.SetColorModeToColorByScalar()
        self.glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
        self.glyph3D.SetInputData(polyData)
        self.glyph3D.ScalingOff() #Needed, otherwise only the red cube is visible
        self.glyph3D.Update()


        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.glyph3D.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.RotateY(45.0)
        actor.RotateZ(45.0)
        actor.RotateX(-45.0)

        self.ren.AddActor(actor)
        self.ren.ResetCamera()

        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.iren.Start()
        self.algo = Eval()

    def update(self, shape, size, one_flag, tri_flag):
        print("update called")
        self.points.Reset()
        self.colors.Reset()

        if(size == ""):
            show_size = 32
        else:
            show_size = int(size)

        shape = shape.detach().numpy()
        shape = self.algo.update(shape, show_size, one_flag, tri_flag)

        for y in range(0, show_size):
            for x in range(0, show_size):
                for z in range(0, show_size):
                    if shape[z][x][y][3] >= 0.5:
                        self.points.InsertNextPoint(z, x, y)
                        self.colors.InsertNextTuple4(shape[z][x][y][0] * 256, shape[z][x][y][1] * 256, shape[z][x][y][2] * 255, 255)

        self.glyph3D.Modified()
        self.ren.ResetCamera()

        return shape