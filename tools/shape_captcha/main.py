
import vtk
import numpy as np
import os
from skimage.transform import resize
from skimage import measure
from scipy import ndimage


if __name__ == "__main__":


    path = '../../dataset/chair_table/shapes/'
    size = 32

    files = os.listdir(path)
    camera = vtk.vtkCamera();
    camera.SetPosition(0, 20, 0);
    camera.SetFocalPoint(0, 0, 0);

    ren = vtk.vtkRenderer()
    ren.SetBackground(1,1,1)
    ren.SetActiveCamera(camera);
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # create a renderwindowinteractor
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)


    points = vtk.vtkPoints()

    # Setup scales. This can also be an Int array
    # char is used since it takes the least memory
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("colors")
    colors.SetNumberOfComponents(4)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.GetPointData().SetScalars(colors)

    # Create anything you want here, we will use a cube for the demo.
    cubeSource = vtk.vtkCubeSource()

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetColorModeToColorByScalar()
    glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
    glyph3D.SetInputData(polyData)
    glyph3D.ScalingOff() #Needed, otherwise only the red cube is visible
    glyph3D.Update()


    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())

    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    ren.AddActor(actor)

    for step, file_name in enumerate(files):
        print('''[%d/%d] %s'''% (step, len(files), file_name))
        i = step

        shape = np.load(path + file_name)
        shape = shape.transpose(1, 2, 3, 0)

        for y in range(0, size):
          for x in range(0, size):
            for z in range(0, size):
              if shape[z][x][y][3] >= 0.5:
                points.InsertNextPoint(z, x, y)
                colors.InsertNextTuple4(shape[z][x][y][0] * 256, shape[z][x][y][1] * 256, shape[z][x][y][2] * 255, 255)

        del shape

        # color the actor

        # assign actor to the renderer

        renWin.Render()
        ren.ResetCamera()

        # screenshot code:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(renWin)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName("./images/" + file_name[:-4] + ".png")
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

        del w2if
        del writer

        points.Reset()
        colors.Reset()


        #enable user intçerface interactor
        # iren.Initialize()
        # iren.Start()