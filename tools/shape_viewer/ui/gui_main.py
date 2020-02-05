from PyQt5.QtWidgets import *
from ui.gui_viewer import GUIViewer
import vtk
import os
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from skimage import measure
import numpy as np
from skimage import measure

class AppWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)


        self.setWindowTitle("viwer")
        self.frame = QFrame()
        self.vl = QVBoxLayout()

        #vtk
        self.viewerWidget = GUIViewer()
        self.vl.addWidget(self.viewerWidget.vtkWidget)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.statusBar  = self.statusBar()

        self.button = QPushButton('Open', self)
        self.button.clicked.connect(self.on_click)
        self.vl.addWidget(self.button)

        self.button = QPushButton('Save', self)
        self.button.clicked.connect(self.save_click)
        self.vl.addWidget(self.button)


        self.show()

    def on_click(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '../../outputs')
        self.shape = np.load(fname[0])
        self.statusBar.showMessage(fname[0])
        self.viewerWidget.update(self.shape, 32)

    def save_click(self):
        verts, faces, normals, values = measure.marching_cubes_lewiner(np.round(self.shape[3]), 0)
        faces = faces + 1

        thefile = open('test.obj', 'w')
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[1],item[0],item[2]))
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[2],item[1],item[0]))

        thefile.close()