from PyQt5.QtWidgets import *
from ui.gui_viewer import GUIViewer
import vtk
import os
import time

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from ui.algo import Eval
from skimage import measure
import numpy as np

class AppWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        #self.setMinimumSize(QSize(440,500)pip install PyQtWebEngine)
        self.setWindowTitle("text2shape")
        self.frame = QFrame()
        self.vl = QVBoxLayout()

        #vtk
        self.viewerWidget = GUIViewer()
        self.vl.addWidget(self.viewerWidget.vtkWidget)


        # Add text field
        self.b = QPlainTextEdit(self)
        self.b.resize(400, 10)
        self.vl.addWidget(self.b)

        self.size = QLineEdit(self)
        self.vl.addWidget(self.size)

        self.one_data= QCheckBox("Single Object",self)
        self.vl.addWidget(self.one_data)

        self.tri = QCheckBox("Auto Scaling",self)
        self.vl.addWidget(self.tri)


        self.button = QPushButton('Generate', self)
        self.button.clicked.connect(self.on_click)
        self.vl.addWidget(self.button)

        self.button = QPushButton('Save', self)
        self.button.clicked.connect(self.save_obj)
        self.vl.addWidget(self.button)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()
        self.algo = Eval()
        self.last_press_time = time.time()
        self.shape = self.algo.create("a chair")
        self.shape = self.viewerWidget.update(self.shape, "", False , False)


    def on_click(self):
        self.shape = self.algo.create(self.b.toPlainText())
        if (self.one_data.checkState() == 2):
            one_flag = True
        else:
            one_flag = False

        if (self.tri.checkState() == 2):
            tri_flag = True
        else:
            tri_flag = False

        self.shape = self.viewerWidget.update(self.shape, self.size.text(), one_flag, tri_flag)


    def save_obj(self):
        shape = self.shape
        verts, faces, normals, values = measure.marching_cubes_lewiner(np.round(shape[3]), 0)
        faces=faces +1

        thefile = open('test.obj', 'w')
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))

        thefile.close()

