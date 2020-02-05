import sys
import qdarkstyle
from PyQt5.QtWidgets import QApplication

import ui.gui_main

import os

if __name__ == "__main__":
  app = QApplication(sys.argv)
  app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
  mainWin = ui.gui_main.AppWindow()
  sys.exit(app.exec_())
  print("Exiting")