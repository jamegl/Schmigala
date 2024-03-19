from PyQt6.QtWidgets import QApplication, QMainWindow

import sys

from main_window import MainWindow

app = QApplication(sys.argv)

window = MainWindow()

app.exec()