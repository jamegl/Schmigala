from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QProgressBar, QAbstractItemView, QTableWidgetItem, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QHBoxLayout, QHeaderView
from solutions_window import SolutionsWindow
from solver import Solver

from solver4 import Solver4


class WarningWindow(QMainWindow):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

        self.setWindowTitle('Warning')
        self.label = QLabel(self)
        self.label.setText("Due to the higher number of nodes the process of finding the solution may take up to 5 min")
        self.label.setWordWrap(True)
        self.okButton = QPushButton("Ok")
        self.okButton.clicked.connect(self.okButtonWasClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.okButton)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        # self.setFixedSize(QSize(200, 100))

    def okButtonWasClicked(self):
        self.solver = Solver4(self.graph)
        self.solver.test_multi()
        self.open_solutions_window()

    def open_solutions_window(self):
        self.solutions_window = SolutionsWindow()
        self.solutions_window.show()
        self.hide()

