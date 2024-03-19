import sys
import os

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QComboBox, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QDialog, QTableWidget, QHBoxLayout, QHeaderView, QLineEdit

class SolutionsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Solutions")

        layout = QVBoxLayout()
        self.files = []
        self.solution_numbers = []

        for f in os.listdir("solutions"):
            self.files.append(f)

        self.get_solutions_with_min_cost()

        comboBox = QComboBox()
        comboBox.addItems(self.solution_numbers)

        comboBox.currentIndexChanged.connect(self.text_changed)

        self.label = QLabel(self)

        graphPixMap = QPixmap("solutions/" + self.files[0])
        self.label.setPixmap(graphPixMap)

        # self.dataWindow = DataWindow()

        self.solutionCostLabel = QLabel(self)


        self.solutionCostLabel.setText("Cost: " + self.get_cost_of_solution(self.files[0]))

        layout = QVBoxLayout()
        layout.addWidget(self.solutionCostLabel)
        layout.addWidget(comboBox)
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def get_cost_of_solution(self, file):
        first_position = file.find("_") + 1
        second_position = file.find(".")
        return file[first_position : second_position]
    
    def get_solutions_with_min_cost(self):
        min_cost = int(self.get_cost_of_solution(self.files[0]))
        for f in self.files:
            cost = int(self.get_cost_of_solution(f))
            if cost < min_cost:
                min_cost = cost

        files = []

        number = 1

        for f in self.files:
            if int(self.get_cost_of_solution(f)) == min_cost:
                files.append(f)
                self.solution_numbers.append(str(number))
                number += 1

        self.files = files
            

    def text_changed(self, i):
        graphPixMap = QPixmap("solutions/" + self.files[i])
        self.solutionCostLabel.setText("Cost: " + self.get_cost_of_solution(self.files[i]))
        self.label.setPixmap(graphPixMap)