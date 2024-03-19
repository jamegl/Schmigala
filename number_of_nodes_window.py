import sys

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QDialog, QTableWidget, QHBoxLayout, QHeaderView, QLineEdit
from PyQt6.QtGui import QIntValidator

class NumberOfNodesWindow(QMainWindow):
    number_entered = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        # self.setWindowTitle("Number of nodes")

        numberOfNodesLabel = QLabel(self)
        numberOfNodesLabel.setText("Enter number of workstations:")
        self.numberOfNodesLineEdit = QLineEdit()
        onlyInt = QIntValidator(2, 10)
        self.numberOfNodesLineEdit.setValidator(onlyInt)

        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.saveData)

        layout = QVBoxLayout()
        layout.addWidget(numberOfNodesLabel)
        layout.addWidget(self.numberOfNodesLineEdit)
        layout.addWidget(self.saveButton)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setFixedSize(QSize(200, 100))

    def saveData(self):
        entered_number = 5
        if self.numberOfNodesLineEdit.text() != '':
            entered_number = int(self.numberOfNodesLineEdit.text())
        if (entered_number > 10) :
            entered_number = 10
        elif (entered_number < 4):
            entered_number = 4
        self.number_entered.emit(entered_number)
        self.hide()
        