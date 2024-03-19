from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QAbstractItemView, QTableWidgetItem, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QHBoxLayout, QHeaderView


class ErrorDialog(QDialog):
    def __init__(self, error_message):
        super().__init__()

        self.setWindowTitle("Error")

        self.okButton = QPushButton("OK")
        self.okButton.clicked.connect(self.okButtonWasClicked)

        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel(error_message))
        self.layout.addWidget(self.okButton)
        self.setLayout(self.layout)

    def okButtonWasClicked(self):
        self.hide()