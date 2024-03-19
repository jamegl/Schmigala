import sys
import numpy as np
import os

from PyQt6.QtCore import QSize, Qt, QModelIndex
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QTableWidgetItem, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QHBoxLayout, QHeaderView
from error_dialog import ErrorDialog
from graph import Graph
from node import Node
from edge import Edge
from numeric_delegate import NumericDelegate
from warning_window import WarningWindow
from solutions_window import SolutionsWindow
from number_of_nodes_window import NumberOfNodesWindow
from solver import Solver
from solver2 import Solver2
from solver3 import Solver3
from solver4 import Solver4
import time

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Triangle method optimization")
        self.number_of_nodes = 5
        self.initUI()

        self.show()

    def initUI(self):
        self.create_buttons()
        self.create_table()
        self.label = QLabel()
        self.label.setText("Serching for solutions...")
        self.label.hide()

        # Create a central widget and set the layout on it
        central_widget = QWidget(self)
        central_layout = QVBoxLayout()
        central_layout.addLayout(self.hBox)  # Add the horizontal layout to the main layout
        central_layout.addWidget(self.label)
        central_layout.addWidget(self.table)  # Add the table to the main layout
        central_widget.setLayout(central_layout)
        
        # Set the central widget for the main window
        self.setCentralWidget(central_widget)
        self.setMinimumSize(QSize(500, 400))

    def create_buttons(self):
        # Create the buttons and add them to a horizontal layout
        self.numberOfNodesButton = QPushButton("Number of workstations")
        self.numberOfNodesButton.clicked.connect(self.numberOfNodesButtonWasClicked)

        # self.graphButton = QPushButton("Show Graph")
        self.calculateButton = QPushButton("Calculate")
        self.calculateButton.clicked.connect(self.calculateButtonWasClicked)
        
        self.hBox = QHBoxLayout()
        self.hBox.addWidget(self.numberOfNodesButton)
        # self.hBox.addWidget(self.graphButton)
        self.hBox.addWidget(self.calculateButton)

    def create_table(self):
        table_labels = []
        for i in range(0, self.number_of_nodes):
            table_labels.append(chr(65 + i))

        self.table = QTableWidget()
        self.table.setRowCount(self.number_of_nodes)
        self.table.setColumnCount(self.number_of_nodes)
        self.table.setHorizontalHeaderLabels(table_labels)
        self.table.setVerticalHeaderLabels(table_labels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)      
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        delegate = NumericDelegate(self.table)
        self.table.setItemDelegate(delegate)
        self.fill_table_with_zeros()

    def fill_table_with_zeros(self):
        for i in range(0, self.table.columnCount()):
            for j in range(0, self.table.columnCount()):
                item = QTableWidgetItem("0")
                if i == j:
                    item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, j, item)

    def numberOfNodesButtonWasClicked(self):
        self.number_of_nodes_window = NumberOfNodesWindow()
        self.number_of_nodes_window.number_entered.connect(self.update_node_number)
        self.number_of_nodes_window.show()

    def calculateButtonWasClicked(self):
        if not os.path.exists("solutions"):
            os.makedirs("solutions")

        if self.table.item(0, 1).isSelected():
            self.table.setCurrentCell(1, 0)
        else:
            self.table.setCurrentCell(0, 1)

        table_is_empty = True
        for i in range(0, self.table.columnCount()):
            for j in range(0, self.table.columnCount()):
                if self.table.item(i, j).text() != "0":
                   table_is_empty = False

        if table_is_empty:
            error_dialog = ErrorDialog("Please input material flow values.")
            error_dialog.exec()
        else:
            self.delete_prev_solutions()
            self.read_data_from_table()
            self.create_graph()
            if self.number_of_nodes < 7:
                self.find_solutions()
                self.open_solutions_window()
            else:
                self.warning_window = WarningWindow(self.graph)
                self.warning_window.show()

    def delete_prev_solutions(self):
        for f in os.listdir("solutions"):
            os.remove("solutions/" + f)

    def update_node_number(self, number):
        self.number_of_nodes = number
        self.initUI()

    def read_data_from_table(self):
        n = self.table.columnCount()
        self.data_from_table = [[0] * n for _ in range(n)]
        if self.table.item(0, 1).isSelected():
            self.table.setCurrentCell(1, 0)
        else:
            self.table.setCurrentCell(0, 1)

        for i in range(0, n):
            for j in range(0, n):
                self.data_from_table[i][j] = int(self.table.item(i, j).text())

    def create_graph(self):
        graph = Graph()

        #creating nodes
        for i in range(0, self.table.columnCount()):
            node = Node(chr(65 + i))
            graph.add_node(node)

        #creating edges
        for i in range(0, len(self.data_from_table)):
            for j in range(0, len(self.data_from_table)):
                first_node = graph.get_node(chr(65 + i))
                second_node = graph.get_node(chr(65 + j))
                edge = Edge(first_node, second_node, self.data_from_table[i][j])
                graph.add_edge(edge)
        
        self.graph = graph

    def find_solutions(self):
        self.solver = Solver4(self.graph)
        self.solver.test_multi()

    def open_solutions_window(self):
        self.solutions_window = SolutionsWindow()
        self.solutions_window.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    app.exec()