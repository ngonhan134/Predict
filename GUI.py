from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog, QFrame
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200, 200, 300, 300)
        self.setWindowTitle("Login")

        self.login_button = QPushButton("Login", self)
        self.login_button.setGeometry(100, 100, 100, 30)
        self.login_button.clicked.connect(self.open_file)

        self.frame = QFrame(self)
        self.frame.setGeometry(50, 50, 200, 200)
        self.frame.hide()

    def open_file(self):
        filename = QFileDialog.getOpenFileName(self, "Open file", "", "Python Files (prediction.py)")
        if filename[0]:
            self.frame.show()
            with open(filename[0], "r") as file:
                print(file.read())

def window():
    app = QApplication(sys.argv)
    win = MyWindow()

    win.show()
    sys.exit(app.exec_())

window()
