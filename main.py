#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication
from app.gui import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 