import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtQuickWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 800)

        self.mainwidget = QtWidgets.QWidget(MainWindow)
        self.mainwidget.setObjectName("mainwidget")
        self.imageWidget = QtQuickWidgets.QQuickWidget(self.mainwidget)

        self.imageWidget.setGeometry(QtCore.QRect(100, 0, 981, 711))
        self.imageWidget.setMouseTracking(False)
        self.imageWidget.setResizeMode(QtQuickWidgets.QQuickWidget.SizeRootObjectToView)
        self.imageWidget.setObjectName("imageWidget")
        self.blurButton = QtWidgets.QPushButton(self.mainwidget)
        self.blurButton.setGeometry(QtCore.QRect(0, 0, 101, 51))
        self.blurButton.setObjectName("blurButton")
        self.horizontalSlider = QtWidgets.QSlider(self.mainwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(100, 720, 481, 21))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        MainWindow.setCentralWidget(self.mainwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 26))
        self.menubar.setObjectName("menubar")
        self.menumenu = QtWidgets.QMenu(self.menubar)
        self.menumenu.setObjectName("menumenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionfile_import = QtWidgets.QAction(MainWindow)
        self.actionfile_import.setObjectName("actionfile_import")
        self.actionfile_save = QtWidgets.QAction(MainWindow)
        self.actionfile_save.setObjectName("actionfile_save")
        self.menumenu.addAction(self.actionfile_import)
        self.menumenu.addAction(self.actionfile_save)
        self.menubar.addAction(self.menumenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.blurButton.setText(_translate("MainWindow", "PushButton"))
        self.menumenu.setTitle(_translate("MainWindow", "menu"))
        self.actionfile_import.setText(_translate("MainWindow", "file_import"))
        self.actionfile_save.setText(_translate("MainWindow", "file_save"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
