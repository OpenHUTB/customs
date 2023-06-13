# -*- coding:utf-8 -*-
# author:peng
# Date：2023/4/18 21:54
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGraphicsPixmapItem, QGraphicsScene, QFileDialog
from PyQt5.QtGui import QIcon, QPalette, QBrush, QPixmap, QImage
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QMovie
from PyQt5 import QtCore

import sys

from PyQt5.uic.properties import QtGui
from matplotlib import pyplot as plt


class Ico(QWidget):
    def __init__(self):
        super().__init__()
        self.lbl = QLabel(self)
        self.lbl2 = QLabel(self)
        self.initui()


    def initui(self):
        font_z = QFont()
        font_z.setFamily('华文行楷')
        font_z.setBold(True)
        font_z.setPointSize(9)
        font_z.setWeight(50)

        self.setGeometry(300 * 2, 300 * 2, 450 * 2, 400 * 2)
        self.setWindowTitle('人脸检测识别系统')
        palette = QPalette()
        palette.setBrush(QPalette.Background, QBrush(QPixmap("./img/img_6.png")))
        self.setPalette(palette)

        x1 = QPushButton('图片录入', self)
        x1.setFont(font_z)
        x1.setIcon(QIcon("./img/img.png"))
        x1.resize(85 + 20, 32 + 10)
        x1.move(50, 130 * 2)
        x1.clicked.connect(self.picLoader)

        x2 = QPushButton('相机录入', self)
        x2.setFont(font_z)
        x2.setIcon(QIcon("./img/img_1.png"))
        x2.resize(85 + 20, 32 + 10)
        x2.move(50, 80 * 2)

        x4 = QPushButton('图片识别', self)
        x4.setFont(font_z)
        x4.setIcon(QIcon("./img/img_2.png"))
        x4.resize(85 + 20, 32 + 10)
        x4.move(50, 230 * 2)

        qbtn = QPushButton('相机识别', self)
        qbtn.setFont(font_z)
        qbtn.setIcon(QIcon("./img/img_3.png"))
        qbtn.clicked.connect(QCoreApplication.instance().quit)
        qbtn.resize(85 + 20, 32 + 10)
        qbtn.move(50, 280 * 2)


        pixmap = QPixmap('./img/zhangxueyou.jpg')  # 按指定路径找到图片
        self.lbl.setPixmap(pixmap)  # 在label上显示图片
        self.lbl.setScaledContents(True)  # 让图片自适应label大小
        self.lbl.resize(400, 350)
        self.lbl.move(300, 20)
        self.lbl.hide()


        # pixmap = QPixmap('./img/zhangxueyou.jpg')  # 按指定路径找到图片
        self.lbl2.setPixmap(pixmap)  # 在label上显示图片
        self.lbl2.setScaledContents(True)  # 让图片自适应label大小
        self.lbl2.resize(400, 350)
        self.lbl2.move(300, 390)
        self.lbl2.hide()

        self.show()

    def picLoader(self):
        imgName, imgType = QFileDialog.getOpenFileName(self,"打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.lbl.width(), self.lbl.height())
        self.lbl.setPixmap(jpg)
        self.lbl.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Ico()
    sys.exit(app.exec_())
