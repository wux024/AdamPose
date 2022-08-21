#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/5/26 12:08
"""
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_AdamPose(object):
    def setupUi(self, AdamPose):
        AdamPose.setObjectName("AdamPose")
        AdamPose.resize(1920, 1080)
        self.layoutWidget = QtWidgets.QWidget(AdamPose)
        self.layoutWidget.setGeometry(QtCore.QRect(50, 100, 281, 31))
        self.layoutWidget.setObjectName("layoutWidget")
        self.GUIContrlLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.GUIContrlLayout.setContentsMargins(0, 0, 0, 0)
        self.GUIContrlLayout.setObjectName("GUIContrlLayout")

        self.Start = QtWidgets.QPushButton(self.layoutWidget)
        self.Start.setObjectName("Start")
        self.GUIContrlLayout.addWidget(self.Start)
        self.Static = QtWidgets.QPushButton(self.layoutWidget)
        self.Static.setObjectName("Static")
        self.GUIContrlLayout.addWidget(self.Static)
        self.Pause = QtWidgets.QPushButton(self.layoutWidget)
        self.Pause.setObjectName("Pause")
        self.GUIContrlLayout.addWidget(self.Pause)
        self.Quit = QtWidgets.QPushButton(self.layoutWidget)
        self.Quit.setObjectName("Quit")
        self.GUIContrlLayout.addWidget(self.Quit)

        self.layoutWidget_2 = QtWidgets.QWidget(AdamPose)
        self.layoutWidget_2.setGeometry(QtCore.QRect(50, 30, 281, 61))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.FileContrlLayout = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.FileContrlLayout.setContentsMargins(0, 0, 0, 0)
        self.FileContrlLayout.setObjectName("FileContrlLayout")
        self.LoadConfigFile = QtWidgets.QPushButton(self.layoutWidget_2)
        self.LoadConfigFile.setObjectName("LoadConfigFile")
        self.FileContrlLayout.addWidget(self.LoadConfigFile)
        self.EditConfigFile = QtWidgets.QPushButton(self.layoutWidget_2)
        self.EditConfigFile.setObjectName("EditConfigFile")
        self.FileContrlLayout.addWidget(self.EditConfigFile)

        self.layoutWidget_3 = QtWidgets.QWidget(AdamPose)
        self.layoutWidget_3.setGeometry(QtCore.QRect(360, 30, 1021, 101))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.KeyPointsSelection = QtWidgets.QGridLayout(self.layoutWidget_3)
        self.KeyPointsSelection.setContentsMargins(0, 0, 0, 0)
        self.KeyPointsSelection.setObjectName("gridLayout")

        self.PoseEstimation2D3D = QtWidgets.QGroupBox(AdamPose)
        self.PoseEstimation2D3D.setGeometry(QtCore.QRect(50, 150, 521, 271))
        self.PoseEstimation2D3D.setObjectName("PoseEstimation2D3D")

        self.Trace3D = QtWidgets.QGroupBox(AdamPose)
        self.Trace3D.setGeometry(QtCore.QRect(590, 150, 331, 271))
        self.Trace3D.setObjectName("Trace3D")

        self.Trace2D = QtWidgets.QGroupBox(AdamPose)
        self.Trace2D.setGeometry(QtCore.QRect(50, 430, 871, 271))
        self.Trace2D.setObjectName("Trace2D")

        self.Velocity = QtWidgets.QGroupBox(AdamPose)
        self.Velocity.setGeometry(QtCore.QRect(970, 430, 871, 271))
        self.Velocity.setObjectName("Velocity")

        self.Fourier = QtWidgets.QGroupBox(AdamPose)
        self.Fourier.setGeometry(QtCore.QRect(970, 720, 871, 271))
        self.Fourier.setObjectName("Fourier")

        self.Displacement = QtWidgets.QGroupBox(AdamPose)
        self.Displacement.setGeometry(QtCore.QRect(970, 150, 871, 271))
        self.Displacement.setObjectName("Displacement")

        self.AngleEstimation = QtWidgets.QGroupBox(AdamPose)
        self.AngleEstimation.setGeometry(QtCore.QRect(50, 720, 871, 271))
        self.AngleEstimation.setObjectName("AngleEstimation")

        self.gridLayoutWidget = QtWidgets.QWidget(AdamPose)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(1440, 30, 401, 101))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.ShowContrlLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.ShowContrlLayout.setContentsMargins(0, 0, 0, 0)
        self.ShowContrlLayout.setObjectName("ShowContrlLayout")
        self.AngleEstimationShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.AngleEstimationShow.setObjectName("AngleEstimationShow")
        self.ShowContrlLayout.addWidget(self.AngleEstimationShow, 4, 0, 1, 1)
        self.Trace3DShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.Trace3DShow.setObjectName("Trace3DShow")
        self.ShowContrlLayout.addWidget(self.Trace3DShow, 2, 0, 1, 1)
        self.Trace2DShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.Trace2DShow.setObjectName("Trace2DShow")
        self.ShowContrlLayout.addWidget(self.Trace2DShow, 2, 1, 1, 1)
        self.VelocityShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.VelocityShow.setObjectName("VelocityShow")
        self.ShowContrlLayout.addWidget(self.VelocityShow, 3, 1, 1, 1)
        self.DispalcementShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.DispalcementShow.setObjectName("DispalcementShow")
        self.ShowContrlLayout.addWidget(self.DispalcementShow, 3, 0, 1, 1)
        self.FourierShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.FourierShow.setObjectName("FourierShow")
        self.ShowContrlLayout.addWidget(self.FourierShow, 4, 1, 1, 1)
        self.PoseEstimationShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.PoseEstimationShow.setObjectName("PoseEstimationShow")
        self.ShowContrlLayout.addWidget(self.PoseEstimationShow, 1, 1, 1, 1)
        self.DefaulatShow = QtWidgets.QRadioButton(self.gridLayoutWidget)
        self.DefaulatShow.setObjectName("DefaulatShow")
        self.ShowContrlLayout.addWidget(self.DefaulatShow, 1, 0, 1, 1)

        self.PoseEstimation2D3D.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget_2.raise_()
        self.Trace3D.raise_()
        self.Trace2D.raise_()
        self.Velocity.raise_()
        self.Fourier.raise_()
        self.Displacement.raise_()
        self.AngleEstimation.raise_()
        self.gridLayoutWidget.raise_()

        self.retranslateUi(AdamPose)
        QtCore.QMetaObject.connectSlotsByName(AdamPose)

    def retranslateUi(self, AdamPose):
        _translate = QtCore.QCoreApplication.translate
        AdamPose.setWindowTitle(_translate("AdamPose", "3D Pose Estimation and Motion Parameter Visualization"))
        self.Start.setText(_translate("AdamPose", "Start"))
        self.Static.setText(_translate("AdamPose", "Static"))
        self.Pause.setText(_translate("AdamPose", "Pause"))
        self.Quit.setText(_translate("AdamPose", "Quit"))
        self.LoadConfigFile.setText(_translate("AdamPose", "Load Configuration File"))
        self.EditConfigFile.setText(_translate("AdamPose", "Edit Configuration File"))
        self.PoseEstimation2D3D.setTitle(_translate("AdamPose", "2D/3D Pose Estimation"))
        self.Trace3D.setTitle(_translate("AdamPose", "3D Motion Trajectory"))
        self.Trace2D.setTitle(_translate("AdamPose", "2D Motion Trajectory"))
        self.Velocity.setTitle(_translate("AdamPose", "Velocity"))
        self.Fourier.setTitle(_translate("AdamPose", "Fourier Analysis"))
        self.Displacement.setTitle(_translate("AdamPose", "Displacement"))
        self.AngleEstimation.setTitle(_translate("AdamPose", "Angle Estimation"))
        self.AngleEstimationShow.setText(_translate("AdamPose", "Angle Estimation"))
        self.Trace3DShow.setText(_translate("AdamPose", "3D Motion Trajectory"))
        self.Trace2DShow.setText(_translate("AdamPose", "2D Motion Trajectory"))
        self.VelocityShow.setText(_translate("AdamPose", "Velocity"))
        self.DispalcementShow.setText(_translate("AdamPose", "Displacement"))
        self.FourierShow.setText(_translate("AdamPose", "Fourier Analysis"))
        self.PoseEstimationShow.setText(_translate("AdamPose", "2D/3D Pose Estimation"))
        self.DefaulatShow.setText(_translate("AdamPose", "Default"))

        # Set font
        font1 = QtGui.QFont()
        font1.setFamily("Times New Roman")
        font1.setPointSize(12)
        font1.setBold(False)

        font2 = QtGui.QFont()
        font2.setFamily("Times New Roman")
        font2.setPointSize(18)
        font2.setBold(False)

        self.Start.setFont(font1)
        self.Static.setFont(font1)
        self.Pause.setFont(font1)
        self.Quit.setFont(font1)
        self.LoadConfigFile.setFont(font1)
        self.EditConfigFile.setFont(font1)
        self.PoseEstimation2D3D.setFont(font2)
        self.Trace3D.setFont(font2)
        self.Trace2D.setFont(font2)
        self.Velocity.setFont(font2)
        self.Fourier.setFont(font2)
        self.Displacement.setFont(font2)
        self.AngleEstimation.setFont(font2)