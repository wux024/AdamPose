#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: Adam Wiveion
@contact: wux024@nenu.edu.cn
@time: 2021/5/26 12:08
"""
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import pandas as pd
import cv2
import yaml
import os
import pyqtgraph as pg
import numpy as np
from operator import itemgetter

from .AdamPoseUI import Ui_AdamPose
from .matplotlib_qt_figure import *
from data_analysis import *


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class AdamPoseMainWindow(QMainWindow, Ui_AdamPose):
    def __init__(self):
        super(AdamPoseMainWindow, self).__init__()
        self.setupUi(self)
        self.parameter_init()
        self.gui_init()

    def parameter_init(self):

        self.pause_flag = False
        self.static_flag = False

        self.key_points = None
        self.key_points_hash = None
        self.angle_key_points = None
        self.body_parts_sub = None

        self.config = None
        self.imgs = None
        self.ixs = None

        self.colors = None

        self.pose_angles_display = False
        self.displacement_display = False
        self.velocity_display = False

        self.data_file_path = None
        self.video_file_path = None

        self.time_seq = None
        self.displacement = None
        self.velocity = None
        self.fourier = None
        self.freq_seq = None
        self.pose_angles = None

        self.cap = None
        self.ts = None
        self.fps = None
        self.frame_number = None
        self.frame_width = None
        self.frame_height = None
        self.scale_width = None

        self.all_gui_contrl = ['Start','Pause','Static','Quit']

        self.all_show_contrl = ['Defaulat', 'PoseEstimation',
                           'Trace3D', 'Trace2D',
                           'Dispalcement', 'Velocity',
                           'AngleEstimation', 'Fourier']

        self.all_show_states = dict(zip(self.all_show_contrl, [False] * len(self.all_show_contrl)))


        self.COLORS = None
        self.mCOLORS = None
        self.axis_flag = False

        self.margin = 1.5
        self.count = 0
        self.video_frame_count = 0


    # Initialize the GUI
    def gui_init(self):
        # Turn off all function keys except Quit
        self.EditConfigFile.setEnabled(False)
        self.Start.setEnabled(False)
        self.Static.setEnabled(False)
        self.Pause.setEnabled(False)
        self.set_button_group()
        self.show_contrl_disable()
        # Link button and callback function
        self.link_file_contrl()
        self.link_gui_contrl()
        self.link_show_contrl()

    def set_button_group(self):
        self.button_group = QButtonGroup()
        for show_contrl_id, show_contrl in enumerate(self.all_show_contrl):
            exec('self.button_group.addButton(self.%sShow, show_contrl_id)'%show_contrl)
        self.button_group.setExclusive(False)

    def show_contrl_disable(self):
        for show_contrl in self.all_show_contrl:
            exec('self.%sShow.setEnabled(False)' % show_contrl)

    def show_contrl_eable(self):
        for show_contrl in self.all_show_contrl:
            exec('self.%sShow.setEnabled(True)' % show_contrl)

    # Link button and callback function
    def link_file_contrl(self):
        self.LoadConfigFile.clicked.connect(self.load_config_file)
        self.EditConfigFile.clicked.connect(self.edit_config_file)

    def link_gui_contrl(self):
        self.Start.clicked.connect(self.start)
        self.Pause.clicked.connect(self.pause_to_continue)
        self.Static.clicked.connect(self.static_to_dynamic)
        self.Quit.clicked.connect(self.quit)

    def link_show_contrl(self):
        self.button_group.buttonClicked[int].connect(self.show_contrl)

    def show_contrl(self, id):
        if id == 0 and not self.all_show_states[self.all_show_contrl[id]]:
            self.all_show_states[self.all_show_contrl[id]] = True
            for show_contrl in self.all_show_contrl[1:]:
                self.all_show_states[show_contrl] = False
                exec('self.%sShow.setChecked(False)' % show_contrl)
                exec('self.%sShow.setEnabled(False)' % show_contrl)
        elif id == 0 and self.all_show_states[self.all_show_contrl[id]]:
            self.all_show_states[self.all_show_contrl[id]] = False
            self.show_contrl_eable()
        else:
            if not self.all_show_states[self.all_show_contrl[id]]:
                self.all_show_states[self.all_show_contrl[id]] = True
            else:
                self.all_show_states[self.all_show_contrl[id]] = False

    # Callbacks for all keys on the GUI
    def load_config_file(self):
        """
        The callback of self.LoadConfigFile
        """
        _translate = QCoreApplication.translate
        self.key_points_selection_clear()
        self.config_file_path, _ = QFileDialog.getOpenFileName(self, 'OpenFile', '.')
        if self.config_file_path:
            self.config_file_load()
        else:
            return
        self.EditConfigFile.setEnabled(True)
        self.pose_parameter_init()
        self.create_keypoints_selection()
        self.create_colors()
        self.show_contrl_eable()
        self.Start.setEnabled(True)

    def edit_config_file(self):
        """
        The callback of self.EditConfigFile
        """
        path = os.getcwd()
        if self.config_file_path:
            md = "C:\\Windows\\System32\\notepad.exe" + " " + self.config_file_path
            os.system(md)
        else:
            return
        os.chdir(path)
        self.Start.setEnabled(True)

    def start(self):
        _translate = QCoreApplication.translate
        p3ds = self.get_data_file()
        self.get_video_file()
        # data pre-process
        self.data_preprocess(p3ds)
        self.Static.setEnabled(True)
        self.Pause.setEnabled(True)
        self.Start.setEnabled(False)
        if self.pause_flag:
            self.body_parts_sub = []
            self.count = 0
            self.video_frame_count = 0
            self.figure_clear()
            self.draw_figure_init()
            self.pause_flag = False
            self.Pause.setText(_translate("AdamPose", "Pause"))
            self.timer.start(int(self.ts * 1000))
            return

        self.visualize_init()
        self.timer_start()

    def key_points_selection_clear(self):
        [self.KeyPointsSelection.itemAt(i).widget().deleteLater()
         for i in range(self.KeyPointsSelection.count())
         ]

    def config_file_load(self):
        file = open(self.config_file_path, 'r', encoding='utf-8')
        cfg = file.read()
        self.config = yaml.full_load(cfg)

    def pose_parameter_init(self):
        self.key_points = self.config['keypoints']
        self.key_points_hash = dict(zip(self.key_points, range(len(self.key_points))))
        self.angle_key_points = self.config['angle_keypoints']
        self.fps = self.config['fps']
        self.ts = 1.0 / self.fps

    # Creates a selectable keypoint array on the GUI based on the information provided by the configuration file
    def create_keypoints_selection(self):
        _translate = QCoreApplication.translate
        # Set font
        font = QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        for keypoint_id, keypoint in enumerate(self.key_points):
            i = keypoint_id // 7
            j = keypoint_id % 7
            exec('self.keypoint%s=QCheckBox(self.layoutWidget_3)' % keypoint_id)
            exec('self.keypoint%s.setObjectName("keypoint%s")' % (keypoint_id, keypoint_id))
            exec('self.KeyPointsSelection.addWidget(self.keypoint%s, %d, %d, 1, 1)' % (keypoint_id, i, j))
            exec('self.keypoint%s.setText(_translate("AdamPose", "%s"))' % (keypoint_id, keypoint))
            exec('self.keypoint%s.setFont(font)' % keypoint_id)
            exec('self.keypoint%s.stateChanged.connect(self.select_keypoints)' % keypoint_id)

    def create_colors(self):
        self.COLORS = dict()
        self.mCOLORS = dict()
        for key_point in self.key_points:
            self.COLORS[key_point] = np.random.randint(0, 255, size=3, dtype="uint8")
            self.mCOLORS[key_point] = np.random.rand(3)

    # The callback of keypoint array
    def select_keypoints(self):
        self.body_parts_sub = []
        for keypoint_id in range(len(self.key_points)):
            exec('self.body_parts_sub += [self.keypoint%s.text()] if self.keypoint%s.isChecked() else []'
                 % (keypoint_id, keypoint_id))


    # Get raw data for pose estimation
    def get_data_file(self):
        self.data_file_path = self.config['data_file_path']
        if self.data_file_path:
            return pd.read_csv(self.data_file_path)
        else:
            return

    # Get raw video for pose estimation
    def get_video_file(self):
        self.video_file_path = self.config['video_file_path']
        if self.video_file_path:
            self.cap = cv2.VideoCapture(self.video_file_path)
            self.frame_number = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            return

    def data_preprocess(self, p3ds):

        # displacement
        self.displacement = [np.array(
            list(itemgetter(*[key_point + '_x', key_point + '_y', key_point + '_z'])(p3ds))).T
                             for key_point in self.key_points
                             ]
        # velocity
        self.time_seq = np.linspace(0, self.displacement[0].shape[0],
                                    self.displacement[0].shape[0], endpoint=False) * self.ts
        self.velocity = [np.array(
            [cal_deriv(self.time_seq, displacement[:,0]),
             cal_deriv(self.time_seq, displacement[:,1]),
             cal_deriv(self.time_seq, displacement[:,2])]).T
                         for displacement in self.displacement
                         ]
        # fourier
        self.fourier = [np.array(
            [fourier_analysis(displacement[:,0], self.fps),
             fourier_analysis(displacement[:,1], self.fps),
             fourier_analysis(displacement[:,2], self.fps)]).T
                        for displacement in self.displacement
                        ]

        self.freq_seq, _ = fourier_analysis(self.displacement[0][:,0], self.fps, retype = True)

        # pose angles
        angle_key_point_val = itemgetter(*self.angle_key_points)(self.key_points_hash)
        vecs = itemgetter(*angle_key_point_val)(self.displacement)
        self.pose_angles = compute_pose_angles(vecs, rand=False)

        self.displacement = np.array(self.displacement)
        self.velocity = np.array(self.velocity)
        self.fourier = np.array(self.fourier)

        # max min
        self.displacement_x_max = np.max(self.displacement[:, :, 0])
        self.displacement_y_max = np.max(self.displacement[:, :, 1])
        self.displacement_z_max = np.max(self.displacement[:, :, 2])
        self.displacement_x_min = np.min(self.displacement[:, :, 0])
        self.displacement_y_min = np.min(self.displacement[:, :, 1])
        self.displacement_z_min = np.min(self.displacement[:, :, 2])

        self.velocity_x_max = np.max(self.velocity[:, :, 0])
        self.velocity_y_max = np.max(self.velocity[:, :, 1])
        self.velocity_z_max = np.max(self.velocity[:, :, 2])
        self.velocity_x_min = np.min(self.velocity[:, :, 0])
        self.velocity_y_min = np.min(self.velocity[:, :, 1])
        self.velocity_z_min = np.min(self.velocity[:, :, 2])

        self.fourier_x_max = np.max(self.fourier[:, :, 0])
        self.fourier_y_max = np.max(self.fourier[:, :, 1])
        self.fourier_z_max = np.max(self.fourier[:, :, 2])
        self.fourier_x_min = np.min(self.fourier[:, :, 0])
        self.fourier_y_min = np.min(self.fourier[:, :, 1])
        self.fourier_z_min = np.min(self.fourier[:, :, 2])

        self.pose_angles_x_max = np.max(self.pose_angles[:, 0])
        self.pose_angles_y_max = np.max(self.pose_angles[:, 1])
        self.pose_angles_z_max = np.max(self.pose_angles[:, 2])
        self.pose_angles_x_min = np.min(self.pose_angles[:, 0])
        self.pose_angles_y_min = np.min(self.pose_angles[:, 1])
        self.pose_angles_z_min = np.min(self.pose_angles[:, 2])

    def pause_to_continue(self):
        _translate = QCoreApplication.translate
        if not self.pause_flag:
            self.timer.stop()
            self.pause_flag = True
            self.Pause.setText(_translate("AdamPose", "Continue"))
        else:
            self.timer.start()
            self.pause_flag = False
            self.Pause.setText(_translate("AdamPose", "Pause"))

    def static_to_dynamic(self):
        _translate = QCoreApplication.translate
        if not self.static_flag:
            self.static_flag = True
            self.Static.setText(_translate("AdamPose", "Dynamic"))
        else:
            self.figure_clear()
            self.static_flag = False
            self.Static.setText(_translate("AdamPose", "Static"))

    def quit(self):
        QApplication.instance().quit()

    def visualize_init(self):
        self.figure_layout_create()
        self.figure_create()
        self.figure_set()
        self.add_figure()

    def figure_layout_create(self):
        self.PoseEstimation = QVBoxLayout(self.PoseEstimation2D3D)
        self.LineFigureLayoutTrace3D = QHBoxLayout(self.Trace3D)
        self.LineFigureLayoutTrace2D = QHBoxLayout(self.Trace2D)
        self.LineFigureLayoutDis = QHBoxLayout(self.Displacement)
        self.LineFigureLayoutVel = QHBoxLayout(self.Velocity)
        self.LineFigureLayoutAngle = QHBoxLayout(self.AngleEstimation)
        self.LineFigureLayoutFou = QHBoxLayout(self.Fourier)

    def figure_create(self):

        self.PoseEstimation2D3DVideo = QLabel()
        self.Trace3DXYZ = Figure3D()

        self.Trace2DXOY = pg.PlotWidget()
        self.Trace2DYOZ = pg.PlotWidget()
        self.Trace2DXOZ = pg.PlotWidget()

        self.DisplacementX = pg.PlotWidget()
        self.DisplacementY = pg.PlotWidget()
        self.DisplacementZ = pg.PlotWidget()

        self.FourierX = pg.PlotWidget()
        self.FourierY = pg.PlotWidget()
        self.FourierZ = pg.PlotWidget()

        self.VelocityX = pg.PlotWidget()
        self.VelocityY = pg.PlotWidget()
        self.VelocityZ = pg.PlotWidget()

        self.AngleFlex = pg.PlotWidget()
        self.AngleAxis = pg.PlotWidget()
        self.AngleCross = pg.PlotWidget()

    def figure_set(self):
        self.PoseEstimation2D3DVideo.setAlignment(Qt.AlignCenter)
        self.draw_figure_init()

    def add_figure(self):
        self.PoseEstimation.addWidget(self.PoseEstimation2D3DVideo)

        self.LineFigureLayoutTrace3D.addWidget(self.Trace3DXYZ)

        self.LineFigureLayoutTrace2D.addWidget(self.Trace2DXOY)
        self.LineFigureLayoutTrace2D.addWidget(self.Trace2DYOZ)
        self.LineFigureLayoutTrace2D.addWidget(self.Trace2DXOZ)

        self.LineFigureLayoutDis.addWidget(self.DisplacementX)
        self.LineFigureLayoutDis.addWidget(self.DisplacementY)
        self.LineFigureLayoutDis.addWidget(self.DisplacementZ)

        self.LineFigureLayoutVel.addWidget(self.VelocityX)
        self.LineFigureLayoutVel.addWidget(self.VelocityY)
        self.LineFigureLayoutVel.addWidget(self.VelocityZ)

        self.LineFigureLayoutAngle.addWidget(self.AngleFlex)
        self.LineFigureLayoutAngle.addWidget(self.AngleAxis)
        self.LineFigureLayoutAngle.addWidget(self.AngleCross)

        self.LineFigureLayoutFou.addWidget(self.FourierX)
        self.LineFigureLayoutFou.addWidget(self.FourierY)
        self.LineFigureLayoutFou.addWidget(self.FourierZ)

    def timer_start(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.frame_count)
        self.timer.start(int(self.ts * 1000))

    def frame_count(self):
        if self.all_show_states['Defaulat'] or self.all_show_states['PoseEstimation']:
            self.video_frame()
        if self.body_parts_sub:
            self.draw_figure()
        if len(self.all_show_states.values())>1:
            self.count += 1
        if self.count == self.frame_number:
            self.count = 0
            self.video_frame_count = 0
            self.cap = cv2.VideoCapture(self.video_file_path)
            self.figure_clear()

    def video_frame(self):
        _, frame = self.cap.read()
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (517, 267))
        Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        self.PoseEstimation2D3DVideo.setPixmap(QPixmap.fromImage(Qframe))

    def draw_figure(self):
        for body_part in self.body_parts_sub:
            body_part_hash = self.key_points_hash[body_part]
            if not self.static_flag:
                displacement = self.displacement[body_part_hash][:self.count, :]
                velocity = self.velocity[body_part_hash][:self.count, :]
                fourier = self.fourier[body_part_hash]
                pose_angles = self.pose_angles[:self.count, :]
                time_seq = self.time_seq[:self.count]
            else:
                body_part_hash = self.key_points_hash[body_part]
                displacement = self.displacement[body_part_hash]
                velocity = self.velocity[body_part_hash]
                fourier = self.fourier[body_part_hash]
                pose_angles = self.pose_angles
                time_seq = self.time_seq
            self.all_show_contrl = ['Defaulat', 'PoseEstimation',
                                    'Trace3D', 'Trace2D',
                                    'Dispalcement', 'Velocity',
                                    'AngleEstimation', 'Fourier']
            if self.all_show_states['Defaulat'] or self.all_show_states['Trace3D']:
                if not self.static_flag:
                    self.draw_3d_trace(body_part, displacement[-2:,:])
                else:
                    self.draw_3d_trace(body_part, displacement)
            if self.all_show_states['Defaulat'] or self.all_show_states['Trace2D']:
                self.draw_2d_trace(body_part, displacement)
            if self.all_show_states['Defaulat'] or self.all_show_states['Dispalcement']:
                self.draw_displacement(time_seq, body_part, displacement)
            if self.all_show_states['Defaulat'] or self.all_show_states['Velocity']:
                self.draw_velocity(time_seq, body_part, velocity)
            if self.all_show_states['Defaulat'] or self.all_show_states['AngleEstimation']:
                self.draw_pose_angle(time_seq, body_part, pose_angles)
            if self.all_show_states['Defaulat'] or self.all_show_states['Fourier']:
                self.draw_fourier(body_part, fourier)

    def draw_figure_init(self):
        self.draw_3d_trace_init()
        self.draw_2d_trace_init()
        self.draw_displacement_init()
        self.draw_velocity_init()
        self.draw_pose_angle_init()
        self.draw_fourier_init()

    def draw_3d_trace_init(self):
        # self.Trace3DXYZ.ax.set_axis_off()
        self.Trace3DXYZ.fig.tight_layout()
        self.Trace3DXYZ.fig.subplots_adjust(top=1, left=0, right=1, hspace=0, wspace=0, bottom=0)
        self.Trace3DXYZ.ax.grid(False)
        # Set the coordinate range
        self.Trace3DXYZ.ax.set_xlim(self.displacement_x_min - self.margin * abs(self.displacement_x_min),
                                    self.displacement_x_max * self.margin)
        self.Trace3DXYZ.ax.set_ylim(self.displacement_y_min - self.margin * abs(self.displacement_y_min),
                                    self.displacement_y_max * self.margin)
        self.Trace3DXYZ.ax.set_zlim(self.displacement_z_min - self.margin * abs(self.displacement_z_min),
                                    self.displacement_z_max * self.margin)

    def draw_3d_axis(self):
        fontdict = {'family': 'Times New Roman', 'size': 19, 'style': 'italic'}
        self.Trace3DXYZ.ax.quiver(0, 0, 0, 1, 0, 0, length=60, normalize=True, color='r', arrow_length_ratio=0.12)
        self.Trace3DXYZ.ax.text(65, 0, 0, 'x', fontdict=fontdict)
        self.Trace3DXYZ.ax.quiver(0, 0, 0, 0, 1, 0, length=60, normalize=True, color='g', arrow_length_ratio=0.12)
        self.Trace3DXYZ.ax.text(0, 65, 0, 'y', fontdict=fontdict)
        self.Trace3DXYZ.ax.quiver(0, 0, 0, 0, 0, 1, length=60, normalize=True, color='b', arrow_length_ratio=0.12)
        self.Trace3DXYZ.ax.text(0, 0, 65, 'z', fontdict=fontdict)
        self.axis_flag = True

    def draw_3d_trace(self, body_part, displacement):
        self.Trace3DXYZ.ax.plot(displacement[:, 0],
                                displacement[:, 1],
                                displacement[:, 2],
                                color=self.mCOLORS[body_part])
        self.Trace3DXYZ.draw()

    def draw_2d_trace_init(self):
        label_style = {"font-family": "Times", "font-size": "12pt"}
        self.Trace2DXOY.setLabel('left', 'Y', 'mm', **label_style)
        self.Trace2DYOZ.setLabel('left', 'Z', 'mm', **label_style)
        self.Trace2DXOZ.setLabel('left', 'Z', 'mm', **label_style)

        self.Trace2DXOY.setLabel('bottom', 'X', 'mm', **label_style)
        self.Trace2DYOZ.setLabel('bottom', 'Y', 'mm', **label_style)
        self.Trace2DXOZ.setLabel('bottom', 'X', 'mm', **label_style)

        self.Trace2DXOY.setXRange(self.displacement_x_min - self.margin * abs(self.displacement_x_min),
                                  self.displacement_x_max * self.margin)
        self.Trace2DYOZ.setXRange(self.displacement_y_min - self.margin * abs(self.displacement_y_min),
                                  self.displacement_y_max * self.margin)
        self.Trace2DXOZ.setXRange(self.displacement_x_min - self.margin * abs(self.displacement_x_min),
                                  self.displacement_x_max * self.margin)

        self.Trace2DXOY.setYRange(self.displacement_y_min - self.margin * abs(self.displacement_y_min),
                                  self.displacement_y_max * self.margin)
        self.Trace2DYOZ.setYRange(self.displacement_z_min - self.margin * abs(self.displacement_z_min),
                                  self.displacement_z_max * self.margin)
        self.Trace2DXOZ.setYRange(self.displacement_z_min - self.margin * abs(self.displacement_z_min),
                                  self.displacement_z_max * self.margin)

    def draw_2d_trace(self, body_part, displacement):
        self.Trace2DXOY.plot(displacement[:, 0], displacement[:, 1], pen=self.COLORS[body_part])
        self.Trace2DYOZ.plot(displacement[:, 1], displacement[:, 2], pen=self.COLORS[body_part])
        self.Trace2DXOZ.plot(displacement[:, 0], displacement[:, 2], pen=self.COLORS[body_part])

    def draw_displacement_init(self):
        label_style = {"font-family": "Times", "font-size": "12pt"}
        self.DisplacementX.setLabel('left', 'displacement', 'mm', **label_style)
        self.DisplacementY.setLabel('left', 'displacement', 'mm', **label_style)
        self.DisplacementZ.setLabel('left', 'displacement', 'mm', **label_style)

        self.DisplacementX.setLabel('bottom', 'Time', 's', **label_style)
        self.DisplacementY.setLabel('bottom', 'Time', 's', **label_style)
        self.DisplacementZ.setLabel('bottom', 'Time', 's', **label_style)

        self.DisplacementX.setXRange(0, self.frame_number/self.fps)
        self.DisplacementY.setXRange(0, self.frame_number/self.fps)
        self.DisplacementZ.setXRange(0, self.frame_number/self.fps)

        self.DisplacementX.setYRange(self.displacement_x_min - self.margin * abs(self.displacement_x_min),
                                     self.displacement_x_max * self.margin)
        self.DisplacementY.setYRange(self.displacement_y_min - self.margin * abs(self.displacement_y_min),
                                     self.displacement_y_max * self.margin)
        self.DisplacementZ.setYRange(self.displacement_z_min - self.margin * abs(self.displacement_z_min),
                                     self.displacement_z_max * self.margin)

    def draw_displacement(self, t, body_part, displacement):
        self.DisplacementX.plot(t, displacement[:, 0], pen=self.COLORS[body_part])
        self.DisplacementY.plot(t, displacement[:, 1], pen=self.COLORS[body_part])
        self.DisplacementZ.plot(t, displacement[:, 2], pen=self.COLORS[body_part])

    def draw_velocity_init(self):
        label_style = {"font-family": "Times", "font-size": "12pt"}
        self.VelocityX.setLabel('left', 'Velocity', 'mm/s', **label_style)
        self.VelocityY.setLabel('left', 'Velocity', 'mm/s', **label_style)
        self.VelocityZ.setLabel('left', 'Velocity', 'mm/s', **label_style)

        self.VelocityX.setLabel('bottom', 'Time', 's', **label_style)
        self.VelocityY.setLabel('bottom', 'Time', 's', **label_style)
        self.VelocityZ.setLabel('bottom', 'Time', 's', **label_style)

        self.VelocityX.setXRange(0, self.frame_number / self.fps)
        self.VelocityY.setXRange(0, self.frame_number / self.fps)
        self.VelocityZ.setXRange(0, self.frame_number / self.fps)

        self.VelocityX.setYRange(self.velocity_x_min - self.margin * abs(self.velocity_x_min),
                                 self.velocity_x_max * self.margin)
        self.VelocityY.setYRange(self.velocity_y_min - self.margin * abs(self.velocity_y_min),
                                 self.velocity_y_max * self.margin)
        self.VelocityZ.setYRange(self.velocity_z_min - self.margin * abs(self.velocity_z_min),
                                 self.velocity_z_max * self.margin)

    def draw_velocity(self, t, body_part, velocity):
        self.VelocityX.plot(t, velocity[:, 0], pen=self.COLORS[body_part])
        self.VelocityY.plot(t, velocity[:, 1], pen=self.COLORS[body_part])
        self.VelocityZ.plot(t, velocity[:, 2], pen=self.COLORS[body_part])

    def draw_pose_angle_init(self):
        label_style = {"font-family": "Times", "font-size": "12pt"}

        self.AngleFlex.setLabel('left', 'Flex Angle', 'rand', **label_style)
        self.AngleAxis.setLabel('left', 'Axis Angle', 'rand', **label_style)
        self.AngleCross.setLabel('left', 'Cross Axis Angle', 'rand', **label_style)

        self.AngleFlex.setLabel('bottom', 'Time', 's', **label_style)
        self.AngleAxis.setLabel('bottom', 'Time', 's', **label_style)
        self.AngleCross.setLabel('bottom', 'Time', 's', **label_style)

        self.AngleFlex.setXRange(0, self.frame_number / self.fps)
        self.AngleAxis.setXRange(0, self.frame_number / self.fps)
        self.AngleCross.setXRange(0, self.frame_number / self.fps)

        self.AngleFlex.setYRange(self.pose_angles_x_min - self.margin * abs(self.pose_angles_x_min),
                                 self.pose_angles_x_max * self.margin)
        self.AngleAxis.setYRange(self.pose_angles_y_min - self.margin * abs(self.pose_angles_y_min),
                                 self.pose_angles_y_max * self.margin)
        self.AngleCross.setYRange(self.pose_angles_y_min - self.margin * abs(self.pose_angles_y_min),
                                  self.pose_angles_y_max * self.margin)


    def draw_pose_angle(self, t, body_part, pose_angles):
        self.AngleFlex.plot(t, pose_angles[:, 0], pen=self.COLORS[body_part])
        self.AngleAxis.plot(t, pose_angles[:, 1], pen=self.COLORS[body_part])
        self.AngleCross.plot(t, pose_angles[:, 2], pen=self.COLORS[body_part])


    def draw_fourier_init(self):
        label_style = {"font-family": "Times", "font-size": "12pt"}
        self.FourierX.setLabel('left', 'Amplitude', 'dB', **label_style)
        self.FourierY.setLabel('left', 'Amplitude', 'dB', **label_style)
        self.FourierZ.setLabel('left', 'Amplitude', 'dB', **label_style)

        self.FourierX.setLabel('bottom', 'Frequency', 'Hz', **label_style)
        self.FourierY.setLabel('bottom', 'Frequency', 'Hz', **label_style)
        self.FourierZ.setLabel('bottom', 'Frequency', 'Hz', **label_style)

        self.FourierX.setXRange(0, self.fps / 2)
        self.FourierY.setXRange(0, self.fps / 2)
        self.FourierZ.setXRange(0, self.fps / 2)

    def draw_fourier(self, body_part, fourier):
        self.FourierX.plot(self.freq_seq, fourier[:, 0], pen=self.COLORS[body_part])
        self.FourierY.plot(self.freq_seq, fourier[:, 1], pen=self.COLORS[body_part])
        self.FourierZ.plot(self.freq_seq, fourier[:, 2], pen=self.COLORS[body_part])

    def figure_clear(self):
        self.Trace3DXYZ.ax.cla()
        self.draw_3d_trace_init()

        self.Trace2DXOY.clear()
        self.Trace2DYOZ.clear()
        self.Trace2DXOZ.clear()

        self.DisplacementX.clear()
        self.DisplacementY.clear()
        self.DisplacementZ.clear()

        self.VelocityX.clear()
        self.VelocityY.clear()
        self.VelocityZ.clear()

        self.Trace2DXOY.clear()
        self.Trace2DYOZ.clear()
        self.Trace2DXOZ.clear()

        self.AngleAxis.clear()
        self.AngleFlex.clear()
        self.AngleCross.clear()

        self.FourierX.clear()
        self.FourierY.clear()
        self.FourierZ.clear()


