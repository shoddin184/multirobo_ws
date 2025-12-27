#!/usr/bin/env python3
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

pyqtgraph.setConfigOption('background', 'w')
pyqtgraph.setConfigOption('foreground', 'k')

ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
AVERAGE_WINDOW = 10


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('graph')
        self.window = window

        for robot_name in ROBOT_NAMES:
            self.create_subscription(
                Float32MultiArray,
                f'/{robot_name}/result',
                lambda msg, rn=robot_name: self.window.receive_data(msg, rn),
                10)


class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle('Result (Multi-Robot)')
        self.setGeometry(50, 50, 600, 650)

        self.episode_count = 0
        self.current_episode_data = {
            rn: {'score': None, 'q_value': None, 'received': False}
            for rn in ROBOT_NAMES
        }

        self.episode_rewards = []
        self.episode_q_values = []

        self.avg_ep = []
        self.avg_rewards = []
        self.avg_q_values = []

        self._init_plots()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True)
        self.ros_thread.start()

    def receive_data(self, msg, robot_name):
        self.current_episode_data[robot_name]['score'] = msg.data[0]
        self.current_episode_data[robot_name]['q_value'] = msg.data[1]
        self.current_episode_data[robot_name]['received'] = True

        if not all(self.current_episode_data[rn]['received'] for rn in ROBOT_NAMES):
            return

        total_reward = sum(self.current_episode_data[rn]['score'] for rn in ROBOT_NAMES)
        total_q_value = sum(self.current_episode_data[rn]['q_value'] for rn in ROBOT_NAMES)

        self.episode_count += 1
        self.episode_rewards.append(total_reward)
        self.episode_q_values.append(total_q_value)

        if self.episode_count % AVERAGE_WINDOW == 0:
            start_idx = self.episode_count - AVERAGE_WINDOW
            self.avg_ep.append(self.episode_count)
            self.avg_rewards.append(np.mean(self.episode_rewards[start_idx:self.episode_count]))
            self.avg_q_values.append(np.mean(self.episode_q_values[start_idx:self.episode_count]))

        for rn in ROBOT_NAMES:
            self.current_episode_data[rn] = {'score': None, 'q_value': None, 'received': False}

    def _init_plots(self):
        self.qValuePlt = pyqtgraph.PlotWidget(
            self, title='Average max Q-value (Sum of all robots, 10-episode avg)')
        self.qValuePlt.setGeometry(0, 320, 600, 300)
        self.qValuePlt.setLabel('bottom', 'Episode')
        self.qValuePlt.setLabel('left', 'Q-value')

        self.rewardsPlt = pyqtgraph.PlotWidget(
            self, title='Total reward (Sum of all robots, 10-episode avg)')
        self.rewardsPlt.setGeometry(0, 10, 600, 300)
        self.rewardsPlt.setLabel('bottom', 'Episode')
        self.rewardsPlt.setLabel('left', 'Reward')

        for pltw in (self.rewardsPlt, self.qValuePlt):
            pltw.showGrid(x=True, y=True, alpha=0.3)
            pltw.showAxis('top')
            pltw.showAxis('right')

            for axis_name in ('bottom', 'left', 'top', 'right'):
                axis = pltw.getAxis(axis_name)
                axis.setPen(pyqtgraph.mkPen('k'))
                show_values = axis_name in ('bottom', 'left')
                axis.setStyle(tickLength=0, showValues=show_values)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(200)

        self.show()

    def _update(self):
        if not self.avg_ep:
            self.rewardsPlt.clear()
            self.qValuePlt.clear()
            return

        max_ep = max(self.avg_ep)
        x_spacing = self._calc_tick_spacing(max_ep)

        if self.avg_rewards:
            reward_range = max(self.avg_rewards) - min(self.avg_rewards)
            reward_max = max(abs(max(self.avg_rewards)), abs(min(self.avg_rewards)))
            y_reward_spacing = self._calc_tick_spacing(max(reward_range, reward_max))
        else:
            y_reward_spacing = 100

        if self.avg_q_values:
            q_range = max(self.avg_q_values) - min(self.avg_q_values)
            q_max = max(abs(max(self.avg_q_values)), abs(min(self.avg_q_values)))
            y_q_spacing = self._calc_tick_spacing(max(q_range, q_max))
        else:
            y_q_spacing = 10

        try:
            self.rewardsPlt.getAxis('bottom').setTickSpacing(major=x_spacing, minor=x_spacing)
            self.rewardsPlt.getAxis('left').setTickSpacing(major=y_reward_spacing, minor=y_reward_spacing)
            self.qValuePlt.getAxis('bottom').setTickSpacing(major=x_spacing, minor=x_spacing)
            self.qValuePlt.getAxis('left').setTickSpacing(major=y_q_spacing, minor=y_q_spacing)
        except AttributeError:
            pass

        self.rewardsPlt.plot(
            self.avg_ep, self.avg_rewards,
            pen=pyqtgraph.mkPen('r', width=2),
            symbol='o', symbolSize=5, symbolBrush='r', clear=True)
        self.qValuePlt.plot(
            self.avg_ep, self.avg_q_values,
            pen=pyqtgraph.mkPen('g', width=2),
            symbol='o', symbolSize=5, symbolBrush='g', clear=True)

    def _calc_tick_spacing(self, value_range):
        if value_range <= 0:
            return 10

        magnitude = 10 ** int(np.floor(np.log10(value_range)))

        if value_range / magnitude <= 2:
            spacing = magnitude / 5
        elif value_range / magnitude <= 5:
            spacing = magnitude / 2
        else:
            spacing = magnitude

        return max(spacing, 1)

    def closeEvent(self, event):
        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        if win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
