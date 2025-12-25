#!/usr/bin/env python
#################################################################################
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
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

pyqtgraph.setConfigOption('background', 'w')  # 背景を白
pyqtgraph.setConfigOption('foreground', 'k')  # 文字・軸・グリッドを黒

# マルチロボット設定
ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
NUM_ROBOTS = len(ROBOT_NAMES)
AVERAGE_WINDOW = 10  # 10エピソードごとの平均


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('graph')

        self.window = window

        # 各ロボットのresultトピックを購読
        self.result_subscriptions = []
        for robot_name in ROBOT_NAMES:
            sub = self.create_subscription(
                Float32MultiArray,
                f'/{robot_name}/result',
                lambda msg, rn=robot_name: self.data_callback(msg, rn),
                10
            )
            self.result_subscriptions.append(sub)

    def data_callback(self, msg, robot_name):
        self.window.receive_data(msg, robot_name)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('Result (Multi-Robot)')
        self.setGeometry(50, 50, 600, 650)

        # エピソードカウンタ
        self.episode_count = 0

        # 各ロボットの現在のエピソードのデータを保持
        # robot_name -> {'score': float, 'q_value': float, 'received': bool}
        self.current_episode_data = {rn: {'score': None, 'q_value': None, 'received': False}
                                     for rn in ROBOT_NAMES}

        # 全ロボット合算のエピソードごとのデータ
        self.episode_rewards = []  # 各エピソードの報酬の和
        self.episode_q_values = []  # 各エピソードのQ値の和

        # 10エピソードごとの平均データ（プロット用）
        self.avg_ep = []  # x軸: 平均を取ったエピソード番号（10, 20, 30, ...）
        self.avg_rewards = []  # 10エピソードごとの平均報酬
        self.avg_q_values = []  # 10エピソードごとの平均Q値

        self.plot()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg, robot_name):
        # 各ロボットからのデータを受信
        score = msg.data[0]
        q_value = msg.data[1]

        self.current_episode_data[robot_name]['score'] = score
        self.current_episode_data[robot_name]['q_value'] = q_value
        self.current_episode_data[robot_name]['received'] = True

        # 全ロボットからデータを受信したかチェック
        all_received = all(self.current_episode_data[rn]['received'] for rn in ROBOT_NAMES)

        if all_received:
            # 全ロボットの報酬とQ値を合算
            total_reward = sum(self.current_episode_data[rn]['score'] for rn in ROBOT_NAMES)
            total_q_value = sum(self.current_episode_data[rn]['q_value'] for rn in ROBOT_NAMES)

            self.episode_count += 1
            self.episode_rewards.append(total_reward)
            self.episode_q_values.append(total_q_value)

            # 10エピソードごとに平均を計算
            if self.episode_count % AVERAGE_WINDOW == 0:
                start_idx = self.episode_count - AVERAGE_WINDOW
                avg_reward = np.mean(self.episode_rewards[start_idx:self.episode_count])
                avg_q = np.mean(self.episode_q_values[start_idx:self.episode_count])

                self.avg_ep.append(self.episode_count)
                self.avg_rewards.append(avg_reward)
                self.avg_q_values.append(avg_q)

            # 次のエピソード用にリセット
            for rn in ROBOT_NAMES:
                self.current_episode_data[rn] = {'score': None, 'q_value': None, 'received': False}

    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title='Average max Q-value (Sum of all robots, 10-episode avg)')
        self.qValuePlt.setGeometry(0, 320, 600, 300)
        self.qValuePlt.setLabel('bottom', 'Episode')
        self.qValuePlt.setLabel('left', 'Q-value')

        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward (Sum of all robots, 10-episode avg)')
        self.rewardsPlt.setGeometry(0, 10, 600, 300)
        self.rewardsPlt.setLabel('bottom', 'Episode')
        self.rewardsPlt.setLabel('left', 'Reward')

        for pltw in (self.rewardsPlt, self.qValuePlt):
            # グリッドON
            pltw.showGrid(x=True, y=True, alpha=0.3)

            # 四辺の軸を取得
            ax_b = pltw.getAxis('bottom')
            ax_l = pltw.getAxis('left')
            ax_t = pltw.getAxis('top')
            ax_r = pltw.getAxis('right')

            # 上側と右側の軸を表示
            pltw.showAxis('top')
            pltw.showAxis('right')

            # ラベルは表示(showValues=True)のまま、tick線だけ消す
            ax_b.setStyle(tickLength=0, showValues=True)
            ax_l.setStyle(tickLength=0, showValues=True)
            ax_t.setStyle(tickLength=0, showValues=False)  # 上側は数値非表示
            ax_r.setStyle(tickLength=0, showValues=False)  # 右側は数値非表示

            # 四辺の軸線を黒で明示
            ax_b.setPen(pyqtgraph.mkPen('k'))
            ax_l.setPen(pyqtgraph.mkPen('k'))
            ax_t.setPen(pyqtgraph.mkPen('k'))
            ax_r.setPen(pyqtgraph.mkPen('k'))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        if len(self.avg_ep) > 0:
            # x軸の適切な間隔を計算
            max_ep = max(self.avg_ep) if self.avg_ep else 100
            x_tick_spacing = self._calculate_tick_spacing(max_ep)

            # 報酬グラフのy軸間隔を計算
            if self.avg_rewards:
                reward_range = max(self.avg_rewards) - min(self.avg_rewards)
                reward_max = max(abs(max(self.avg_rewards)), abs(min(self.avg_rewards)))
                y_reward_spacing = self._calculate_tick_spacing(max(reward_range, reward_max))
            else:
                y_reward_spacing = 100

            # Q値グラフのy軸間隔を計算
            if self.avg_q_values:
                q_range = max(self.avg_q_values) - min(self.avg_q_values)
                q_max = max(abs(max(self.avg_q_values)), abs(min(self.avg_q_values)))
                y_q_spacing = self._calculate_tick_spacing(max(q_range, q_max))
            else:
                y_q_spacing = 10

            # 軸の間隔を設定
            try:
                self.rewardsPlt.getAxis('bottom').setTickSpacing(major=x_tick_spacing, minor=x_tick_spacing)
                self.rewardsPlt.getAxis('left').setTickSpacing(major=y_reward_spacing, minor=y_reward_spacing)
                self.qValuePlt.getAxis('bottom').setTickSpacing(major=x_tick_spacing, minor=x_tick_spacing)
                self.qValuePlt.getAxis('left').setTickSpacing(major=y_q_spacing, minor=y_q_spacing)
            except AttributeError:
                pass

            # プロット（10エピソード平均）
            self.rewardsPlt.plot(self.avg_ep, self.avg_rewards, pen=pyqtgraph.mkPen('r', width=2),
                                 symbol='o', symbolSize=5, symbolBrush='r', clear=True)
            self.qValuePlt.plot(self.avg_ep, self.avg_q_values, pen=pyqtgraph.mkPen('g', width=2),
                                symbol='o', symbolSize=5, symbolBrush='g', clear=True)
        else:
            # データがない場合はクリア
            self.rewardsPlt.clear()
            self.qValuePlt.clear()

    def _calculate_tick_spacing(self, value_range):
        """適切なtick間隔を計算する"""
        if value_range <= 0:
            return 10

        # 値の範囲に基づいて適切な間隔を決定
        # おおよそ5-10個のtickになるようにする
        magnitude = 10 ** int(np.floor(np.log10(value_range)))

        if value_range / magnitude <= 2:
            spacing = magnitude / 5
        elif value_range / magnitude <= 5:
            spacing = magnitude / 2
        else:
            spacing = magnitude

        # 最小間隔を設定
        if spacing < 1:
            spacing = 1

        return spacing

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
        print('shutdown')
        if win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()