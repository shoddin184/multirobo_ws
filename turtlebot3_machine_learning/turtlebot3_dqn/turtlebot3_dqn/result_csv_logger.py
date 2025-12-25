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
# CSV Logger for Multi-Robot DQN Training Results
# Saves team reward and Q-value every 100 episodes for comparison across
# different stages and DQN agent designs.

import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# マルチロボット設定
ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
SAVE_INTERVAL = 100  # 100エピソードごとに保存

# デフォルトの保存ディレクトリ
# このファイルから2階層上 -> src/turtlebot3_machine_learning/turtlebot3_dqn
# そこから turtlebot3_machine_learning/saved_model/logs
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # turtlebot3_machine_learning
DEFAULT_SAVE_DIR = os.path.join(_PACKAGE_ROOT, 'turtlebot3_machine_learning', 'saved_model', 'logs')


class ResultCSVLogger(Node):

    def __init__(self):
        super().__init__('result_csv_logger')

        # パラメータの宣言
        self.declare_parameter('save_dir', DEFAULT_SAVE_DIR)
        self.declare_parameter('stage_name', 'stage_unknown')
        self.declare_parameter('agent_design', 'default')
        self.declare_parameter('experiment_name', '')

        # パラメータの取得
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.stage_name = self.get_parameter('stage_name').get_parameter_value().string_value
        self.agent_design = self.get_parameter('agent_design').get_parameter_value().string_value
        self.experiment_name = self.get_parameter('experiment_name').get_parameter_value().string_value

        # 保存ディレクトリの作成
        os.makedirs(self.save_dir, exist_ok=True)

        # CSVファイル名の生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.experiment_name:
            filename = f'{self.experiment_name}_{self.stage_name}_{self.agent_design}_{timestamp}.csv'
        else:
            filename = f'{self.stage_name}_{self.agent_design}_{timestamp}.csv'
        self.csv_path = os.path.join(self.save_dir, filename)

        # エピソードカウンタ
        self.episode_count = 0

        # 各ロボットの現在のエピソードのデータを保持
        self.current_episode_data = {rn: {'score': None, 'q_value': None, 'received': False}
                                     for rn in ROBOT_NAMES}

        # エピソードごとのデータ（100エピソード分を蓄積）
        self.episode_rewards = []
        self.episode_q_values = []

        # 100エピソードごとの平均データ
        self.logged_data = []

        # 各ロボットのresultトピックを購読
        self.result_subs = []
        for robot_name in ROBOT_NAMES:
            sub = self.create_subscription(
                Float32MultiArray,
                f'/{robot_name}/result',
                lambda msg, rn=robot_name: self.data_callback(msg, rn),
                10
            )
            self.result_subs.append(sub)

        # CSVファイルのヘッダーを書き込み
        self._write_csv_header()

        self.get_logger().info(f'Result CSV Logger started')
        self.get_logger().info(f'Stage: {self.stage_name}, Agent Design: {self.agent_design}')
        self.get_logger().info(f'Saving to: {self.csv_path}')

    def _write_csv_header(self):
        """CSVファイルのヘッダーを書き込む"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # メタデータをコメントとして記録
            writer.writerow(['# Stage:', self.stage_name])
            writer.writerow(['# Agent Design:', self.agent_design])
            writer.writerow(['# Experiment:', self.experiment_name or 'N/A'])
            writer.writerow(['# Created:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['# Robots:', ', '.join(ROBOT_NAMES)])
            writer.writerow([])
            # データヘッダー
            writer.writerow(['episode', 'avg_team_reward', 'avg_team_q_value',
                           'min_reward', 'max_reward', 'min_q_value', 'max_q_value'])

    def data_callback(self, msg, robot_name):
        """各ロボットからのデータを受信"""
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

            # 100エピソードごとに平均を計算してCSVに保存
            if self.episode_count % SAVE_INTERVAL == 0:
                self._save_interval_data()

            # 次のエピソード用にリセット
            for rn in ROBOT_NAMES:
                self.current_episode_data[rn] = {'score': None, 'q_value': None, 'received': False}

    def _save_interval_data(self):
        """100エピソードごとの平均データをCSVに保存"""
        start_idx = self.episode_count - SAVE_INTERVAL
        interval_rewards = self.episode_rewards[start_idx:self.episode_count]
        interval_q_values = self.episode_q_values[start_idx:self.episode_count]

        avg_reward = sum(interval_rewards) / len(interval_rewards)
        avg_q_value = sum(interval_q_values) / len(interval_q_values)
        min_reward = min(interval_rewards)
        max_reward = max(interval_rewards)
        min_q_value = min(interval_q_values)
        max_q_value = max(interval_q_values)

        # CSVに追記
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                f'{avg_reward:.4f}',
                f'{avg_q_value:.4f}',
                f'{min_reward:.4f}',
                f'{max_reward:.4f}',
                f'{min_q_value:.4f}',
                f'{max_q_value:.4f}'
            ])

        self.get_logger().info(
            f'Episode {self.episode_count}: '
            f'Avg Reward={avg_reward:.2f}, Avg Q-value={avg_q_value:.4f}'
        )

    def destroy_node(self):
        """ノード終了時に残りのデータを保存"""
        # 100エピソード未満の残りデータがあれば保存
        remaining = self.episode_count % SAVE_INTERVAL
        if remaining > 0 and len(self.episode_rewards) > 0:
            start_idx = self.episode_count - remaining
            interval_rewards = self.episode_rewards[start_idx:]
            interval_q_values = self.episode_q_values[start_idx:]

            if interval_rewards:
                avg_reward = sum(interval_rewards) / len(interval_rewards)
                avg_q_value = sum(interval_q_values) / len(interval_q_values)
                min_reward = min(interval_rewards)
                max_reward = max(interval_rewards)
                min_q_value = min(interval_q_values)
                max_q_value = max(interval_q_values)

                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f'{self.episode_count} (partial: {remaining} episodes)',
                        f'{avg_reward:.4f}',
                        f'{avg_q_value:.4f}',
                        f'{min_reward:.4f}',
                        f'{max_reward:.4f}',
                        f'{min_q_value:.4f}',
                        f'{max_q_value:.4f}'
                    ])

                self.get_logger().info(f'Saved partial data ({remaining} episodes)')

        self.get_logger().info(f'CSV Logger stopped. Data saved to: {self.csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = ResultCSVLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
