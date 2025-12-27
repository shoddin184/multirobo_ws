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

import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
SAVE_INTERVAL = 100

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
DEFAULT_SAVE_DIR = os.path.join(_PACKAGE_ROOT, 'turtlebot3_machine_learning', 'saved_model', 'logs')


class ResultCSVLogger(Node):

    def __init__(self):
        super().__init__('result_csv_logger')

        self.declare_parameter('save_dir', DEFAULT_SAVE_DIR)
        self.declare_parameter('stage_name', 'stage_unknown')
        self.declare_parameter('agent_design', 'default')
        self.declare_parameter('experiment_name', '')

        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        self.stage_name = self.get_parameter('stage_name').get_parameter_value().string_value
        self.agent_design = self.get_parameter('agent_design').get_parameter_value().string_value
        self.experiment_name = self.get_parameter('experiment_name').get_parameter_value().string_value

        os.makedirs(self.save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.experiment_name:
            filename = f'{self.experiment_name}_{self.stage_name}_{self.agent_design}_{timestamp}.csv'
        else:
            filename = f'{self.stage_name}_{self.agent_design}_{timestamp}.csv'
        self.csv_path = os.path.join(self.save_dir, filename)

        self.episode_count = 0
        self.current_episode_data = {
            rn: {'score': None, 'q_value': None, 'received': False}
            for rn in ROBOT_NAMES
        }
        self.episode_rewards = []
        self.episode_q_values = []

        for robot_name in ROBOT_NAMES:
            self.create_subscription(
                Float32MultiArray,
                f'/{robot_name}/result',
                lambda msg, rn=robot_name: self._data_callback(msg, rn),
                10)

        self._write_csv_header()

        self.get_logger().info(f'CSV Logger started: {self.csv_path}')

    def _write_csv_header(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['# Stage:', self.stage_name])
            writer.writerow(['# Agent Design:', self.agent_design])
            writer.writerow(['# Experiment:', self.experiment_name or 'N/A'])
            writer.writerow(['# Created:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['# Robots:', ', '.join(ROBOT_NAMES)])
            writer.writerow([])
            writer.writerow([
                'episode', 'avg_team_reward', 'avg_team_q_value',
                'min_reward', 'max_reward', 'min_q_value', 'max_q_value'])

    def _data_callback(self, msg, robot_name):
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

        if self.episode_count % SAVE_INTERVAL == 0:
            self._save_interval_data()

        for rn in ROBOT_NAMES:
            self.current_episode_data[rn] = {'score': None, 'q_value': None, 'received': False}

    def _save_interval_data(self):
        start_idx = self.episode_count - SAVE_INTERVAL
        rewards = self.episode_rewards[start_idx:self.episode_count]
        q_values = self.episode_q_values[start_idx:self.episode_count]

        avg_reward = sum(rewards) / len(rewards)
        avg_q_value = sum(q_values) / len(q_values)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                f'{avg_reward:.4f}', f'{avg_q_value:.4f}',
                f'{min(rewards):.4f}', f'{max(rewards):.4f}',
                f'{min(q_values):.4f}', f'{max(q_values):.4f}'])

        self.get_logger().info(
            f'Episode {self.episode_count}: Avg Reward={avg_reward:.2f}, Avg Q={avg_q_value:.4f}')

    def destroy_node(self):
        remaining = self.episode_count % SAVE_INTERVAL
        if remaining > 0 and self.episode_rewards:
            start_idx = self.episode_count - remaining
            rewards = self.episode_rewards[start_idx:]
            q_values = self.episode_q_values[start_idx:]

            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                avg_q_value = sum(q_values) / len(q_values)

                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f'{self.episode_count} (partial: {remaining} episodes)',
                        f'{avg_reward:.4f}', f'{avg_q_value:.4f}',
                        f'{min(rewards):.4f}', f'{max(rewards):.4f}',
                        f'{min(q_values):.4f}', f'{max(q_values):.4f}'])

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
