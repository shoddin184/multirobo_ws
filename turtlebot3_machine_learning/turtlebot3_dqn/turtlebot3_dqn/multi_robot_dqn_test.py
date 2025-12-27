#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
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
# Multi-Robot DQN Test
# Tests trained models for multiple robots

import os
import sys
import time
import threading
from statistics import mean

import numpy
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_srvs.srv import Empty
from std_msgs.msg import Int32
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import RMSprop

from turtlebot3_msgs.srv import Dqn

# マルチロボット設定
ROBOT_NAMES = ['robot1', 'robot2', 'robot3']
NUM_ROBOTS = len(ROBOT_NAMES)

# デフォルト設定
DEFAULT_EVAL_EPISODES = 100
DEFAULT_SUCCESS_THR = None
DEFAULT_COLLISION_THR = None
_INTERNAL_MAX_STEPS = 100000

# モデル保存ディレクトリのベースパス
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
SAVED_MODEL_BASE = os.path.join(_PACKAGE_ROOT, 'turtlebot3_machine_learning', 'saved_model')


class RobotTester:
    """個別ロボットのテスト処理を管理するクラス"""

    def __init__(self, node, robot_name, stage, load_episode, success_thr, collision_thr):
        self.node = node
        self.robot_name = robot_name
        self.stage = stage
        self.load_episode = load_episode
        self.success_thr = success_thr
        self.collision_thr = collision_thr

        self.state_size = 182
        self.action_size = 5

        # モデル読み込み
        self.model = self._build_model()
        model_path = os.path.join(
            SAVED_MODEL_BASE,
            robot_name,
            f'stage{stage}_episode{load_episode}.keras'
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model not found: {model_path}')

        loaded_model = load_model(model_path, compile=False, custom_objects={'mse': MeanSquaredError()})
        self.model.set_weights(loaded_model.get_weights())
        node.get_logger().info(f'{robot_name}: Loaded model from {model_path}')

        # エピソードの状態
        self.current_state = None
        self.done = False
        self.total_return = 0.0
        self.last_reward = 0.0
        self.steps = 0
        self.episode_start_time = 0.0

        # 結果記録
        self.episode_returns = []
        self.outcomes = []
        self.success_times = []

    def _build_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        q_values = self.model.predict(state, verbose=0)
        return int(numpy.argmax(q_values[0]))

    def classify_terminal(self, terminal_reward, total_return):
        if self.success_thr is not None and terminal_reward >= self.success_thr:
            return 'success'
        if self.collision_thr is not None and terminal_reward <= self.collision_thr:
            return 'collision'

        if terminal_reward > 0.0:
            return 'success'
        if terminal_reward < 0.0:
            return 'collision'

        if total_return > 0.0:
            return 'success'
        if total_return < 0.0:
            return 'collision'

        return 'other'

    def reset_episode(self):
        self.done = False
        self.total_return = 0.0
        self.last_reward = 0.0
        self.steps = 0
        self.episode_start_time = time.time()

    def finish_episode(self):
        duration = time.time() - self.episode_start_time
        outcome = self.classify_terminal(self.last_reward, self.total_return) if self.done else 'other'

        if outcome == 'success':
            self.success_times.append(duration)

        self.episode_returns.append(self.total_return)
        self.outcomes.append(outcome)

        return outcome, duration


class MultiRobotDQNTest(Node):
    def __init__(self, stage, load_episode, eval_episodes, success_thr, collision_thr):
        super().__init__('multi_robot_dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.eval_episodes = int(eval_episodes)
        self.success_thr = None if success_thr is None else float(success_thr)
        self.collision_thr = None if collision_thr is None else float(collision_thr)

        self.callback_group = ReentrantCallbackGroup()

        # 各ロボットのテスターを作成
        self.testers = {}
        for robot_name in ROBOT_NAMES:
            try:
                self.testers[robot_name] = RobotTester(
                    self, robot_name, self.stage, self.load_episode,
                    self.success_thr, self.collision_thr
                )
            except FileNotFoundError as e:
                self.get_logger().error(str(e))
                raise

        # サービスクライアント（各ロボット用）
        self.rl_clients = {}
        self.reset_clients = {}
        self.make_env_clients = {}

        for robot_name in ROBOT_NAMES:
            self.rl_clients[robot_name] = self.create_client(
                Dqn, f'/{robot_name}/rl_agent_interface',
                callback_group=self.callback_group
            )
            self.reset_clients[robot_name] = self.create_client(
                Dqn, f'/{robot_name}/reset_environment',
                callback_group=self.callback_group
            )
            self.make_env_clients[robot_name] = self.create_client(
                Empty, f'/{robot_name}/make_environment',
                callback_group=self.callback_group
            )

        # エピソード完了状態
        self.robots_done = {rn: False for rn in ROBOT_NAMES}
        self.current_episode = 0

        self.get_logger().info(
            f'MultiRobotDQNTest initialized: stage={self.stage}, '
            f'episode={self.load_episode}, eval_episodes={self.eval_episodes}'
        )

    def env_make(self):
        """全ロボットの環境を構築"""
        for robot_name in ROBOT_NAMES:
            client = self.make_env_clients[robot_name]
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f'{robot_name}: make_environment not available, waiting...')
            client.call_async(Empty.Request())
        time.sleep(1.0)
        self.get_logger().info('All environments created')

    def env_reset(self, robot_name):
        """指定ロボットの環境をリセット"""
        client = self.reset_clients[robot_name]
        tester = self.testers[robot_name]

        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'{robot_name}: reset_environment not available, waiting...')

        future = client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            raise RuntimeError(f'{robot_name}: Reset failed: {future.exception()}')

        state = numpy.reshape(numpy.asarray(future.result().state), [1, tester.state_size])
        tester.current_state = state
        tester.reset_episode()
        return state

    def step(self, robot_name, action):
        """指定ロボットでアクションを実行"""
        client = self.rl_clients[robot_name]
        tester = self.testers[robot_name]

        req = Dqn.Request()
        req.action = int(action)

        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'{robot_name}: rl_agent_interface not available, waiting...')

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            raise RuntimeError(f'{robot_name}: Step failed: {future.exception()}')

        next_state = numpy.reshape(numpy.asarray(future.result().state), [1, tester.state_size])
        reward = float(future.result().reward)
        done = bool(future.result().done)

        return next_state, reward, done

    def run_episode_for_robot(self, robot_name):
        """1つのロボットのエピソードを実行"""
        tester = self.testers[robot_name]

        while not tester.done and tester.steps < _INTERNAL_MAX_STEPS:
            tester.steps += 1

            # 最初のステップは前進
            if tester.steps == 1:
                action = 2
            else:
                action = tester.get_action(tester.current_state)

            next_state, reward, done = self.step(robot_name, action)
            tester.total_return += reward
            tester.last_reward = reward
            tester.current_state = next_state
            tester.done = done

            time.sleep(0.003)

        self.robots_done[robot_name] = True

    def evaluate(self):
        """全ロボットを使って評価"""
        self.env_make()

        for ep in range(1, self.eval_episodes + 1):
            self.current_episode = ep
            self.robots_done = {rn: False for rn in ROBOT_NAMES}

            # 全ロボットの環境をリセット
            for robot_name in ROBOT_NAMES:
                self.env_reset(robot_name)

            time.sleep(0.2)

            # 各ロボットのエピソードを並列実行
            threads = []
            for robot_name in ROBOT_NAMES:
                t = threading.Thread(target=self.run_episode_for_robot, args=(robot_name,))
                threads.append(t)
                t.start()

            # 全スレッドの完了を待つ
            for t in threads:
                t.join()

            # 結果を記録
            episode_results = []
            total_team_return = 0.0

            for robot_name in ROBOT_NAMES:
                tester = self.testers[robot_name]
                outcome, duration = tester.finish_episode()
                total_team_return += tester.total_return
                episode_results.append(f'{robot_name}:{outcome}({tester.total_return:.1f})')

            self.get_logger().info(
                f'[Episode {ep}/{self.eval_episodes}] {" | ".join(episode_results)} '
                f'| Team Total: {total_team_return:.2f}'
            )

        # 最終結果を集計
        self.print_results()

    def print_results(self):
        """結果を出力"""
        print('\n' + '=' * 60)
        print('Multi-Robot DQN Test Results')
        print('=' * 60)

        total_success = 0
        total_collision = 0
        total_returns = []
        total_success_times = []

        for robot_name in ROBOT_NAMES:
            tester = self.testers[robot_name]
            n = len(tester.outcomes)
            n_success = sum(1 for o in tester.outcomes if o == 'success')
            n_collision = sum(1 for o in tester.outcomes if o == 'collision')

            success_rate = (n_success / n * 100) if n else 0.0
            collision_rate = (n_collision / n * 100) if n else 0.0
            avg_return = mean(tester.episode_returns) if tester.episode_returns else float('nan')
            avg_time = mean(tester.success_times) if tester.success_times else float('nan')

            print(f'\n--- {robot_name} ---')
            print(f'  Success Rate: {success_rate:.2f}%')
            print(f'  Collision Rate: {collision_rate:.2f}%')
            print(f'  Avg Return: {avg_return:.3f}')
            print(f'  Avg Arrival Time: {avg_time:.3f}s')

            total_success += n_success
            total_collision += n_collision
            total_returns.extend(tester.episode_returns)
            total_success_times.extend(tester.success_times)

        # チーム全体の結果
        total_n = len(total_returns)
        team_success_rate = (total_success / total_n * 100) if total_n else 0.0
        team_collision_rate = (total_collision / total_n * 100) if total_n else 0.0
        team_avg_return = mean(total_returns) if total_returns else float('nan')
        team_avg_time = mean(total_success_times) if total_success_times else float('nan')

        print('\n' + '=' * 60)
        print('TEAM TOTAL')
        print('=' * 60)
        print(f'  Total Episodes: {self.eval_episodes} x {NUM_ROBOTS} robots = {total_n}')
        print(f'  Team Success Rate: {team_success_rate:.2f}%')
        print(f'  Team Collision Rate: {team_collision_rate:.2f}%')
        print(f'  Team Avg Return: {team_avg_return:.3f}')
        print(f'  Team Avg Arrival Time: {team_avg_time:.3f}s')
        print('=' * 60)


def _parse_arg(idx, default):
    if len(sys.argv) > idx:
        val = sys.argv[idx]
        if isinstance(default, (float, type(None))):
            if val == 'None':
                return None
            try:
                return float(val)
            except Exception:
                return default
        try:
            return type(default)(val)
        except Exception:
            return default
    return default


def main(args=None):
    rclpy.init(args=args if args else sys.argv)

    stage = _parse_arg(1, '1')
    load_episode = _parse_arg(2, '1000')
    eval_episodes = int(_parse_arg(3, DEFAULT_EVAL_EPISODES))
    success_thr = _parse_arg(4, DEFAULT_SUCCESS_THR)
    collision_thr = _parse_arg(5, DEFAULT_COLLISION_THR)

    try:
        node = MultiRobotDQNTest(stage, load_episode, eval_episodes, success_thr, collision_thr)
        node.evaluate()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        print(f'Model files should be in: {SAVED_MODEL_BASE}/<robot_name>/')
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
