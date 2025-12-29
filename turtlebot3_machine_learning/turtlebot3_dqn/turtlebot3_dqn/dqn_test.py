#!/usr/bin/env python3
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

import collections
import os
import sys
import time
from statistics import mean

import numpy
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import RMSprop

from turtlebot3_msgs.srv import Dqn

DEFAULT_SUCCESS_THR = None
DEFAULT_COLLISION_THR = None
MAX_STEPS = 100000


class DQNTest(Node):

    def __init__(self, stage, load_episode, eval_episodes, success_thr, collision_thr):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.eval_episodes = int(eval_episodes)
        self.state_size = 182
        self.action_size = 5

        self.success_thr = None if success_thr is None else float(success_thr)
        self.collision_thr = None if collision_thr is None else float(collision_thr)

        self.memory = collections.deque(maxlen=1000000)
        self.model = self._build_model()
        self._load_model()

        self.rl_agent_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_env_client = self.create_client(Empty, 'make_environment')
        self.reset_env_client = self.create_client(Dqn, 'reset_environment')

        self._make_env()

    def _build_model(self):
        model = Sequential([
            Dense(512, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'),
            Dense(256, activation='relu', kernel_initializer='lecun_uniform'),
            Dense(128, activation='relu', kernel_initializer='lecun_uniform'),
            Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'),
        ])
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model', f'stage{self.stage}_episode{self.load_episode}.keras')
        loaded = load_model(model_path, compile=False, custom_objects={'mse': MeanSquaredError()})
        self.model.set_weights(loaded.get_weights())

    def _make_env(self):
        while not self.make_env_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for make_environment service...')
        self.make_env_client.call_async(Empty.Request())
        time.sleep(1.0)

    def _reset_env(self):
        while not self.reset_env_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for reset_environment service...')
        future = self.reset_env_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f'Reset service failed: {future.exception()}')
        return numpy.reshape(numpy.asarray(future.result().state), [1, self.state_size])

    def _step(self, action):
        req = Dqn.Request()
        req.action = int(action)
        while not self.rl_agent_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for rl_agent_interface service...')
        future = self.rl_agent_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f'rl_agent_interface failed: {future.exception()}')
        result = future.result()
        next_state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
        return next_state, float(result.reward), bool(result.done)

    def _get_action(self, state):
        q_values = self.model.predict(state, verbose=0)
        return int(numpy.argmax(q_values[0]))

    def _classify_terminal(self, terminal_reward, total_return):
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

    def evaluate(self):
        env_returns = []
        outcomes = []
        success_times = []

        for ep in range(1, self.eval_episodes + 1):
            state = self._reset_env()
            done = False
            total_return = 0.0
            last_reward = 0.0
            steps = 0
            t0 = time.time()

            time.sleep(0.2)

            while not done and steps < MAX_STEPS:
                steps += 1
                action = 2 if steps == 1 else self._get_action(state)
                next_state, reward, done = self._step(action)
                total_return += reward
                last_reward = reward
                state = next_state
                time.sleep(0.003)

            duration = time.time() - t0
            outcome = self._classify_terminal(last_reward, total_return) if done else 'other'
            if outcome == 'success':
                success_times.append(duration)

            env_returns.append(total_return)
            outcomes.append(outcome)

            self.get_logger().info(
                f'[Episode {ep}/{self.eval_episodes}] outcome={outcome} '
                f'duration={duration:.2f}s return={total_return:.3f}')

        n = len(outcomes)
        n_success = sum(1 for o in outcomes if o == 'success')
        n_collision = sum(1 for o in outcomes if o == 'collision')

        success_rate = (n_success / n) if n else 0.0
        collision_rate = (n_collision / n) if n else 0.0
        avg_return = mean(env_returns) if env_returns else float('nan')
        avg_arrival_time = mean(success_times) if success_times else float('nan')

        print(f"SuccessRate(%)={success_rate * 100:.2f}")
        print(f"AvgArrivalTime_sec={avg_arrival_time:.3f}")
        print(f"CollisionRate(%)={collision_rate * 100:.2f}")
        print(f"AvgCumulativeReturn={avg_return:.3f}")

        return {
            'success_rate': success_rate,
            'avg_arrival_time': avg_arrival_time,
            'collision_rate': collision_rate,
            'avg_return': avg_return,
        }


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
    load_episode = _parse_arg(2, '600')
    eval_episodes = int(_parse_arg(3, int(load_episode)))
    success_thr = _parse_arg(4, DEFAULT_SUCCESS_THR)
    collision_thr = _parse_arg(5, DEFAULT_COLLISION_THR)

    node = DQNTest(stage, load_episode, eval_episodes, success_thr, collision_thr)

    try:
        node.evaluate()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
