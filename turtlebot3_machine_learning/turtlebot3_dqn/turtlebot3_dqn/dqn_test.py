#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
# (License header omitted for brevity: unchanged)
#################################################################################

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

# しきい値（任意指定）。None のときは「符号」で判定する。
DEFAULT_SUCCESS_THR = None      # 例: 90.0 にすれば「>=90 を成功」
DEFAULT_COLLISION_THR = None    # 例: -40.0 にすれば「<=-40 を衝突」

# 安全上限（無限ループ保険）
_INTERNAL_MAX_STEPS = 100000


class DQNTest(Node):
    def __init__(self, stage, load_episode, eval_episodes, success_thr, collision_thr):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.eval_episodes = int(eval_episodes)

        self.state_size = 182
        self.action_size = 5

        # None の場合は符号判定にフォールバック
        self.success_thr = None if success_thr is None else float(success_thr)
        self.collision_thr = None if collision_thr is None else float(collision_thr)

        self.memory = collections.deque(maxlen=1000000)

        # モデル読み込み（.keras）
        self.model = self.build_model()
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            f'stage{self.stage}_episode{self.load_episode}.keras'
        )
        loaded_model = load_model(model_path, compile=False, custom_objects={'mse': MeanSquaredError()})
        self.model.set_weights(loaded_model.get_weights())

        # サービスクライアント
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        # 環境構築
        self.env_make()

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Environment make client not available, waiting...')
        self.make_environment_client.call_async(Empty.Request())
        time.sleep(1.0)

    def env_reset(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset environment client not available, waiting...')
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f'Reset service failed: {future.exception()}')
        state = numpy.reshape(numpy.asarray(future.result().state), [1, self.state_size])
        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = int(action)
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('rl_agent_interface not available, waiting...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            raise RuntimeError(f'rl_agent_interface failed: {future.exception()}')
        next_state = numpy.reshape(numpy.asarray(future.result().state), [1, self.state_size])
        reward = float(future.result().reward)
        done = bool(future.result().done)
        return next_state, reward, done

    def get_action(self, state):
        # Greedy（評価用）
        q_values = self.model.predict(state, verbose=0)
        return int(numpy.argmax(q_values[0]))

    def _classify_terminal(self, terminal_reward: float, total_return: float):
        """
        終端時の判定：
          1) しきい値が与えられていればそれを優先
          2) なければ「終端報酬の符号」で判定（>0 成功, <0 衝突）
          3) 0 ちょうどの場合は合計報酬の符号で補助
        """
        # 明示しきい値が設定されていれば優先
        if self.success_thr is not None and terminal_reward >= self.success_thr:
            return 'success'
        if self.collision_thr is not None and terminal_reward <= self.collision_thr:
            return 'collision'

        # フォールバック：符号判定
        if terminal_reward > 0.0:
            return 'success'
        if terminal_reward < 0.0:
            return 'collision'

        # 0 ちょうどなら合計報酬の符号で補助
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
            state = self.env_reset()
            done = False
            total_return = 0.0
            last_reward = 0.0
            steps = 0
            t0 = time.time()

            time.sleep(0.2)  # 落ち着かせる

            while not done and steps < _INTERNAL_MAX_STEPS:
                steps += 1
                action = 2 if steps == 1 else self.get_action(state)
                next_state, reward, done = self.step(action)
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
                f'duration={duration:.2f}s total_return={total_return:.3f} last_reward={last_reward:.3f}'
            )

        n = len(outcomes)
        n_success = sum(1 for o in outcomes if o == 'success')
        n_collision = sum(1 for o in outcomes if o == 'collision')

        success_rate = (n_success / n) if n else 0.0
        collision_rate = (n_collision / n) if n else 0.0
        avg_return = mean(env_returns) if env_returns else float('nan')
        avg_arrival_time = mean(success_times) if success_times else float('nan')

        # 指標のみ出力
        print(f"SuccessRate(%)={success_rate*100:.2f}")
        print(f"AvgArrivalTime_sec={avg_arrival_time:.3f}")
        print(f"CollisionRate(%)={collision_rate*100:.2f}")
        print(f"AvgCumulativeReturn={avg_return:.3f}")

        return {
            'success_rate': success_rate,
            'avg_arrival_time': avg_arrival_time,
            'collision_rate': collision_rate,
            'avg_return': avg_return,
        }


def _parse_arg(idx, default):
    # "None" 文字列なら None を返す
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
    success_thr = _parse_arg(4, DEFAULT_SUCCESS_THR)      # 例: 90.0
    collision_thr = _parse_arg(5, DEFAULT_COLLISION_THR)  # 例: -40.0

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
