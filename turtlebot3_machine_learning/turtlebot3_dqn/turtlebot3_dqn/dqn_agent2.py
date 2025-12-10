#!/usr/bin/env python3
#################################################################################
# DQN Agent for Robot 2 - DEBUG VERSION
# デバッグログを追加してフリーズの原因を特定
#################################################################################

import collections
import datetime
import json
import math
import os
import random
import sys
import time
import threading

import numpy
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray, String, Int32
from std_srvs.srv import Empty
import tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn


tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class DQNMetric(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super(DQNMetric, self).__init__(name=name)
        self.loss = self.add_weight(name='loss', initializer='zeros')
        self.episode_step = self.add_weight(name='step', initializer='zeros')

    def update_state(self, y_true, y_pred=0, sample_weight=None):
        self.loss.assign_add(y_true)
        self.episode_step.assign_add(1)

    def result(self):
        return self.loss / self.episode_step

    def reset_states(self):
        self.loss.assign(0)
        self.episode_step.assign(0)


class DQNAgent1(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent_robot2')

        self.robot_name = 'robot2'
        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 182
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = 6000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000

        self.model = self.create_qnetwork()
        self.target_model = self.create_qnetwork()
        self.update_target_model()
        self.update_target_after = 5000
        self.target_update_after_counter = 0

        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            self.robot_name
        )
        os.makedirs(self.model_dir_path, exist_ok=True)

        self.model_path = os.path.join(
            self.model_dir_path,
            'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.keras'
        )

        if self.load_model and os.path.exists(self.model_path):
            self.model.set_weights(load_model(self.model_path).get_weights())
            json_path = os.path.join(
                self.model_dir_path,
                'stage' + str(self.stage) + '_episode' + str(self.load_episode) + '.json'
            )
            if os.path.exists(json_path):
                with open(json_path) as outfile:
                    param = json.load(outfile)
                    self.epsilon = param.get('epsilon')
                    self.step_counter = param.get('step_counter')

        if LOGGING:
            tensorboard_file_name = current_time + f'_dqn_stage{self.stage}_{self.robot_name}_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = tensorflow.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        # ReentrantCallbackGroup を追加
        self.cb_group = ReentrantCallbackGroup()

        # Publishers and subscribers with robot namespace
        self.status_pub = self.create_publisher(String, f'/{self.robot_name}/status', 10)
        self.action_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/result', 10)

        # Subscribe to coordinator signals with callback group
        self.start_episode_sub = self.create_subscription(
            Int32, '/start_episode', 
            self.start_episode_callback, 
            10,
            callback_group=self.cb_group
        )

        # Service clients with robot namespace and callback group
        self.rl_agent_interface_client = self.create_client(
            Dqn, f'/{self.robot_name}/rl_agent_interface',
            callback_group=self.cb_group
        )
        self.get_state_client = self.create_client(
            Dqn, f'/{self.robot_name}/get_state',
            callback_group=self.cb_group
        )

        self.episode_start_requested = False
        self.current_episode = 0

        # ★ DEBUG: コールバック受信カウンター ★
        self.callback_count = 0

        self.get_logger().info(f'{self.robot_name} DQN Agent initialized')
        self.get_logger().info(f'{self.robot_name} Starting process() method in separate thread...')

        # process()を別スレッドで実行
        self.process_thread = threading.Thread(target=self.process_wrapper)
        self.process_thread.daemon = True
        self.process_thread.start()

        # ★ DEBUG: 定期的に状態を出力するタイマー ★
        self.debug_timer = self.create_timer(5.0, self.debug_status_callback)

    def debug_status_callback(self):
        """定期的に内部状態を出力（デバッグ用）"""
        self.get_logger().info(
            f'[DEBUG] {self.robot_name} status: '
            f'episode_start_requested={self.episode_start_requested}, '
            f'current_episode={self.current_episode}, '
            f'callback_count={self.callback_count}, '
            f'process_thread.is_alive={self.process_thread.is_alive()}'
        )

    def process_wrapper(self):
        """Wrapper for process() to handle exceptions in thread"""
        try:
            self.process()
        except Exception as e:
            self.get_logger().error(f'{self.robot_name} Exception in process(): {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def start_episode_callback(self, msg):
        """Callback for episode start signal from coordinator"""
        self.callback_count += 1  # ★ DEBUG: カウンターをインクリメント ★
        self.current_episode = msg.data
        self.episode_start_requested = True
        self.get_logger().info(
            f'{self.robot_name} - ★ CALLBACK RECEIVED ★ episode {self.current_episode} '
            f'(callback_count={self.callback_count})'
        )

    def publish_status(self):
        """Publish current status to coordinator"""
        status = {
            'done': self.done,
            'succeeded': self.succeed,
            'failed': self.fail
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def process(self):
        self.get_logger().info(f'{self.robot_name} - process() started')
        self.get_logger().info(f'{self.robot_name} - Waiting for coordinator to start episodes...')

        # 初回起動時に少し待機してexecutorのspinが開始されるのを待つ
        time.sleep(1.0)
        
        last_processed_episode = 0  # 最後に処理したエピソード番号を記録

        # Wait for coordinator to start managing episodes
        while rclpy.ok():
            self.get_logger().info(
                f'{self.robot_name} - [WAIT LOOP] Waiting for episode start signal... '
                f'episode_start_requested={self.episode_start_requested}, '
                f'current_episode={self.current_episode}, '
                f'last_processed_episode={last_processed_episode}'
            )

            # ★ DEBUG: 待機ループの詳細ログ ★
            wait_count = 0
            while (not self.episode_start_requested or 
                   self.current_episode <= last_processed_episode) and rclpy.ok():
                wait_count += 1
                if wait_count % 50 == 0:  # 5秒ごとにログ出力
                    self.get_logger().info(
                        f'{self.robot_name} - [WAITING] count={wait_count}, '
                        f'episode_start_requested={self.episode_start_requested}, '
                        f'current_episode={self.current_episode}, '
                        f'last_processed_episode={last_processed_episode}, '
                        f'condition1={not self.episode_start_requested}, '
                        f'condition2={self.current_episode <= last_processed_episode}'
                    )
                time.sleep(0.1)

            if not rclpy.ok():
                break

            episode = self.current_episode
            self.get_logger().info(f'{self.robot_name} - ★ Starting episode {episode} ★')
            
            # フラグをリセットして処理済みエピソードを記録
            self.episode_start_requested = False
            last_processed_episode = episode

            # Get initial state from environment (already reset by coordinator)
            self.get_logger().info(f'{self.robot_name} - Getting initial state...')
            state = self.get_current_state()
            self.get_logger().info(f'{self.robot_name} - Received state, shape: {state.shape}')

            local_step = 0
            score = 0
            sum_max_q = 0.0

            # Run episode
            self.get_logger().info(f'{self.robot_name} - Entering episode loop...')
            while True:
                local_step += 1

                q_values = self.model.predict(state, verbose=0)
                sum_max_q += float(numpy.max(q_values))

                action = int(self.get_action(state))

                # ★ DEBUG: step()呼び出し前のログ ★
                if local_step % 100 == 0:
                    self.get_logger().info(f'{self.robot_name} - step {local_step}, calling step({action})...')

                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                # Publish status
                self.publish_status()

                if done:
                    self.get_logger().info(f'{self.robot_name} - ★ Episode {episode} DONE ★')

                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode
                            )
                        self.dqn_reward_metric.reset_states()

                    self.get_logger().info(
                        f'{self.robot_name} - Episode: {episode}, '
                        f'score: {score}, '
                        f'memory length: {len(self.replay_memory)}, '
                        f'epsilon: {self.epsilon:.3f}'
                    )

                    # Episode complete - coordinator will handle reset
                    self.get_logger().info(
                        f'{self.robot_name} - Episode complete. '
                        f'Before break: episode_start_requested={self.episode_start_requested}, '
                        f'current_episode={self.current_episode}, '
                        f'last_processed_episode will be {episode}'
                    )
                    break

                time.sleep(0.01)

            self.get_logger().info(
                f'{self.robot_name} - Exited episode loop. '
                f'Now waiting for next episode. '
                f'episode_start_requested={self.episode_start_requested}, '
                f'current_episode={self.current_episode}, '
                f'last_processed_episode={last_processed_episode}'
            )

            # Save model periodically
            if self.train_mode and episode % 1000 == 0:
                param_keys = ['epsilon', 'step_counter']
                param_values = [self.epsilon, self.step_counter]
                param_dictionary = dict(zip(param_keys, param_values))

                self.model_path = os.path.join(
                    self.model_dir_path,
                    'stage' + str(self.stage) + '_episode' + str(episode) + '.keras')
                self.model.save(self.model_path)
                with open(
                    os.path.join(
                        self.model_dir_path,
                        'stage' + str(self.stage) + '_episode' + str(episode) + '.json'
                    ),
                    'w'
                ) as outfile:
                    json.dump(param_dictionary, outfile)
                self.get_logger().info(f'{self.robot_name} - Model saved at episode {episode}')

    def get_current_state(self):
        """Get current state from environment (called after coordinator resets environment)"""
        req = Dqn.Request()

        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{self.robot_name} - Waiting for get_state service...')

        future = self.get_state_client.call_async(req)
        
        # rclpy.spin_until_future_complete()を使わない（デッドロック回避）
        while not future.done() and rclpy.ok():
            time.sleep(0.01)

        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
            return state
        else:
            self.get_logger().error(f'{self.robot_name} - Failed to get current state')
            return numpy.zeros([1, self.state_size])

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                result = random.randint(0, self.action_size - 1)
            else:
                result = numpy.argmax(self.model.predict(state, verbose=0))
        else:
            result = numpy.argmax(self.model.predict(state, verbose=0))

        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{self.robot_name} - rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        # rclpy.spin_until_future_complete()を使わない（デッドロック回避）
        while not future.done() and rclpy.ok():
            time.sleep(0.01)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                f'{self.robot_name} - Exception while calling service: {future.exception()}')
            # エラー時のデフォルト値を返す
            next_state = numpy.zeros([1, self.state_size])
            reward = 0.0
            done = True

        return next_state, reward, done

    def create_qnetwork(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        self.get_logger().info(f'{self.robot_name} - *Target model updated*')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states, verbose=0)

        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states, verbose=0)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                future_reward = numpy.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            tensorflow.convert_to_tensor(x_train, tensorflow.float32),
            tensorflow.convert_to_tensor(y_train, tensorflow.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    dqn_agent = DQNAgent1(stage_num, max_training_episodes)
    
    # MultiThreadedExecutorを使用
    executor = MultiThreadedExecutor()
    executor.add_node(dqn_agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        dqn_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()