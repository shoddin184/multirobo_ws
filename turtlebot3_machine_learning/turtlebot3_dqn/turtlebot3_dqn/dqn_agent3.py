#!/usr/bin/env python3
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
import tensorflow
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam

from turtlebot3_msgs.srv import Dqn

tensorflow.config.set_visible_devices([], 'GPU')

LOGGING = True
CURRENT_TIME = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class DQNMetric(tensorflow.keras.metrics.Metric):

    def __init__(self, name='dqn_metric'):
        super().__init__(name=name)
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


class DQNAgent(Node):

    def __init__(self, robot_name, stage_num, max_training_episodes):
        super().__init__(f'dqn_agent_{robot_name}')

        self.robot_name = robot_name
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

        self.model = self._create_qnetwork()
        self.target_model = self._create_qnetwork()
        self._update_target_model()
        self.update_target_after = 5000
        self.target_update_counter = 0

        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model', self.robot_name)
        os.makedirs(self.model_dir_path, exist_ok=True)

        self._load_saved_model()
        self._init_logging()

        self.cb_group = ReentrantCallbackGroup()
        self._init_publishers()
        self._init_subscribers()
        self._init_service_clients()

        self.episode_start_requested = False
        self.current_episode = 0
        self.callback_count = 0

        # Debug tracking
        self.debug_phase = 'initializing'
        self.debug_local_step = 0
        self.debug_last_activity = time.time()

        self.process_thread = threading.Thread(target=self._process_wrapper)
        self.process_thread.daemon = True
        self.process_thread.start()

        self.debug_timer = self.create_timer(30.0, self._debug_status_callback)
        self.get_logger().info(f'{robot_name} DQN Agent initialized')

    def _load_saved_model(self):
        model_path = os.path.join(
            self.model_dir_path,
            f'stage{self.stage}_episode{self.load_episode}.keras')

        if self.load_model and os.path.exists(model_path):
            self.model.set_weights(load_model(model_path).get_weights())
            json_path = os.path.join(
                self.model_dir_path,
                f'stage{self.stage}_episode{self.load_episode}.json')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    param = json.load(f)
                    self.epsilon = param.get('epsilon')
                    self.step_counter = param.get('step_counter')

    def _init_logging(self):
        if not LOGGING:
            return
        log_name = f'{CURRENT_TIME}_dqn_stage{self.stage}_{self.robot_name}_reward'
        log_dir = os.path.join(
            os.path.expanduser('~'), 'turtlebot3_dqn_logs', 'gradient_tape', log_name)
        self.dqn_reward_writer = tensorflow.summary.create_file_writer(log_dir)
        self.dqn_reward_metric = DQNMetric()

    def _init_publishers(self):
        self.status_pub = self.create_publisher(String, f'/{self.robot_name}/status', 10)
        self.action_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/result', 10)

    def _init_subscribers(self):
        self.create_subscription(
            Int32, '/start_episode', self._start_episode_callback, 10,
            callback_group=self.cb_group)

    def _init_service_clients(self):
        self.rl_agent_interface_client = self.create_client(
            Dqn, f'/{self.robot_name}/rl_agent_interface', callback_group=self.cb_group)
        self.get_state_client = self.create_client(
            Dqn, f'/{self.robot_name}/get_state', callback_group=self.cb_group)

    def _debug_status_callback(self):
        elapsed = time.time() - self.debug_last_activity
        self.get_logger().warn(
            f'[DEBUG] {self.robot_name}: phase={self.debug_phase}, '
            f'episode={self.current_episode}, step={self.debug_local_step}, '
            f'idle={elapsed:.1f}s, mem={len(self.replay_memory)}, eps={self.epsilon:.3f}')

    def _process_wrapper(self):
        try:
            self._process()
        except Exception as e:
            self.get_logger().error(f'{self.robot_name} Exception in process(): {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _start_episode_callback(self, msg):
        self.callback_count += 1
        self.current_episode = msg.data
        self.episode_start_requested = True
        self.get_logger().info(f'{self.robot_name}: Episode {self.current_episode} callback received')

    def _publish_status(self):
        msg = String()
        msg.data = json.dumps({
            'done': self.done, 'succeeded': self.succeed, 'failed': self.fail})
        self.status_pub.publish(msg)

    def _process(self):
        self.get_logger().info(f'{self.robot_name}: Process started, waiting for coordinator...')
        self.debug_phase = 'waiting_coordinator'
        self.debug_last_activity = time.time()
        time.sleep(1.0)

        last_processed_episode = 0
        episode_wait_timeout = 60.0

        while rclpy.ok():
            self.debug_phase = 'waiting_episode_signal'
            self.debug_last_activity = time.time()
            wait_start = time.time()
            timed_out = False
            while self.current_episode <= last_processed_episode and rclpy.ok():
                if time.time() - wait_start > episode_wait_timeout:
                    self.get_logger().error(
                        f'{self.robot_name}: Timeout waiting for episode start signal '
                        f'(last={last_processed_episode}, current={self.current_episode}). Forcing next episode.')
                    timed_out = True
                    break
                time.sleep(0.1)

            if not rclpy.ok():
                break

            if timed_out:
                last_processed_episode += 1
                self.current_episode = last_processed_episode
                self.get_logger().warn(f'{self.robot_name}: Forced episode to {last_processed_episode}')

            episode = self.current_episode
            last_processed_episode = episode
            self.episode_start_requested = False

            self.get_logger().info(f'{self.robot_name}: Starting episode {episode}')

            self.debug_phase = 'get_initial_state'
            self.debug_last_activity = time.time()
            state = self._get_current_state()
            local_step = 0
            score = 0
            sum_max_q = 0.0

            while True:
                local_step += 1
                self.debug_local_step = local_step
                self.debug_last_activity = time.time()

                self.debug_phase = 'predict'
                q_values = self.model.predict(state, verbose=0)
                sum_max_q += float(numpy.max(q_values))

                action = int(self._get_action(state))
                self.debug_phase = f'step_action_{action}'
                next_state, reward, done = self._step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.debug_phase = 'training'
                    self.replay_memory.append((state, action, reward, next_state, done))
                    self._train_model(done)

                state = next_state
                self._publish_status()

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode)
                        self.dqn_reward_metric.reset_states()

                    self.get_logger().info(
                        f'{self.robot_name}: Episode {episode}, score={score:.1f}, '
                        f'memory={len(self.replay_memory)}, epsilon={self.epsilon:.3f}')
                    break

                time.sleep(0.01)

            if self.train_mode and episode % 1000 == 0:
                self._save_model(episode)

    def _get_current_state(self):
        while not self.get_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{self.robot_name}: Waiting for get_state service...')

        future = self.get_state_client.call_async(Dqn.Request())
        while not future.done() and rclpy.ok():
            time.sleep(0.01)

        if future.result() is not None:
            state = future.result().state
            return numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(f'{self.robot_name}: Failed to get current state')
            return numpy.zeros([1, self.state_size])

    def _get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            if random.random() > (1 - self.epsilon):
                return random.randint(0, self.action_size - 1)
        return numpy.argmax(self.model.predict(state, verbose=0))

    def _step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{self.robot_name}: Waiting for rl_agent_interface...')

        future = self.rl_agent_interface_client.call_async(req)
        while not future.done() and rclpy.ok():
            time.sleep(0.01)

        if future.result() is not None:
            result = future.result()
            next_state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
            return next_state, result.reward, result.done
        else:
            self.get_logger().error(f'{self.robot_name}: Step service call failed')
            return numpy.zeros([1, self.state_size]), 0.0, True

    def _create_qnetwork(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear'),
        ])
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0
        self.get_logger().info(f'{self.robot_name}: Target model updated')

    def _train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([t[0] for t in batch]).squeeze()
        current_qvalues = self.model.predict(current_states, verbose=0)

        next_states = numpy.array([t[3] for t in batch]).squeeze()
        next_qvalues = self.target_model.predict(next_states, verbose=0)

        x_train, y_train = [], []

        for i, (state, action, reward, _, done) in enumerate(batch):
            q_values = current_qvalues[i].copy()
            if done:
                q_values[action] = reward
            else:
                q_values[action] = reward + self.discount_factor * numpy.max(next_qvalues[i])
            x_train.append(state)
            y_train.append(q_values)

        x_train = numpy.reshape(numpy.array(x_train), [self.batch_size, self.state_size])
        y_train = numpy.reshape(numpy.array(y_train), [self.batch_size, self.action_size])

        self.model.fit(
            tensorflow.convert_to_tensor(x_train, tensorflow.float32),
            tensorflow.convert_to_tensor(y_train, tensorflow.float32),
            batch_size=self.batch_size, verbose=0)

        self.target_update_counter += 1
        if self.target_update_counter > self.update_target_after and terminal:
            self._update_target_model()

    def _save_model(self, episode):
        model_path = os.path.join(self.model_dir_path, f'stage{self.stage}_episode{episode}.keras')
        json_path = os.path.join(self.model_dir_path, f'stage{self.stage}_episode{episode}.json')

        self.model.save(model_path)
        with open(json_path, 'w') as f:
            json.dump({'epsilon': self.epsilon, 'step_counter': self.step_counter}, f)
        self.get_logger().info(f'{self.robot_name}: Model saved at episode {episode}')


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_episodes = args[2] if len(args) > 2 else '1000'

    rclpy.init(args=args)
    agent = DQNAgent('robot3', stage_num, max_episodes)

    executor = MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
