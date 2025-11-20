#!/usr/bin/env python3
#################################################################################
# DQN Agent for Robot 3
# Independent DQN agent with its own neural network and learning
#################################################################################

import collections
import datetime
import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Bool
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


class DQNAgent3(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('dqn_agent_robot3')

        self.robot_name = 'robot3'
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

        # Publishers and subscribers with robot namespace
        self.status_pub = self.create_publisher(String, f'/{self.robot_name}/status', 10)
        self.action_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, f'/{self.robot_name}/result', 10)

        self.reset_sub = self.create_subscription(Bool, '/reset_episode', self.reset_callback, 10)

        # Service clients with robot namespace
        self.rl_agent_interface_client = self.create_client(Dqn, f'/{self.robot_name}/rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, f'/{self.robot_name}/make_environment')
        self.reset_environment_client = self.create_client(Dqn, f'/{self.robot_name}/reset_environment')

        self.episode_reset_requested = False

        self.get_logger().info(f'{self.robot_name} DQN Agent initialized')
        self.get_logger().info(f'{self.robot_name} Starting process() method...')

        try:
            self.process()
        except Exception as e:
            self.get_logger().error(f'{self.robot_name} Exception in process(): {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def reset_callback(self, msg):
        """Callback for episode reset signal from coordinator"""
        if msg.data:
            self.episode_reset_requested = True

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

        self.get_logger().info(f'{self.robot_name} - Calling env_make()')
        self.env_make()
        self.get_logger().info(f'{self.robot_name} - env_make() completed')

        time.sleep(1.0)

        episode_num = self.load_episode
        self.get_logger().info(
            f'{self.robot_name} - Starting training loop: episodes {self.load_episode + 1} to {self.max_training_episodes}'
        )

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            self.get_logger().info(f'{self.robot_name} - Episode {episode} starting')
            self.get_logger().info(f'{self.robot_name} - Calling reset_environment()')
            state = self.reset_environment()
            self.get_logger().info(f'{self.robot_name} - reset_environment() completed, state shape: {state.shape}')
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1

                q_values = self.model.predict(state, verbose=0)
                sum_max_q += float(numpy.max(q_values))

                action = int(self.get_action(state))
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
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            tensorflow.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            )
                        self.dqn_reward_metric.reset_states()

                    self.get_logger().info(
                        f'{self.robot_name} - Episode: {episode}, '
                        f'score: {score}, '
                        f'memory length: {len(self.replay_memory)}, '
                        f'epsilon: {self.epsilon:.3f}'
                    )

                    # Wait for all robots to finish
                    self.get_logger().info(f'{self.robot_name} - Waiting for all robots to complete...')
                    self.episode_reset_requested = False
                    while not self.episode_reset_requested:
                        rclpy.spin_once(self, timeout_sec=0.1)
                    self.get_logger().info(f'{self.robot_name} - Reset signal received, continuing to next episode')

                    param_keys = ['epsilon', 'step_counter']
                    param_values = [self.epsilon, self.step_counter]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.01)

            if self.train_mode:
                if episode % 250 == 0:
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

    def env_make(self):
        self.get_logger().info(f'{self.robot_name} - env_make: Waiting for make_environment service...')

        timeout_count = 0
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            timeout_count += 1
            self.get_logger().warn(
                f'{self.robot_name} - Environment make client failed to connect to the server (attempt {timeout_count}), try again ...'
            )
            if timeout_count > 10:
                self.get_logger().error(f'{self.robot_name} - Failed to connect to make_environment service after 10 attempts!')
                return

        self.get_logger().info(f'{self.robot_name} - make_environment service found, calling...')
        self.make_environment_client.call_async(Empty.Request())
        self.get_logger().info(f'{self.robot_name} - make_environment service called')

    def reset_environment(self):
        self.get_logger().info(f'{self.robot_name} - reset_environment: Waiting for service...')

        timeout_count = 0
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            timeout_count += 1
            self.get_logger().warn(
                f'{self.robot_name} - Reset environment client failed to connect to the server (attempt {timeout_count}), try again ...'
            )
            if timeout_count > 10:
                self.get_logger().error(f'{self.robot_name} - Failed to connect to reset_environment service after 10 attempts!')
                raise Exception('Failed to connect to reset_environment service')

        self.get_logger().info(f'{self.robot_name} - reset_environment service found, calling...')
        future = self.reset_environment_client.call_async(Dqn.Request())

        self.get_logger().info(f'{self.robot_name} - Waiting for reset_environment response...')
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            state = future.result().state
            self.get_logger().info(f'{self.robot_name} - Received state with length: {len(state)}')
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
            self.get_logger().info(f'{self.robot_name} - State reshaped to: {state.shape}')
        else:
            self.get_logger().error(
                f'{self.robot_name} - Exception while calling service: {future.exception()}')
            raise Exception(f'reset_environment service call failed: {future.exception()}')

        return state

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

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                f'{self.robot_name} - Exception while calling service: {future.exception()}')

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

    dqn_agent = DQNAgent3(stage_num, max_training_episodes)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
