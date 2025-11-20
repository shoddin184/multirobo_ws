#!/usr/bin/env python3
#################################################################################
# Multi-Robot DQN Coordinator
# Centralized coordinator that manages all robots' environments and episodes
#################################################################################

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Int32
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn
import json
import time


class MultiRobotCoordinator(Node):

    def __init__(self, num_robots=3, max_episodes=1000):
        super().__init__('multi_robot_coordinator')

        self.num_robots = num_robots
        self.max_episodes = max_episodes
        self.current_episode = 0

        self.robot_names = [f'robot{i+1}' for i in range(num_robots)]

        self.robot_status = {
            name: {
                'done': False,
                'succeeded': False,
                'failed': False,
                'ready': False
            } for name in self.robot_names
        }

        # Subscribe to status updates from each robot
        self.status_subscribers = []
        for robot_name in self.robot_names:
            sub = self.create_subscription(
                String,
                f'/{robot_name}/status',
                lambda msg, name=robot_name: self.status_callback(msg, name),
                10
            )
            self.status_subscribers.append(sub)

        # Publisher for episode control signals
        self.reset_episode_pub = self.create_publisher(Bool, '/reset_episode', 10)
        self.start_episode_pub = self.create_publisher(Int32, '/start_episode', 10)

        # Service clients for environment management
        self.make_env_clients = {}
        self.reset_env_clients = {}

        for robot_name in self.robot_names:
            self.make_env_clients[robot_name] = self.create_client(
                Empty, f'/{robot_name}/make_environment'
            )
            self.reset_env_clients[robot_name] = self.create_client(
                Dqn, f'/{robot_name}/reset_environment'
            )

        self.get_logger().info(f'Multi-Robot Coordinator initialized for {num_robots} robots')
        self.get_logger().info(f'Max episodes: {max_episodes}')

    def initialize_all_environments(self):
        """Initialize environments for all robots"""
        self.get_logger().info('Initializing environments for all robots...')

        for robot_name in self.robot_names:
            self.get_logger().info(f'Waiting for {robot_name} make_environment service...')
            if not self.make_env_clients[robot_name].wait_for_service(timeout_sec=10.0):
                self.get_logger().error(f'Service {robot_name}/make_environment not available!')
                return False

            future = self.make_env_clients[robot_name].call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                self.get_logger().info(f'{robot_name} environment initialized')
            else:
                self.get_logger().error(f'Failed to initialize {robot_name} environment')
                return False

        self.get_logger().info('All environments initialized successfully')
        return True

    def reset_all_environments(self):
        """Reset environments for all robots and return states"""
        self.get_logger().info('Resetting environments for all robots...')
        states = {}

        for robot_name in self.robot_names:
            if not self.reset_env_clients[robot_name].wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f'Service {robot_name}/reset_environment not available!')
                return None

            future = self.reset_env_clients[robot_name].call_async(Dqn.Request())
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                states[robot_name] = future.result().state
                self.get_logger().info(f'{robot_name} environment reset, state size: {len(states[robot_name])}')
            else:
                self.get_logger().error(f'Failed to reset {robot_name} environment')
                return None

        return states

    def status_callback(self, msg, robot_name):
        """Receive status updates from individual robots"""
        try:
            status_data = json.loads(msg.data)
            self.robot_status[robot_name]['done'] = status_data.get('done', False)
            self.robot_status[robot_name]['succeeded'] = status_data.get('succeeded', False)
            self.robot_status[robot_name]['failed'] = status_data.get('failed', False)
            self.robot_status[robot_name]['ready'] = status_data.get('ready', False)

            # Check if all robots are done
            if self.check_all_done():
                self.handle_episode_complete()

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse status from {robot_name}: {e}')

    def check_all_done(self):
        """Check if all robots have completed their tasks"""
        return all(status['done'] for status in self.robot_status.values())

    def check_all_ready(self):
        """Check if all robots are ready to start"""
        return all(status['ready'] for status in self.robot_status.values())

    def handle_episode_complete(self):
        """Handle episode completion when all robots are done"""
        num_succeeded = sum(1 for status in self.robot_status.values() if status['succeeded'])
        num_failed = sum(1 for status in self.robot_status.values() if status['failed'])

        self.get_logger().info(
            f'Episode {self.current_episode} Complete! '
            f'Succeeded: {num_succeeded}/{self.num_robots}, '
            f'Failed: {num_failed}/{self.num_robots}'
        )

        # Reset all robot statuses
        for robot_name in self.robot_status:
            self.robot_status[robot_name] = {
                'done': False,
                'succeeded': False,
                'failed': False,
                'ready': False
            }

        # Check if training is complete
        if self.current_episode >= self.max_episodes:
            self.get_logger().info(f'Training complete! Reached {self.max_episodes} episodes')
            return

        # Reset environments for next episode
        time.sleep(0.5)
        states = self.reset_all_environments()

        if states is not None:
            # Start next episode
            self.current_episode += 1
            self.get_logger().info(f'Starting episode {self.current_episode}...')

            start_msg = Int32()
            start_msg.data = self.current_episode
            self.start_episode_pub.publish(start_msg)

    def run_training(self):
        """Main training loop controlled by coordinator"""
        self.get_logger().info('Starting training coordination...')

        # Initialize all environments
        if not self.initialize_all_environments():
            self.get_logger().error('Failed to initialize environments')
            return

        time.sleep(2.0)

        # Start first episode
        self.current_episode = 1
        self.get_logger().info('Resetting environments for episode 1...')
        states = self.reset_all_environments()

        if states is None:
            self.get_logger().error('Failed to reset environments')
            return

        time.sleep(1.0)

        # Send start signal for first episode
        self.get_logger().info(f'Starting episode {self.current_episode}')
        start_msg = Int32()
        start_msg.data = self.current_episode
        self.start_episode_pub.publish(start_msg)

        # Now spin and handle episode completions
        self.get_logger().info('Training loop running, waiting for episode completions...')

    def get_episode_stats(self):
        """Get current episode statistics"""
        return {
            'current_episode': self.current_episode,
            'total_robots': self.num_robots,
            'completed': sum(1 for status in self.robot_status.values() if status['done']),
            'succeeded': sum(1 for status in self.robot_status.values() if status['succeeded']),
            'failed': sum(1 for status in self.robot_status.values() if status['failed'])
        }


def main(args=None):
    rclpy.init(args=args)

    import sys
    num_robots = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    coordinator = MultiRobotCoordinator(num_robots=num_robots, max_episodes=max_episodes)

    # Run the training coordination
    coordinator.run_training()

    try:
        rclpy.spin(coordinator)
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
