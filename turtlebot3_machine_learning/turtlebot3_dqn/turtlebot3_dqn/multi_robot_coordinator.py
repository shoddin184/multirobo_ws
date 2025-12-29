#!/usr/bin/env python3
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Int32
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn, Goal


class MultiRobotCoordinator(Node):

    def __init__(self, num_robots=3, max_episodes=1000):
        super().__init__('multi_robot_coordinator')

        self.num_robots = num_robots
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.robot_names = [f'robot{i+1}' for i in range(num_robots)]

        self.cb_group = ReentrantCallbackGroup()
        self.is_transitioning = False
        self._episode_complete_flag = threading.Event()
        self._pending_notifications = {name: None for name in self.robot_names}

        self._init_robot_status()
        self._init_publishers()
        self._init_service_clients()
        self._init_service_servers()

        self.get_logger().info(
            f'Coordinator initialized: {num_robots} robots, {max_episodes} episodes')

    def _init_robot_status(self):
        self.robot_status = {
            name: {'done': False, 'succeeded': False, 'failed': False, 'ready': False}
            for name in self.robot_names
        }

    def _init_publishers(self):
        self.start_pub = self.create_publisher(Int32, '/start_episode', 10)

    def _init_service_clients(self):
        self.make_env_clients = {}
        self.reset_env_clients = {}

        for name in self.robot_names:
            self.make_env_clients[name] = self.create_client(
                Empty, f'/{name}/make_environment', callback_group=self.cb_group)
            self.reset_env_clients[name] = self.create_client(
                Dqn, f'/{name}/reset_environment', callback_group=self.cb_group)

    def _init_service_servers(self):
        for name in self.robot_names:
            self.create_service(
                Goal, f'/{name}/coordinator/task_succeed',
                lambda req, res, n=name: self._task_succeed_callback(req, res, n),
                callback_group=self.cb_group)
            self.create_service(
                Goal, f'/{name}/coordinator/task_failed',
                lambda req, res, n=name: self._task_failed_callback(req, res, n),
                callback_group=self.cb_group)

    def _task_succeed_callback(self, request, response, robot_name):
        if self.is_transitioning:
            self.get_logger().warn(f'{robot_name}: Task succeeded (queued - transitioning)')
            self._pending_notifications[robot_name] = 'succeeded'
            response.success = True
            response.pose_x = 0.0
            response.pose_y = 0.0
            return response

        self.get_logger().info(f'{robot_name}: Task succeeded')
        status = self.robot_status[robot_name]
        status['done'] = True
        status['succeeded'] = True

        response.success = True
        response.pose_x = 0.0
        response.pose_y = 0.0

        if self._all_done():
            self.get_logger().info('All robots done - setting episode complete flag')
            self._episode_complete_flag.set()

        return response

    def _task_failed_callback(self, request, response, robot_name):
        if self.is_transitioning:
            self.get_logger().warn(f'{robot_name}: Task failed (queued - transitioning)')
            self._pending_notifications[robot_name] = 'failed'
            response.success = True
            response.pose_x = 0.0
            response.pose_y = 0.0
            return response

        self.get_logger().info(f'{robot_name}: Task failed')
        status = self.robot_status[robot_name]
        status['done'] = True
        status['failed'] = True

        response.success = True
        response.pose_x = 0.0
        response.pose_y = 0.0

        if self._all_done():
            self.get_logger().info('All robots done - setting episode complete flag')
            self._episode_complete_flag.set()

        return response

    def initialize_all_environments(self):
        for name in self.robot_names:
            if not self.make_env_clients[name].wait_for_service(timeout_sec=10.0):
                self.get_logger().error(f'{name}/make_environment not available')
                return False

            future = self.make_env_clients[name].call_async(Empty.Request())
            while not future.done() and rclpy.ok():
                time.sleep(0.01)

            if future.result() is None:
                self.get_logger().error(f'Failed to init {name} environment')
                return False

        self.get_logger().info('All environments initialized')
        return True

    def reset_all_environments(self):
        states = {}
        for name in self.robot_names:
            if not self.reset_env_clients[name].wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f'{name}/reset_environment service not available')
                return None

            future = self.reset_env_clients[name].call_async(Dqn.Request())
            while not future.done() and rclpy.ok():
                time.sleep(0.01)

            if future.result() is None:
                self.get_logger().error(f'Failed to reset {name} environment')
                return None
            states[name] = future.result().state

        return states

    def _process_pending_notifications(self):
        processed_count = 0
        for robot_name, notification in self._pending_notifications.items():
            if notification is not None:
                self.get_logger().info(f'Processing queued: {robot_name} -> {notification}')
                status = self.robot_status[robot_name]
                status['done'] = True
                if notification == 'succeeded':
                    status['succeeded'] = True
                elif notification == 'failed':
                    status['failed'] = True
                self._pending_notifications[robot_name] = None
                processed_count += 1

        if processed_count > 0:
            self.get_logger().info(f'Processed {processed_count} queued notifications')

    def _all_done(self):
        return all(s['done'] for s in self.robot_status.values())

    def _handle_episode_complete(self):
        self.is_transitioning = True

        self.get_logger().info(f'=== Episode {self.current_episode} complete ===')
        succeeded = sum(1 for s in self.robot_status.values() if s['succeeded'])
        failed = sum(1 for s in self.robot_status.values() if s['failed'])
        self.get_logger().info(
            f'Episode {self.current_episode}: {succeeded}/{self.num_robots} succeeded, '
            f'{failed}/{self.num_robots} failed')

        if self.current_episode >= self.max_episodes:
            self.get_logger().info('Training complete')
            self.is_transitioning = False
            return

        self._clear_pending_notifications('stale')

        self.get_logger().info('Starting environment reset...')
        time.sleep(0.5)
        states = self.reset_all_environments()

        if states is None:
            self.get_logger().error('reset_all_environments() returned None')
            self.is_transitioning = False
            return

        self.get_logger().info(f'Environment reset successful, got {len(states)} states')

        for name in self.robot_status:
            self.robot_status[name] = {
                'done': False, 'succeeded': False, 'failed': False, 'ready': False}

        self._clear_pending_notifications('during reset')

        self.current_episode += 1
        self.get_logger().info(f'Episode incremented to {self.current_episode}')

        self.is_transitioning = False
        self.get_logger().info('Transition flag cleared')

        time.sleep(0.3)
        self._publish_start_with_retry(self.current_episode)
        self.get_logger().info(f'=== Episode {self.current_episode} started ===')

    def _clear_pending_notifications(self, context):
        for name in self._pending_notifications:
            if self._pending_notifications[name] is not None:
                self.get_logger().warn(
                    f'Discarding {context} notification from {name}: '
                    f'{self._pending_notifications[name]}')
            self._pending_notifications[name] = None

    def _publish_start(self, episode):
        self.get_logger().info(f'Publishing episode {episode}')
        msg = Int32()
        msg.data = episode
        self.start_pub.publish(msg)

    def _publish_start_with_retry(self, episode, retries=5, interval=0.2):
        for _ in range(retries):
            self._publish_start(episode)
            time.sleep(interval)
        self.get_logger().info(f'Published start signal {retries} times for episode {episode}')

    def run_training(self):
        if not self.initialize_all_environments():
            return

        time.sleep(2.0)
        self.current_episode = 1
        states = self.reset_all_environments()

        if states is None:
            self.get_logger().error('Failed to reset environments')
            return

        time.sleep(1.0)
        self._publish_start_with_retry(self.current_episode)
        self.get_logger().info('Training started')

    def episode_monitor_loop(self):
        self.get_logger().info('Episode monitor loop started')
        while rclpy.ok():
            if self._episode_complete_flag.wait(timeout=0.5):
                self._episode_complete_flag.clear()
                self.get_logger().info('Episode complete flag received')
                self._handle_episode_complete()


def main(args=None):
    rclpy.init(args=args)

    num_robots = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    coordinator = MultiRobotCoordinator(num_robots, max_episodes)

    executor = MultiThreadedExecutor()
    executor.add_node(coordinator)

    executor_thread = threading.Thread(target=executor.spin)
    executor_thread.daemon = True
    executor_thread.start()

    time.sleep(0.5)

    monitor_thread = threading.Thread(target=coordinator.episode_monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()

    coordinator.run_training()

    try:
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        coordinator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
