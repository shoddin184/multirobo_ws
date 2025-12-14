#!/usr/bin/env python3
"""Multi-Robot DQN Coordinator - Manages all robots' environments and episodes."""

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
    """Centralized coordinator for multi-robot DQN training."""

    def __init__(self, num_robots=3, max_episodes=1000):
        super().__init__('multi_robot_coordinator')

        self.num_robots = num_robots
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.robot_names = [f'robot{i+1}' for i in range(num_robots)]

        # ReentrantCallbackGroup for concurrent service handling
        self.cb_group = ReentrantCallbackGroup()
        
        # Episode transition lock
        self.is_transitioning = False
        
        # ★ FIX: エピソード遷移を別スレッドで処理するためのフラグ ★
        self._episode_complete_flag = threading.Event()

        self._init_robot_status()
        self._init_publishers()
        self._init_service_clients()
        self._init_service_servers()

        self.get_logger().info(
            f'Coordinator initialized: {num_robots} robots, {max_episodes} episodes')

    def _init_robot_status(self):
        """Initialize robot status tracking."""
        self.robot_status = {
            name: {'done': False, 'succeeded': False, 'failed': False, 'ready': False}
            for name in self.robot_names
        }

    def _init_publishers(self):
        """Initialize episode control publishers."""
        self.start_pub = self.create_publisher(Int32, '/start_episode', 10)

    def _init_service_clients(self):
        """Initialize service clients for environment management."""
        self.make_env_clients = {}
        self.reset_env_clients = {}

        for name in self.robot_names:
            self.make_env_clients[name] = self.create_client(
                Empty, f'/{name}/make_environment',
                callback_group=self.cb_group)
            self.reset_env_clients[name] = self.create_client(
                Dqn, f'/{name}/reset_environment',
                callback_group=self.cb_group)

    def _init_service_servers(self):
        """Initialize service servers to receive episode end notifications."""
        for name in self.robot_names:
            # task_succeed service server
            self.create_service(
                Goal, f'/{name}/task_succeed',
                lambda req, res, n=name: self._task_succeed_callback(req, res, n),
                callback_group=self.cb_group)
            
            # task_failed service server
            self.create_service(
                Goal, f'/{name}/task_failed',
                lambda req, res, n=name: self._task_failed_callback(req, res, n),
                callback_group=self.cb_group)

    # --- Service Callbacks ---

    def _task_succeed_callback(self, request, response, robot_name):
        """Handle task success notification from robot."""
        # Ignore notifications during episode transition
        if self.is_transitioning:
            self.get_logger().warn(
                f'{robot_name}: Task succeeded (ignored - transitioning)')
            response.success = True
            response.pose_x = 0.0
            response.pose_y = 0.0
            return response
        
        self.get_logger().info(f'{robot_name}: Task succeeded (received)')
        
        status = self.robot_status[robot_name]
        status['done'] = True
        status['succeeded'] = True
        
        # Return new goal position (placeholder - implement goal generation logic)
        response.success = True
        response.pose_x = 0.0
        response.pose_y = 0.0
        
        # ★ FIX: コールバック内で直接処理せず、フラグをセットするだけ ★
        if self._all_done():
            self.get_logger().info('All robots done - setting episode complete flag')
            self._episode_complete_flag.set()
        
        return response

    def _task_failed_callback(self, request, response, robot_name):
        """Handle task failure notification from robot."""
        # Ignore notifications during episode transition
        if self.is_transitioning:
            self.get_logger().warn(
                f'{robot_name}: Task failed (ignored - transitioning)')
            response.success = True
            response.pose_x = 0.0
            response.pose_y = 0.0
            return response
        
        self.get_logger().info(f'{robot_name}: Task failed (received)')
        
        status = self.robot_status[robot_name]
        status['done'] = True
        status['failed'] = True
        
        # Return new goal position (placeholder - implement goal generation logic)
        response.success = True
        response.pose_x = 0.0
        response.pose_y = 0.0
        
        # ★ FIX: コールバック内で直接処理せず、フラグをセットするだけ ★
        if self._all_done():
            self.get_logger().info('All robots done - setting episode complete flag')
            self._episode_complete_flag.set()
        
        return response

    # --- Environment Management ---

    def initialize_all_environments(self):
        """Initialize environments for all robots."""
        for name in self.robot_names:
            if not self.make_env_clients[name].wait_for_service(timeout_sec=10.0):
                self.get_logger().error(f'{name}/make_environment not available')
                return False

            future = self.make_env_clients[name].call_async(Empty.Request())
            
            # ★ FIX: spin_until_future_completeの代わりにポーリング ★
            while not future.done() and rclpy.ok():
                time.sleep(0.01)

            if future.result() is None:
                self.get_logger().error(f'Failed to init {name} environment')
                return False

        self.get_logger().info('All environments initialized')
        return True

    def reset_all_environments(self):
        """Reset all robot environments and return states."""
        states = {}

        for name in self.robot_names:
            if not self.reset_env_clients[name].wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f'{name}/reset_environment service not available')
                return None

            future = self.reset_env_clients[name].call_async(Dqn.Request())
            
            # ★ FIX: spin_until_future_completeの代わりにポーリング ★
            while not future.done() and rclpy.ok():
                time.sleep(0.01)

            if future.result() is None:
                self.get_logger().error(f'Failed to reset {name} environment')
                return None
            states[name] = future.result().state

        return states

    # --- Episode Management ---

    def _all_done(self):
        """Check if all robots completed their tasks."""
        return all(s['done'] for s in self.robot_status.values())

    def _handle_episode_complete(self):
        """Handle episode completion and start next episode."""
        self.is_transitioning = True
        
        self.get_logger().info(f'=== Episode {self.current_episode} complete handler started ===') 
        succeeded = sum(1 for s in self.robot_status.values() if s['succeeded'])
        failed = sum(1 for s in self.robot_status.values() if s['failed'])

        self.get_logger().info(
            f'Episode {self.current_episode}: {succeeded}/{self.num_robots} succeeded, '
            f'{failed}/{self.num_robots} failed')

        if self.current_episode >= self.max_episodes:
            self.get_logger().info('Training complete')
            self.is_transitioning = False
            return

        self.get_logger().info('Starting environment reset...')
        time.sleep(0.5)
        states = self.reset_all_environments()

        if states is None:
            self.get_logger().error('reset_all_environments() returned None - Episode loop stopped!')
            self.is_transitioning = False
            return

        self.get_logger().info(f'Environment reset successful, got {len(states)} states')

        # Reset status AFTER reset_all_environments completes
        for name in self.robot_status:
            self.robot_status[name] = {
                'done': False, 'succeeded': False, 'failed': False, 'ready': False}

        # Increment episode
        self.current_episode += 1
        self.get_logger().info(f'Episode incremented to {self.current_episode}')
        
        # Wait a bit for agents to be ready
        time.sleep(0.2)
        
        # Publish start signal multiple times
        self._publish_start_with_retry(self.current_episode)

        # Wait a bit longer to ensure all old notifications are processed
        time.sleep(0.5)

        self.is_transitioning = False
        self.get_logger().info(f'=== Episode {self.current_episode} transition complete ===')

    def _publish_start(self, episode):
        """Publish episode start signal."""
        self.get_logger().info(
        f'★ PUBLISHING ★ episode {episode} at {time.time()}'
        )
        msg = Int32()
        msg.data = episode
        self.start_pub.publish(msg)

    def _publish_start_with_retry(self, episode, retries=3, interval=0.1):
        """Publish episode start signal with retries to ensure delivery."""
        for i in range(retries):
            self._publish_start(episode)
            if i < retries - 1:
                time.sleep(interval)
        self.get_logger().info(f'Published start signal {retries} times for episode {episode}')

    # --- Training ---

    def run_training(self):
        """Run the training loop."""
        if not self.initialize_all_environments():
            return

        time.sleep(2.0)

        # Start first episode
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
            # エピソード完了フラグを待つ
            if self._episode_complete_flag.wait(timeout=0.5):
                self._episode_complete_flag.clear()
                self.get_logger().info('Episode complete flag received - handling transition')
                self._handle_episode_complete()


def main(args=None):
    rclpy.init(args=args)

    num_robots = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    coordinator = MultiRobotCoordinator(num_robots, max_episodes)
    
    # ★ FIX: Executorを先にセットアップして別スレッドでspinを開始 ★
    executor = MultiThreadedExecutor()  
    executor.add_node(coordinator)
    
    # executorを別スレッドでspin
    executor_thread = threading.Thread(target=executor.spin)
    executor_thread.daemon = True
    executor_thread.start()
    
    # executorが開始されるのを少し待つ
    time.sleep(0.5)
    
    # エピソード監視ループを別スレッドで開始
    monitor_thread = threading.Thread(target=coordinator.episode_monitor_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # トレーニング開始（これはブロッキングしない）
    coordinator.run_training()

    # メインスレッドはexecutorスレッドの終了を待つ
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