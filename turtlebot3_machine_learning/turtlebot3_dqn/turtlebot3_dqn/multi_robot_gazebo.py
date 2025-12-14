#!/usr/bin/env python3
#################################################################################
# Multi-Robot Gazebo Interface
# Manages goal spawning and deletion for multiple robots
#################################################################################

import os
import random
import subprocess
import sys
import time
import threading

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')
if ROS_DISTRO == 'humble':
    from gazebo_msgs.srv import DeleteEntity
    from gazebo_msgs.srv import SpawnEntity
    from gazebo_msgs.srv import SetEntityState
    from gazebo_msgs.msg import EntityState
    from geometry_msgs.msg import Pose


class MultiRobotGazeboInterface(Node):

    def __init__(self, stage_num, robot_name='robot1'):
        super().__init__(f'gazebo_interface_{robot_name}')
        self.stage = int(stage_num)
        self.robot_name = robot_name

        self.get_logger().info(f'Initializing Gazebo interface for {robot_name}, stage {stage_num}')

        # Map robot names to Gazebo entity names
        # robot1 -> burger_1, robot2 -> burger_2, robot3 -> burger_3
        gazebo_robot_name_map = {
            'robot1': 'burger_1',
            'robot2': 'burger_2',
            'robot3': 'burger_3'
        }
        self.gazebo_robot_name = gazebo_robot_name_map.get(robot_name, robot_name)
        self.get_logger().info(f'Mapped {robot_name} -> Gazebo entity: {self.gazebo_robot_name}')

        self.entity_name = f'goal_box_{robot_name}'
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        # Lock for thread-safe service calls
        self._reset_lock = threading.Lock()
        self._reset_complete = threading.Event()

        if ROS_DISTRO == 'humble':
            self.entity = None
            self.open_entity()
            self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
            self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
            self.set_entity_state_client = self.create_client(SetEntityState, 'set_entity_state')
            self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        self.callback_group = MutuallyExclusiveCallbackGroup()

        # Create services with robot namespace
        self.initialize_env_service = self.create_service(
            Goal,
            f'/{robot_name}/initialize_env',
            self.initialize_env_callback,
            callback_group=self.callback_group
        )
        self.task_succeed_service = self.create_service(
            Goal,
            f'/{robot_name}/task_succeed',
            self.task_succeed_callback,
            callback_group=self.callback_group
        )
        self.task_failed_service = self.create_service(
            Goal,
            f'/{robot_name}/task_failed',
            self.task_failed_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info(f'Gazebo interface for {robot_name} initialized')

    def open_entity(self):
        try:
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            with open(model_path, 'r') as f:
                self.entity = f.read()
            self.get_logger().info('Loaded entity from: ' + model_path)
        except Exception as e:
            self.get_logger().error('Failed to load entity file: {}'.format(e))
            raise e

    def spawn_entity(self):
        if ROS_DISTRO == 'humble':
            entity_pose = Pose()
            entity_pose.position.x = self.entity_pose_x
            entity_pose.position.y = self.entity_pose_y

            spawn_req = SpawnEntity.Request()
            spawn_req.name = self.entity_name
            spawn_req.xml = self.entity
            spawn_req.initial_pose = entity_pose

            while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('service for spawn_entity is not available, waiting ...')
            future = self.spawn_entity_client.call_async(spawn_req)
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info(
                f'Spawn Goal for {self.robot_name} at ({self.entity_pose_x}, {self.entity_pose_y}, {0.0})'
            )
        else:
            service_name = '/world/dqn/create'
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            req = (
                f'sdf_filename: "{model_path}", '
                f'name: "{self.entity_name}", '
                f'pose: {{ position: {{ '
                f'x: {self.entity_pose_x}, '
                f'y: {self.entity_pose_y}, '
                f'z: 0.0 }} }}'
            )
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                self.get_logger().info(
                    f'Spawn Goal for {self.robot_name} at ({self.entity_pose_x}, {self.entity_pose_y}, {0.0})'
                )
            except subprocess.CalledProcessError:
                pass

    def delete_entity(self):
        if ROS_DISTRO == 'humble':
            delete_req = DeleteEntity.Request()
            delete_req.name = self.entity_name

            while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn('service for delete_entity is not available, waiting ...')
            future = self.delete_entity_client.call_async(delete_req)
            rclpy.spin_until_future_complete(self, future)
            self.get_logger().info(f'Delete Goal for {self.robot_name}')
        else:
            service_name = '/world/dqn/remove'
            req = f'name: "{self.entity_name}", type: 2'
            cmd = [
                'gz', 'service',
                '-s', service_name,
                '--reqtype', 'gz.msgs.Entity',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '1000',
                '--req', req
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                self.get_logger().info(f'Delete Goal for {self.robot_name}')
            except subprocess.CalledProcessError:
                pass

    def reset_simulation(self):
        """Reset single robot position (for Humble) - Non-blocking version"""
        self.get_logger().info(f'[{self.robot_name}] reset_simulation() called - Resetting ONLY this robot')

        # Reset completion event
        self._reset_complete.clear()

        # Start reset in separate thread to avoid blocking ROS2 spin
        reset_thread = threading.Thread(target=self._reset_robot_position_sync)
        reset_thread.start()

        # Wait for completion with timeout
        if not self._reset_complete.wait(timeout=5.0):
            self.get_logger().warn(f'[{self.robot_name}] Reset timed out, continuing anyway')
        else:
            self.get_logger().info(f'[{self.robot_name}] Reset completed successfully')

    def _reset_robot_position_sync(self):
        """Synchronous robot position reset (runs in separate thread)"""
        with self._reset_lock:
            try:
                # Get initial position for this robot
                initial_positions = {
                    'robot1': {'x': 2.0, 'y': 1.0},
                    'robot2': {'x': -2.0, 'y': 1.0},
                    'robot3': {'x': 0.0, 'y': -2.0}
                }
                pos = initial_positions.get(self.robot_name, {'x': 0.0, 'y': 0.0})
                self.get_logger().info(f'[{self.robot_name}] Target position: ({pos["x"]}, {pos["y"]})')

                # Check service availability (non-blocking)
                if not self.set_entity_state_client.service_is_ready():
                    self.get_logger().warn(f'[{self.robot_name}] set_entity_state service not ready, waiting...')
                    # Wait with timeout in thread (doesn't block main spin)
                    ready = self.set_entity_state_client.wait_for_service(timeout_sec=3.0)
                    if not ready:
                        self.get_logger().error(f'[{self.robot_name}] Service not available after timeout')
                        return

                # Prepare request
                req = SetEntityState.Request()
                state = EntityState()
                state.name = self.gazebo_robot_name
                state.pose.position.x = pos['x']
                state.pose.position.y = pos['y']
                state.pose.position.z = 0.01
                state.pose.orientation.w = 1.0
                state.twist.linear.x = 0.0
                state.twist.linear.y = 0.0
                state.twist.linear.z = 0.0
                state.twist.angular.x = 0.0
                state.twist.angular.y = 0.0
                state.twist.angular.z = 0.0
                req.state = state

                self.get_logger().info(f'[{self.robot_name}] Calling set_entity_state for {self.gazebo_robot_name}')

                # Call service asynchronously
                future = self.set_entity_state_client.call_async(req)

                # Wait for result with timeout (in this thread, not blocking main)
                timeout_sec = 3.0
                start_time = time.time()
                while not future.done():
                    if time.time() - start_time > timeout_sec:
                        self.get_logger().warn(f'[{self.robot_name}] Service call timed out')
                        break
                    time.sleep(0.05)

                if future.done():
                    try:
                        result = future.result()
                        self.get_logger().info(
                            f'[{self.robot_name}] Reset {self.gazebo_robot_name} to ({pos["x"]}, {pos["y"]}) - Success: {result.success}'
                        )
                    except Exception as e:
                        self.get_logger().warn(f'[{self.robot_name}] Service call exception: {e}')
                else:
                    self.get_logger().info(
                        f'[{self.robot_name}] Reset {self.gazebo_robot_name} to ({pos["x"]}, {pos["y"]}) - Async call sent'
                    )

            except Exception as e:
                self.get_logger().error(f'[{self.robot_name}] Reset failed with exception: {e}')
            finally:
                # Signal completion
                self._reset_complete.set()

    def reset_robot(self):
        """Reset robot position to initial state"""
        service_name_delete = '/world/dqn/remove'
        req_delete = f'name: "{self.gazebo_robot_name}", type: 2'
        cmd_delete = [
            'gz', 'service',
            '-s', service_name_delete,
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_delete
        ]
        try:
            subprocess.run(cmd_delete, check=True, stdout=subprocess.DEVNULL)
            self.get_logger().info(f'Delete {self.gazebo_robot_name}')
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.2)

        service_name_spawn = '/world/dqn/create'
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(package_share, 'models', 'turtlebot3_burger', 'model.sdf')

        # Different initial positions for each robot
        # Match the positions from multi_robot.launch.py: [[2, 1], [-2, 1], [0, -2]]
        initial_positions = {
            'robot1': {'x': 2.0, 'y': 1.0},
            'robot2': {'x': -2.0, 'y': 1.0},
            'robot3': {'x': 0.0, 'y': -2.0}
        }
        pos = initial_positions.get(self.robot_name, {'x': 0.0, 'y': 0.0})

        req_spawn = (
            f'sdf_filename: "{model_path}", '
            f'name: "{self.gazebo_robot_name}", '
            f'pose: {{ position: {{ x: {pos["x"]}, y: {pos["y"]}, z: 0.0 }} }}'
        )
        cmd_spawn = [
            'gz', 'service',
            '-s', service_name_spawn,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_spawn
        ]
        try:
            subprocess.run(cmd_spawn, check=True, stdout=subprocess.DEVNULL)
            self.get_logger().info(f'Spawn {self.gazebo_robot_name} at ({pos["x"]}, {pos["y"]})')
        except subprocess.CalledProcessError:
            pass

    def task_succeed_callback(self, request, response):
        self.get_logger().info(f'[{self.robot_name}] Task succeed callback - Goal reached!')
        self.delete_entity()
        time.sleep(0.2)
        self.generate_goal_pose()
        self.get_logger().info(f'[{self.robot_name}] New goal generated at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def task_failed_callback(self, request, response):
        self.get_logger().info(f'[{self.robot_name}] Task failed callback - Collision/Timeout detected!')
        self.delete_entity()
        time.sleep(0.2)
        if ROS_DISTRO == 'humble':
            self.get_logger().info(f'[{self.robot_name}] Resetting robot position (Humble mode)')
            self.reset_simulation()
        else:
            self.get_logger().info(f'[{self.robot_name}] Resetting robot position (Gazebo mode)')
            self.reset_robot()
        time.sleep(0.2)
        self.generate_goal_pose()
        self.get_logger().info(f'[{self.robot_name}] New goal generated at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info(f'[{self.robot_name}] Task failed callback complete')
        return response

    def initialize_env_callback(self, request, response):
        self.get_logger().info(f'[{self.robot_name}] Initialize environment callback')
        self.delete_entity()
        time.sleep(0.2)
        if ROS_DISTRO == 'humble':
            self.get_logger().info(f'[{self.robot_name}] Resetting robot position (Humble mode)')
            self.reset_simulation()
        else:
            self.get_logger().info(f'[{self.robot_name}] Resetting robot position (Gazebo mode)')
            self.reset_robot()
        time.sleep(0.2)
        self.generate_goal_pose()
        self.get_logger().info(f'[{self.robot_name}] Initial goal generated at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info(f'[{self.robot_name}] Initialize environment callback complete')
        return response

    def generate_goal_pose(self):
        """Generate goal position avoiding other robots' goals"""
        if self.stage != 4:
            # Random position in the environment
            self.entity_pose_x = random.randrange(-21, 21) / 10
            self.entity_pose_y = random.randrange(-21, 21) / 10
        else:
            # Predefined positions for stage 4
            goal_pose_list = [
                [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 1.5], [0.5, 2.0], [-1.5, 2.1],
                [-2.0, 0.5], [-2.0, -0.5], [-1.5, -2.0], [-0.5, -1.0], [2.0, -0.5], [-1.0, -1.0]
            ]
            rand_index = random.randint(0, len(goal_pose_list) - 1)
            self.entity_pose_x = goal_pose_list[rand_index][0]
            self.entity_pose_y = goal_pose_list[rand_index][1]


def main(args=None):
    rclpy.init(args=sys.argv)
    stage_num = sys.argv[1] if len(sys.argv) > 1 else '1'
    robot_name = sys.argv[2] if len(sys.argv) > 2 else 'robot1'

    if len(sys.argv) < 3:
        print(f'WARNING: robot_name not specified, defaulting to "{robot_name}"')
        print(f'Usage: ros2 run turtlebot3_dqn multi_robot_gazebo <stage_num> <robot_name>')
        print(f'Example: ros2 run turtlebot3_dqn multi_robot_gazebo 1 robot2')

    gazebo_interface = MultiRobotGazeboInterface(stage_num, robot_name)
    try:
        while rclpy.ok():
            rclpy.spin_once(gazebo_interface, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gazebo_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()