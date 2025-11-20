#!/usr/bin/env python3
#################################################################################
# Multi-Robot RL Environment
# Manages environment for individual robot with namespace support
#################################################################################

import math
import os

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')


class MultiRobotRLEnvironment(Node):

    def __init__(self, robot_name='robot1'):
        super().__init__(f'rl_environment_{robot_name}')

        self.robot_name = robot_name

        # Map robot names to Gazebo namespaces
        # robot1 -> TB3_1, robot2 -> TB3_2, robot3 -> TB3_3
        gazebo_namespace_map = {
            'robot1': 'TB3_1',
            'robot2': 'TB3_2',
            'robot3': 'TB3_3'
        }
        self.gazebo_namespace = gazebo_namespace_map.get(robot_name, robot_name)

        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 800

        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.is_front_min_actual_front = False

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        qos = QoSProfile(depth=10)

        # Publishers and subscribers - use Gazebo namespace for actual robot topics
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.gazebo_namespace}/cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, f'/{self.gazebo_namespace}/cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.gazebo_namespace}/odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            f'/{self.gazebo_namespace}/scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f'RL Environment for {robot_name} initialized (Gazebo namespace: {self.gazebo_namespace})'
        )

        # Service clients with robot namespace
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            f'/{robot_name}/task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            f'/{robot_name}/task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            f'/{robot_name}/initialize_env',
            callback_group=self.clients_callback_group
        )

        # Services with robot namespace
        self.rl_agent_interface_service = self.create_service(
            Dqn,
            f'/{robot_name}/rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            f'/{robot_name}/make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            f'/{robot_name}/reset_environment',
            self.reset_environment_callback
        )
        self.get_state_service = self.create_service(
            Dqn,
            f'/{robot_name}/get_state',
            self.get_state_callback
        )

    def make_environment_callback(self, request, response):
        self.get_logger().info(f'{self.robot_name} - Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self.robot_name} - service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error(f'{self.robot_name} - initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                f'{self.robot_name} - goal initialized at [{self.goal_pose_x:.2f}, {self.goal_pose_y:.2f}]'
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.get_logger().info(
            f'{self.robot_name} - Reset environment: state size = {len(state)}, '
            f'front_ranges size = {len(self.front_ranges)}'
        )
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state

        return response

    def get_state_callback(self, request, response):
        """Get current state without taking any action"""
        state = self.calculate_state_without_termination_check()
        response.state = state
        response.reward = 0.0
        response.done = False
        return response

    def calculate_state_without_termination_check(self):
        """Calculate state without checking for episode termination"""
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))

        # If LiDAR data is not ready yet, fill with default values
        if len(self.front_ranges) == 0:
            # Fill with 180 scan points with max range
            for _ in range(180):
                state.append(3.5)
        else:
            for var in self.front_ranges:
                state.append(float(var))

        return state

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self.robot_name} - service for task succeed is not available, waiting ...'
            )
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info(f'{self.robot_name} - service for task succeed finished')
        else:
            self.get_logger().error(f'{self.robot_name} - task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f'{self.robot_name} - service for task failed is not available, waiting ...'
            )
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info(f'{self.robot_name} - service for task failed finished')
        else:
            self.get_logger().error(f'{self.robot_name} - task failed service call failed')

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        self.front_distance = scan.ranges[0]

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        self.min_obstacle_distance = min(self.scan_ranges) if self.scan_ranges else 10.0
        self.front_min_obstacle_distance = min(self.front_ranges) if self.front_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))

        # If LiDAR data is not ready yet, fill with default values
        if len(self.front_ranges) == 0:
            # Fill with 180 scan points with max range
            for _ in range(180):
                state.append(3.5)
        else:
            for var in self.front_ranges:
                state.append(float(var))

        self.local_step += 1

        if self.goal_distance < 0.20:
            self.get_logger().info(f'{self.robot_name} - Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info(f'{self.robot_name} - Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.max_step:
            self.get_logger().info(f'{self.robot_name} - Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        return state

    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 6
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights

    def compute_weighted_obstacle_reward(self):
        if not self.front_ranges or not self.front_angles:
            return 0.0

        front_ranges = numpy.array(self.front_ranges)
        front_angles = numpy.array(self.front_angles)

        valid_mask = front_ranges <= 0.5
        if not numpy.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = numpy.unwrap(front_angles)
        relative_angles[relative_angles > numpy.pi] -= 2 * numpy.pi

        weights = self.compute_directional_weights(relative_angles, max_weight=10.0)

        safe_dists = numpy.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = numpy.exp(-3.0 * safe_dists)

        weighted_decay = numpy.dot(weights, decay)

        reward = - (1.0 + 4.0 * weighted_decay)

        return reward

    def calculate_reward(self):
        yaw_reward = 1 - (2 * abs(self.goal_angle) / math.pi)
        obstacle_reward = self.compute_weighted_obstacle_reward()

        reward = yaw_reward + obstacle_reward

        if self.succeed:
            reward = 100.0
        elif self.fail:
            reward = -50.0

        return reward

    def rl_agent_interface_callback(self, request, response):
        action = request.action
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = 0.2
            msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            msg.twist.linear.x = 0.2
            msg.twist.angular.z = self.angular_vel[action]

        self.cmd_vel_pub.publish(msg)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    import sys
    rclpy.init(args=args)
    robot_name = sys.argv[1] if len(sys.argv) > 1 else 'robot1'

    rl_environment = MultiRobotRLEnvironment(robot_name)
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
