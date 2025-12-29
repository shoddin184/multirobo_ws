#!/usr/bin/env python3
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from std_msgs.msg import Bool
from turtlebot3_msgs.srv import Dqn, Goal

ROS_DISTRO = os.environ.get('ROS_DISTRO')
GAZEBO_NS_MAP = {'robot1': 'TB3_1', 'robot2': 'TB3_2', 'robot3': 'TB3_3'}

GOAL_THRESHOLD = 0.20
COLLISION_THRESHOLD = 0.15
OBSTACLE_DETECT_RANGE = 0.5
MAX_LIDAR_RANGE = 3.5
LINEAR_VELOCITY = 0.2
ANGULAR_VELOCITIES = [1.5, 0.75, 0.0, -0.75, -1.5]


class MultiRobotRLEnvironment(Node):

    def __init__(self, robot_name='robot1'):
        super().__init__(f'rl_environment_{robot_name}')
        self.robot_name = robot_name
        self.gazebo_ns = GAZEBO_NS_MAP.get(robot_name, robot_name)
        self.cb_group = ReentrantCallbackGroup()

        self._init_state_variables()
        self._init_publishers_subscribers()
        self._init_service_clients()
        self._init_services()

        self.get_logger().info(f'{robot_name} environment initialized')

    def _init_state_variables(self):
        self.goal_pose = (0.0, 0.0)
        self.robot_pose = (0.0, 0.0, 0.0)
        self.goal_distance = 1.0
        self.goal_angle = 0.0
        self.init_goal_distance = 0.5
        self.prev_goal_distance = 0.5

        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []
        self.min_obstacle_distance = 10.0

        self.done = False
        self.fail = False
        self.succeed = False
        self.local_step = 0
        self.max_step = 800
        self.stop_timer = None
        self.is_resetting = False

    def _init_publishers_subscribers(self):
        qos = QoSProfile(depth=10)
        cmd_vel_type = Twist if ROS_DISTRO == 'humble' else TwistStamped

        self.cmd_vel_pub = self.create_publisher(
            cmd_vel_type, f'/{self.gazebo_ns}/cmd_vel', qos)
        self.succeed_pub = self.create_publisher(
            Bool, f'/{self.robot_name}/episode_succeed', qos)
        self.create_subscription(
            Odometry, f'/{self.gazebo_ns}/odom', self._odom_callback, qos)
        self.create_subscription(
            LaserScan, f'/{self.gazebo_ns}/scan', self._scan_callback,
            qos_profile_sensor_data)

    def _init_service_clients(self):
        ns = self.robot_name
        self.task_succeed_client = self.create_client(
            Goal, f'/{ns}/coordinator/task_succeed', callback_group=self.cb_group)
        self.task_failed_client = self.create_client(
            Goal, f'/{ns}/coordinator/task_failed', callback_group=self.cb_group)
        self.init_env_client = self.create_client(
            Goal, f'/{ns}/initialize_env', callback_group=self.cb_group)

    def _init_services(self):
        ns = self.robot_name
        self.create_service(
            Dqn, f'/{ns}/rl_agent_interface', self._agent_callback,
            callback_group=self.cb_group)
        self.create_service(
            Empty, f'/{ns}/make_environment', self._make_env_callback,
            callback_group=self.cb_group)
        self.create_service(
            Dqn, f'/{ns}/reset_environment', self._reset_env_callback,
            callback_group=self.cb_group)
        self.create_service(
            Dqn, f'/{ns}/get_state', self._get_state_callback,
            callback_group=self.cb_group)

    def _odom_callback(self, msg):
        pos = msg.pose.pose.position
        _, _, theta = self._euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = (pos.x, pos.y, theta)

        gx, gy = self.goal_pose
        rx, ry, rt = self.robot_pose

        self.goal_distance = math.hypot(gx - rx, gy - ry)
        angle = math.atan2(gy - ry, gx - rx) - rt
        self.goal_angle = self._normalize_angle(angle)

    def _scan_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        for i, dist in enumerate(scan.ranges):
            angle = scan.angle_min + i * scan.angle_increment
            dist = self._sanitize_distance(dist)
            self.scan_ranges.append(dist)

            if self._is_front_angle(angle):
                self.front_ranges.append(dist)
                self.front_angles.append(angle)

        self.min_obstacle_distance = min(self.scan_ranges) if self.scan_ranges else 10.0

    def _agent_callback(self, request, response):
        self._publish_velocity(request.action)
        self._reset_stop_timer()

        response.state = self._calculate_state()
        response.reward = self._calculate_reward()
        response.done = self.done

        if self.done:
            self._reset_episode_flags()

        return response

    def _make_env_callback(self, request, response):
        self._wait_for_service(self.init_env_client, 'initialize_env')
        result = self.init_env_client.call(Goal.Request())
        if result and result.success:
            self.goal_pose = (result.pose_x, result.pose_y)
        return response

    def _reset_env_callback(self, request, response):
        self.get_logger().info(f'{self.robot_name}: Resetting environment...')

        self.is_resetting = True
        self._reset_episode_flags()
        self.local_step = 0
        self._stop_robot()

        try:
            self._call_service(self.init_env_client)
            self.get_logger().info(f'{self.robot_name}: New goal: {self.goal_pose}')
        except Exception as e:
            self.get_logger().error(f'{self.robot_name}: Failed to initialize env: {e}')

        time.sleep(0.5)
        self.min_obstacle_distance = MAX_LIDAR_RANGE

        state = self._build_state_vector()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance

        time.sleep(0.3)
        self.is_resetting = False

        response.state = state
        self.get_logger().info(
            f'{self.robot_name}: Reset complete. min_obstacle={self.min_obstacle_distance:.2f}')
        return response

    def _get_state_callback(self, request, response):
        response.state = self._build_state_vector()
        response.reward = 0.0
        response.done = False
        return response

    def _calculate_state(self):
        state = self._build_state_vector()
        self.local_step += 1

        if self.is_resetting:
            return state

        if self.goal_distance < GOAL_THRESHOLD:
            self._handle_success()
        elif self.min_obstacle_distance < COLLISION_THRESHOLD:
            self._handle_collision()
        elif self.local_step >= self.max_step:
            self._handle_timeout()

        return state

    def _build_state_vector(self):
        state = [float(self.goal_distance), float(self.goal_angle)]
        if not self.front_ranges:
            state.extend([MAX_LIDAR_RANGE] * 180)
        else:
            state.extend(float(r) for r in self.front_ranges)
        return state

    def _calculate_reward(self):
        if self.succeed:
            return 100.0
        if self.fail:
            return -50.0

        yaw_reward = 1 - (2 * abs(self.goal_angle) / math.pi)
        obstacle_reward = self._compute_obstacle_reward()
        return yaw_reward + obstacle_reward

    def _compute_obstacle_reward(self):
        if not self.front_ranges or not self.front_angles:
            return 0.0

        ranges = np.array(self.front_ranges)
        angles = np.array(self.front_angles)

        mask = ranges <= OBSTACLE_DETECT_RANGE
        if not np.any(mask):
            return 0.0

        ranges, angles = ranges[mask], angles[mask]
        angles = np.unwrap(angles)
        angles[angles > np.pi] -= 2 * np.pi

        weights = self._compute_direction_weights(angles)
        safe_dist = np.clip(ranges - 0.25, 1e-2, MAX_LIDAR_RANGE)
        decay = np.exp(-3.0 * safe_dist)

        return -(1.0 + 4.0 * np.dot(weights, decay))

    def _compute_direction_weights(self, angles, max_weight=10.0):
        raw = np.cos(angles) ** 6 + 0.1
        scaled = raw * (max_weight / np.max(raw))
        return scaled / np.sum(scaled)

    def _handle_success(self):
        self.get_logger().info(f'{self.robot_name}: Goal reached')
        self._end_episode(succeed=True)
        self._call_service(self.task_succeed_client)

    def _handle_collision(self):
        self.get_logger().info(f'{self.robot_name}: Collision')
        self._end_episode(fail=True)
        self._call_service(self.task_failed_client)

    def _handle_timeout(self):
        self.get_logger().info(f'{self.robot_name}: Timeout')
        self._end_episode(fail=True)
        self._call_service(self.task_failed_client)

    def _end_episode(self, succeed=False, fail=False):
        self.get_logger().warn('Episode ended')
        self.succeed = succeed
        self.fail = fail
        self.done = True
        self.local_step = 0
        self._stop_robot()

        succeed_msg = Bool()
        succeed_msg.data = succeed
        self.succeed_pub.publish(succeed_msg)

    def _reset_episode_flags(self):
        self.done = False
        self.succeed = False
        self.fail = False

    def _publish_velocity(self, action):
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = LINEAR_VELOCITY
            msg.angular.z = ANGULAR_VELOCITIES[action]
        else:
            msg = TwistStamped()
            msg.twist.linear.x = LINEAR_VELOCITY
            msg.twist.angular.z = ANGULAR_VELOCITIES[action]
        self.cmd_vel_pub.publish(msg)

    def _stop_robot(self):
        msg = Twist() if ROS_DISTRO == 'humble' else TwistStamped()
        self.cmd_vel_pub.publish(msg)

    def _reset_stop_timer(self):
        if self.stop_timer:
            self.destroy_timer(self.stop_timer)
        self.stop_timer = self.create_timer(0.8, self._stop_timer_callback)

    def _stop_timer_callback(self):
        self._stop_robot()
        self.destroy_timer(self.stop_timer)

    def _call_service(self, client):
        if not client.service_is_ready():
            self._wait_for_service(client, client.srv_name)

        try:
            response = client.call(Goal.Request())
            if response:
                self.goal_pose = (response.pose_x, response.pose_y)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def _wait_for_service(self, client, name):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Waiting for {name} service...')

    @staticmethod
    def _sanitize_distance(dist):
        if dist == float('Inf'):
            return MAX_LIDAR_RANGE
        if np.isnan(dist):
            return 0.0
        return dist

    @staticmethod
    def _is_front_angle(angle):
        return (0 <= angle <= math.pi / 2) or (3 * math.pi / 2 <= angle <= 2 * math.pi)

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def _euler_from_quaternion(q):
        sinr = 2 * (q.w * q.x + q.y * q.z)
        cosr = 1 - 2 * (q.x * q.x + q.y * q.y)
        roll = np.arctan2(sinr, cosr)

        pitch = np.arcsin(2 * (q.w * q.y - q.z * q.x))

        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny, cosy)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    robot_name = sys.argv[1] if len(sys.argv) > 1 else 'robot1'

    node = MultiRobotRLEnvironment(robot_name)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
