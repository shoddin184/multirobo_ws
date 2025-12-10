#!/usr/bin/env python3
"""Multi-Robot RL Environment - Corrected for Deadlocks and Reset Issues."""

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
from turtlebot3_msgs.srv import Dqn, Goal

# Constants
ROS_DISTRO = os.environ.get('ROS_DISTRO')
GAZEBO_NS_MAP = {'robot1': 'TB3_1', 'robot2': 'TB3_2', 'robot3': 'TB3_3'}

GOAL_THRESHOLD = 0.20
COLLISION_THRESHOLD = 0.15
OBSTACLE_DETECT_RANGE = 0.5
MAX_LIDAR_RANGE = 3.5
LINEAR_VELOCITY = 0.2
ANGULAR_VELOCITIES = [1.5, 0.75, 0.0, -0.75, -1.5]


class MultiRobotRLEnvironment(Node):
    """RL Environment node for a single robot in multi-robot setup."""

    def __init__(self, robot_name='robot1'):
        super().__init__(f'rl_environment_{robot_name}')
        self.robot_name = robot_name
        self.gazebo_ns = GAZEBO_NS_MAP.get(robot_name, robot_name)

        # [FIX] ReentrantCallbackGroupを作成
        # これにより、コールバック実行中でも別のスレッドが割り込んで処理可能になる
        self.cb_group = ReentrantCallbackGroup()

        self._init_state_variables()
        self._init_publishers_subscribers()
        self._init_service_clients()
        self._init_services()

        self.get_logger().info(f'{robot_name} environment initialized (Multi-Threaded)')

    def _init_state_variables(self):
        """Initialize state tracking variables."""
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
        
        # [NEW] リセット中フラグ - リセット中は衝突判定をスキップ
        self.is_resetting = False

    def _init_publishers_subscribers(self):
        """Initialize ROS publishers and subscribers."""
        qos = QoSProfile(depth=10)
        cmd_vel_type = Twist if ROS_DISTRO == 'humble' else TwistStamped

        self.cmd_vel_pub = self.create_publisher(
            cmd_vel_type, f'/{self.gazebo_ns}/cmd_vel', qos)
        
        # Topic購読は非同期で軽いのでデフォルトのままでも良いが、
        # 念のためReentrantに入れても害はない（今回は指定なし＝デフォルト）
        self.create_subscription(
            Odometry, f'/{self.gazebo_ns}/odom', self._odom_callback, qos)
        self.create_subscription(
            LaserScan, f'/{self.gazebo_ns}/scan', self._scan_callback,
            qos_profile_sensor_data)

    def _init_service_clients(self):
        """Initialize service clients."""
        ns = self.robot_name

        # [FIX] クライアントも cb_group (Reentrant) に所属させる
        self.task_succeed_client = self.create_client(
            Goal, f'/{ns}/task_succeed', callback_group=self.cb_group)
        self.task_failed_client = self.create_client(
            Goal, f'/{ns}/task_failed', callback_group=self.cb_group)
        self.init_env_client = self.create_client(
            Goal, f'/{ns}/initialize_env', callback_group=self.cb_group)

    def _init_services(self):
        """Initialize ROS services."""
        ns = self.robot_name
        
        # [FIX] ★ここが最重要修正★
        # サービスサーバー自体を ReentrantCallbackGroup に所属させる。
        # これにより、_agent_callback 実行中でも、クライアントからの返答(別スレッド)を受け入れ可能になる。
        self.create_service(Dqn, f'/{ns}/rl_agent_interface', self._agent_callback, callback_group=self.cb_group)
        self.create_service(Empty, f'/{ns}/make_environment', self._make_env_callback, callback_group=self.cb_group)
        self.create_service(Dqn, f'/{ns}/reset_environment', self._reset_env_callback, callback_group=self.cb_group)
        self.create_service(Dqn, f'/{ns}/get_state', self._get_state_callback, callback_group=self.cb_group)

    # --- Callbacks ---

    def _odom_callback(self, msg):
        """Update robot pose and calculate goal metrics."""
        pos = msg.pose.pose.position
        _, _, theta = self._euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_pose = (pos.x, pos.y, theta)

        gx, gy = self.goal_pose
        rx, ry, rt = self.robot_pose

        self.goal_distance = math.hypot(gx - rx, gy - ry)
        angle = math.atan2(gy - ry, gx - rx) - rt
        self.goal_angle = self._normalize_angle(angle)

    def _scan_callback(self, scan):
        """Process LiDAR scan data."""
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
        """Handle RL agent action request."""
        # 1. Action
        self._publish_velocity(request.action)
        self._reset_stop_timer()

        # 2. State Calculation & Termination Check
        # 内部で衝突判定 -> _call_service が呼ばれる。
        # Reentrant設定のおかげで、ここでブロックしてもデッドロックしない。
        response.state = self._calculate_state()
        
        # 3. Reward & Done
        response.reward = self._calculate_reward()
        response.done = self.done  # ここには正しいTrue/Falseが入る

        # 4. Reset Flags if done
        if self.done:
            self._reset_episode_flags()

        # 5. Return Response
        # デッドロックが解消されたので、必ずここまで到達して送信される
        return response

    def _make_env_callback(self, request, response):
        """Initialize environment with goal position."""
        self._wait_for_service(self.init_env_client, 'initialize_env')
        
        # [FIX] call_async + wait ではなく、同期 call() を使用
        req = Goal.Request()
        result = self.init_env_client.call(req)

        if result and result.success:
            self.goal_pose = (result.pose_x, result.pose_y)
        return response

    # [FIX] ★★★ 完全に書き直した _reset_env_callback ★★★
    def _reset_env_callback(self, request, response):
        """Reset environment and return initial state."""
        self.get_logger().info(f'{self.robot_name}: Resetting environment...')
        
        # 1. リセット中フラグを立てる（衝突判定をスキップするため）
        self.is_resetting = True
        
        # 2. エピソードフラグをリセット
        self._reset_episode_flags()
        
        # 3. ステップカウンタをリセット
        self.local_step = 0
        
        # 4. ロボットを停止
        self._stop_robot()
        
        # 5. 新しいゴール位置を取得（環境初期化サービスを呼ぶ）
        #    これにより Stage Node がロボットをテレポートする
        try:
            self._call_service(self.init_env_client)
            self.get_logger().info(f'{self.robot_name}: New goal position: {self.goal_pose}')
        except Exception as e:
            self.get_logger().error(f'{self.robot_name}: Failed to initialize env: {e}')
        
        # 6. センサーデータが更新されるまで待機
        #    ロボットがテレポートされた後、LiDARデータが更新されるのを待つ
        time.sleep(0.5)
        
        # 7. 障害物距離を安全な値にリセット（古いcollision状態をクリア）
        self.min_obstacle_distance = MAX_LIDAR_RANGE
        
        # 8. 状態を取得（衝突判定なしで _build_state_vector を使用）
        state = self._build_state_vector()
        
        # 9. 距離を初期化
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        
        # 10. リセット中フラグを解除
        self.is_resetting = False
        
        response.state = state
        self.get_logger().info(f'{self.robot_name}: Environment reset complete. '
                               f'min_obstacle_distance={self.min_obstacle_distance:.2f}')
        return response

    def _get_state_callback(self, request, response):
        """Get current state without action."""
        response.state = self._build_state_vector()
        response.reward = 0.0
        response.done = False
        return response

    # --- Core Logic ---

    # [FIX] _calculate_state にリセット中チェックを追加
    def _calculate_state(self):
        """Calculate state and check termination conditions."""
        state = self._build_state_vector()
        self.local_step += 1

        # リセット中は衝突判定をスキップ
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
        """Build state vector from current observations."""
        state = [float(self.goal_distance), float(self.goal_angle)]

        if not self.front_ranges:
            state.extend([MAX_LIDAR_RANGE] * 180)
        else:
            state.extend(float(r) for r in self.front_ranges)

        return state

    def _calculate_reward(self):
        """Calculate reward for current state."""
        if self.succeed:
            return 100.0
        if self.fail:
            return -50.0

        yaw_reward = 1 - (2 * abs(self.goal_angle) / math.pi)
        obstacle_reward = self._compute_obstacle_reward()
        return yaw_reward + obstacle_reward

    def _compute_obstacle_reward(self):
        """Compute weighted obstacle avoidance reward."""
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
        """Compute directional weights for obstacle avoidance."""
        raw = np.cos(angles) ** 6 + 0.1
        scaled = raw * (max_weight / np.max(raw))
        return scaled / np.sum(scaled)

    # --- Episode Handling ---

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
        """End current episode."""
        # デバッグログ
        self.get_logger().warn('def _end_episode started!') 
        self.succeed = succeed
        self.fail = fail
        self.done = True
        self.local_step = 0
        self._stop_robot()

    def _reset_episode_flags(self):
        """Reset episode state flags."""
        self.done = False
        self.succeed = False
        self.fail = False

    # --- Utility Methods ---

    def _publish_velocity(self, action):
        """Publish velocity command."""
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
        """Stop robot movement."""
        msg = Twist() if ROS_DISTRO == 'humble' else TwistStamped()
        self.cmd_vel_pub.publish(msg)

    def _reset_stop_timer(self):
        """Reset velocity stop timer."""
        if self.stop_timer:
            self.destroy_timer(self.stop_timer)
        self.stop_timer = self.create_timer(0.8, self._stop_timer_callback)

    def _stop_timer_callback(self):
        self._stop_robot()
        self.destroy_timer(self.stop_timer)

    def _call_service(self, client):
        """Call a Goal service synchronously."""
        # サービスが見つかるまで待機
        if not client.service_is_ready():
            self._wait_for_service(client, client.srv_name)

        req = Goal.Request()

        try:
            response = client.call(req)
            if response:
                self.goal_pose = (response.pose_x, response.pose_y)
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def _wait_for_service(self, client, name):
        """Wait for service to be available."""
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Waiting for {name} service...')

    @staticmethod
    def _sanitize_distance(dist):
        """Sanitize LiDAR distance value."""
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

    # [FIX] MultiThreadedExecutor を使用
    # num_threadsはPCのスペックに合わせて調整（デフォルトはCPUコア数）
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