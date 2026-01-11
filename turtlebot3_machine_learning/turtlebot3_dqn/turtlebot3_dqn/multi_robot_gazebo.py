#!/usr/bin/env python3
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
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
    from gazebo_msgs.msg import EntityState
    from geometry_msgs.msg import Pose

GAZEBO_ROBOT_MAP = {'robot1': 'burger_1', 'robot2': 'burger_2', 'robot3': 'burger_3'}

INITIAL_POSITIONS = {
    'default': {
        'robot1': {'x': 1.0, 'y': 0.0},
        'robot2': {'x': -0.5, 'y': 1.0},
        'robot3': {'x': -0.5, 'y': -1.0},
    },
    'stage4': {
        'robot1': {'x': 1.0, 'y': 1.0},
        'robot2': {'x': -1.0, 'y': 1.0},
        'robot3': {'x': 0.0, 'y': -1.0},
    },
}

STAGE4_GOAL_POSITIONS = [
    [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 1.5], [0.5, 2.0], [-1.5, 2.1],
    [-2.0, 0.5], [-2.0, -0.5], [-1.5, -2.0], [-0.5, -1.0], [2.0, -0.5], [-1.0, -1.0],
]


class MultiRobotGazeboInterface(Node):

    def __init__(self, stage_num, robot_name='robot1'):
        super().__init__(f'gazebo_interface_{robot_name}')
        self.stage = int(stage_num)
        self.robot_name = robot_name
        self.gazebo_robot_name = GAZEBO_ROBOT_MAP.get(robot_name, robot_name)
        self.entity_name = f'goal_box_{robot_name}'
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        self._reset_lock = threading.Lock()
        self._reset_complete = threading.Event()

        if ROS_DISTRO == 'humble':
            self.entity = self._load_goal_entity()
            self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
            self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
            self.set_entity_state_client = self.create_client(SetEntityState, 'set_entity_state')
            self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self._init_services()

        self.get_logger().info(f'{robot_name} Gazebo interface initialized (stage {stage_num})')

    def _load_goal_entity(self):
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(
            package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf')
        with open(model_path, 'r') as f:
            return f.read()

    def _init_services(self):
        self.create_service(
            Goal, f'/{self.robot_name}/initialize_env',
            self._initialize_env_callback, callback_group=self.callback_group)
        self.create_service(
            Goal, f'/{self.robot_name}/task_succeed',
            self._task_succeed_callback, callback_group=self.callback_group)
        self.create_service(
            Goal, f'/{self.robot_name}/task_failed',
            self._task_failed_callback, callback_group=self.callback_group)

    def _initialize_env_callback(self, request, response):
        self.get_logger().info(f'{self.robot_name}: Initialize environment')
        self._delete_entity()
        time.sleep(0.2)
        self._reset_robot_position()
        time.sleep(0.2)
        self._generate_goal_pose()
        self.get_logger().info(
            f'{self.robot_name}: Goal at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self._spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def _task_succeed_callback(self, request, response):
        self.get_logger().info(f'{self.robot_name}: Task succeeded')
        self._delete_entity()
        time.sleep(0.2)
        self._generate_goal_pose()
        self.get_logger().info(
            f'{self.robot_name}: New goal at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self._spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def _task_failed_callback(self, request, response):
        self.get_logger().info(f'{self.robot_name}: Task failed')
        self._delete_entity()
        time.sleep(0.2)
        self._reset_robot_position()
        time.sleep(0.2)
        self._generate_goal_pose()
        self.get_logger().info(
            f'{self.robot_name}: New goal at ({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')
        time.sleep(0.2)
        self._spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def _spawn_entity(self):
        if ROS_DISTRO == 'humble':
            self._spawn_entity_humble()
        else:
            self._spawn_entity_gazebo()

    def _spawn_entity_humble(self):
        entity_pose = Pose()
        entity_pose.position.x = self.entity_pose_x
        entity_pose.position.y = self.entity_pose_y

        req = SpawnEntity.Request()
        req.name = self.entity_name
        req.xml = self.entity
        req.initial_pose = entity_pose

        self._wait_for_service(self.spawn_entity_client)
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(
            f'{self.robot_name}: Spawned goal at '
            f'({self.entity_pose_x:.2f}, {self.entity_pose_y:.2f})')

    def _spawn_entity_gazebo(self):
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(
            package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf')
        req = (
            f'sdf_filename: "{model_path}", '
            f'name: "{self.entity_name}", '
            f'pose: {{ position: {{ x: {self.entity_pose_x}, y: {self.entity_pose_y}, z: 0.0 }} }}'
        )
        self._run_gz_service('/world/dqn/create', 'gz.msgs.EntityFactory', req)

    def _delete_entity(self):
        if ROS_DISTRO == 'humble':
            self._delete_entity_humble()
        else:
            self._delete_entity_gazebo()

    def _delete_entity_humble(self):
        req = DeleteEntity.Request()
        req.name = self.entity_name
        self._wait_for_service(self.delete_entity_client)
        future = self.delete_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info(f'{self.robot_name}: Deleted goal')

    def _delete_entity_gazebo(self):
        req = f'name: "{self.entity_name}", type: 2'
        self._run_gz_service('/world/dqn/remove', 'gz.msgs.Entity', req)

    def _reset_robot_position(self):
        if ROS_DISTRO == 'humble':
            self._reset_robot_position_humble()
        else:
            self._reset_robot_position_gazebo()

    def _reset_robot_position_humble(self):
        self._reset_complete.clear()
        reset_thread = threading.Thread(target=self._reset_robot_position_sync)
        reset_thread.start()

        if not self._reset_complete.wait(timeout=5.0):
            self.get_logger().warn(f'{self.robot_name}: Reset timed out')
        else:
            self.get_logger().info(f'{self.robot_name}: Reset completed')

    def _reset_robot_position_sync(self):
        with self._reset_lock:
            try:
                pos = self._get_initial_position()
                self.get_logger().info(f'{self.robot_name}: Target position: ({pos["x"]}, {pos["y"]})')

                if not self.set_entity_state_client.service_is_ready():
                    if not self.set_entity_state_client.wait_for_service(timeout_sec=3.0):
                        self.get_logger().error(f'{self.robot_name}: Service not available')
                        return

                req = SetEntityState.Request()
                state = EntityState()
                state.name = self.gazebo_robot_name
                state.pose.position.x = pos['x']
                state.pose.position.y = pos['y']
                state.pose.position.z = 0.01
                state.pose.orientation.w = 1.0
                req.state = state

                future = self.set_entity_state_client.call_async(req)

                start_time = time.time()
                while not future.done():
                    if time.time() - start_time > 3.0:
                        self.get_logger().warn(f'{self.robot_name}: Service call timed out')
                        break
                    time.sleep(0.05)

                if future.done():
                    result = future.result()
                    self.get_logger().info(
                        f'{self.robot_name}: Reset to ({pos["x"]}, {pos["y"]}) - Success: {result.success}')
            except Exception as e:
                self.get_logger().error(f'{self.robot_name}: Reset failed: {e}')
            finally:
                self._reset_complete.set()

    def _reset_robot_position_gazebo(self):
        req_delete = f'name: "{self.gazebo_robot_name}", type: 2'
        self._run_gz_service('/world/dqn/remove', 'gz.msgs.Entity', req_delete)
        time.sleep(0.2)

        pos = self._get_initial_position()
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(package_share, 'models', 'turtlebot3_burger', 'model.sdf')
        req_spawn = (
            f'sdf_filename: "{model_path}", '
            f'name: "{self.gazebo_robot_name}", '
            f'pose: {{ position: {{ x: {pos["x"]}, y: {pos["y"]}, z: 0.0 }} }}'
        )
        self._run_gz_service('/world/dqn/create', 'gz.msgs.EntityFactory', req_spawn)
        self.get_logger().info(f'{self.robot_name}: Respawned at ({pos["x"]}, {pos["y"]})')

    def _get_initial_position(self):
        positions = INITIAL_POSITIONS['stage4'] if self.stage == 4 else INITIAL_POSITIONS['default']
        return positions.get(self.robot_name, {'x': 0.0, 'y': 0.0})

    def _generate_goal_pose(self):
        if self.stage != 4:
            self.entity_pose_x = random.randrange(-21, 21) / 10
            self.entity_pose_y = random.randrange(-21, 21) / 10
        else:
            goal = random.choice(STAGE4_GOAL_POSITIONS)
            self.entity_pose_x = goal[0]
            self.entity_pose_y = goal[1]

    def _wait_for_service(self, client):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Waiting for {client.srv_name} service...')

    def _run_gz_service(self, service_name, req_type, req):
        cmd = [
            'gz', 'service', '-s', service_name,
            '--reqtype', req_type, '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', req,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass


def main(args=None):
    rclpy.init(args=sys.argv)
    stage_num = sys.argv[1] if len(sys.argv) > 1 else '1'
    robot_name = sys.argv[2] if len(sys.argv) > 2 else 'robot1'

    if len(sys.argv) < 3:
        print(f'WARNING: robot_name not specified, defaulting to "{robot_name}"')
        print(f'Usage: ros2 run turtlebot3_dqn multi_robot_gazebo <stage_num> <robot_name>')

    node = MultiRobotGazeboInterface(stage_num, robot_name)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
