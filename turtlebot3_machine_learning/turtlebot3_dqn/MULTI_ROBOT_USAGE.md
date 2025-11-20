# マルチロボットDQNシステムの使用方法

## 概要

このシステムでは、3台のロボットが独立したDQNエージェントを持ち、それぞれが独自のゴールに向かって学習を行います。すべてのロボットがゴールに到達または衝突したらエピソードが終了します。

## ファイル構成

### エージェントファイル
- `dqn_agent1.py` - Robot1用のDQNエージェント
- `dqn_agent2.py` - Robot2用のDQNエージェント
- `dqn_agent3.py` - Robot3用のDQNエージェント

### 環境ファイル
- `multi_robot_environment.py` - 各ロボットの環境管理（センサー、オドメトリ、報酬計算）
- `multi_robot_gazebo.py` - 各ロボットのゴール生成・管理

### 中央制御
- `multi_robot_coordinator.py` - 全ロボットの状態を監視し、エピソード終了を管理

## 起動方法

### 1. ビルド

```bash
cd ~/01_research/multirobo_ws
colcon build --packages-select turtlebot3_dqn
source install/setup.bash
```

### 2. Gazeboシミュレーションの起動

まず、マルチロボット用のGazebo環境を起動します：

```bash
# 別のターミナルで
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo multi_robot.launch.py
```

### 3. 各ノードの起動

#### ターミナル1: Coordinator
```bash
ros2 run turtlebot3_dqn multi_robot_coordinator 3
```
- 引数: ロボット数（デフォルト: 3）

#### ターミナル2-4: Robot1
```bash
# Gazeboインターフェース
ros2 run turtlebot3_dqn multi_robot_gazebo 1 robot1

# 環境ノード
ros2 run turtlebot3_dqn multi_robot_environment robot1

# DQNエージェント
ros2 run turtlebot3_dqn dqn_agent1 1 1000
```
- 第1引数: ステージ番号
- 第2引数: 最大エピソード数

#### ターミナル5-7: Robot2
```bash
# Gazeboインターフェース
ros2 run turtlebot3_dqn multi_robot_gazebo 1 robot2

# 環境ノード
ros2 run turtlebot3_dqn multi_robot_environment robot2

# DQNエージェント
ros2 run turtlebot3_dqn dqn_agent2 1 1000
```

#### ターミナル8-10: Robot3
```bash
# Gazeboインターフェース
ros2 run turtlebot3_dqn multi_robot_gazebo 1 robot3

# 環境ノード
ros2 run turtlebot3_dqn multi_robot_environment robot3

# DQNエージェント
ros2 run turtlebot3_dqn dqn_agent3 1 1000
```

## システムアーキテクチャ

```
┌─────────────────────────────────┐
│  Multi-Robot Coordinator        │
│  - 全ロボットの状態監視         │
│  - エピソード終了判定           │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┬────────────┐
    │                 │            │
┌───▼─────┐    ┌─────▼──┐   ┌────▼─────┐
│ Robot1  │    │ Robot2 │   │ Robot3   │
├─────────┤    ├────────┤   ├──────────┤
│ Agent1  │    │ Agent2 │   │ Agent3   │
│ Env1    │    │ Env2   │   │ Env3     │
│ Gazebo1 │    │ Gazebo2│   │ Gazebo3  │
└─────────┘    └────────┘   └──────────┘
```

## トピック構成

### Robot1
- `/robot1/status` - ステータス情報（done, succeeded, failed）
- `/robot1/cmd_vel` - 速度指令
- `/robot1/odom` - オドメトリ
- `/robot1/scan` - LiDARデータ
- `/robot1/get_action` - 選択されたアクション
- `/robot1/result` - エピソード結果

### Robot2, Robot3も同様

### グローバル
- `/reset_episode` - エピソードリセット信号

## サービス構成

各ロボットは以下のサービスを持ちます：

- `/{robot_name}/rl_agent_interface` - エージェントと環境のインターフェース
- `/{robot_name}/make_environment` - 環境初期化
- `/{robot_name}/reset_environment` - 環境リセット
- `/{robot_name}/initialize_env` - Gazebo環境初期化
- `/{robot_name}/task_succeed` - ゴール到達時の処理
- `/{robot_name}/task_failed` - 衝突時の処理

## モデル保存

各ロボットのモデルは独立して保存されます：

```
saved_model/
├── robot1/
│   ├── stage1_episode250.keras
│   └── stage1_episode250.json
├── robot2/
│   ├── stage1_episode250.keras
│   └── stage1_episode250.json
└── robot3/
    ├── stage1_episode250.keras
    └── stage1_episode250.json
```

## エピソード終了条件

すべてのロボットが以下のいずれかの状態になった時にエピソードが終了します：

1. **ゴール到達**: ゴールから0.2m以内に到達
2. **衝突**: 障害物から0.15m以内に接近
3. **タイムアウト**: 800ステップ経過

## ログとTensorBoard

各ロボットのログは以下に保存されます：

```bash
~/turtlebot3_dqn_logs/gradient_tape/
├── [時刻]_dqn_stage1_robot1_reward/
├── [時刻]_dqn_stage1_robot2_reward/
└── [時刻]_dqn_stage1_robot3_reward/
```

TensorBoardで確認：
```bash
tensorboard --logdir ~/turtlebot3_dqn_logs/gradient_tape/
```

## トラブルシューティング

### サービスが見つからない
```bash
ros2 service list | grep robot1
```

### トピックの確認
```bash
ros2 topic list | grep robot
ros2 topic echo /robot1/status
```

### ノードの確認
```bash
ros2 node list
```

## カスタマイズ

### ロボット数の変更

ロボット数を変更する場合は、以下のファイルを作成：
- `dqn_agent{N}.py` - 新しいロボット用エージェント
- `setup.py` に新しいエントリポイントを追加

### ハイパーパラメータの調整

各エージェントファイル内で調整可能：
- `learning_rate`: 学習率（デフォルト: 0.0007）
- `epsilon_decay`: ε減衰率（デフォルト: 6000 × stage）
- `batch_size`: バッチサイズ（デフォルト: 128）
- `discount_factor`: 割引率（デフォルト: 0.99）

### 初期位置の変更

[multi_robot_gazebo.py:199-205](multi_robot_gazebo.py#L199-L205) の `initial_positions` を編集してください。
