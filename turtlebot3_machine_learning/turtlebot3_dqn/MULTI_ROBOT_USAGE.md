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

**⚠️ 起動順序が重要です。以下の順番で起動してください。**

#### ターミナル1: 学習進捗モニタリング（result_graph）
```bash
ros2 run turtlebot3_dqn result_graph
```
- 3台のロボットの合計報酬とQ値をリアルタイムグラフ表示
- 100エピソードごとの平均をCSVファイルに自動保存
- CSVファイル名: `multi_robot_training_results_YYYYMMDDHHMMSS.csv`

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

#### ターミナル11: Coordinator（最後に起動）
```bash
ros2 run turtlebot3_dqn multi_robot_coordinator 3
```
- 引数: ロボット数（デフォルト: 3）
- **⚠️ 必ず全ロボットのノードが起動してから最後に起動すること**

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
- `/robot1/get_action` - 選択されたアクション（action, score, reward）
- `/robot1/result` - エピソード結果（total_score, avg_max_q）

### Robot2, Robot3も同様

### グローバル
- `/reset_episode` - エピソードリセット信号

### result_graph.pyがサブスクライブするトピック
- `/robot1/result` - Robot1のエピソード結果
- `/robot2/result` - Robot2のエピソード結果
- `/robot3/result` - Robot3のエピソード結果

**データフォーマット:**
```python
Float32MultiArray.data = [total_score, avg_max_q]
```
- `total_score`: エピソード累積報酬
- `avg_max_q`: 平均最大Q値

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

## ログとデータ保存

### TensorBoard

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

### 学習進捗データ（CSV）

`result_graph.py` が自動的にCSVファイルを生成します：

**ファイル名:** `multi_robot_training_results_YYYYMMDDHHMMSS.csv`

**保存場所:** `result_graph.py` を実行したディレクトリ

**フォーマット:**
```csv
Episode_Range,Avg_Total_Q_Value,Avg_Total_Reward
1-100,450.23,1250.50
101-200,520.67,1380.75
201-300,610.45,1520.30
...
```

**内容:**
- `Episode_Range`: エピソード範囲（100エピソードごと）
- `Avg_Total_Q_Value`: 3台のロボットのQ値合計の平均
- `Avg_Total_Reward`: 3台のロボットの報酬合計の平均

**データの意味:**
- 各値は3台のロボットの合計値
- 例: Avg_Total_Reward=1250.50 → Robot1(400) + Robot2(450) + Robot3(400.5) の100エピソード平均

## トラブルシューティング

### result_graph.pyがデータを受信しない

1. トピックが存在するか確認：
```bash
ros2 topic list | grep result
```

2. データが流れているか確認：
```bash
ros2 topic echo /robot1/result
```

3. ロボット名が正しいか確認：
```bash
# デフォルトはrobot1, robot2, robot3
ros2 run turtlebot3_dqn result_graph --robots robot1 robot2 robot3
```

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

### GUIウィンドウが表示されない

PyQt5がインストールされているか確認：
```bash
pip3 show PyQt5
pip3 show pyqtgraph
```

インストールされていない場合：
```bash
pip3 install PyQt5 pyqtgraph
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

[multi_robot_gazebo.py:201-214](multi_robot_gazebo.py#L201-L214) の `initial_positions` を編集してください。

**Stage 1-3の初期位置:**
```python
initial_positions = {
    'robot1': {'x': 2.0, 'y': 0.0},
    'robot2': {'x': -0.5, 'y': 2.0},
    'robot3': {'x': -0.5, 'y': -2.0}
}
```

**Stage 4の初期位置:**
```python
initial_positions = {
    'robot1': {'x': 2.0, 'y': 1.0},
    'robot2': {'x': -2.0, 'y': 1.0},
    'robot3': {'x': 0.0, 'y': -2.0}
}
```

### result_graph.pyのカスタマイズ

**エピソード集計間隔の変更:**

`result_graph.py` の84行目を編集：
```python
self.episode_interval = 100  # デフォルト: 100エピソードごと
```

**追跡するロボット数の変更:**
```bash
# 2台のみ
ros2 run turtlebot3_dqn result_graph --robots robot1 robot2

# 4台以上（要agent4.py等の実装）
ros2 run turtlebot3_dqn result_graph --robots robot1 robot2 robot3 robot4
```

## 実験ワークフロー

### 30,000エピソードの長期実験

```bash
# ステージ1、30,000エピソード
ros2 run turtlebot3_dqn dqn_agent1 1 30000
ros2 run turtlebot3_dqn dqn_agent2 1 30000
ros2 run turtlebot3_dqn dqn_agent3 1 30000
```

**予想実行時間:** 約3日（研究進捗.mdより）

**生成されるCSV:**
- 300行のデータ（30,000 ÷ 100）
- ファイルサイズ: 約10-15KB

**データ活用:**
1. CSVをExcel/Pythonで読み込み
2. エピソード範囲ごとの学習曲線をプロット
3. 報酬設計の比較分析（通常 vs 新設計）
