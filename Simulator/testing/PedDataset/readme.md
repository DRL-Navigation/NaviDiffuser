# 跑Ped Dataset

## 基本说明

### 原理

将真实的行人数据集放到2d仿真器中进行测试，行人数据集包括每一帧包括世界坐标系下：<x,y,yaw,vx,vy>

#### cfg配置文件

`drlnav_env/envs/cfg/ped_dataset_cfg`

#### wrapper类 

`drlnav_env/envs/wrapper/evaluation_wrapper/PedTrajectoryDatasetWrapper.py.py:~PedTrajectoryDatasetWrapper.py`

## 使用手册
#### 测试Online PPO 模型
**需要用drlnav_frame, 注意**

- 注意确保`nn/nav_encoder.py`里1d激光输出维度和输入的激光对应上
  如果打开gazebo后不动，看`out/pre.log`,大概率是这前向维度不对。

- 注意ppo训练环境的.yaml和`eth.yaml` 等里面大部分参数一直，比如laser_norm, laser_max, time_max, control_hz等。

- `config/base_config.py`把`LOAD_CHECKPOINT`改成True，`LOAD_CHECKPOINT_PATH`改成训练模型绝对路径

  

#### 1.
在`drlnav_frame/sh/machines/all.sh`将 ENV_FILES 改成如下

```
ENV_FILES="
machines/ped_dataset_test/eth-test.sh
"
```

#### 2.

```
source drlnav_frame/USTC_lab/env/drlnav_env/devel/setup.bash
cd sh/
bash start.sh
```

#### 测试DWA
```

```

### 测试其他Learning模型





