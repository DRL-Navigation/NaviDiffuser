# 跑[BarnData benchmark](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9292572)

cfg配置文件`drlnav_env/envs/cfg/gazebo_cfg`

wrapper类 `drlnav_env/envs/wrapper/evaluation_wrapper/BarnDataSetWrapper.py:~BarnDataSetWrapper`

robot_model:
`drlnav_env/src/robots_model/README.MD`,根据readme编译对应的robot仿真模型，随后source



## 使用手册
#### 测试Online PPO 模型
需要用drlnav_frame, 注意
- 注意确保`nn/nav_encoder.py`里1d激光输出维度和输入的激光对应上
如果打开gazebo后不动，看`out/pre.log`,大概率是这前向维度不对。
- 注意ppo训练环境的.yaml和XXbarndataset.yaml 里面大部分参数一直，比如laser_norm, laser_max, time_max, control_hz等。
```
turtlebot3 改激光参数：drlnav_env/src/robots_model/turtlebot3_simulations/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro  
jackal_clearpath 改激光参数：drlnav_env/src/robots_model/jackal_simulations/jackal/jackal_description/urdf/accessories/hokuyo_ust10.urdf.xacro
```
1.
`agent/agent.py` 将
`class Agents(Process)`改成`class Agents(Thread):`
2.
在`drlnav_frame/sh/machines/all.sh`将 ENV_FILES 改成如下
```
ENV_FILES="
gazebo/barndataset.sh
"
```

3.
```
cd drlnav_frame/USTC_lab/env/drlnav_env/
source devel/setup.bash
cd sh/
bash start.sh
```

#### 测试DWA
```
cd testing/BarnData_Benchmark
python run_dwa.py
```

### 测试其他Learning模型





