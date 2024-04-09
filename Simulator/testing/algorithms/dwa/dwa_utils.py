import matplotlib.pyplot as plt
import tf
import math
import numpy as np
import copy

import sys
from geometry_msgs.msg import Pose, Quaternion, Point

from testing.algorithms.dwa import RobotProp


def q_to_rpy(q):
    if type(q) == list or type(q) == tuple:
        quaternion = q
    elif type(q) == Pose:
        quaternion = (q.orientation.x, q.orientation.y,
                      q.orientation.z, q.orientation.w)
    elif type(q) == Quaternion:
        quaternion = (q.x, q.y, q.z, q.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    roll = euler[0]
    pitch = euler[1]
    yaw = euler[2]
    return (roll, pitch, yaw)

def rpy_to_q(rpy):
    return tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

def matrix_from_t_q(t, q):
    return tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(t),\
            tf.transformations.quaternion_matrix(q))

def matrix_from_t_y(t, y):
    q = rpy_to_q([0, 0, y])
    return matrix_from_t_q(t, q)

def matrix_from_t_m(t, m):
    return np.array([[m[0][0], m[0][1], m[0][2], t[0]],
                     [m[1][0], m[1][1], m[1][2], t[1]],
                     [m[2][0], m[2][1], m[2][2], t[2]],
                     [0,       0,       0,       1]
                    ])

def matrix_from_pose(pose):
    if type(pose) == list or type(pose) == tuple:
        return matrix_from_t_q(pose[:3], pose[3:])
    elif type(pose) == Pose:
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        position = (pose.position.x, pose.position.y, pose.position.z)
        return matrix_from_t_q(position, quaternion)

def q_from_matrix(m):
    return tf.transformations.quaternion_from_matrix(m)

def t_from_matrix(m):
    return tf.transformations.translation_from_matrix(m)

def rpy_from_matrix(m):
    return tf.transformations.euler_from_matrix(m)

def inverse(m):
    return tf.transformations.inverse_matrix(m)

def transform_point(m, point):
    xyz = tuple(np.dot(m, np.array([point[0], point[1], point[2], 1.0])))[:3]
    return xyz

def mul_matrix(m1, m2):
    return np.dot(m1, m2)

show_animation = True  # 动画




def scan_to_obs(scan, min_angle, max_angle, max_range, filter_scale = 10):
    """
    激光转障碍物list
    :param scan:
    :param min_angle:
    :param max_angle:
    :param max_range:
    """
    # print(scan, max_range, flush=True)
    obs_list = []
    range_total = scan.size // filter_scale
    assert range_total > 0
    angle_step = (max_angle - min_angle) / range_total
    cur_angle = min_angle
    for i in range(range_total):
        j = i * filter_scale
        if scan[j] <= max_range:
            x = scan[j] * math.cos(cur_angle)
            y = scan[j] * math.sin(cur_angle)
            obs_list.append([x,y])
        cur_angle += angle_step
    return np.array(obs_list)


def calc_dynamic_window(x, robot_prop):
    """
    计算动态窗口
    :param: 机器人坐标、机器人参数
    :return: 动态窗口：最小速度，最大速度，最小角速度 最大角速度速度
    """
    # 车子速度的最大最小范围依次为：最小速度 最大速度 最小角速度 最大角速度速度
    vs = [robot_prop.minV, robot_prop.maxV, robot_prop.minW, robot_prop.maxW]
    # vs = [-robot_prop.maxV, robot_prop.maxV, -robot_prop.maxW, robot_prop.maxW]

    # 根据当前速度以及加速度限制计算的动态窗口  依次为：最小速度 最大速度 最小角速度 最大角速度速度
    vd = [x[3] - robot_prop.accV * robot_prop.dt,
          x[3] + robot_prop.accV * robot_prop.dt,
          x[4] - robot_prop.accW * robot_prop.dt,
          x[4] + robot_prop.accW * robot_prop.dt]
    # 最终的Dynamic Window
    vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
          max(vs[2], vd[2]), min(vs[3], vd[3])]

    return vr


def motion(x, u, robot_prop):
    """
    Motion Model 根据当前状态推算下一个控制周期（dt）的状态
    :param x: 机器人当前位置
    :param u: 当前速度
    :param robot_prop: 参数
    :return: 机器人下一刻的位置
    """
    dt = robot_prop.dt
    y = np.zeros_like(x)

    y[0] = x[0] + u[0] * math.cos(x[2]) * dt  # x方向位移
    y[1] = x[1] + u[0] * math.sin(x[2]) * dt  # y
    y[2] = x[2] + u[1] * dt  # 航向角
    y[3] = u[0]  # 速度v
    y[4] = u[1]  # 角速度w
    return y


def calc_traj(x, v, w, robot_prop, goal):
    """
    轨迹生成
    :param: 机器人坐标，速度，角速度，机器人参数
    :return: 机器人在T时间内的轨迹
    """
    time = 0
    tmp_index = 0
    trajs = [x]
    tmp = calc_to_goal_cost(trajs, goal)
    i = 1
    while time <= robot_prop.T:
        time += robot_prop.dt
        x = motion(x, [v, w], robot_prop)
        trajs.append(x)
        # cost = calc_to_goal_cost(trajs, goal)
        # if cost < tmp:
        #     tmp = cost
        #     tmp_index = i
        i += 1
    return trajs[-1], np.array(trajs)
    # return trajs[tmp_index], np.array(trajs[:tmp_index+1])


def calc_heading(x, goal):
    """
    航向参数得分  当前车的航向和相对于目标点的航向 偏离程度越小 分数越高
    :param x:
    :param goal:
    :return:
    """
    if x[0]**2+x[1]**2 >= 0:
        goal = list(goal)
        t = list(rpy_to_q([0,0,x[2]]))
        T = inverse(matrix_from_pose([x[0],x[1],0] + t))
        tmp = transform_point(T, goal+[0])
        heading_tmp = math.atan2(tmp[1], tmp[0])
        # if heading_tmp < 0:
        #     heading_tmp += math.pi

        return abs(heading_tmp)
    else:
        cos_theta = (x[0]*goal[0]+x[1]*goal[1]) /  ( math.sqrt(x[0]**2+x[1]**2) * math.sqrt(goal[0]**2 + goal[1]**2) )
    # goal_theta = math.atan2(goal[1] - x[1], goal[0] - x[0])
    # heading = abs(x[2] -  goal_theta)
        return math.acos(cos_theta)

def calc_to_goal_cost(traj, goal, heading=0):
    # print("goal cost", math.sqrt((traj[-1][0]-goal[0])**2+(traj[-1][1]-goal[1])**2), heading)
    cost = math.sqrt((traj[-1][0]-goal[0])**2+(traj[-1][1]-goal[1])**2) + heading
    return cost

def calc_score_v_w(vt, wt, robot_prop):
    """
    :param vt:
    :param wt:
    :param robot_prop:
    :return:
    """
    # 计算线速度得分
    # vi = robot_prop.kv * robot_prop.maxV * math.cos(alpha) * math.tanh(ro / robot_prop.kro)
    # score_v = 1 - abs(vt - vi) / (2 * robot_prop.maxV)
    #
    # # 计算角速度得分
    # wi = robot_prop.kalpha * alpha + vi * math.sin(alpha) / ro
    # score_w = 1 - abs(wt - wi) / (2 * robot_prop.maxW)
    cost = abs(abs(vt) - robot_prop.maxV)
    return cost


def calc_score_dis2obs(traj, obs, inf_r, robot_prop):
    """
    障碍物距离评价函数  （机器人在当前轨迹上与最近的障碍物之间的距离，如果没有障碍物则设定一个常数）
    :param traj:
    :param obs:
    :param inf_r： 膨胀半径
    :return:
    """
    dis2obs = 1000.0
    #print "obs:",obs
    # 提取轨迹上的机器人x y坐标
    robotx = traj[1:, 0:2]
    if len(obs)!=0:
        for it in range(0, len(robotx[:, 1])):
            # print("robot :",robotx[it])
            for io in range(0, len(obs[:, 0])):
                # print("obs: ", obs[io])
                dx = obs[io, 0] - robotx[it, 0]
                dy = obs[io, 1] - robotx[it, 1]
                disttemp = math.sqrt(dx ** 2 + dy ** 2) - robot_prop.radius

                if disttemp < dis2obs:
                    dis2obs = disttemp

    dis2obs = max(dis2obs, 0.00001)
    # print( "mindist", dis2obs)
    cost = round(1.0 / dis2obs, 2)
    return cost


def calc_breaking_dist(vt, robot_prop):
    """
    :param vt:
    :param robot_prop:
    :return:
    """
    stopdist = vt ** 2 / (2 * robot_prop.accV) + robot_prop.radius
    return stopdist


def evaluation(x, vr, goal, obs, inf_r, robot_prop):
    """
    评价函数 内部负责产生可用轨迹
    :param x:
    :param vr:
    :param goal:
    :param obs:
    :param inf_r:
    :param robot_prop:
    :return:
    """
    # robot_score = np.array([0, 0, 0, 0, 0])
    robot_score = []
    robot_trajectory = []
    # print("+++++++++++++++")
    for vt in np.arange(vr[0], vr[1]+0.001, robot_prop.resolV):
        vt = round(vt, 2)
        for wt in np.arange(vr[2], vr[3]+0.001, robot_prop.resolW):
            # 计算机器人的轨迹
            wt = round(wt, 2)
            xt, traj = calc_traj(copy.deepcopy(x), vt, wt, robot_prop, goal)
            # print(traj, flush=True)
            if len(traj) == 1:
                continue
            # 机器人线速度及角速度惩罚
            cost_v = calc_score_v_w(vt, wt, robot_prop)

            #机器人与目标点距离惩罚
            heading =  calc_heading(traj[-1]-traj[-2], goal)
            # print("heading",heading,flush=True)
            # print(vt, wt, traj)
            # print("he", heading)
            cost_dis2goal = calc_to_goal_cost(traj, goal, 0)

            # 机器人障碍物得分
            cost_dis2obs = calc_score_dis2obs(traj, obs, inf_r, robot_prop)
            # print(vt, wt, cost_v, cost_dis2goal, cost_dis2obs, heading, flush=True)
            if cost_dis2obs < 100000:
                #print "vt=", round(vt,3), "wt=", round(wt,3), "cost_v=", round(cost_v,3), "cost_dis2goal=", round(cost_dis2goal,3), "cost_dis2obs", round(cost_dis2obs,3), "total=", round(cost_v+cost_dis2goal+cost_dis2obs,3)
                robot_score.append([vt, wt, cost_v, cost_dis2goal, cost_dis2obs])
                # robot_score = np.vstack((robot_score, [vt, wt, score_v, score_w, score_dis2obs]))
                robot_trajectory.append(np.transpose(traj))

    robot_score = np.array(robot_score)
    return robot_score, robot_trajectory


def normalization(score):
    """
    归一化处理
    :param score:
    :return:
    """
    if sum(score[:, 2]) != 0:
        score[:, 2] = score[:, 2] / sum(score[:, 2])

    if sum(score[:, 3]) != 0:
        score[:, 3] = score[:, 3] / sum(score[:, 3])

    if sum(score[:, 4]) != 0:
        score[:, 4] = score[:, 4] / sum(score[:, 4])

    return score


def dwa_control(x, goal, obs, inf_r, robot_prop):
    """
    DWA算法实现
    :param x:
    :param goal:
    :param obs:
    :param inf_r:
    :param robot_prop:
    :return:
    """
    score_list = []
    # Dynamic Window: Vr = [vmin, vmax, wmin, wmax] 最小速度 最大速度 最小角速度 最大角速度速度
    # 根据当前状态 和 运动模型 计算当前的参数允许范围
    vr = calc_dynamic_window(x, robot_prop)
    robot_score, robot_trajectory = evaluation(x, vr, goal, obs, inf_r, robot_prop)
    #print "robot_score:", robot_score
    if len(robot_score) == 0:
        print('no path to goal')
        u = np.transpose([0, 0])
        return u, False
    else:
        #score = normalization(robot_score)
        score = robot_score

        for ii in range(0, len(score[:, 0])):
            weights = np.mat([robot_prop.weightV, robot_prop.weightG, robot_prop.weightObs])
            scoretemp = weights * (np.mat(score[ii, 2:5])).T
            score_list.append(scoretemp)

        max_score_id = np.argmin(score_list)
        u = score[max_score_id, 0:2]
        # print(u, flush=True)
        trajectory = robot_trajectory[int(max_score_id)]
        trajectory = np.array(trajectory)
        trajectory = np.transpose(trajectory)
    # print(u)
    return u, trajectory


def choose_dwa_action(lasers, state, vw, min_angle=-1.570795, max_angle=1.570795, max_range=1.0, inf_r=0.17):
    """
    DWA　image_cpp_env调用接口
    :param lasers:
    :param min_angle:
    :param max_angle:
    :param max_range:
    :param state:
    :param vw:
    :param inf_r:膨胀半径
    :return:
    """
    is_no_way = False
    robot_prop = RobotProp()
    # scan = lasers[0,:,-1]
    obs = scan_to_obs(lasers, min_angle, max_angle, max_range)
    goal = np.array([state[0], state[1]])
    # goal = np.array([state[0], state[1]])

    x = np.array([0.0, 0.0, 0.0, vw[0], vw[1]])
    u, traj = dwa_control(x, goal, obs, inf_r, robot_prop)
    # print "goal:", goal
    #print "traj:", traj
    #print "obs:", obs
    # print("state",state, vw)
    # print("out", u)
    if type(traj) == bool:
        is_no_way = True
    return u, is_no_way

def plot_arrow(a, x, y, yaw, length=0.5, width=0.1):
    a.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=1.5 * width, head_width=width)
    a.plot(x, y)


def plot_circle(a, obs, inf_r, robot_r):
    for i in range(len(obs[:, 0])):
        theta = np.arange(0, 2*np.pi, 0.01)
        x = obs[i, 0] + inf_r * np.cos(theta)
        y = obs[i, 1] + inf_r * np.sin(theta)
        a.plot(x, y, color = 'red')
        x_robot = obs[i, 0] + (inf_r + robot_r) * np.cos(theta)
        y_robot = obs[i, 1] + (inf_r + robot_r) * np.sin(theta)
        a.plot(x_robot, y_robot, color = 'red', linestyle = '--')


def draw_dynamic_search(a, root, canvas, best_trajectory, x, goal, ob, inf_r, robot_r, mapx, mapy):
    # 设置图形尺寸与质量
    a.cla()  # 清除上次绘制图像
    a.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
    a.plot(x[0], x[1], "xr")
    a.plot(0, 0, "og")
    a.plot(goal[0], goal[1], "ro")
    a.plot(ob[:, 0], ob[:, 1], "bs")
    plot_arrow(a, x[0], x[1], x[2])
    plot_circle(a, ob, inf_r, robot_r)
    a.axis('equal')
    plt.xlim(mapx)
    plt.ylim(mapy)
    plt.grid(True)  # 添加网格

    canvas.draw()
    root.update()
    # time.sleep(0.05)  # 让程序休息二十分之一秒（0.05秒），然后再继续


def draw_path(trajectory, goal, ob, x, inf_r, robot_r):
    # 创建图形
    f = plt.figure(2, figsize=(4, 4), dpi=100)
    a = f.add_subplot(111)

    a.plot(x[0], x[1], "xr")
    a.plot(0, 0, "og")
    a.plot(goal[0], goal[1], "ro")
    a.plot(ob[:, 0], ob[:, 1], "bs")
    plot_arrow(a, x[0], x[1], x[2])
    plot_circle(a, ob, inf_r, robot_r)
    a.axis("equal")
    plt.grid(True)
    a.plot(trajectory[:, 0], trajectory[:, 1], 'g')
    plt.show()

