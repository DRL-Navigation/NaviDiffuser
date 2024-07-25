# NaviDiffuser

This is the open-source repo for our thesis:

**<center>NaviDiffuser: Tackling Multi-Objective Robot Navigation by Diffusion Model Decision-Making</center>**            
<center>Xuyang Zhang, Ziyang Feng, Quecheng Qiu, Jie Peng, Haoyu Li, and Jianmin Ji</center>

## Overview

The data-driven paradigm has recently shown great potential in solving many planning tasks. In the robot navigation realm, it sparked a new trend. People believe powerful data-driven methods can learn efficient and general navigation policies. 

However, robot navigation tasks differ from common planning tasks and present unique challenges. They often involve multi-objective optimization and need to meet complex and ever-changing human preferences in real-world applications. 

Furthermore, effective navigation requires a durable action sequence output to overcome short-sightedness and a high planning frequency to respond to environment changes. Both are challenging for data-driven methods. 

In this work, we integrate the diffusion model into robot navigation to address these challenges. Our proposed approach, NaviDiffuser, utilizes Classifier Guidance Diffusion Model for multi-objective solving and Transformer backbone for long-horizon planning. It also includes distillation skills to achieve high planning frequency and output quality. 

We have conducted experiments in both simulated and real-world scenarios to evaluate our approach. The results indicate that NaviDiffuser can produce diverse navigation policies that align with human preferences when maintaining a high arrival rate.

![real-world example](./doc/real_world.jpeg)