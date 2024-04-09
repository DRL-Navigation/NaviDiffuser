FROM pytorch/pytorch:latest

CMD bash

# setup timezone
RUN echo 'Asia/Shanghai' > /etc/timezone \
    && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# change to USTC apt mirror
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sed -i 's/http:/https:/g' /etc/apt/sources.list \
    && apt-get update && apt-get install -q -y --no-install-recommends curl sudo

# create user
RUN echo 'docker ALL=(ALL)ALL' >> /etc/sudoers \
    && addgroup --gid 1000 docker \
    && useradd  --uid 1000 --gid docker --home-dir /home/docker --create-home --shell /bin/sh --password $(openssl passwd -1 kai) --comment "" docker \
    && USER=docker \
    && GROUP=docker \
    && curl -SsL https://kgithub.com/boxboat/fixuid/releases/download/v0.5.1/fixuid-0.5.1-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - \
    && chown root:root /usr/local/bin/fixuid \
    && chmod 4755 /usr/local/bin/fixuid \
    && mkdir -p /etc/fixuid \
    && printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER docker:docker

# install base packages
RUN echo kai | sudo -S apt-get update \
    && sudo apt-get install -q -y --no-install-recommends \
    lsb-release \
    tzdata \
    vim \
    git \
    at \
    dirmngr \
    gnupg2 \
    openssl \
    openssh-server \
    && sudo rm -rf /var/lib/apt/lists/*

# install ros packages
RUN echo kai | sudo -S gpg --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && sudo gpg --export C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 | sudo tee /usr/share/keyrings/ros.gpg > /dev/null \
    && sudo echo "deb [signed-by=/usr/share/keyrings/ros.gpg] https://mirrors.ustc.edu.cn/ros/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros-latest.list \
    && sudo apt-get update \
    && sudo apt-get install -q -y --no-install-recommends \
    libedit-dev \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    ros-noetic-ros-base \
    ros-noetic-controller-manager \
    ros-noetic-interactive-marker-twist-server \
    ros-noetic-diff-drive-controller \
    ros-noetic-joint-state-controller \
    ros-noetic-robot-localization \
    ros-noetic-teleop-twist-joy \
    ros-noetic-stage \
    ros-noetic-nav-core \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins \
    ros-noetic-pcl-conversions \
    ros-noetic-joint-state-publisher \
    ros-noetic-rqt-console \
    ros-noetic-rqt-robot-monitor \
    ros-noetic-rviz \
    ros-noetic-pcl-ros \
    ros-noetic-rosdoc-lite \
    ros-noetic-flir-camera-description \
    ros-noetic-gmapping \
    ros-noetic-map-server \
    ros-noetic-move-base \
    && sudo rm -rf /var/lib/apt/lists/*

# install redis and mongodb
RUN echo kai | sudo -S apt-get update \
    && sudo curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg \
    && sudo echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list \
    && sudo curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor \
    && sudo echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list \
    && sudo apt-get update \
    && sudo apt-get install -q -y --no-install-recommends\
    mongodb-org \
    redis \
    && sudo rm -rf /var/lib/apt/lists/* 
    
# install python packages
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple pip -U \
    && pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple \
    && pip install numpy==1.21.3 catkin_pkg rospkg empy gym==0.25.2 opencv-python pycryptodomex gnupg imageio matplotlib pandas scipy tensorboard multiprocess pyglet redis pymongo transformers

# update bashrc
RUN echo kai | sudo -S echo 'source "/opt/ros/noetic/setup.bash"' >> ~/.bashrc \
    && sudo echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH' >> ~/.bashrc \
    && sudo echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5' >> ~/.bashrc \
    && sudo echo 'export NCCL_P2P_DISABLE=1' >> ~/.bashrc
ENTRYPOINT ["fixuid"]