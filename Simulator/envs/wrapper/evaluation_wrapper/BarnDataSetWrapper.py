import gym, numpy, os
import xml.etree.ElementTree as ET
from envs.utils import EnvPos


class BarnDataSetWrapper(gym.Wrapper):
    """

    for one robot
    """
    def __init__(self, env, cfg):
        super(BarnDataSetWrapper, self).__init__(env)
        self.cfg = cfg
        self.repeated_time_per_env = cfg.get("repeated_time_per_env", 2)
        self.max_worlds = cfg.get("max_worlds", 300)
        self.file_path = cfg.get("file_path", '../../../src/gazebo_env/worlds/BARN/world_{}.world')
        self.file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.file_path)
        self.map_scale = cfg['global_map']['resolution'] / 0.1
        self.cur_world = 0
        self.change_world()
        self.cur_repeated_time_per_env -= 1

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)

    def reset(self, **kwargs):
        print("BarnDataSetWrapper Reset", flush=True)
        if self.cur_world >= self.max_worlds:
            print("[BarnDataSetWrapper]: Run Over.", flush=True)
        else:
            if self.cur_repeated_time_per_env >= self.repeated_time_per_env:
                self.change_world()
            else:
                self.cur_repeated_time_per_env += 1
        return self.env.reset(**kwargs)

    def change_world(self):
        print("[BarnDataSetWrapper]: cur_world", self.cur_world, flush=True)
        self.world_to_cfg()
        self.map_scale_up()
        self.env.env_pose = EnvPos(self.env.cfg)
        self.cur_world += 1
        self.cur_repeated_time_per_env = 1

    def world_to_cfg(self):
        tree = ET.parse(self.file_path.format(self.cur_world))
        root = tree.getroot()
        
        obstacles_info = {
            "total": 0,
            "shape": [],
            "size_range": [],
            "poses_type": [],
            "poses": []
        }

        for model in root.findall(".//model"):
            name = model.get('name')
            if name and name.startswith('unit_cylinder'):
                pose = model.find('pose').text.split()
                position = list(map(float, pose[:2]))
                geometry = model.find(".//link/collision/geometry/cylinder")
                if geometry is not None:
                    radius = float(geometry.find('radius').text)
                    obstacles_info["total"] += 1
                    obstacles_info["shape"].append("circle")
                    obstacles_info["size_range"].append([radius, radius])
                    obstacles_info["poses_type"].append("fix")
                    obstacles_info["poses"].append(position)

        poses = numpy.array(obstacles_info["poses"])
        poses_x_min = numpy.min(poses[:,0])
        poses_y_min = numpy.min(poses[:,1])
        for i in range(len(poses)):
            poses[i][0], poses[i][1] = poses[i][0]-poses_x_min+0.5, poses[i][1]-poses_y_min+0.5
        obstacles_info["poses"] = poses.tolist()

        self.env.cfg['object'] = obstacles_info
    
    def map_scale_up(self):
        self.env.cfg['object']['size_range'] = (numpy.array(self.env.cfg['object']['size_range']) * self.map_scale).tolist()
        self.env.cfg['object']['poses'] = (numpy.array(self.env.cfg['object']['poses']) * self.map_scale).tolist()