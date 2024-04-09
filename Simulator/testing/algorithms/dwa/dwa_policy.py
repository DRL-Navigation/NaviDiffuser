import random

from testing.algorithms.dwa.dwa_utils import choose_dwa_action


class DWAPolicy4Nav:
    def __init__(self, n, cfg):
        self.n = n
        self.vw = [(0, 0)] * n
        self.config_env = cfg
        self.laser_norm = cfg.get('laser_norm', True)
        self.laser_max = cfg['laser_max']

    def reverse_laser(self, laser):
        if self.laser_norm:
            return laser * self.laser_max
        else:
            return laser

    def gen_action(self, state):
        out = []

        for i in range(self.n):
            laser_scan = self.reverse_laser(state[0][i][0])
            vw  = choose_dwa_action(laser_scan, state[1][i], self.vw[i], self.config_env['view_angle_begin'],
                              self.config_env['view_angle_end'], self.config_env['laser_max'])[0]
            # print(vw, flush=True)
            self.vw[i] = tuple(vw[:2])
            out.append( (vw[0], vw[1], 0 ) )

        # print(out)
        return out