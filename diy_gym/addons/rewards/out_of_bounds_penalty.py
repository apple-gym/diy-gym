import numpy as np
import pybullet as p
from diy_gym.addons.addon import Addon


class OutOfBoundsPenalty(Addon):
    def __init__(self, parent, config):
        super(OutOfBoundsPenalty, self).__init__(parent, config)

        self.source_model = parent#.models[config.get('source_model')]
        self.min_xyz = config.get('min_xyz', [0., 0., 0.])
        self.max_xyz = config.get('max_xyz', [0., 0., 0.])

        self.source_frame_id = self.source_model.get_frame_id(
            config.get('source_frame')) if 'source_frame' in config else -1

        self.tolerance = config.get('tolerance', 0.05)
        self.penalty = config.get('penalty', -1)

    def outside(self):
        """Is it out of bounds."""
        source_xyz = p.getLinkState(
            self.source_model.uid,
            self.source_frame_id)[4] if self.source_frame_id >= 0 else p.getBasePositionAndOrientation(
                self.source_model.uid)[0]
        source_xyz = np.array(source_xyz)
        return (source_xyz<self.min_xyz).any() or (source_xyz>self.max_xyz).any()

    def is_terminal(self):
        if self.outside() > self.tolerance:
            return self.penalty
        else:
            return 0
