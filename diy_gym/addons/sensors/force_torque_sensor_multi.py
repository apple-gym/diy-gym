import numpy as np
import pybullet as p
from gym import spaces
from diy_gym.addons.addon import Addon


class ForceTorqueSensorMulti(Addon):
    """
    Senses Force and Torque on multiple frames in a body.
    """
    def __init__(self, parent, config):
        super().__init__(parent, config)
        
        # (optional) filter for child frames
        if 'frame_filter' in config:
            self.frame_ids = parent.find_frames_ids(config.get('frame_filter'))
        else:
            self.frame_ids = list(range(p.getNumJoints(parent.uid)))
        self.uid = parent.uid

        for frame_id in self.frame_ids:
            p.enableJointForceTorqueSensor(self.uid, frame_id, enableSensor=True)

        num_joints = len(self.frame_ids)
        self.observation_space = spaces.Dict({
            'force': spaces.Box(-10, 10, shape=(num_joints, 3, ), dtype='float32'),
            'torque': spaces.Box(-10, 10, shape=(num_joints, 3, ), dtype='float32'),
        })

    def observe(self):
        states = p.getJointStates(self.uid, self.frame_ids)
        reaction_forces = np.array([s[2] for s in states])
        return {'forces': reaction_forces[:, :3], 'torques': reaction_forces[:, 3:]}
