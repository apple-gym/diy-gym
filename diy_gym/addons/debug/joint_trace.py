import pybullet as p

from gym import spaces
import pybullet_planning as pbp
import numpy as np
from diy_gym.addons.addon import Addon


class JointTrace(Addon):
    """
    JointTrace

    Trace the follows a joints movements
    """
    def __init__(self, parent, config):
        super().__init__(parent, config)

        self.uid = parent.uid
        joint_info = [p.getJointInfo(self.uid, i) for i in range(p.getNumJoints(self.uid))]

        if 'joint' in config:
            joints = [config.get('joint')]
        elif 'joints' in config:
            joints = config.get('joints')
        else:
            joints = [info[1].decode('UTF-8') for info in joint_info]

        self.joint_ids = [info[0] for info in joint_info if info[1].decode('UTF-8') in joints and info[3] > -1]
        self.last = None

    def reset(self):        
        p.removeAllUserDebugItems()
        self.last = None

    def update(self, action):
        # A colored trace for each joint
        joint_pos = np.array([pbp.get_link_pose(self.uid, i)[0] for i in self.joint_ids])
        if self.last is not None:
            m = len(joint_pos)
            for i in range(n):
                p.addUserDebugLine(
                            lineFromXYZ=joint_pos[i], 
                            lineToXYZ=self.last[i], lineColorRGB=[(n-i)/(n+1), 0.9, i/(n+1)], lineWidth=1, lifeTime=360)
        self.last = joint_pos
