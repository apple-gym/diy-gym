import pybullet as p

from gym import spaces
import pybullet_planning as pbp
import numpy as np
from diy_gym.addons.addon import Addon


class JointGainController(Addon):
    """
    JointController with Gain

    if gain <0, it's automatic (scaled based on distance to target)
    if gain >0 it's controlled by agent

    Desired position or velocity or torque
    """
    def __init__(self, parent, config):
        super(JointGainController, self).__init__(parent, config)

        self.uid = parent.uid

        self.position_gain = config.get('position_gain', 0.015)
        self.velocity_gain = config.get('velocity_gain', 1.0)
        self.scaling = config.get('action_scaling', 1.0)

        self.control_mode = {
            'position': p.POSITION_CONTROL,
            'velocity': p.VELOCITY_CONTROL,
            'torque': p.TORQUE_CONTROL
        }[config.get('control_mode', 'velocity')]

        joint_info = [p.getJointInfo(self.uid, i) for i in range(p.getNumJoints(self.uid))]

        if 'joint' in config:
            joints = [config.get('joint')]
        elif 'joints' in config:
            joints = config.get('joints')
        else:
            joints = [info[1].decode('UTF-8') for info in joint_info]

        self.joint_ids = [info[0] for info in joint_info if info[1].decode('UTF-8') in joints and info[3] > -1]
        self.rest_position = config.get('rest_position', [0] * len(self.joint_ids))

        self.torque_limit = np.array([p.getJointInfo(self.uid, joint_id)[10] for joint_id in self.joint_ids])
        self.vel_limit = np.array([p.getJointInfo(self.uid, joint_id)[11] for joint_id in self.joint_ids])

        if self.control_mode == p.TORQUE_CONTROL:
            tl = np.array(list(self.torque_limit)+[1])
            self.action_space = spaces.Box(-tl, tl, dtype='float32')
        elif self.control_mode == p.VELOCITY_CONTROL:
            self.action_space = spaces.Box(-self.vel_limit/self.scaling, self.vel_limit/self.scaling, dtype='float32')
        else:
            low = np.array([p.getJointInfo(self.uid, joint_id)[8] for joint_id in self.joint_ids]+[-6,])/self.scaling
            high = np.array([p.getJointInfo(self.uid, joint_id)[9] for joint_id in self.joint_ids]+[6,])/self.scaling
            self.action_space = spaces.Box(low, high, shape=(len(low), ), dtype='float32')
        self.torque_limit

        self.random_reset = config.get('reset_range', [0.] * len(self.joint_ids))

    def reset(self):        
        random_delta = np.random.random(len(self.joint_ids)) * self.random_reset
        for joint_id, angle, d_angle in zip(self.joint_ids, self.rest_position, random_delta):
            p.resetJointState(self.uid, joint_id, angle + d_angle)

    def update(self, action):
        
        action = tuple(a * self.scaling for a in action)
        gain = np.abs(action[-1])/6
        action = action[:-1]

        pGain = self.position_gain
        vGain = self.velocity_gain

        kwargs = {}

        if self.control_mode == p.POSITION_CONTROL:
            # To be able to move close, we need a tighter fit (higher gain) as we get closer
            if gain <= 0:
                joint_state = pbp.get_joint_positions(self.uid, self.joint_ids)
                # distance in joint space, per joint
                dist = np.abs(np.array(action)-joint_state)/np.abs(self.action_space.high-self.action_space.low)[:-1]
                pGain = pGain / (dist + 1e-2) 
                # but 1<vgain>pGain for stability
                vGain = vGain + pGain ** 2
            else:
                joint_state = pbp.get_joint_positions(self.uid, self.joint_ids)
                dist = np.abs(np.array(action)-joint_state)/np.abs(self.action_space.high-self.action_space.low)[:-1]
                pGain = pGain * gain / (dist + 1e-2)
                vGain = vGain + pGain ** 2

            kwargs['targetPositions'] = action
            kwargs['targetVelocities'] = [0.0] * len(action)
            kwargs['forces'] = self.torque_limit
        elif self.control_mode == p.VELOCITY_CONTROL:
            kwargs['targetVelocities'] = action
            kwargs['forces'] = self.torque_limit
            pGain = [pGain] * len(action)
            vGain = [vGain] * len(action)
        else:
            # To be able to move close, we need a tighter fit (higher gain) as we get closer
            if gain > 0:
                pGain = pGain * gain
                vGain = vGain + pGain ** 2
            kwargs['forces'] = action
            pGain = [pGain] * len(action)
            vGain = [vGain] * len(action)

        p.setJointMotorControlArray(self.uid,
                                    self.joint_ids,
                                    self.control_mode,
                                    positionGains=pGain,
                                    velocityGains=vGain,
                                    **kwargs)
