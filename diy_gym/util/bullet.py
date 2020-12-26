"""
some wrapper around pybullet to make it easier
"""
import pandas as pd
import pybullet as p
import numpy as np
from collections import namedtuple

# TODO just use https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/utils.py
from pybullet_planning.utils import get_joint_info

# def find_all_frames(uid, key):
#     joint_info = [p.getJointInfo(uid, i) for i in range(p.getNumJoints(uid))]
#     return [j[0] for j in joint_info if key in j[1].decode()]
    
# # FIXME replace with ppu.joint_from_name()
# def get_frame_id(uid, frame):
#     frames = [p.getJointInfo(uid, i)[1].decode('utf-8') for i in range(p.getNumJoints(uid))]
#     return frames.index(frame) if frame in frames else - 1
    
def joint_dataframe(uid):
    # see https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c#L5077
    joint_info = [p.getJointInfo(uid, i) for i in range(p.getNumJoints(uid))]
    df = pd.DataFrame(joint_info, columns=['index', 'name', 'type', 'qIndex', 'uIndex', 'flags', 'damping', 'friction', 'p_lower_limit', 'p_upper_limit', 'torque_limit', 'velocity', 'link', 'axis', 'pos', 'orn', 'parentIndex'])
    
    # rename enums
    jointTypeList = dict(enumerate(["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]))
    df['type'] = df['type'].replace(jointTypeList)
    
    # decode strings    
    df['name'] = df['name'].str.decode('utf8')
    df['link'] = df['link'].str.decode('utf8')
    return df

# CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
#                                                        'geometry_type', 'dimensions', 'filename',
#                                                        'local_frame_pos', 'local_frame_orn'])

# DynamicsInfo = namedtuple('DynamicsInfo',  ['mass', 'lateral_friction', 'local_inertia_diagonal', 
#          'inertial_position', 'inertial_orientation', 'restitution', 
#          'rolling_friction', 'spinning_friction',
#          'contact_damping', 'contact_stiffness'])

# def get_dynamics_info(*args, **kwargs):
#     r = p.getDynamicsInfo(*args, **kwargs)
#     r = [np.array(rr) for rr in r]
#     return  DynamicsInfo(*r)


# LinkState = namedtuple('LinkState',  ['linkWorldPosition', 'linkWorldOrientation', 'localInertialFramePosition', 
#          'localInertialFrameOrientation', 'worldLinkFramePosition', 'worldLinkFrameOrientation', 
#          'worldLinkLinearVelocity', 'worldLinkAngularVelocity'])

# def getLinkState(*args, **kwargs):
#     r = p.getLinkState(*args, **kwargs)
#     r = [np.array(rr) for rr in r]
#     if len(r) == 6:
#         r += [[0, 0,0 ], [0,0,0]]
#     return LinkState(*r)

# PositionAndOrientation = namedtuple('PositionAndOrientation',  ['xyz', 'rpy'])

# def getBasePositionAndOrientation(*args, **kwargs):
#     target_xyz, target_rpy = p.getBasePositionAndOrientation(*args, **kwargs)
#     return PositionAndOrientation(xyz=target_xyz, rpy=target_rpy)

def quaternion_multiply(q1, q0):
    """Return multiplication of two quaternions."""
    return np.array((q1[0] * q0[3] + q1[1] * q0[2] - q1[2] * q0[1] + q1[3] * q0[0], -q1[0] * q0[2] + q1[1] * q0[3] +
                        q1[2] * q0[0] + q1[3] * q0[1], q1[0] * q0[1] - q1[1] * q0[0] + q1[2] * q0[3] + q1[3] * q0[2],
                        -q1[0] * q0[0] - q1[1] * q0[1] - q1[2] * q0[2] + q1[3] * q0[3]),
                    dtype=np.float64)
