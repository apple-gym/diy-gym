"""
some wrapper around pybullet to make it easier
"""
import pandas as pd
import pybullet as p
import pybullet_planning as pbp
import numpy as np
from collections import namedtuple

    
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

ContactResult = namedtuple('ContactResult', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB',
                                         'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                         'contactNormalOnB', 'contactDistance', 'normalForce',
                                             'lateralFriction1', 'lateralFrictionDir1', 'lateralFriction2',
                                             'lateralFrictionDir2'
                                            ])

def get_contact_points(*args, **kwargs):
    cs = p.getContactPoints(*args, **kwargs)
    cs = [ContactResult(*c) for c in cs] 
    return cs

def normal_force_between_bodies(body_id, other_id):
    force = 0
    for other_link_id in pbp.get_links(other_id):
        for body_link_id in pbp.get_links(body_id):
            cs = get_contact_points(body_id, body_link_id, other_id, other_link_id)
            force += sum([c.normalForce for c in cs])
    return force

def quaternion_multiply(q1, q0):
    """Return multiplication of two quaternions."""
    return np.array((q1[0] * q0[3] + q1[1] * q0[2] - q1[2] * q0[1] + q1[3] * q0[0], -q1[0] * q0[2] + q1[1] * q0[3] +
                        q1[2] * q0[0] + q1[3] * q0[1], q1[0] * q0[1] - q1[1] * q0[0] + q1[2] * q0[3] + q1[3] * q0[2],
                        -q1[0] * q0[0] - q1[1] * q0[1] - q1[2] * q0[2] + q1[3] * q0[3]),
                    dtype=np.float64)
