"""
some wrapper around pybullet to make it easier
"""
import pandas as pd
import pybullet as p

def find_all_frames(uid, key):
    joint_info = [p.getJointInfo(uid, i) for i in range(p.getNumJoints(uid))]
    return [j[0] for j in joint_info if key in j[1].decode()]
    
    
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

def getDynamicsInfo(*args):
    r = p.getDynamicsInfo(*args)
    k = ['mass', 'lateral_friction', 'local_inertia_diagonal', 
         'inertial_position', 'inertial_orientation', 'restitution', 
         'rolling_friction', 'spinning_friction',
         'contact_damping', 'contact_stiffness']
    return dict(zip(k, r))

def getLinkState(*args):
    r = p.getLinkState(self.uid, self.end_effector_joint_id)
    raise Exception('not implemented')
    k = ['mass', 'lateral_friction', 'local_inertia_diagonal', 
         'inertial_position', 'inertial_orientation', 'restitution', 
         'rolling_friction', 'spinning_friction',
         'contact_damping', 'contact_stiffness']
    return dict(zip(k, r))

def getBasePositionAndOrientation(*args):
    target_xyz, target_rpy = p.getBasePositionAndOrientation(i)
    return dict(xyz=target_xyz, rpy=target_rpy)
