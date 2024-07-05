import mujoco
import numpy as np
from .utils import quat2rpy
from tqdm import tqdm

XML_PATH = '../model/xarm6_scene.xml'
SITE_ID = 'attachment_site'

CONSTRAINT = np.array([[0, 0, 0, 1, 1, 0]]) 

class Robot:
    
    def __init__(self):
        
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
    def compute_jacobian(self, joint_pos):
        
        jac = np.zeros((6, ))
        data.qpos = joint_pos
        mujoco.mj_forward(self.model, self.data)
        
    def compute_constraint(self, joint_pos):
        
        if joint_pos.ndim > 1:
            c = np.zeros((joint_pos.shape[1], 1))
            td = tqdm(range(c.shape[0]), colour='blue', desc='Computing constraints...')
            for i in td:
                self.data.qpos = joint_pos[0, i, :].reshape((6,))
                mujoco.mj_forward(self.model, self.data)
                state = self.compute_state(self.data)
                c[i, :] = self.constraint_function(state)
            return c
        else:    
            self.data.qpos = joint_pos
            mujoco.mj_forward(self.model, self.data)
            state = self.compute_state(self.data)
            return self.constraint_function(state)
        
    def compute_state(self, data) -> np.ndarray:
        ef_pos = self.data.site(SITE_ID).xpos.reshape((1,3))
        ef_rot = self.data.site(SITE_ID).xmat
        
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, ef_rot)
        ef_euler = quat2rpy(site_quat)
        
        ef_state = np.concatenate((ef_pos, ef_euler), axis=1)
        return ef_state

    def constraint_function(self, state) -> int:
        vec = np.zeros((1,6))
        np.multiply(CONSTRAINT, state, out=vec)
        f = np.linalg.norm(vec)
        return f
        