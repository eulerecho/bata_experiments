import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True

class PretouchGripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        # placeholder
        self.object_sid = -2
        self.object_top_sid = -3
        self.target_sid = -1
        self.env_timestep = 0
        self.length = 1
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/pretouch_gripper.xml', 2)
        # self.init_qpos = np.array([0.08 ,0.1, 0.07, 1 , 0, 0, 0, -0.789906, -0.0177654, -0.0148775, 0.789957, 0.0178698, 0.0148437])
        self.init_qpos = np.array([-0.08 ,0.1, 0.07, 1 , 0, 0, 0, -0.789906, -0.0177654, -0.0148775, 0.789957, 0.0178698, 0.0148437])
        self.object_sid = self.model.site_name2id("object")
        self.target_sid = self.model.site_name2id("target")
        self.object_top_sid = self.model.site_name2id("object_top")
        self.length = np.linalg.norm(self.data.site_xpos[self.object_top_sid] -self.data.site_xpos[self.object_sid])
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):

        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a

        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()

        current_pos = self.data.site_xpos[self.object_sid]
        current_top_pos = self.data.site_xpos[self.object_top_sid]
        target_pos = self.data.site_xpos[self.target_sid]

        obj_orien = (current_top_pos-current_pos)/self.length
        target_orien = np.array([0,0,1]) #Unit vector along z axis

        # get to goal
        reward = -3*np.linalg.norm(current_pos - target_pos)

        orien_similarity = np.dot(obj_orien, target_orien)
        reward += 4*orien_similarity

        # velocity penalty
        reward -= 0.003*np.linalg.norm(self.data.qvel.ravel())

        # control penalty
        reward -= 0.05 * np.linalg.norm(a)

        # if ADD_BONUS_REWARDS:
        #     # bonus for same orientation
        #     # if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
        #     #     reward += 2
        #     # bonus for reaching the goal
        #     if (np.linalg.norm(current_pos - target_pos) < 0.020):
        #         reward += 25
        #     if (np.linalg.norm(current_pos - target_pos) < 0.010):
        #         reward += 75

        return ob, reward, False , self.get_env_infos()


    def get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0:3],
            self.sim.data.qpos.flat[7:13],
            self.sim.data.qvel.flat[6:12],
        ])

    # --------------------------------
    # resets and randomization
    # --------------------------------

    def robot_reset(self):
        self.set_state(self.init_qpos,self.init_qvel)

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        self.robot_reset()
        return self.get_obs()

    # --------------------------------
    # get and set states
    # --------------------------------

    def get_env_state(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),target_pos=target_pos, timestep=self.env_timestep)

    def set_env_state(self, state):
        self.sim.reset()
        qp = state['qp'].copy()
        qv = state['qv'].copy()
        target_pos = state['target_pos']
        self.env_timestep = state['timestep']
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.sim.forward()

    # --------------------------------
    # utility functions
    # --------------------------------

    def get_env_infos(self):
        return dict(state=self.get_env_state())

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.type = 1
        self.sim.forward()
        self.viewer.cam.distance = self.model.stat.extent * 2.0
