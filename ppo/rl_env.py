from f1tenth_gym.envs.track.utils import nearest_point_on_trajectory, find_track_dir
from scipy.interpolate import CubicSpline
# from f1tenth_gym.envsutils import find_track_dir
from f1tenth_gym.envs.rendering import make_renderer
from f1tenth_gym.envs import F110Env
import gymnasium as gym
import numpy as np


from typing import List
import os
import cv2
import time

class OpponentDriver:
    def __init__(self, **kwargs):
        """Wrapper class for opponent policies"""
        pass

    def drive(self, obs):
        """Drive the car: implemented in subclasses"""
        return np.zeros(2)

class F110Ego(gym.Wrapper):
    def __init__(self, env, opps: List[OpponentDriver] = None):
        """
        f1tenth env wrapper: action space only for ego,
        supports self-play against fixed
        """
        super().__init__(env)
        self.env = env
        self.ego_idx = env.unwrapped.ego_idx
        self.num_agents = env.unwrapped.num_agents

        self.action_space = env.unwrapped.action_type.space
        self.observation_type = env.unwrapped.observation_type
        self.observation_space = env.unwrapped.observation_space

        self.opp_idxs = [i for i in range(self.num_agents) if i != self.ego_idx]
        self.opps = opps if opps else [OpponentDriver()] * (self.num_agents - 1)

        # reward scaling
        self.r_max = self.env.unwrapped.config['params']['v_max'] * self.env.unwrapped.timestep # max distance that can be traveled in one step call

    def step(self, action: np.ndarray):
        """Steps using provided action + opponent policies"""
        actions = np.zeros((self.num_agents, 2))
        actions[self.ego_idx] = action

        opp_idx = 0
        obs = self.observation_type.observe()
        for i in self.opp_idxs:
            actions[i] = self.opps[opp_idx].drive(obs)

        return self.env.step(actions)

    def update_opponent(self, opp, idx):
        """Update opponent policy"""
        self.opps[idx] = opp

    def _get_reward(self):
        """
        Same logic as f110 applied to ego agent only
        Will definitely want to play with this later
        (ex: laptime reward upon loop completion, consecutive loop completion, etc)
        """
        if not hasattr(self, "last_s"):
            self.last_s = 0.0

        current_s, _ = (
            self.track.centerline.spline.calc_arclength_inaccurate(
                self.poses_x[self.ego_idx], self.poses_y[self.ego_idx]
            )
        )

        prog = current_s - self.last_s
        if prog > 0.9 * self.track.centerline.spline.s[-1]:
            prog = (self.track.centerline.spline.s[-1] - self.last_s[i]) + current_s

        self.last_s = current_s
        return prog / self.r_max if not self.collisions[self.ego_idx] else -1.0

class F110EnvDR(F110Env):
    def __init__(
        self,
        config: dict,
        render_mode: str = None,
        **kwargs         
    ):
        """
        F110Env with support for domain randomization

        Enabled by setting 'param': {'min': val, 'max': val} in config.yml
        instead of static values
        """
        self.config_input = config
        self.params_input = config['params']
        self.num_obstacles = config["num_obstacles"]
        
        if os.path.exists(config['map']) and os.path.isdir(config['map']):
            tracks = [d for d in os.listdir(config['map']) if os.path.isdir(os.path.join(config['map'], d))]
        else:
            tracks = []

        if len(tracks) > 0:
            self.use_trackgen = True
            self.tracks = tracks
        else:
            self.use_trackgen = False
            self.tracks = None

        config = self._sample_dict(self.config_input)
        config['params'] = self._sample_dict(self.params_input)
        super().__init__(config, render_mode, **kwargs)
        self.centerline = self._update_centerline(config['map'])
        raceline = self._update_raceline(config['map'])
        # self.vspline = self._get_velocity_spline(raceline)
        self.last_action = np.zeros((self.num_agents, 2))
        self.stag_count = 0 #np.zeros((self.num_agents,))
        self.total_prog = 0 #np.zeros((self.num_agents,))

        # print(self.action_space)

    def _get_velocity_spline(self, raceline_info):
        sv_values = []
        for i in range(raceline_info.shape[0]):
            x = raceline_info[i, 1]
            y = raceline_info[i, 2]
            sv_values.append([self.track.centerline.spline.calc_arclength_inaccurate(x, y)[0], 
                              raceline_info[i, 5]])
        sv_values = np.array(sv_values)
        # print(sv_values)
        sv_values = sv_values[sv_values[:, 0].argsort()]
        ## noticed that last value repeats
        sv_values = sv_values[:-1]
        return CubicSpline(sv_values[:,0], sv_values[:,1])

    def _sample_dict(self, params: dict):
        """Sample parameters for domain randomization"""
        pcopy = params.copy()
        for key, val in pcopy.items():
            if isinstance(val, dict) and 'min' in val and 'max' in val: # sample numeric
                pcopy[key] = np.random.uniform(val['min'], val['max'])
            elif self.use_trackgen and key == 'map': # sample track
                pcopy[key] = os.path.join(self.config_input['map'], np.random.choice(self.tracks))
        return pcopy

    def _update_raceline(self, track):
        """
        sets up [x, y, width_left, width_right] centerline attr for current track:
        used to ensure obstacles leave room for ego
        """
        track_dir = find_track_dir(track)
        centerline_file = os.path.join(track_dir, f"{track}_raceline.csv")
        return np.loadtxt(centerline_file, delimiter=';').astype(np.float32)
    
    def _update_centerline(self, track):
        """
        sets up [x, y, width_left, width_right] centerline attr for current track:
        used to ensure obstacles leave room for ego
        """
        track_dir = find_track_dir(track)
        centerline_file = os.path.join(track_dir, f"{track}_centerline.csv")
        return np.loadtxt(centerline_file, delimiter=',').astype(np.float32)

    def _update_map_from_track(self):
        self.sim.set_map(self.track)

    ## NOTE: a lot of these functions are implemented in a way that implicityl assumes 1 agent
    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)

        # observation
        obs = self.observation_type.observe()

        # times
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state()

        # rendering observation
        self.render_obs = {
            "ego_idx": self.sim.ego_idx,
            "poses_x": self.sim.agent_poses[:, 0],
            "poses_y": self.sim.agent_poses[:, 1],
            "poses_theta": self.sim.agent_poses[:, 2],
            "steering_angles": self.sim.agent_steerings,
            "lap_times": self.lap_times,
            "lap_counts": self.lap_counts,
            "collisions": self.sim.collisions,
            "sim_time": self.current_time,
        }

        # check done
        done, toggle_list = self._check_done()
        truncated = False
        info = {"checkpoint_done": toggle_list}

        # calc reward
        reward, timeout = self._get_reward(action)
        done = done or timeout
        info['timeout'] = int(done) # 0/1 for T/F
        info['total_progress'] = self.total_prog

        return obs, reward, done, truncated, info


    def _get_reward(self, action):
        """
        Get the reward for the current step
        action - np.array (num_agents, 2)
        """
        # reward for progress compared to last step - imported from base environment
        # penalty for each crashed agent - imported from base environment
        # penalty for norm of action differences from last time step - NEW
        # stagnation penalty - every timestep, add a penalty if we've moved below a certain amount - NEW
        # ^ for above, will also call an episode termination if we've incurred this penalty consecutively
        # for self.STAG_CUTOFF time steps
        # penalty for not tracking reference velocities - NEW #TODO: have not added this yet
        # Purely collaborative reward if more than 1 agent


        ACTION_CHANGE_PENALTY = -0.01
        STAGNATION_PENALTY = -1
        STAGNATION_CUTOFF = 0.02 # delta s as a fraction of total track length
        STAG_TIMEOUT = 20 # number of consecutive stag penalties required to trigger a timeout
        VELOCITY_PENALTY = -0.05

        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents

        reward = 0.0
        timeout = False
        for i in range(self.num_agents):
            current_s, _ = (
                self.track.centerline.spline.calc_arclength_inaccurate(
                    self.poses_x[i], self.poses_y[i]
                )
            )

            prog = current_s - self.last_s[i]
            if prog > 0.9 * self.track.centerline.spline.s[-1]:
                prog = (self.track.centerline.spline.s[-1] - self.last_s[i]) + current_s
            reward += prog
            self.total_prog += prog

            if (prog) / self.track.centerline.spline.s[-1] < STAGNATION_CUTOFF:
                reward += STAGNATION_PENALTY
                self.stag_count += 1
                if self.stag_count == STAG_TIMEOUT:
                    timeout = True
            else:
                self.stag_count = 0
            
            reward += ACTION_CHANGE_PENALTY * np.linalg.norm(action[i] - self.last_action[i], 2)
            
            if self.collisions[i]:
                reward -= 1.
            
            # v_ref = self.vspline(current_s)
            # reward += VELOCITY_PENALTY * abs(action[i, 1] - v_ref)

            

            self.last_s[i] = current_s
        return reward, timeout

    def _reset_pos(self, seed=None, options=None):
        '''
        Resets the pose (position and orientation) of the car. To be called in reset() and
        copied over from the base F110Env to handle the few cases where obstacles spawn on top 
        of the car due to the fact that super.reset() was previously being called AFTER we spawned
        obtacles
        '''
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be a numpy array of shape (num_agents, 3)"

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # call reset to simulator
        self.sim.reset(poses)

    def reset(self, seed=None, options=None):
        """resets agents, randomizes params"""
        if hasattr(self, 'config_input') and hasattr(self, 'params_input'):
            config = self._sample_dict(self.config_input)
            config['params'] = self._sample_dict(self.params_input)
            self.configure({'params': config['params']})

            for k, v in config.items():
                if k != 'params' and hasattr(self, k):
                    setattr(self, k, v)

        
        if self.use_trackgen:
            self.update_map(config['map'])
            self.centerline = self._update_centerline(config['map'])

        self._reset_pos(seed=seed, options=options)

        # regenerate the map to the original without obstacles anyways to ensure that obstacles don't clutter over time
        self.update_map(config['map'])
        for _ in range(self.num_obstacles):
            self._spawn_obstacle()
        self._update_map_from_track()
        # get no input observations
        self.last_action = np.zeros((self.num_agents, 2))
        self.total_prog = 0
        obs, _, _, _, info = self.step(self.last_action)

        ## updated to support changing maps, create new renederer with most up to date info
        # self.renderer, self.render_spec = make_renderer(
        #     params=self.params,
        #     track=self.track,
        #     agent_ids=self.agent_ids,
        #     render_mode=self.render_mode,
        #     render_fps=self.metadata["render_fps"],
        # )
        return obs, info

    def _spawn_obstacle(
        self, 
        room=10, 
        r_min=0.2,
        margin=0.4
    ):
        """
        spawns a random box on track room away from ego
        only draws circles for now, with low lidar resolution should be fine

        Args:
            room (int): minimum distance in indices from ego to spawn location 
            r_min (float): minimum obstacle size
            margin (float): how much track width to leave on either side of the circle
        """
        ego_x, ego_y = self.start_xs[self.ego_idx], self.start_ys[self.ego_idx] #, self.poses_yaw[self.ego_idx]
        pt = np.array([ego_x, ego_y])
        _, _, _, n_idx = nearest_point_on_trajectory(pt.astype(np.float64), self.centerline[:, :2].astype(np.float64))

        # deletes indices in B_r(pt) from selection pool
        # TODO: idk if these checks are necessary,
        # agent reset might account for updated occupancy map
        idxs = np.arange(self.centerline.shape[0])
        close = idxs.take(np.arange(n_idx - room, n_idx + room + 1), mode='wrap')
        idxs = np.delete(idxs, close)

        # randomly select (s, ey) from remaining indices
        xc, yc = self.centerline[np.random.choice(idxs),:2]
        wl, wr = self.centerline[np.random.choice(idxs),2:4] # track width at (xc, yc)
        s, _ = self.track.centerline.spline.calc_arclength_inaccurate(xc, yc)
        ey = np.random.uniform(-wr, wl)
        xc, yc, _ = self.track.frenet_to_cartesian(s, ey, 0.0)

        # select appropriate radius using track width and ey
        if ey < 0: # leave RHS clear
            r_max = wr - np.abs(ey) - margin
        else: # leave LHS clear
            r_max = wl - np.abs(ey) - margin
        r_max = max(r_min, r_max) # to ensure nonnegative
        r = np.random.uniform(r_min, r_max)
        self._draw_circle(xc, yc, r)

    def _draw_circle(self, x, y, r):
        """draws circle on the occupancy grid"""
        scale = self.track.spec.resolution # conversion faactor pixel -> m
        ox, oy, yaw = self.track.spec.origin
        if r < 0.0:
            r = 0.0
        r = int(r / scale)
        dx = x - ox
        dy = y - oy
        c = np.cos(-yaw)
        s = np.sin(-yaw)
        x = c * dx - s * dy
        y = s * dx + c * dy
        x = int(x / scale) 
        y = int(y / scale)
        self.track.occupancy_map = cv2.circle(self.track.occupancy_map, (x, y), r, 0.0, -1)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)
        # print('rendering!')
        if self.render_mode not in self.metadata["render_modes"]:
            return
        # update to the most recent occupancy grid
        # print('rendering')
        # self.renderer.update_occupancy(self.track)
        self.renderer.update(state=self.render_obs)
        return self.renderer.render()

class F110EgoDR(F110Ego):
    def __init__(
        self,
        config: dict,
        opps: List[OpponentDriver] = None,
        render_mode: str = None,
        **kwargs
    ):
        """
        F110Ego with support for domain randomization
        """
        env = F110EnvDR(config, render_mode, **kwargs)
        super().__init__(env, opps)
        self.config_input = config
        self.params_input = config['params']

    