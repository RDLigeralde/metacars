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

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,2),
            dtype=np.float32,
        )

        # reward scaling
        self.r_max = self.env.unwrapped.config['params']['v_max'] * self.env.unwrapped.timestep # max distance that can be traveled in one step call

    def step(self, action: np.ndarray):
        """Steps using provided action + opponent policies"""
        actions = np.zeros((self.num_agents, 2))
        actions *= np.array([self.env.unwrapped.params['s_max'], self.env.unwrapped.params['v_max']])
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
        self.render_mode = render_mode

        if config['normalize_input']:
            print('running with normalized input')
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,2),
                dtype=np.float32,
            )
        self.action_range = np.array([self.params_input['s_max'], self.params_input['v_max']])

        
        self.centerline = self._update_centerline(config['map'])
        raceline = self._update_raceline(config['map'])
        self.vspline = self._get_velocity_spline(raceline)
        self.yaw_spline = self._get_yaw_spline(raceline)
        self.last_action = np.zeros((self.num_agents, 2))
        self.stag_count = 0 #np.zeros((self.num_agents,))
        self.total_prog = 0 #np.zeros((self.num_agents,))

        # crash penalty for rewards that will gradually get stricter
        self.crash_penalty = -1.0
        self.total_timesteps = 0

        self.MILESTONE_INCREMEMENT = 0.1
        self.milestone = 0.1 # percentage progress that will trigger a large positive reward

        # print(self.action_space)
        # for logging
        self.n_timeouts = 0
        self.n_crashes = 0
        self.last_run_progress = 0.0
        self.n_laps = 0
        self.last_checkpoint_time = 0.0
        print(self.action_space, self.action_range)


    def _get_yaw_spline(self, raceline_info):
        data = np.zeros((raceline_info.shape[0], 2))
        data[:, 0] = raceline_info[:, 0] # s
        data[:, 1] = raceline_info[:, 3] # yaw
        data = data[data[:, 0].argsort()]
        data = data[:-1]
        return CubicSpline(data[:, 0], data[:, 1])
    
    def _get_velocity_spline(self, raceline_info):
        sv_values = []
        for i in range(raceline_info.shape[0]):
            x = raceline_info[i, 1]
            y = raceline_info[i, 2]
            sv_values.append([self.track.centerline.spline.calc_arclength_inaccurate(x, y)[0],
                            raceline_info[i, 5]])
        sv_values = np.array(sv_values)
        sv_values = sv_values[sv_values[:, 0].argsort()]
        sv_values = sv_values[:-1]

        s_diff = np.diff(sv_values[:, 0])
        mask = np.ones(len(sv_values), dtype=bool)
        mask[1:] = s_diff > 0

        cleaned_values = sv_values[mask]

        if len(cleaned_values) < 4:
            print("Warning: Not enough unique points for cubic spline. Using linear interpolation.")
            from scipy.interpolate import interp1d
            return interp1d(cleaned_values[:, 0], cleaned_values[:, 1], 
                            bounds_error=False, fill_value="extrapolate")

        return CubicSpline(cleaned_values[:, 0], cleaned_values[:, 1])
    
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
        if self.config_input['normalize_input']:
            sim_action = action * self.action_range
        else:
            sim_action = action
        # print(sim_action)
        self.sim.step(sim_action)

        # observation
        obs = self.observation_type.observe()

        # times
        self.current_time = self.current_time + self.timestep
        self.total_timesteps += 1

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
        reward, reward_info = self._get_reward(action)
        self.last_action = action
        # add in new timeout condition after 1 minute
        timeout = ((self.current_time / self.timestep) >= (60.0 / self.timestep)) 
        self.n_timeouts += int(timeout) # hope to see this get bigger overtime
        done = done or timeout
        self.last_run_progress = self.total_prog if done else self.last_run_progress
        info['custom/timeouts'] = self.n_timeouts
        info['custom/most_recent_progress'] = self.last_run_progress # now tracks max total progress at any timestep
        info['custom/crashes'] = self.n_crashes
        info['custom/num_laps'] = self.n_laps
        info.update(reward_info)

        return obs, reward, done, truncated, info


    def _sigmoid(self, x):
        # To get rid of warning
        if x < -1e3:
            return 0
        if x > 1e3:
            return 1
        return 1.0 / (1.0 + np.exp(-x))
    VEL_ACTION_CHANGE_PENALTY = 0 #-0.5
    STEER_ACTION_CHANGE_PENALTY = 0 #-1.0
    STAGNATION_PENALTY = -0.1
    STAGNATION_CUTOFF = 0.02 # delta s as a fraction of total track length
    # STAG_TIMEOUT = 20 # number of consecutive stag penalties required to trigger a timeout (not using anymore)
    VELOCITY_REWARD_SCALE = 0.0
    HEADING_PENALTY = -1.0 # ended up slowing down training significantly
    # CRASH_PENALTY = -1000
    PROGRESS_WEIGHT = 100
    CRASH_CURRICULUM = int(1e5)
    DELTA_U_CURRICULUM = int(1e6)
    V_REF_CURRICULUM = int(1e6)
    MILESTONE_REWARD = 5
    DECAY_INTERVAL = 1e5
    MAX_CRASH_PENALTY = 1 # now staying constant
    TURN_SPEED_PENALTY = 0 #-0.1
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
        # penalty for not tracking reference velocities - NEW
        # penalty for being significantly off from reference yaw - NEW (essentially want to prevent turning around)
        # Purely collaborative reward if more than 1 agent

        # NOTE: can't make some penalties negative b/c otherwise agent learns to just crash to get a quick
        # negative reward
        self.crash_penalty = -(1 + (self.MAX_CRASH_PENALTY - 1) * 
                               np.tanh(self.total_timesteps / self.CRASH_CURRICULUM)) # gradually grows between 1 and max value

        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents

        reward = 0.0
        reward_info = {}
        for i in range(self.num_agents):
            current_s, _ = (
                self.track.centerline.spline.calc_arclength_inaccurate(
                    self.poses_x[i], self.poses_y[i]
                )
            )

            prog = current_s - self.last_s[i]
            # account for lapping
            if current_s < 0.1 * self.track.centerline.spline.s[-1] and self.last_s[i] > 0.9 * self.track.centerline.spline.s[-1]:
                prog += self.track.centerline.spline.s[-1]
            # looped backward
            elif self.last_s[i] < 0.1 * self.track.centerline.spline.s[-1] and current_s > 0.9 * self.track.centerline.spline.s[-1]:
                prog -= self.track.centerline.spline.s[-1]
    

            pcnt = prog / self.track.centerline.spline.s[-1]

            prog_reward = pcnt * self.PROGRESS_WEIGHT
            reward += prog_reward
            reward_info['custom/reward_terms/prog'] = prog_reward
            # want to see this grow during training, stores percentage of track traveleed
            self.total_prog += pcnt
            if self.total_prog > self.milestone:
                self.milestone += self.MILESTONE_INCREMEMENT
                try:
                    reward += self.MILESTONE_REWARD #/ (self.current_time - self.last_checkpoint_time)
                    reward_info['custom/reward_terms/milestone'] = self.MILESTONE_REWARD #/ (self.current_time - self.last_checkpoint_time)
                    self.last_checkpoint_time = self.current_time
                except:
                    print(f'problem pose: {self.poses_x[i]}, {self.poses_y[i]}')
                    print(f'staring pose: {self.start_xs[i]}, {self.start_ys[i]}')
                    print(f'{current_s}, {self.last_s[i]}, {self.current_time}, {self.last_checkpoint_time} on fail')
                    raise Exception('div by 0')
            else:
               reward_info['custom/reward_terms/milestone'] = 0.0 
                
            # reward += self.ACTION_CHANGE_PENALTY * 1 / (np.linalg.norm(action[i] - self.last_action[i], 2) + 1)
            
            # rework the action change penalty so that it works its way up from 0 at the start of training
            # using a sigmoid
            # print(np.abs(self.last_action[i, 0] - action[i, 0]))
            time_increase_factor = self._sigmoid(((self.total_timesteps - self.DELTA_U_CURRICULUM) / self.DECAY_INTERVAL))
            steer_delta_pen = self.STEER_ACTION_CHANGE_PENALTY * np.abs(self.last_action[i, 0] - action[i, 0]) * time_increase_factor
            reward += steer_delta_pen
            reward_info['custom/reward_terms/delta_steer'] = steer_delta_pen

            vel_delta_pen = self.VEL_ACTION_CHANGE_PENALTY * np.abs(self.last_action[i, 1] - action[i, 1]) * time_increase_factor
            reward += vel_delta_pen
            reward_info['custom/reward_terms/delta_v'] = vel_delta_pen

            turn_speed_pen = self.TURN_SPEED_PENALTY * np.abs((action[i, 0] * action[i, 1])) * time_increase_factor
            reward += turn_speed_pen
            reward_info['custom/reward_terms/turning_spped'] = turn_speed_pen

            if self.collisions[i]:
                reward += self.crash_penalty
                reward_info['custom/reward_terms/collision'] = self.crash_penalty
                self.n_crashes += 1 # hope to see this eventually stagnate in logging
            else:
                reward_info['custom/reward_terms/collision'] = 0.0
            
            ## velocity tracing for a warm start - want to wean this off to let policy eventually learn potentially
            ## better 
            time_decrease_factor = self._sigmoid(-((self.total_timesteps - self.V_REF_CURRICULUM) / self.DECAY_INTERVAL))
            # v_ref = self.vspline(current_s)
            # maxes out when v_ref = v_action
            # vel_track_reward = self.VELOCITY_REWARD_SCALE * np.exp(-(v_ref - action[i, 1]) ** 2)  * time_decrease_factor
            # reward += vel_track_reward
            # reward_info['custom/reward_terms/vel_tracking'] = vel_track_reward

            # yaw_ref = self.yaw_spline(current_s)
            # if abs(self.poses_theta[i] - yaw_ref) > np.deg2rad(75):
            #     reward += HEADING_PENALTY
            # print(action)
            if np.abs(action[i, 1]) < 1e-3 :
                # print('stagnating')
                stag_penalty = self.crash_penalty # self.timestep * 
                reward += stag_penalty # make stagnating for a second just as bad as crashing
                reward_info['custom/reward_terms/stagnation'] = stag_penalty
            else:
                reward_info['custom/reward_terms/stagnation'] = 0.0

            reward_info['custom/reward_terms/total_timestep_reward'] = reward
            
            self.last_s[i] = current_s
            
        return reward, reward_info

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
        self.total_prog = 0.0
        self.milestone = self.MILESTONE_INCREMEMENT
        self.last_checkpoint_time = 0.0

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

        self.poses_x = self.start_xs
        self.poses_y = self.start_ys

         ## makre sure to recalculate track position
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        for i in range(self.num_agents):
            self.last_s[i], _ = self.track.centerline.spline.calc_arclength_inaccurate(
                    self.poses_x[i], self.poses_y[i]
                )

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

        # update laps from last trial
        self.n_laps += int(self.total_prog)
        self._reset_pos(seed=seed, options=options)

        # regenerate the map to the original without obstacles anyways to ensure that obstacles don't clutter over time
        self.update_map(config['map'])
        self._spawn_obstacle(self.num_obstacles)
        self._update_map_from_track()
        # get no input observations
        self.last_action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(self.last_action)

        ## updated to support changing maps, create new renederer with most up to date info
        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=self.render_mode,
            render_fps=self.metadata["render_fps"],
        )
        return obs, info

    def _spawn_obstacle(
        self, 
        n_obs,
        obs_room = 30,
        room=30, 
        r_min=0.15,
        r_max=0.3,
        margin=0.6,
    ):
        """
        spawns a random box on track room away from ego
        only draws circles for now, with low lidar resolution should be fine

        Args:
            obs_room: minimum number of indices separating the sampled centerline points for the obstacles
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
        # track.centerline.
        curr = self.track.occupancy_map
        idxs = np.arange(len(self.centerline))
        remove_window = np.arange(n_idx - room, n_idx + room + 1)
        remove_window[remove_window < 0] += self.centerline.shape[0]
        remove_window[remove_window > self.centerline.shape[0]] -= self.centerline.shape[0]
        idxs = np.setdiff1d(idxs, remove_window)
        for i in range(n_obs):

            # randomly select (s, ey) from remaining indices
            rand_idx = np.random.choice(idxs)

            # exclude next ones in next iteration
            remove_window = np.arange(rand_idx - obs_room, rand_idx + obs_room + 1)
            remove_window[remove_window < 0] += self.centerline.shape[0]
            remove_window[remove_window > self.centerline.shape[0]] -= self.centerline.shape[0]
            idxs = np.setdiff1d(idxs, remove_window)

            # print(rand_idx)
            xc, yc = self.centerline[rand_idx, :2]
            s, _ = self.track.centerline.spline.calc_arclength_inaccurate(xc, yc)
            yaw = self.yaw_spline(s)    
            wl, wr = self.centerline[rand_idx, 2:4] # track width at (xc, yc)
            ey = np.random.uniform(-wr, wl)

            dx = -ey * np.sin(yaw)
            dy = ey * np.cos(yaw)
            x = xc + dx
            y = yc + dy
            r = np.random.uniform(r_min, r_max)
            curr = self._draw_circle(x, y, r)
        return curr

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
    
    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] * 2 + temp_y*2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        ## NEW -using self.total_prog to judge laps, will terminate episode after 3 laps
        done = (self.collisions[self.ego_idx]) or int(self.total_prog) >= 3 # or np.all(self.toggle_list >= 4)
        # self.n_laps += int(np.all(self.toggle_list >= 4)) # this is wrong becuse it counts collisions too (not sure about this comment)

        


        return bool(done), self.toggle_list >= 4

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