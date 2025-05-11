from f1tenth_gym.envs.track.utils import find_track_dir
from f1tenth_gym.envs.rendering import make_renderer
from f1tenth_gym.envs import F110Env

from opponents.opponent import OpponentDriver
from opponents.loader import PolicyLoader

from scipy.interpolate import CubicSpline
import gymnasium as gym
import numpy as np

from typing import List, Optional, Tuple
import os

class F110Multi(F110Env):
    def __init__(
        self,
        config: dict,
        render_mode: str = None, 
        **kwargs         
    ):
        """
        F110Env with support for domain randomization and overtaking rewards

        Enabled by setting 'param': {'min': val, 'max': val} in config.yml
        instead of static values
        """
        self.config_input = config
        self.params_input = config['params']

        self.reward_params = config.get('reward_params', {})
        self._init_reward_params()

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
        self.render_mode = render_mode
        super().__init__(config, render_mode, **kwargs)
        

        if config['normalize_input']:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_agents,2),
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
        
        self.total_timesteps = 0
        self.n_timeouts = 0
        self.n_crashes = 0
        self.last_run_progress = 0.0
        self.n_laps = 0
        self.last_checkpoint_time = 0.0

    def _init_reward_params(self) -> None:
        self.MILESTONE_INCREMENT = self.reward_params.get('milestone_increment')
        self.milestone = self.reward_params.get('initial_milestone')  # percentage progress that will trigger a large positive reward
        self.crash_penalty = self.reward_params.get('initial_crash_penalty')
        self.VEL_ACTION_CHANGE_PENALTY = self.reward_params.get('vel_action_change_penalty')
        self.STEER_ACTION_CHANGE_PENALTY = self.reward_params.get('steer_action_change_penalty')
        self.STAGNATION_PENALTY = self.reward_params.get('stagnation_penalty')
        self.STAGNATION_CUTOFF = self.reward_params.get('stagnation_cutoff')
        self.VELOCITY_REWARD_SCALE = self.reward_params.get('velocity_reward_scale')
        self.HEADING_PENALTY = self.reward_params.get('heading_penalty')
        self.PROGRESS_WEIGHT = self.reward_params.get('progress_weight')
        self.CRASH_CURRICULUM = self.reward_params.get('crash_curriculum')
        self.DELTA_U_CURRICULUM = self.reward_params.get('delta_u_curriculum')
        self.V_REF_CURRICULUM = self.reward_params.get('v_ref_curriculum')
        self.MILESTONE_REWARD = self.reward_params.get('milestone_reward')
        self.DECAY_INTERVAL = self.reward_params.get('decay_interval')
        self.MAX_CRASH_PENALTY = self.reward_params.get('max_crash_penalty')
        self.TURN_SPEED_PENALTY = self.reward_params.get('turn_speed_penalty')
        self.OVERTAKE_REWARD = self.reward_params.get('overtake_reward')

    def _get_yaw_spline(self, raceline_info) -> CubicSpline:
        data = np.zeros((raceline_info.shape[0], 2))
        data[:, 0] = raceline_info[:, 0] # s
        data[:, 1] = raceline_info[:, 3] # yaw
        data = data[data[:, 0].argsort()]
        data = data[:-1]
        return CubicSpline(data[:, 0], data[:, 1])
    
    def _get_velocity_spline(self, raceline_info) -> CubicSpline:
        svs = np.zeros((len(raceline_info), 2), dtype=np.float32)
        for i in range(len(raceline_info)):
            x, y = raceline_info[i, 1], raceline_info[i, 2]
            svs[i] = np.array([
                self.track.centerline.spline.calc_arclength_inaccurate(x, y)[0],
                raceline_info[i, 5]
            ])
        svs = svs[svs[:, 0].argsort()][:-1]

        s_diff = np.diff(svs[:, 0])
        mask = np.ones(len(svs), dtype=bool)
        mask[1:] = s_diff > 0.0

        masked = svs[mask]
        return CubicSpline(masked[:, 0], masked[:, 1])
    
    def _sample_dict(self, params: dict) -> dict:
        """Sample parameters for domain randomization"""
        pcopy = params.copy()
        for key, val in pcopy.items():
            if isinstance(val, dict) and 'min' in val and 'max' in val: # sample numeric
                pcopy[key] = np.random.uniform(val['min'], val['max'])
            elif self.use_trackgen and key == 'map': # sample track
                pcopy[key] = os.path.join(self.config_input['map'], np.random.choice(self.tracks))
        return pcopy

    def _update_raceline(self, track) -> np.ndarray:
        track_dir = find_track_dir(track)
        centerline_file = os.path.join(track_dir, f"{track}_raceline.csv")
        return np.loadtxt(centerline_file, delimiter=';').astype(np.float32)
    
    def _update_centerline(self, track):
        track_dir = find_track_dir(track)
        centerline_file = os.path.join(track_dir, f"{track}_centerline.csv")
        return np.loadtxt(centerline_file, delimiter=',').astype(np.float32)

    def _update_map_from_track(self):
        self.sim.set_map(self.track)

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        if self.config_input['normalize_input']:
            sim_action = action * self.action_range
        else:
            sim_action = action

        self.sim.step(sim_action)
        obs = self.observation_type.observe()

        self.current_time = self.current_time + self.timestep
        self.total_timesteps += 1
        self._update_state()

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

        done, truncated, toggle_list = self._check_done()
        info = {"checkpoint_done": toggle_list}

        reward, reward_info = self._get_reward(action)
        self.last_action = action

        timeout = self.current_time >= self.config.get('max_len', 60.0)
        self.n_timeouts += int(timeout) # hope to see this get bigger overtime
        done = done or timeout

        self.last_run_progress = self.total_prog if done else self.last_run_progress
        info['custom/timeouts'] = self.n_timeouts
        info['custom/most_recent_progress'] = self.last_run_progress # now tracks max total progress at any timestep
        info['custom/crashes'] = self.n_crashes
        info['custom/num_laps'] = self.n_laps
        info.update(reward_info)

        return obs, reward, done, truncated, info
    
    def _sigmoid(self, x: float) -> float:
        if x < -1e3:
            return 0
        if x > 1e3:
            return 1
        return 1.0 / (1.0 + np.exp(-x))

    def _get_progress_reward(self, current_s: float) -> Tuple[float, float]:
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        
        prog = current_s - self.last_s[self.ego_idx]
        
        if current_s < 0.1 * self.track.centerline.spline.s[-1] and self.last_s[self.ego_idx] > 0.9 * self.track.centerline.spline.s[-1]:
            prog += self.track.centerline.spline.s[-1]

        elif self.last_s[self.ego_idx] < 0.1 * self.track.centerline.spline.s[-1] and current_s > 0.9 * self.track.centerline.spline.s[-1]:
            prog -= self.track.centerline.spline.s[-1]
        
        pcnt = prog / self.track.centerline.spline.s[-1]
        prog_reward = pcnt * self.PROGRESS_WEIGHT
        
        self.total_prog += pcnt
        return prog_reward, pcnt
    
    def _get_milestone_reward(self) -> float:
        if self.total_prog > self.milestone:
            self.milestone += self.MILESTONE_INCREMENT
            try:
                milestone_reward = self.MILESTONE_REWARD
                self.last_checkpoint_time = self.current_time
                return milestone_reward
            except:
                raise Exception('div by 0')
        else:
            return 0.0

    def _get_steering_change_penalty(self, action: np.ndarray) -> float:
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.DELTA_U_CURRICULUM) / self.DECAY_INTERVAL)
        )
        steer_delta_pen = self.STEER_ACTION_CHANGE_PENALTY * \
                        np.abs(self.last_action[self.ego_idx, 0] - action[self.ego_idx, 0]) * \
                        time_increase_factor
        return steer_delta_pen

    def _get_velocity_change_penalty(self, action: np.ndarray) -> float:
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.DELTA_U_CURRICULUM) / self.DECAY_INTERVAL)
        )
        vel_delta_pen = self.VEL_ACTION_CHANGE_PENALTY * \
                        np.abs(self.last_action[self.ego_idx, 1] - action[self.ego_idx, 1]) * \
                        time_increase_factor
        return vel_delta_pen

    def _get_turn_speed_penalty(self, action: np.ndarray) -> float:
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.DELTA_U_CURRICULUM) / self.DECAY_INTERVAL)
        )
        turn_speed_pen = self.TURN_SPEED_PENALTY * \
                        np.abs((action[self.ego_idx, 0] * action[self.ego_idx, 1])) * \
                        time_increase_factor
        return turn_speed_pen

    def _get_collision_penalty(self) -> float:
        if self.collisions[self.ego_idx]:
            self.n_crashes += 1
            return self.crash_penalty
        else:
            return 0.0

    def _get_stagnation_penalty(self, action: np.ndarray) -> float:
        if np.abs(action[self.ego_idx, 1]) < 1e-3:
            return self.crash_penalty  # Same magnitude as crash penalty
        else:
            return 0.0

    def _get_overtaking_reward(self, current_s_all: List[float]) -> float:
        if self.num_agents == 1:
            return 0.0
        
        if not hasattr(self, "last_s"):
            return 0.0
        
        ego_idx = self.ego_idx
        other_idx = 1 - ego_idx # assumes 2 agents at most
        track_length = self.track.centerline.spline.s[-1]
        
        ego_progress = current_s_all[ego_idx] - self.last_s[ego_idx]
        other_progress = current_s_all[other_idx] - self.last_s[other_idx]
        
        if current_s_all[ego_idx] < 0.1 * track_length and self.last_s[ego_idx] > 0.9 * track_length:
            ego_progress += track_length
        elif self.last_s[ego_idx] < 0.1 * track_length and current_s_all[ego_idx] > 0.9 * track_length:
            ego_progress -= track_length
            
        if current_s_all[other_idx] < 0.1 * track_length and self.last_s[other_idx] > 0.9 * track_length:
            other_progress += track_length
        elif self.last_s[other_idx] < 0.1 * track_length and current_s_all[other_idx] > 0.9 * track_length:
            other_progress -= track_length
        
        was_behind = self.last_s[ego_idx] < self.last_s[other_idx]
        is_ahead = current_s_all[ego_idx] > current_s_all[other_idx]
        
        if abs(self.last_s[ego_idx] - self.last_s[other_idx]) > 0.5 * track_length:
            was_behind = not was_behind
        if abs(current_s_all[ego_idx] - current_s_all[other_idx]) > 0.5 * track_length:
            is_ahead = not is_ahead
            
        if was_behind and is_ahead:
            return self.reward_params.get('OVERTAKE_REWARD', 1.0)
        elif not was_behind and not is_ahead:
            return -self.reward_params.get('OVERTAKE_REWARD', 1.0)
        
        return 0.0

    def _get_reward(self, action: np.ndarray) -> Tuple[float, dict]:
        self._update_crash_penalty()
        
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        
        current_s_all = []
        for i in range(self.num_agents):
            current_s, _ = self.track.centerline.spline.calc_arclength_inaccurate(
                self.poses_x[i], self.poses_y[i]
            )
            current_s_all.append(current_s)
        
        overtake_reward = self._get_overtaking_reward(current_s_all)
        
        total_reward = 0.0
        reward_info = {}

        current_s = current_s_all[self.ego_idx]
        
        prog_reward, pcnt = self._get_progress_reward(current_s)
        milestone_reward = self._get_milestone_reward()
        steer_penalty = self._get_steering_change_penalty(action)
        vel_penalty = self._get_velocity_change_penalty(action)
        turn_speed_penalty = self._get_turn_speed_penalty(action)
        collision_penalty = self._get_collision_penalty()
        stagnation_penalty = self._get_stagnation_penalty(action)
        
        ego_reward = (
            prog_reward +
            milestone_reward +
            steer_penalty +
            vel_penalty +
            turn_speed_penalty +
            collision_penalty +
            stagnation_penalty +
            overtake_reward
        )

        total_reward = ego_reward

        reward_info['custom/reward_terms/prog'] = prog_reward
        reward_info['custom/reward_terms/milestone'] = milestone_reward
        reward_info['custom/reward_terms/delta_steer'] = steer_penalty
        reward_info['custom/reward_terms/delta_v'] = vel_penalty
        reward_info['custom/reward_terms/turning_speed'] = turn_speed_penalty
        reward_info['custom/reward_terms/collision'] = collision_penalty
        reward_info['custom/reward_terms/stagnation'] = stagnation_penalty
        reward_info['custom/reward_terms/overtaking'] = overtake_reward
        reward_info['custom/reward_terms/total_timestep_reward'] = total_reward

        for i in range(self.num_agents):
            self.last_s[i] = current_s_all[i]

        return total_reward, reward_info

    def _reset_pos(self, seed=None, options=None) -> None:
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.total_prog = 0.0
        self.milestone = self.MILESTONE_INCREMENT
        self.last_checkpoint_time = 0.0

        if options is not None and "poses" in options:
            poses = options["poses"]
        else:
            poses = self.reset_fn.sample()

        assert isinstance(poses, np.ndarray) and poses.shape == (
            self.num_agents,
            3,
        ), "Initial poses must be np.ndarray of shape (num_agents, 3)"

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

        self.sim.reset(poses)
        self.poses_x = self.start_xs
        self.poses_y = self.start_ys

        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        for i in range(self.num_agents):
            self.last_s[i], _ = self.track.centerline.spline.calc_arclength_inaccurate(
                    self.poses_x[i], self.poses_y[i]
                )

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
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

        self.n_laps += int(self.total_prog)
        self._reset_pos(seed=seed, options=options)

        self.update_map(config['map'])
        self._update_map_from_track()
        self.last_action = np.zeros((self.num_agents, 2))
        obs, _, _, _, info = self.step(self.last_action)

        self.renderer, self.render_spec = make_renderer(
            params=self.params,
            track=self.track,
            agent_ids=self.agent_ids,
            render_mode=self.render_mode,
            render_fps=self.metadata["render_fps"],
        )
        return obs, info

    def _check_done(self) -> Tuple[bool, bool, list[int]]:
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

        done = int(self.total_prog) >= 4
        truncated = bool(self.collisions[self.ego_idx])
        return done, truncated, self.toggle_list >= 4

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
        self.renderer.update(state=self.render_obs)
        return self.renderer.render()
    
    def _update_crash_penalty(self):
        """Update the crash penalty value based on curriculum"""
        self.crash_penalty = -(1 + (self.MAX_CRASH_PENALTY - 1) * 
                            np.tanh(self.total_timesteps / self.CRASH_CURRICULUM))

class F110MultiView(gym.Wrapper):
    def __init__(
        self,
        env: F110Multi,
        opp_dir: str=None,
        opp_cfg: dict=None,
        **kwargs
    ):
        """Interface for single-agent policy to race against opponent population"""
        super().__init__(env)
        self.env = env
        self.ego_idx = env.unwrapped.sim.ego_idx
        self.opponent_idx = 1 - self.ego_idx

        if opp_dir is not None and opp_cfg is not None:
            entropy_min, entropy_max = opp_cfg.get('entropy_min'), opp_cfg.get('entropy_max')
            self.loader = PolicyLoader(
                opp_dir, 
                entropy_min=entropy_min, 
                entropy_max=entropy_max,
                seed=self.env.unwrapped.seed
            )
            self.opponent, self.opp_vmax = self.loader.sample()
        else:
            self.loader = None
            self.opponent = None

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,2),
            dtype=np.float32,
        )
        self.action_range = self.env.unwrapped.action_range

        # hardcoded to frenet_marl
        large_num = 1e30
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5

        self.observation_space = gym.spaces.Dict({
            'scan': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
            'pose': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
            'vel': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
            'heading': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 2), dtype=np.float32),
        })

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        if self.loader is not None:
            self.opponent, self.opp_vmax = self.loader.sample()

        obs, info = self.env.reset(seed=seed, options=options)
        return self._ego_observe(obs, self.ego_idx), info

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        obs = self.env.unwrapped.observation_type.observe()
        opp_obs = self._ego_observe(obs, self.opponent_idx)
        opponent_action = self.opponent(opp_obs) if self.opponent else None

        if action.shape[0] == 2: # hack for recurrent policy
            action = np.expand_dims(action, axis=0)

        if opponent_action is not None:
            # ensures same normalization step correctly scales both agent speeds
            opponent_action[:, 1] *= (self.opp_vmax / self.action_range[1])
            action = np.concatenate((action, opponent_action), axis=0)

        obs, reward, done, truncated, info = self.env.step(action)
        obs = self._ego_observe(obs, self.ego_idx)
        return obs, reward, done, truncated, info
    
    def _ego_observe(self, obs, i) -> dict:
        """
        currently hardcoded to frenet_marl
        shouldn't be hard to change if you want, say lidar_conv
        """
        ego_obs = {
            'scan': obs['scan'][i:i+1],
            'pose': obs['pose'][i:i+1],
            'vel': obs['vel'][i:i+1],
            'heading': obs['heading'][i:i+1],
        }
        return ego_obs