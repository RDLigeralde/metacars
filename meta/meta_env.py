from f1tenth_gym.envs.track.utils import find_track_dir
from meta.opponents.opponent import OpponentDriver
from f1tenth_gym.envs import F110Env
import gymnasium as gym

from scipy.interpolate import CubicSpline, interp1d
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import os

class F110Multi(F110Env):
    def __init__(
        self,
        config: dict,
        render_mode: str = None,
        **kwargs
    ):
        """
        F110Env set up for 1v1 racing

        Args:
            config (dict): 
            render_mode (str, optional): _description_. Defaults to None.
        """
        self.config = config
        self.params = config['params']
        self.reward_coefs = config['reward']
        self.ego_idx, self.opp_idx = 0, 1 # hardcoded harmlessly
        self.last_action = np.zeros((2, 2))

        self.render_mode = render_mode
        super().__init__(config=config, render_mode=render_mode, **kwargs)
        
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        self.action_range = np.array([self.params['s_max'], self.params['v_max']])
        self.action_last = np.zeros_like(self.action_range)

        self.vspline, self.yaw_spline = self._init_splines()
        self._init_reward_params()

        # custom metrics
        self.total_timesteps = 0
        self.n_timeouts = 0
        self.n_crashes = 0
        self.last_run_progress = 0.0
        self.n_laps = 0
        self.last_checkpoint_time = 0.0

    def _init_splines(self):
        track = self.config['map']
        track_dir = find_track_dir(track)
        raceline = os.path.join(track_dir, f"{track}_raceline.csv")
        raceline_arr = np.loadtxt(raceline, delimiter=';').astype(np.float32)
        
        vspline = self._get_velocity_spline(raceline_arr)
        yaw_spline = self._get_yaw_spline(raceline_arr)
        return vspline, yaw_spline

    def _init_reward_params(self):
        self.total_prog = 0.0
        for key, value in self.reward_coefs.items():
            setattr(self, key, value)

    def _get_yaw_spline(self, raceline_info: np.ndarray) -> CubicSpline:
        data = np.zeros((raceline_info.shape[0], 2))
        data[:, 0] = raceline_info[:, 0] # s
        data[:, 1] = raceline_info[:, 3] # yaw
        data = data[data[:, 0].argsort()]
        data = data[:-1]
        return CubicSpline(data[:, 0], data[:, 1])
    
    def _get_velocity_spline(self, raceline_info: np.ndarray) -> CubicSpline:
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
            return interp1d(cleaned_values[:, 0], cleaned_values[:, 1], 
                            bounds_error=False, fill_value="extrapolate")

        return CubicSpline(cleaned_values[:, 0], cleaned_values[:, 1])
    
    def _get_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        if not hasattr(self, "last_s"):
            self.last_s = np.zeros(2)
        
        self.crash_penalty = -(1 + (self.max_crash_penalty - 1) * 
                            np.tanh(self.total_timesteps / self.crash_curriculum))
        
        agent_reward = 0.0
        reward_info = {}
        
        prog_reward, current_s, pcnt = self._calculate_progress_reward()
        agent_reward += prog_reward
        self.last_s[0] = current_s
        reward_info['custom/reward_terms/prog'] = prog_reward
        
        milestone_reward = self._calculate_milestone_reward(pcnt)
        agent_reward += milestone_reward
        reward_info['custom/reward_terms/milestone'] = milestone_reward
        
        action_penalties = self._calculate_action_penalties(action)
        for penalty_key, penalty_value in action_penalties.items():
            agent_reward += penalty_value
            reward_info[f'custom/reward_terms/{penalty_key}'] = penalty_value
        
        collision_penalty = self._calculate_collision_penalty()
        agent_reward += collision_penalty
        reward_info['custom/reward_terms/collision'] = collision_penalty
        
        #vel_tracking_reward = self._calculate_velocity_tracking(action, current_s)
        #agent_reward += vel_tracking_reward
        #reward_info['custom/reward_terms/vel_tracking'] = vel_tracking_reward
        
        stagnation_penalty = self._calculate_stagnation_penalty(action)
        agent_reward += stagnation_penalty
        reward_info['custom/reward_terms/stagnation'] = stagnation_penalty

        distance_reward, overtaking_reward = self._position_reward()
        agent_reward += distance_reward
        agent_reward += overtaking_reward
        reward_info['custom/reward_terms/opponent_distance'] = distance_reward
        reward_info['custom/reward_terms/overtaking'] = overtaking_reward
        
        reward_info['custom/reward_terms/total_reward'] = agent_reward
        
        return agent_reward, reward_info

    def _calculate_progress_reward(self):
        """Calculate reward for making progress along the track."""
        current_s, _ = (
            self.track.centerline.spline.calc_arclength_inaccurate(
                self.poses_x[self.ego_idx], self.poses_y[self.ego_idx]
            )
        )
        
        prog = current_s - self.last_s[self.ego_idx]
        
        if current_s < 0.1 * self.track.centerline.spline.s[-1] and self.last_s[self.ego_idx] > 0.9 * self.track.centerline.spline.s[-1]:
            prog += self.track.centerline.spline.s[-1]
        elif self.last_s[self.ego_idx] < 0.1 * self.track.centerline.spline.s[-1] and current_s > 0.9 * self.track.centerline.spline.s[-1]:
            prog -= self.track.centerline.spline.s[-1]
        
        pcnt = prog / self.track.centerline.spline.s[-1]
        prog_reward = pcnt * self.progress_weight
        
        return prog_reward, current_s, pcnt

    def _calculate_milestone_reward(self, pcnt):
        self.total_prog += pcnt
        if self.total_prog > self.milestone:
            self.milestone += self.milestone_increment
            self.last_checkpoint_time = self.current_time
            return self.milestone_reward
        return 0.0

    def _calculate_action_penalties(self, action):
        penalties = {}

        time_increase_factor = 1.0 / (1.0 + np.exp(-((self.total_timesteps - self.delta_u_curriculum) / self.decay_interval)))
        
        steer_delta_pen = self.steer_action_change_penalty * np.abs(self.last_action[self.ego_idx, 0] - action[self.ego_idx, 0]) * time_increase_factor
        penalties['delta_steer'] = steer_delta_pen
        
        vel_delta_pen = self.vel_action_change_penalty * np.abs(self.last_action[self.ego_idx, 1] - action[self.ego_idx, 1]) * time_increase_factor
        penalties['delta_v'] = vel_delta_pen
        
        turn_speed_pen = self.turn_speed_penalty * np.abs((action[self.ego_idx, 0] * action[self.ego_idx, 1])) * time_increase_factor
        penalties['turning_speed'] = turn_speed_pen
        
        return penalties

    def _calculate_collision_penalty(self):
        """Calculate penalty for collisions."""
        if self.collisions[self.ego_idx]:
            self.n_crashes += 1
            return self.crash_penalty
        return 0.0

    def _calculate_stagnation_penalty(self, action):
        """Calculate penalty for not moving (stagnation)."""
        if np.abs(action[self.ego_idx, 1]) < 1e-3:
            return self.crash_penalty  # Make stagnating as bad as crashing
        return 0.0
    
    def _position_delta(self):
        opponent_s, _ = (
            self.track.centerline.spline.calc_arclength_inaccurate(
                self.poses_x[1], self.poses_y[1]
            )
        )
        self.last_s[1] = opponent_s
        s_diff = self.last_s[0] - self.last_s[1]
        track_length = self.track.centerline.spline.s[-1]

        if s_diff > track_length / 2:
            s_diff -= track_length
        elif s_diff < -track_length / 2:
            s_diff += track_length

        return s_diff

    def _position_reward(self) -> Tuple[float, float]:
        """
        Reward for closing distance / gettting further away from opponent
        and overtake event

        Returns:
            Tuple[float, float]: (distance reward, overtaking reward)
        """
        if not hasattr(self, "last_opp_dist"):
            self.last_opp_dist = self._position_delta()
            self.leading = self.last_opp_dist > 0
            return 0.0, 0.0
        
        s_diff = self._position_delta()
        distance = s_diff - self.last_opp_dist
        distance_reward = distance * self.position_weight

        leading = s_diff > 0
        if leading and not self.leading:
            overtaking_reward = self.overtake_reward
        elif not leading and self.leading:
            overtaking_reward = -self.overtake_reward
        else:
            overtaking_reward = 0.0

        self.last_opp_dist = s_diff
        self.leading = leading

        return distance_reward, overtaking_reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Overriding step for reward calculation purposes"""
        action *= self.action_range
        self.sim.step(action)

        obs = self.observation_type.observe()
        self.current_time = self.current_time + self.timestep
        self.total_timesteps += 1

        self._update_state()

        self.render_obs = { # TODO: figure out how to render opponent
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
        done, toggle_list = self._check_done()
        truncated = False
        info = {'checkpoint_done': toggle_list}

        reward, reward_info = self._get_reward(action)
        self.last_action = action

        timeout = ((self.current_time / self.timestep) >= (60.0 / self.timestep)) # TODO: make this a config param
        self.n_timeouts += int(timeout)
        done = done or timeout
        self.last_run_progress = self.total_prog if done else self.last_run_progress
        info['custom/timeouts'] = self.n_timeouts
        info['custom/most_recent_progress'] = self.last_run_progress
        info['custom/crashes'] = self.n_crashes
        info.update(reward_info)

        return obs, reward, done, truncated, info

class F110MultiView(gym.Wrapper):
    def __init__(self, env: F110Multi, opponents: List[OpponentDriver], agent_idx: int = 0):
        """Interface for a single-agent policy to interact with F110Multi"""
        super().__init__(env)
        self.env = env
        self.opponents = opponents
        self.opponent = np.random.choice(opponents)
        self.agent_idx = agent_idx

        self.ego_idx, self.opp_idx = 0, 1 # hardcoded harmlessly

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        self.action_range = np.array([self.env.unwrapped.params['s_max'], self.env.unwrapped.params['v_max']])

        large_num = 1e30
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        self.observation_space = gym.spaces.Dict({
            'scans': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
            'poses': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
            'vels': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
            'headings': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 2), dtype=np.float32),
        })

    def reset(self, opponent_idx=None, seed=None, options=None) -> Tuple[dict, dict]:
        self.opponent = self.opponents[opponent_idx] if opponent_idx else np.random.choice(self.opponents)
        obs, info = super().reset(seed=seed, options=options)
        return self._ego_observe(obs, self.ego_idx), info
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        obs = self.env.unwrapped.observation_type.observe()
        opp_obs = self._ego_observe(obs, self.opp_idx)
        opponent_action = self.opponent(opp_obs)

        action_full = np.zeros((2, 2), dtype=np.float32)
        action_full[0, :] = action
        action_full[1, :] = opponent_action

        obs_t1, reward, done, truncated, info = self.env.step(action_full)
        obs = self._ego_observe(obs_t1, self.ego_idx)
        return obs, reward, done, truncated, info

    def _ego_observe(self, obs: dict, idx: int) -> dict:
        return {
            'scans': obs['scans'][idx:idx+1],
            'poses': obs['poses'][idx:idx+1],
            'vels': obs['vels'][idx:idx+1],
            'headings': obs['headings'][idx:idx+1]
        }



        
        






