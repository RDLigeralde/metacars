from f1tenth_gym.envs.track.utils import find_track_dir
from meta.opponents.opponent import OpponentDriver
from f1tenth_gym.envs import F110Env
import gymnasium as gym

from scipy.interpolate import CubicSpline, interp1d
from typing import Tuple, Dict, Any, List, Optional
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
        # Ensure config specifies 2 agents
        if 'num_agents' not in config:
            config['num_agents'] = 2  # Force 2 agents for 1v1 racing
        
        self.config = config
        self.params = config['params']
        self.reward_coefs = config['reward']
        self.ego_idx, self.opp_idx = 0, 1 # hardcoded harmlessly
        self.last_action = np.zeros((config['num_agents'], 2))

        self.render_mode = render_mode
        super().__init__(config=config, render_mode=render_mode, **kwargs)
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_agents, 2),
            dtype=np.float32,
        )
        self.action_range = np.array([self.params['s_max'], self.params['v_max']])
        self.action_last = np.zeros_like(self.action_range)

        self.vspline, self.yaw_spline = self._init_splines()
        self._init_reward_params()

        # metrics for wandb
        self.total_timesteps = 0
        self.n_timeouts = 0
        self.n_crashes = 0
        self.last_run_progress = 0.0
        self.n_laps = 0
        self.last_checkpoint_time = 0.0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        # Initialize tracking variables
        self.last_s = np.zeros(self.num_agents)
        self.total_prog = 0.0
        self.milestone = self.milestone_increment if hasattr(self, 'milestone_increment') else 0.1
        
        # Reset metrics
        self.total_timesteps = 0
        self.n_timeouts = 0
        self.n_crashes = 0
        self.last_run_progress = 0.0
        self.n_laps = 0
        self.last_checkpoint_time = 0.0
        
        # Reset reward tracking attributes
        if hasattr(self, 'last_opp_dist'):
            delattr(self, 'last_opp_dist')
        if hasattr(self, 'leading'):
            delattr(self, 'leading')
        
        self.last_action = np.zeros((self.num_agents, 2))
        
        return obs, info

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
    
    def _sigmoid(self, x):
        if x < -1e3:
            return 0
        if x > 1e3:
            return 1
        return 1.0 / (1.0 + np.exp(-x))

    def _calculate_progress_reward(self, agent_idx):
        """Calculate reward for making progress along the track."""
        current_s, _ = (
            self.track.raceline.spline.calc_arclength_inaccurate(
                self.poses_x[agent_idx], self.poses_y[agent_idx]
            )
        )

        prog = current_s - self.last_s[agent_idx]

        if current_s < 0.1 * self.track.raceline.spline.s[-1] and self.last_s[agent_idx] > 0.9 * self.track.raceline.spline.s[-1]:
            prog += self.track.raceline.spline.s[-1]
        elif self.last_s[agent_idx] < 0.1 * self.track.raceline.spline.s[-1] and current_s > 0.9 * self.track.raceline.spline.s[-1]:
            prog -= self.track.raceline.spline.s[-1]

        pcnt = prog / self.track.raceline.spline.s[-1]
        prog_reward = pcnt * self.PROGRESS_WEIGHT

        return prog_reward, current_s, pcnt

    def _calculate_milestone_reward(self, pcnt):
        """Calculate reward for reaching milestones."""
        self.total_prog += pcnt
        if self.total_prog > self.milestone:
            self.milestone += self.MILESTONE_INCREMENT
            try:
                self.last_checkpoint_time = self.current_time
                return self.MILESTONE_REWARD
            except:
                print('Error calculating milestone reward')
                raise Exception('div by 0')
        return 0.0

    def _calculate_action_penalties(self, action, agent_idx):
        """Calculate penalties for action changes."""
        penalties = {}
        
        time_increase_factor = self._sigmoid(((self.total_timesteps - self.DELTA_U_CURRICULUM) / self.DECAY_INTERVAL))
        
        steer_delta_pen = self.STEER_ACTION_CHANGE_PENALTY * np.abs(self.last_action[agent_idx, 0] - action[agent_idx, 0]) * time_increase_factor
        penalties['delta_steer'] = steer_delta_pen
        
        vel_delta_pen = self.VEL_ACTION_CHANGE_PENALTY * np.abs(self.last_action[agent_idx, 1] - action[agent_idx, 1]) * time_increase_factor
        penalties['delta_v'] = vel_delta_pen
        
        turn_speed_pen = self.TURN_SPEED_PENALTY * np.abs((action[agent_idx, 0] * action[agent_idx, 1])) * time_increase_factor
        penalties['turning_speed'] = turn_speed_pen
        
        return penalties

    def _calculate_collision_penalty(self, agent_idx):
        """Calculate penalty for collisions."""
        if self.collisions[agent_idx]:
            self.n_crashes += 1
            return self.crash_penalty
        return 0.0

    def _calculate_stagnation_penalty(self, action, agent_idx):
        """Calculate penalty for not moving (stagnation)."""
        if np.abs(action[agent_idx, 1]) < 1e-3:
            return self.crash_penalty  # Make stagnating as bad as crashing
        return 0.0

    def _calculate_velocity_tracking(self, action, agent_idx, current_s):
        """Calculate reward for tracking reference velocity."""
        time_decrease_factor = self._sigmoid(-((self.total_timesteps - self.V_REF_CURRICULUM) / self.DECAY_INTERVAL))
        v_ref = self.vspline(current_s)
        return self.VELOCITY_REWARD_SCALE * np.exp(-(v_ref - action[agent_idx, 1]) ** 2) * time_decrease_factor

    def _get_reward(self, action):
        """
        Get the reward for the current step
        action - np.array (num_agents, 2)
        """
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents

        # Calculate crash penalty that grows over time
        self.crash_penalty = -(1 + (self.MAX_CRASH_PENALTY - 1) * 
                            np.tanh(self.total_timesteps / self.CRASH_CURRICULUM))

        reward = 0.0
        reward_info = {}
        
        for i in range(self.num_agents):
            current_s, _ = (
                self.track.raceline.spline.calc_arclength_inaccurate(
                    self.poses_x[i], self.poses_y[i]
                )
            )
            self.last_s[i] = current_s
        
        i = self.ego_idx
        
        prog_reward, current_s, pcnt = self._calculate_progress_reward(i)
        reward += prog_reward
        reward_info['custom/reward_terms/prog'] = prog_reward
        
        milestone_reward = self._calculate_milestone_reward(pcnt)
        reward += milestone_reward
        reward_info['custom/reward_terms/milestone'] = milestone_reward
        
        action_penalties = self._calculate_action_penalties(action, i)
        for penalty_key, penalty_value in action_penalties.items():
            reward += penalty_value
            reward_info[f'custom/reward_terms/{penalty_key}'] = penalty_value
        
        collision_penalty = self._calculate_collision_penalty(i)
        reward += collision_penalty
        reward_info['custom/reward_terms/collision'] = collision_penalty
        
        vel_tracking_reward = self._calculate_velocity_tracking(action, i, current_s)
        reward += vel_tracking_reward
        reward_info['custom/reward_terms/vel_tracking'] = vel_tracking_reward
        
        stagnation_penalty = self._calculate_stagnation_penalty(action, i)
        reward += stagnation_penalty
        reward_info['custom/reward_terms/stagnation'] = stagnation_penalty
        
        reward_info['custom/reward_terms/total_timestep_reward'] = reward
        
        return reward, reward_info
    
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
    def __init__(self, env: F110Multi, opponents: Optional[List[OpponentDriver]], agent_idx: int = 0):
        """Interface for a single-agent policy to interact with F110Multi"""
        super().__init__(env)
        self.env = env
        self.opponents = opponents
        self.opponent = np.random.choice(opponents) if opponents else None
        self.agent_idx = agent_idx

        self.ego_idx, self.opp_idx = 0, 1 # hardcoded harmlessly

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 2),
            dtype=np.float32,
        )
        self.action_range = np.array([self.env.unwrapped.params['s_max'], self.env.unwrapped.params['v_max']])

        large_num = 1e30
        odom_length = 8
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5


        self.observation_type = self.env.unwrapped.config['observation_config']['type']
        if self.observation_type == 'lidar_conv':
            self.observation_space = gym.spaces.Dict({
                'scan': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
                'odometry': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, odom_length), dtype=np.float32),
            })
        elif self.observation_type == 'mlp':
            self.observation_space = gym.spaces.Box(
                low=-large_num, high=large_num, shape=(1, scan_size + odom_length), dtype=np.float32
            )
        elif self.observation_type == 'frenet_marl':
            self.observation_space = gym.spaces.Dict({
                'scan': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
                'pose': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
                'vel': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
                'heading': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 2), dtype=np.float32),
            })

    def reset(self, opponent_idx=None, seed=None, options=None) -> Tuple[dict, dict]:
        if self.opponents:
            self.opponent = self.opponents[opponent_idx] if opponent_idx else np.random.choice(self.opponents)
        
        obs, info = self.env.reset(seed=seed, options=options)
        return self._ego_observe(obs, self.ego_idx), info
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        obs = self.env.unwrapped.observation_type.observe()
        opp_obs = self._ego_observe(obs, self.opp_idx)
        opponent_action = self.opponent(opp_obs) if self.opponent else np.zeros(2)

        action_full = np.zeros((2, 2), dtype=np.float32)
        action_full[0, :] = action
        action_full[1, :] = opponent_action

        obs, reward, done, truncated, info = self.env.step(action_full)
        obs = self._ego_observe(obs, self.ego_idx)
        return obs, reward, done, truncated, info

    def _ego_observe(self, obs: dict, idx: int) -> dict:
        if self.observation_type == 'lidar_conv':
            return {
                'scan': obs['scan'][idx:idx+1],
                'odometry': obs['odometry'][idx:idx+1],
            }
        elif self.observation_type == 'mlp':
            return obs[idx:idx+1]
        elif self.observation_type == 'frenet_marl':
            return {
                'scan': obs['scan'][idx:idx+1],
                'pose': obs['pose'][idx:idx+1],
                'vel': obs['vel'][idx:idx+1],
                'heading': obs['heading'][idx:idx+1],
            }
        else:
            return obs