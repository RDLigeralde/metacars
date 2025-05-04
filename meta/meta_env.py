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
        config: Dict[str, Any],
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """F110Env with adversarial reward structure"""
        super().__init__(config=config, render_mode=render_mode)
        self.config = config
        self.render_mode = render_mode
        self._init_rews()

        self.norm_input = config.get('normalize_input', False)
        if self.norm_input:
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_agents, 2),
                dtype=np.float32,
            )
            self.action_range = np.array([
                config['params']['s_max'],
                config['params']['v_max'],
            ])

        # to reward following raceline
        self.raceline, self.centerline = self._set_lines(config['map'])
        self.vspline = self._calc_vspline()
        self.yspline = self._calc_yspline()

        # to reward stable driving
        self.last_action = np.zeros((self.num_agents, 2), dtype=np.float32)

        # progress tracking
        self.stag_count = 0
        self.total_prog = 0
        self.total_timesteps = 0

        self.timeouts = 0
        self.crashes = 0
        self.last_progress = 0.0
        self.laps = 0
        self.last_ckpt_time = 0.0

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Overriden for action-dependent rewards and WandB logging"""
        if self.norm_input:
            action *= self.action_range

        self.sim.step(action)
        obs = self.observation_type.observe()
        self.current_time += self.timestep
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

        timeout = self.current_time >= 60 # TODO: make this configurable
        self.timeouts += int(timeout)
        done = done or timeout

        self.last_progress = self.total_prog if done else self.last_progress
        info['custom/timeouts'] = self.timeouts
        info['custom/last_progress'] = self.last_progress
        info['custom/crashes'] = self.crashes
        info['custom/laps'] = self.laps
        info.update(reward_info)

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.laps += int(self.total_prog)
        obs, info = super().reset(seed=seed, options=options)
        
        self.last_action = np.zeros((self.num_agents, 2))
        self.stag_count = 0
        self.total_prog = 0
        self.total_timesteps = 0
        self.crashes = 0
        self.last_progress = 0.0
        self.last_ckpt_time = 0.0
        
        obs, _, _, _, info = self.step(self.last_action)
        
        return obs, info

    def _get_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Reward function with individual terms and overtake rewards"""
        
        # Update crash penalty based on curriculum
        self._update_crash_penalty()
        
        # Initialize tracking variables if needed
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        
        reward_info = {}
        
        i = self.ego_idx if hasattr(self, 'ego_idx') else 0
        
        current_s_ego, _ = self.track.centerline.spline.calc_arclength_inaccurate(
            self.poses_x[i], self.poses_y[i]
        )
        
        prog_reward, pcnt = self._get_progress_reward(i, current_s_ego)
        milestone_reward = self._get_milestone_reward()
        steer_penalty = self._get_steering_change_penalty(i, action)
        vel_penalty = self._get_velocity_change_penalty(i, action)
        turn_speed_penalty = self._get_turn_speed_penalty(i, action)
        collision_penalty = self._get_collision_penalty(i)
        stagnation_penalty = self._get_stagnation_penalty(i, action) 
        overtake_reward = self._get_overtake_reward(i, current_s_ego)
              
        
        total_reward = (
            prog_reward +
            milestone_reward +
            steer_penalty +
            vel_penalty +
            turn_speed_penalty +
            collision_penalty +
            stagnation_penalty +
            overtake_reward
        )
        
        reward_info['custom/reward_terms/prog'] = prog_reward
        reward_info['custom/reward_terms/milestone'] = milestone_reward
        reward_info['custom/reward_terms/delta_steer'] = steer_penalty
        reward_info['custom/reward_terms/delta_v'] = vel_penalty
        reward_info['custom/reward_terms/turning_speed'] = turn_speed_penalty
        reward_info['custom/reward_terms/collision'] = collision_penalty
        reward_info['custom/reward_terms/stagnation'] = stagnation_penalty
        reward_info['custom/reward_terms/overtake'] = overtake_reward
        reward_info['custom/reward_terms/total_timestep_reward'] = total_reward
        
        # Update state for next iteration
        self.last_s[i] = current_s_ego
        
        return total_reward, reward_info

    def _get_overtake_reward(self, ego_idx, current_s_ego):
        if not hasattr(self, "last_s_opponent"):
            self.last_s_opponent = None
        
        overtake_reward = 0.0 # implicitly skipped for one agent case
        opp_idx = 1 - ego_idx if self.num_agents == 2 else None
        
        if opp_idx is not None:
            current_s_opp, _ = self.track.centerline.spline.calc_arclength_inaccurate(
                self.poses_x[opp_idx], self.poses_y[opp_idx]
            )
            
            if self.last_s_opponent is None:
                self.last_s_opponent = current_s_opp
                return 0.0
            
            last_s_opp = self.last_s_opponent
            
            if self.last_s[ego_idx] < last_s_opp and current_s_ego > current_s_opp:
                if not (current_s_ego < 0.1 * self.track.centerline.spline.s[-1] and 
                        last_s_opp > 0.9 * self.track.centerline.spline.s[-1]):
                    overtake_reward = self.overtake_reward
            
            elif self.last_s[ego_idx] > last_s_opp and current_s_ego < current_s_opp:
                if not (last_s_opp < 0.1 * self.track.centerline.spline.s[-1] and 
                        current_s_ego > 0.9 * self.track.centerline.spline.s[-1]):
                    overtake_reward = -self.overtake_reward
            
            self.last_s_opponent = current_s_opp
        
        return overtake_reward

    def _sigmoid(self, x):
        """Helper function for smooth transitions"""
        if x < -1e3:
            return 0
        if x > 1e3:
            return 1
        return 1.0 / (1.0 + np.exp(-x))

    def _get_progress_reward(self, i, current_s):
        """Calculate progress reward for agent i"""
        if not hasattr(self, "last_s"):
            self.last_s = [0.0] * self.num_agents
        
        prog = current_s - self.last_s[i]
        
        # Account for lapping
        if current_s < 0.1 * self.track.centerline.spline.s[-1] and self.last_s[i] > 0.9 * self.track.centerline.spline.s[-1]:
            prog += self.track.centerline.spline.s[-1]
        # Looped backward
        elif self.last_s[i] < 0.1 * self.track.centerline.spline.s[-1] and current_s > 0.9 * self.track.centerline.spline.s[-1]:
            prog -= self.track.centerline.spline.s[-1]
        
        pcnt = prog / self.track.centerline.spline.s[-1]
        prog_reward = pcnt * self.progress_weight
        
        # Update total progress
        self.total_prog += pcnt
        
        return prog_reward, pcnt

    def _get_milestone_reward(self):
        """Calculate milestone reward if threshold is passed"""
        if self.total_prog > self.milestone:
            self.milestone += self.milestone_increment
            try:
                milestone_reward = self.milestone_reward
                self.last_checkpoint_time = self.current_time
                return milestone_reward
            except:
                print(f'Error in milestone reward calculation')
                raise
        else:
            return 0.0

    def _get_steering_change_penalty(self, i, action):
        """Calculate penalty for steering action changes"""
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.delta_u_curriculum) / self.decay_interval)
        )
        steer_delta_pen = self.steer_action_change_penalty * \
                        np.abs(self.last_action[i, 0] - action[i, 0]) * \
                        time_increase_factor
        return steer_delta_pen

    def _get_velocity_change_penalty(self, i, action):
        """Calculate penalty for velocity action changes"""
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.delta_u_curriculum) / self.decay_interval)
        )
        vel_delta_pen = self.vel_action_change_penalty * \
                        np.abs(self.last_action[i, 1] - action[i, 1]) * \
                        time_increase_factor
        return vel_delta_pen

    def _get_turn_speed_penalty(self, i, action):
        """Calculate penalty for turning at high speeds"""
        time_increase_factor = self._sigmoid(
            ((self.total_timesteps - self.delta_u_curriculum) / self.decay_interval)
        )
        turn_speed_pen = self.turn_speed_penalty * \
                        np.abs((action[i, 0] * action[i, 1])) * \
                        time_increase_factor
        return turn_speed_pen

    def _get_collision_penalty(self, i):
        """Calculate collision penalty for agent i"""
        if self.collisions[i]:
            self.crashes += 1 if hasattr(self, 'crashes') else setattr(self, 'crashes', 1)
            return self.crash_penalty
        else:
            return 0.0

    def _get_stagnation_penalty(self, i, action):
        """Calculate stagnation penalty for low velocity"""
        if np.abs(action[i, 1]) < 1e-3:
            return self.crash_penalty  # Same magnitude as crash penalty
        else:
            return 0.0

    def _update_crash_penalty(self):
        """Update the crash penalty value based on curriculum"""
        self.crash_penalty = -(1 + (self.max_crash_penalty - 1) * 
                            np.tanh(self.total_timesteps / self.crash_curriculum))

    def _init_rews(self):
        for k, v in self.config['reward'].items():
            setattr(self, k, v)

    def _set_lines(self, track: str) -> Tuple[np.ndarray, np.ndarray]:
        track_dir = find_track_dir(track)
        raceline_file = os.path.join(track_dir, f'{track}_raceline.csv')
        centerline_file = os.path.join(track_dir, f'{track}_centerline.csv')
        return np.loadtxt(raceline_file, delimiter=';').astype(np.float32), np.loadtxt(centerline_file, delimiter=',').astype(np.float32)

    def _calc_vspline(self) -> CubicSpline:
        svs = np.zeros((len(self.raceline), 2), dtype=np.float32)
        for i in range(len(self.raceline)):
            x, y = self.raceline[i, 1], self.raceline[i, 2]
            svs[i] = np.array([
                self.track.centerline.spline.calc_arclength_inaccurate(x, y)[0],
                self.raceline[i, 5]
            ])
        svs = svs[svs[:, 0].argsort()][:-1]

        s_diff = np.diff(svs[:, 0])
        mask = np.ones(len(svs), dtype=bool)
        mask[1:] = s_diff > 0.0

        masked = svs[mask]
        return CubicSpline(masked[:, 0], masked[:, 1])

    def _calc_yspline(self) -> CubicSpline:
        ys = np.zeros((len(self.raceline), 2), dtype=np.float32)
        ys[:, 0] = self.raceline[:, 0] # s
        ys[:, 1] = self.raceline[:, 3] # yaw
        ys = ys[ys[:, 0].argsort()][:-1]
        return CubicSpline(ys[:, 0], ys[:, 1])

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
        # self.laps += int(np.all(self.toggle_list >= 4)) # this is wrong becuse it counts collisions too (not sure about this comment)

        return bool(done), self.toggle_list >= 4
         
class F110MultiView(gym.Wrapper):
    """Single-agent interface for F110Multi environment"""
    
    def __init__(
        self, 
        env: F110Multi, 
        opponents: 
        Optional[List[OpponentDriver]], 
    ):
        super().__init__(env)
        self.env = env
        self.opponents = opponents
        self.opponent = np.random.choice(opponents) if opponents else None
        self.ego_idx, self.opp_idx = 0, 1
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1, 2),
            dtype=np.float32,
        )
        
        self.observation_type = self.env.unwrapped.config.get('observation_config', {}).get('type', 'frenet_rl')
        self._setup_observation_space()
    
    def _setup_observation_space(self):
        """Configure observation space based on observation type"""
        large_num = 1e30
        odom_length = 8
        scan_size = self.env.unwrapped.sim.agents[0].scan_simulator.num_beams
        scan_range = self.env.unwrapped.sim.agents[0].scan_simulator.max_range + 0.5
        
        if self.observation_type == 'lidar_conv':
            self.observation_space = gym.spaces.Dict({
                'scan': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
                'odometry': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, odom_length), dtype=np.float32),
            })
        elif self.observation_type == 'mlp':
            self.observation_space = gym.spaces.Box(
                low=-large_num, high=large_num, shape=(1, scan_size + odom_length), dtype=np.float32
            )
        elif self.observation_type == 'frenet_rl' or self.observation_type == 'frenet_marl':
            self.observation_space = gym.spaces.Dict({
                'scan': gym.spaces.Box(low=0, high=scan_range, shape=(1, scan_size), dtype=np.float32),
                'pose': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
                'vel': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 3), dtype=np.float32),
                'heading': gym.spaces.Box(low=-large_num, high=large_num, shape=(1, 2), dtype=np.float32),
            })
    
    def reset(self, opponent_idx=None, seed=None, options=None) -> Tuple[dict, dict]:
        """Reset environment and select opponent"""
        if self.opponents:
            self.opponent = self.opponents[opponent_idx] if opponent_idx is not None else np.random.choice(self.opponents)
        
        obs, info = self.env.reset(seed=seed, options=options)
        return self._ego_observe(obs, self.ego_idx), info
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Step environment with ego action and opponent policy"""
        obs = self.env.unwrapped.observation_type.observe()
        
        opp_obs = self._ego_observe(obs, self.opp_idx)
        opponent_action = self.opponent(opp_obs) if self.opponent else np.zeros(2)
        
        action_full = np.zeros((2, 2), dtype=np.float32)
        action_full[self.ego_idx, :] = action.squeeze()
        action_full[self.opp_idx, :] = opponent_action
        
        # Step environment
        obs, reward, done, truncated, info = self.env.step(action_full)
        obs = self._ego_observe(obs, self.ego_idx)
        
        return obs, reward, done, truncated, info
    
    def _ego_observe(self, obs: dict, idx: int) -> dict:
        """Extract single agent observation from multi-agent observation"""
        if self.observation_type == 'lidar_conv':
            return {
                'scan': obs['scan'][idx:idx+1],
                'odometry': obs['odometry'][idx:idx+1],
            }
        elif self.observation_type == 'mlp':
            return obs[idx:idx+1]
        elif self.observation_type == 'frenet_rl' or self.observation_type == 'frenet_marl':
            return {
                'scan': obs['scan'][idx:idx+1],
                'pose': obs['pose'][idx:idx+1],
                'vel': obs['vel'][idx:idx+1],
                'heading': obs['heading'][idx:idx+1],
            }
        else:
            return obs