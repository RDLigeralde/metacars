from f1tenth_gym.envs.track.utils import find_track_dir
from meta.opponents.opponent import OpponentDriver
from ppo.utils import get_cfg_dicts
import gymnasium as gym

from numba import njit
import numpy as np

@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

# @njit(fastmath=False, cache=True)
# TODO: njit here gives type error right now, fix later
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    Find the first point on the trajectory that intersects with a circle
    centered at point with radius radius.
    
    Note: All input arrays must have the same dtype (float32 or float64)
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    point = point.astype(trajectory.dtype)  # Ensure point has same dtype as trajectory
    
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
            
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
            
    # wrap around to the beginning of the trajectory if no intersection is found
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, action_range):
    """
    Returns actuation
    """
    # Ensure consistent dtype
    position = position.astype(np.float32)
    pose_theta = np.float32(pose_theta)
    
    waypoint_y = np.dot(
        np.array([np.sin(-pose_theta), np.cos(-pose_theta)], dtype=np.float32), 
        lookahead_point[0:2].astype(np.float32) - position
    )
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle / action_range[0], speed / action_range[1] # Use environment's action range for normalization

class PurePursuit(OpponentDriver):
    def __init__(self, conf):
        self.conf = conf
        self.wheelbase = np.float32(conf['wheelbase'])
        self.lookahead_distance = np.float32(conf['lookahead'])
        self.max_reacquire = np.float32(conf['max_reacquire'])
        self.action_range = np.array([conf['s_max'], conf['v_max']], dtype=np.float32)

        track = conf['waypoints']
        waypoints_dir = find_track_dir(track)
        waypoints_file = f'{waypoints_dir}/{track}_raceline_pp.csv'
        self.waypoints = self._load_waypoints(waypoints_file)

    def _load_waypoints(self, waypoints_file):
        waypoints = np.loadtxt(waypoints_file, delimiter=';').astype(np.float32)
        return waypoints

    def _get_current_waypoint(self, position):
        """
        gets the current waypoint to follow
        """
        # Ensure position is float32 to match waypoints
        position = position.astype(np.float32)
        
        wpts = self.waypoints[:, 1:3]
        _, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < self.lookahead_distance:
            _, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, self.lookahead_distance, wpts, i+t, wrap=True
            )
            if i2 is None:
                return None
            current_waypoint = np.empty((3, ), dtype=np.float32)
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, -2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, -2]).astype(np.float32)
        else:
            return None
        
    def _call_single(self, pos, **args):
        """Actuates using observation space"""
        # Ensure pos is float32 for consistent dtype
        pos = pos.astype(np.float32)
        
        lookahead = self._get_current_waypoint(pos[:2])
        if lookahead is None:
            # Return safe values if no waypoint found
            return np.array([0.0, 0.3], dtype=np.float32)
    
        return get_actuation(
            pos[2], lookahead, pos[0:2], self.lookahead_distance, self.wheelbase, self.action_range
        )

    def __call__(self, obs):
        positions = obs['pose']
        B = positions.shape[0]
        controls = np.zeros((B, 2), dtype=np.float32)
        for i in range(B):
            controls[i] = self._call_single(positions[i])
        controls[:, 1] *= self.conf['synthbrake']
        return controls
    
def main():
    gym.register(
        id="chainer",
        entry_point="ppo.rl_env:F110Multi",  # Updated to F110Multi
    )
    synthbrake = 0.35
    
    conf = {
        'wheelbase': 0.5,
        'lookahead': 0.4,
        'max_reacquire': 1.0,
        'synthbrake': synthbrake,
    }
    config = '/Users/robbyligeralde/metacars/ppo/exp_cfgs/clean.yml'
    env_args, *_ = get_cfg_dicts(config)
    
    # Get action ranges from environment parameters
    conf['s_max'] = env_args['params']['s_max']
    conf['v_max'] = env_args['params']['v_max'] # Apply slowdown to max speed
    conf['waypoints'] = env_args['map']  # Add track name for waypoints
    
    pp = PurePursuit(conf)
    env = gym.make(
        'chainer', 
        config=env_args, 
        render_mode='human'
    )

    obs, _ = env.reset()
    done = False
    while not done:
        action = pp(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()

if __name__ == "__main__":
    main()