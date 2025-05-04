from numba import njit
import numpy as np

#@njit(fastmath=False, cache=True)
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

#@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
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

#@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuit:
    def __init__(self, conf):
        self.conf = conf
        self.wheelbase = conf['wheelbase']
        self.lookahead_distance = np.float32(conf['lookahead'])
        self.max_reacquire = conf['max_reacquire']

        waypoints_file = f'../../maps/{conf["waypoints"]}/{conf["waypoints"]}_raceline.csv'
        self.waypoints = self._load_waypoints(waypoints_file)

    def _load_waypoints(self, waypoints_file):
        waypoints = np.loadtxt(waypoints_file, delimiter=';').astype(np.float64)
        return waypoints

    def _get_current_waypoint(self, position):
        """
        gets the current waypoint to follow
        """
        wpts = self.waypoints[:, 1:3]
        _, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < self.lookahead_distance:
            _, i2, _ = first_point_on_trajectory_intersecting_circle(
                position, self.lookahead_distance, wpts, i+t, wrap=True
            )
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, -2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, -2])
        else:
            return None
        
    def _call_single(self, pos, **args):
        """Actuates using observation space"""
        lookahead = self._get_current_waypoint(pos[:2])

        if lookahead is None:
            return 4.0, 0.0
    
        return get_actuation(
            pos[2], lookahead, pos[0:2], self.lookahead_distance, self.wheelbase
        )

    def __call__(self, obs, states, done):
        positions = obs['odometry'][...,:3]
        B = positions.shape[0]
        controls = np.zeros((B, 2), dtype=np.float32)
        for i in range(B):
            controls[i] = self._call_single(positions[i])
        return controls, states
    
def main():
    conf = {
        'wheelbase': 0.5,
        'lookahead': 0.5,
        'max_reacquire': 1.0,
        'waypoints': 'race3'
    }
    pp = PurePursuit(conf)
    obs = {'odometry': np.array([[0, 0, 0], [1, 1, np.pi/4]])}
    states = None
    done = None
    controls, states = pp(obs, states, done)
    print(controls)

if __name__ == "__main__":
    main()