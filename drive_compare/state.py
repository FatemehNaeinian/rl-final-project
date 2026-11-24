"""Single source of truth for what every controller is allowed to know.

`extract_world_state(env, route)` reads the simulator once per step and returns a `WorldState`:
  * ego:     x, y, yaw, speed, previous steering command, size
  * traffic: every other vehicle within SENSE_RADIUS (position, yaw, speed, velocity vector, size)
  * route:   a dense polyline from the start to the destination, plus the ego's progress along it

drive_compare.obs also encodes it as a fixed-size vector; the MPC reads the WorldState directly.
Both are derived from the same WorldState, so all methods get the same information.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SENSE_RADIUS = 50.0  # m; vehicles farther than this are invisible to every controller
ROUTE_SPACING = 0.5  # m between route polyline points


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Route:
    """Reference path through the intersection, sampled every ROUTE_SPACING metres.

    points: (N, 3) array of [x, y, yaw] along the lane centreline.
    s:      (N,) arc length of each point from the route start.
    """
    points: np.ndarray
    s: np.ndarray
    lane_width: float
    turn: str  # "left" | "straight" | "right" (from the total heading change), for stratifying results
    # drivable band around the centreline (m, + = left): the solid line left of lane 0 to the road edge right
    # of the last lane. Touching either ends the episode as out_of_road. Account for the car's half-width!
    lat_min: float = -1.75
    lat_max: float = 1.75
    _last_idx: int = 0

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, xy) -> tuple[float, float, float, int]:
        """Return (progress s, signed lateral offset, route yaw, index) of the closest point.

        The search is restricted to a window ahead of the last match, so the projection cannot jump to a
        different branch where the route passes close to itself.
        """
        lo = max(self._last_idx - 20, 0)
        hi = min(self._last_idx + 200, len(self.points))
        d = np.linalg.norm(self.points[lo:hi, :2] - np.asarray(xy)[None], axis=1)
        i = lo + int(np.argmin(d))
        self._last_idx = i
        px, py, pyaw = self.points[i]
        dx, dy = xy[0] - px, xy[1] - py
        lateral = -np.sin(pyaw) * dx + np.cos(pyaw) * dy  # positive = left of the route
        return float(self.s[i]), float(lateral), float(pyaw), i

    def lookahead(self, s_from: float, distances) -> np.ndarray:
        """Route points [x, y, yaw] at arc lengths s_from + distances (clamped to the route end).

        Useful for building an MPC reference trajectory, e.g. distances = v_ref * dt * arange(H).
        """
        query = np.clip(s_from + np.asarray(distances, dtype=float), 0.0, self.s[-1])
        idx = np.clip(np.searchsorted(self.s, query), 0, len(self.s) - 1)
        return self.points[idx]

    def reset_tracking(self):
        self._last_idx = 0


@dataclass
class WorldState:
    t: float  # seconds since episode start
    # ego
    x: float
    y: float
    yaw: float
    v: float
    steer_prev: float  # previous normalized steering command (acts with a one-step delay, see vehicle_model)
    length: float
    width: float
    # route-relative ego state
    s: float  # progress along route (m)
    lateral: float  # signed offset from the route centreline (m, + = left)
    heading_error: float  # ego yaw - route yaw (rad)
    route: Route
    # traffic: arrays of shape (N,) / (N, 2); N may be 0
    traffic_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    traffic_vel: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    traffic_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    traffic_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    traffic_length: np.ndarray = field(default_factory=lambda: np.zeros(0))
    traffic_width: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def ego_state(self) -> np.ndarray:
        """State vector used by drive_compare.vehicle_model: [x, y, yaw, v, steer_prev]."""
        return np.array([self.x, self.y, self.yaw, self.v, self.steer_prev])

    @property
    def num_traffic(self) -> int:
        return len(self.traffic_xy)

    @property
    def remaining(self) -> float:
        return self.route.length - self.s


def build_route(env) -> Route:
    """Centreline of the ego's starting lane index, followed through every road on the planned route.

    MetaDrive counts arrival on any lane of the final road, so keeping the starting lane index is a valid
    route; controllers are still free to deviate from it (e.g. to avoid a stopped car).
    """
    vehicle = env.agent
    nav = vehicle.navigation
    graph = env.current_map.road_network.graph
    lane_idx = vehicle.lane_index[2]
    pts = []
    widths = []
    num_lanes = min(len(graph[a][b]) for a, b in zip(nav.checkpoints[:-1], nav.checkpoints[1:]))
    for a, b in zip(nav.checkpoints[:-1], nav.checkpoints[1:]):
        lanes = graph[a][b]
        lane = lanes[min(lane_idx, len(lanes) - 1)]
        widths.append(lane.width)
        n = max(int(np.ceil(lane.length / ROUTE_SPACING)), 1)
        for s in np.linspace(0.0, lane.length, n, endpoint=False):
            x, y = lane.position(s, 0.0)
            pts.append((x, y, lane.heading_theta_at(s)))
    x, y = lane.position(lane.length, 0.0)
    pts.append((x, y, lane.heading_theta_at(lane.length)))
    points = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    dyaw = wrap_angle(points[-1, 2] - points[0, 2])
    turn = "straight" if abs(dyaw) < np.pi / 4 else ("left" if dyaw > 0 else "right")
    lw = float(np.mean(widths))
    i = min(lane_idx, num_lanes - 1)
    # lane 0 is leftmost; higher indices are further right (negative lateral)
    lat_max = lw / 2 + i * lw
    lat_min = -(lw / 2 + (num_lanes - 1 - i) * lw)
    return Route(points=points, s=s, lane_width=lw, turn=turn, lat_min=lat_min, lat_max=lat_max)


def _other_vehicles(env):
    from metadrive.component.vehicle.base_vehicle import BaseVehicle
    ego = env.agent
    return [o for o in env.engine.get_objects().values() if isinstance(o, BaseVehicle) and o is not ego]


def extract_world_state(env, route: Route, steer_prev: float) -> WorldState:
    ego = env.agent
    x, y = ego.position
    s, lateral, route_yaw, _ = route.project((x, y))
    xy, vel, yaw, spd, ln, wd = [], [], [], [], [], []
    for o in _other_vehicles(env):
        p = np.asarray(o.position, dtype=float)
        if np.hypot(p[0] - x, p[1] - y) > SENSE_RADIUS:
            continue
        xy.append(p)
        vel.append(np.asarray(o.velocity, dtype=float))
        yaw.append(o.heading_theta)
        spd.append(o.speed)
        ln.append(o.LENGTH)
        wd.append(o.WIDTH)
    return WorldState(
        t=env.episode_step * env.config["physics_world_step_size"] * env.config["decision_repeat"],
        x=float(x),
        y=float(y),
        yaw=float(ego.heading_theta),
        v=float(ego.speed),
        steer_prev=float(steer_prev),
        length=float(ego.LENGTH),
        width=float(ego.WIDTH),
        s=s,
        lateral=lateral,
        heading_error=float(wrap_angle(ego.heading_theta - route_yaw)),
        route=route,
        traffic_xy=np.asarray(xy, dtype=float).reshape(-1, 2),
        traffic_vel=np.asarray(vel, dtype=float).reshape(-1, 2),
        traffic_yaw=np.asarray(yaw, dtype=float),
        traffic_speed=np.asarray(spd, dtype=float),
        traffic_length=np.asarray(ln, dtype=float),
        traffic_width=np.asarray(wd, dtype=float),
    )
