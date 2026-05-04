import math

SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
BOARD = 100.0
ROTATION_RADIUS_LIMIT = 50.0  # orbital_radius + planet_radius < 50 => orbiting


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    return min(1.0 + 5.0 * (math.log(ships) / math.log(1000)) ** 1.5, 6.0)


def travel_time(d, ships):
    return math.ceil(d / fleet_speed(ships))


def dist2(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_orbiting(p):
    """p is a Planet namedtuple or list: [id,owner,x,y,radius,ships,production]."""
    orbital_radius = dist2(p[2], p[3], SUN_X, SUN_Y)
    return orbital_radius + p[4] < ROTATION_RADIUS_LIMIT


def planet_pos(p, initial_p, angular_velocity, turn):
    """
    Current position of planet p at given turn.
    initial_p: same planet's data from initial_planets (has original x,y).
    angular_velocity: from obs, radians/turn.
    """
    if not is_orbiting(initial_p):
        return p[2], p[3]  # static — use current x,y (they don't change)
    # Rotate initial position around sun by angular_velocity * turn
    ix, iy = initial_p[2], initial_p[3]
    angle = angular_velocity * turn
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rx, ry = ix - SUN_X, iy - SUN_Y
    x = SUN_X + rx * cos_a - ry * sin_a
    y = SUN_Y + rx * sin_a + ry * cos_a
    return x, y


def predict_pos(p, initial_p, angular_velocity, current_turn, ships, src_x, src_y):
    """Iteratively converge on where planet will be when fleet arrives."""
    if not is_orbiting(initial_p):
        return p[2], p[3]
    px, py = planet_pos(p, initial_p, angular_velocity, current_turn)
    for _ in range(6):
        d = dist2(src_x, src_y, px, py)
        eta = travel_time(d, ships)
        px, py = planet_pos(p, initial_p, angular_velocity, current_turn + eta)
    return px, py


def angle_to(sx, sy, dx, dy):
    return math.atan2(dy - sy, dx - sx)


def path_hits_sun(sx, sy, dx, dy):
    """True if segment from (sx,sy) to (dx,dy) passes within SUN_RADIUS of sun."""
    ex, ey = dx - sx, dy - sy
    fx, fy = SUN_X - sx, SUN_Y - sy
    t = (fx * ex + fy * ey) / (ex * ex + ey * ey + 1e-9)
    t = max(0.0, min(1.0, t))
    cx = sx + t * ex - SUN_X
    cy = sy + t * ey - SUN_Y
    return math.hypot(cx, cy) < SUN_RADIUS


def comet_future_pos(comet_group, planet_id, steps_ahead):
    """
    comet_group: one entry from obs['comets'] with keys planet_ids, paths, path_index.
    Returns (x, y) of the comet at path_index + steps_ahead, or None if off path.
    """
    try:
        idx = comet_group["planet_ids"].index(planet_id)
        path = comet_group["paths"][idx]
        future_idx = comet_group["path_index"] + steps_ahead
        if future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
    except (ValueError, IndexError):
        pass
    return None
