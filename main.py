import math
from utils import (
    fleet_speed, travel_time, dist2, planet_pos,
    predict_pos, angle_to, path_hits_sun, comet_future_pos
)

P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = 0, 1, 2, 3, 4, 5, 6
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 0, 1, 2, 3, 4, 5, 6
NEUTRAL  = -1
GARRISON = 5   # minimum ships to keep on any planet


# Fleet size per phase — from 38-game replay analysis
def scaled_min(turn):
    if turn <= 20:  return 17
    if turn <= 50:  return 31
    if turn <= 100: return 43
    if turn <= 200: return 73
    if turn <= 300: return 86
    return 126


class Agent:
    def __init__(self):
        self._turn            = 0
        self._initial_planets = {}
        self._angular_velocity = 0.0

    def __call__(self, obs):
        if self._turn == 0:
            self._angular_velocity = obs.get('angular_velocity', 0.0)
            for p in obs.get('initial_planets', []):
                self._initial_planets[p[P_ID]] = p

        turn  = self._turn
        self._turn += 1

        my_id        = obs.get('player', 0)
        planets      = {p[P_ID]: p for p in obs.get('planets', [])}
        fleets       = obs.get('fleets', [])
        comet_ids    = set(obs.get('comet_planet_ids', []))
        comet_groups = obs.get('comets', [])

        actions      = []
        used_sources = set()
        used_targets = set()

        # Planets already being targeted by our in-transit fleets — don't double-send
        for f in fleets:
            if f[F_OWNER] == my_id:
                tgt = self._guess_target(f, planets, turn)
                if tgt:
                    used_targets.add(tgt)

        # --- DEFENSE first ---
        threats = self._detect_threats(planets, fleets, my_id, turn)
        for pid, incoming in threats.items():
            p         = planets[pid]
            shortfall = incoming - p[P_SHIPS]
            if shortfall <= 0:
                continue
            helper = self._find_reinforcer(
                planets, pid, shortfall + GARRISON, my_id, turn, used_sources)
            if helper:
                actions.append(helper)
                used_sources.add(helper[0])

        # --- ATTACK from every planet that has ships to spare ---
        # Sort by most ships first so strongest planets get best targets
        my_planets = sorted(
            [p for p in planets.values()
             if p[P_OWNER] == my_id and p[P_ID] not in used_sources],
            key=lambda p: -p[P_SHIPS]
        )

        candidates = [p for p in planets.values()
                      if p[P_OWNER] != my_id
                      and p[P_ID] not in comet_ids]

        for src in my_planets:
            sx, sy    = self._pos(src, turn)
            available = src[P_SHIPS] - GARRISON
            if available <= 0:
                continue

            # Send 90% of available, but at least the turn-scaled minimum
            # If we can't meet the minimum, send whatever we have (don't skip)
            send = max(int(available * 0.90), min(scaled_min(turn), available))
            if send <= 0:
                continue

            best_score  = -1e9
            best_action = None
            best_tgt_id = None

            for tgt in candidates:
                if tgt[P_ID] in used_targets:
                    continue

                tx, ty = self._predict(tgt, sx, sy, send, turn)
                if path_hits_sun(sx, sy, tx, ty):
                    continue

                d   = dist2(sx, sy, tx, ty)
                eta = travel_time(d, send)

                extra    = tgt[P_PROD] * eta if tgt[P_OWNER] != NEUTRAL else 0
                garrison = tgt[P_SHIPS] + extra

                # Core score — production value, penalise distance and garrison
                score = tgt[P_PROD] * 80 - d * 1.5 - garrison * 2

                # Enemy bonus — matches observed 66% enemy targeting
                if tgt[P_OWNER] != NEUTRAL:
                    score += 120

                # Early game: heavily favour close cheap planets to snowball fast
                if turn <= 50:
                    score -= d * 2.0        # extra distance penalty early
                    score += max(0, 30 - garrison) * 3  # bonus for low garrison

                # Late game: favour high production to maximise ship generation
                if turn >= 200:
                    score += tgt[P_PROD] * 60

                if score > best_score:
                    best_score  = score
                    best_action = [src[P_ID], angle_to(sx, sy, tx, ty), send]
                    best_tgt_id = tgt[P_ID]

            if best_action:
                actions.append(best_action)
                used_sources.add(src[P_ID])
                used_targets.add(best_tgt_id)

        # --- COMETS: grab with small dedicated fleets ---
        for group in comet_groups:
            for cid in group.get('planet_ids', []):
                if cid not in planets or planets[cid][P_OWNER] == my_id:
                    continue
                future = comet_future_pos(group, cid, 3)
                cx, cy = future if future else (planets[cid][P_X], planets[cid][P_Y])
                for src in sorted(
                    [p for p in planets.values()
                     if p[P_OWNER] == my_id and p[P_ID] not in used_sources],
                    key=lambda p: dist2(*self._pos(p, turn), cx, cy)
                ):
                    sx, sy = self._pos(src, turn)
                    send   = planets[cid][P_SHIPS] + 5
                    avail  = src[P_SHIPS] - GARRISON
                    if avail < send or path_hits_sun(sx, sy, cx, cy):
                        continue
                    actions.append([src[P_ID], angle_to(sx, sy, cx, cy), send])
                    used_sources.add(src[P_ID])
                    break

        return actions

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    def _pos(self, p, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return planet_pos(p, ip, self._angular_velocity, turn)

    def _predict(self, p, sx, sy, ships, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return predict_pos(p, ip, self._angular_velocity, turn, ships, sx, sy)

    def _detect_threats(self, planets, fleets, my_id, turn):
        threats = {}
        my_pids = {pid for pid, p in planets.items() if p[P_OWNER] == my_id}
        for f in fleets:
            if f[F_OWNER] == my_id:
                continue
            tgt = self._guess_target(f, planets, turn)
            if tgt not in my_pids:
                continue
            d       = dist2(f[F_X], f[F_Y], *self._pos(planets[tgt], turn))
            eta     = travel_time(d, f[F_SHIPS])
            garrison = planets[tgt][P_SHIPS] + planets[tgt][P_PROD] * eta
            if f[F_SHIPS] > garrison:
                threats[tgt] = threats.get(tgt, 0) + f[F_SHIPS]
        return threats

    def _guess_target(self, fleet, planets, turn):
        fx, fy, fa = fleet[F_X], fleet[F_Y], fleet[F_ANGLE]
        best, best_diff = None, math.pi
        for pid, p in planets.items():
            px, py = self._pos(p, turn)
            a    = angle_to(fx, fy, px, py)
            diff = abs((a - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff, best = diff, pid
        return best if best_diff < 0.25 else None

    def _find_reinforcer(self, planets, target_id, needed, my_id, turn, used):
        tgt    = planets[target_id]
        tx, ty = self._pos(tgt, turn)
        best, best_d = None, 1e9
        for pid, p in planets.items():
            if p[P_OWNER] != my_id or pid == target_id or pid in used:
                continue
            avail = p[P_SHIPS] - GARRISON
            if avail < needed:
                continue
            sx, sy = self._pos(p, turn)
            d = dist2(sx, sy, tx, ty)
            if d < best_d and not path_hits_sun(sx, sy, tx, ty):
                best_d = d
                best   = [pid, angle_to(sx, sy, tx, ty), needed]
        return best


_agent = Agent()

def agent(obs):
    return _agent(obs)
