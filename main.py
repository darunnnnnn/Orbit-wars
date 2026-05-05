import math
from utils import (
    fleet_speed, travel_time, dist2, planet_pos,
    predict_pos, angle_to, path_hits_sun, comet_future_pos
)

P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = 0, 1, 2, 3, 4, 5, 6
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 0, 1, 2, 3, 4, 5, 6
NEUTRAL = -1

# ---------------------------------------------------------------
# FLEET SIZE SCALING — from 38-game replay analysis
# avg fleet size per phase: 17 / 31 / 43 / 73 / 86 / 126
# ---------------------------------------------------------------
def scaled_fleet_size(turn):
    if turn <= 20:  return 17
    if turn <= 50:  return 31
    if turn <= 100: return 43
    if turn <= 200: return 73
    if turn <= 300: return 86
    return 126

# ---------------------------------------------------------------
# SEND RATIO — 84% of the time sends 81-100% of available ships
# Multi-launch only when ships >= 378 (avg trigger from data)
# ---------------------------------------------------------------
SEND_RATIO       = 0.90   # send 90% of available ships
MULTI_LAUNCH_MIN = 378    # minimum ships before launching 2+ fleets
GARRISON_MIN     = 5      # always keep at least 5 ships


class Agent:
    def __init__(self):
        self._turn = 0
        self._initial_planets = {}
        self._angular_velocity = 0.0

    def __call__(self, obs):
        if self._turn == 0:
            self._angular_velocity = obs.get('angular_velocity', 0.0)
            for p in obs.get('initial_planets', []):
                self._initial_planets[p[P_ID]] = p

        turn = self._turn
        self._turn += 1

        my_id   = obs.get('player', 0)
        planets = {p[P_ID]: p for p in obs.get('planets', [])}
        fleets  = obs.get('fleets', [])
        comet_ids    = set(obs.get('comet_planet_ids', []))
        comet_groups = obs.get('comets', [])

        my_planets  = [p for p in planets.values() if p[P_OWNER] == my_id]
        my_total    = sum(p[P_SHIPS] for p in my_planets)

        # How many launches allowed this turn
        # Data: 1 fleet 74%, 2 fleets 25%, more rarely — only when ships >= 378
        max_launches = 1
        if my_total >= MULTI_LAUNCH_MIN:
            max_launches = 2
        if my_total >= MULTI_LAUNCH_MIN * 2:
            max_launches = 3

        actions      = []
        used_sources = set()
        used_targets = set()

        # --- DEFENSE: reinforce planets under threat ---
        threats = self._detect_threats(planets, fleets, my_id, turn)
        for pid, incoming in threats.items():
            if len(actions) >= max_launches:
                break
            p = planets[pid]
            shortfall = incoming - p[P_SHIPS]
            if shortfall <= 0:
                continue
            helper = self._find_reinforcer(
                planets, pid, shortfall + GARRISON_MIN, my_id, turn, used_sources)
            if helper:
                actions.append(helper)
                used_sources.add(helper[0])

        # --- ATTACK: score and pick best targets ---
        remaining = max_launches - len(actions)
        if remaining > 0:
            attacks = self._pick_attacks(
                planets, my_id, turn, comet_ids,
                used_sources, used_targets, remaining)
            actions.extend(attacks)

        return actions

    # ----------------------------------------------------------
    # POSITION HELPERS
    # ----------------------------------------------------------
    def _pos(self, p, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return planet_pos(p, ip, self._angular_velocity, turn)

    def _predict(self, p, sx, sy, ships, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return predict_pos(p, ip, self._angular_velocity, turn, ships, sx, sy)

    # ----------------------------------------------------------
    # THREAT DETECTION
    # ----------------------------------------------------------
    def _detect_threats(self, planets, fleets, my_id, turn):
        threats = {}
        my_pids = {pid for pid, p in planets.items() if p[P_OWNER] == my_id}
        for f in fleets:
            if f[F_OWNER] == my_id:
                continue
            tgt = self._guess_fleet_target(f, planets, turn)
            if tgt not in my_pids:
                continue
            d   = dist2(f[F_X], f[F_Y], *self._pos(planets[tgt], turn))
            eta = travel_time(d, f[F_SHIPS])
            garrison = planets[tgt][P_SHIPS] + planets[tgt][P_PROD] * eta
            if f[F_SHIPS] > garrison:
                threats[tgt] = threats.get(tgt, 0) + f[F_SHIPS]
        return threats

    def _guess_fleet_target(self, fleet, planets, turn):
        fx, fy, fa = fleet[F_X], fleet[F_Y], fleet[F_ANGLE]
        best, best_diff = None, math.pi
        for pid, p in planets.items():
            px, py = self._pos(p, turn)
            a    = angle_to(fx, fy, px, py)
            diff = abs((a - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff, best = diff, pid
        return best if best_diff < 0.25 else None

    # ----------------------------------------------------------
    # ATTACK SELECTION
    # Replicates bowwowforeach scoring from data:
    #   - attacks enemy 66% / neutral 34%  → enemy bonus
    #   - uses predicted position 79%
    #   - sends 90% of available ships
    #   - fleet size scaled by turn
    # ----------------------------------------------------------
    def _pick_attacks(self, planets, my_id, turn, comet_ids,
                      used_sources, used_targets, max_count):
        results = []

        # Sort source planets: most ships first (strongest attacks first)
        my_planets = sorted(
            [p for p in planets.values()
             if p[P_OWNER] == my_id and p[P_ID] not in used_sources],
            key=lambda p: -p[P_SHIPS]
        )

        candidates = [p for p in planets.values()
                      if p[P_OWNER] != my_id
                      and p[P_ID] not in used_targets
                      and p[P_ID] not in comet_ids]

        for src in my_planets:
            if len(results) >= max_count:
                break

            sx, sy   = self._pos(src, turn)
            available = src[P_SHIPS] - GARRISON_MIN
            if available <= 0:
                continue

            # How many ships to send: 90% of available, scaled by turn minimum
            send_base = max(int(available * SEND_RATIO), scaled_fleet_size(turn))
            if send_base > available:
                send_base = available
            if send_base <= 0:
                continue

            best_score  = -1e9
            best_action = None
            best_tgt_id = None

            for tgt in candidates:
                if tgt[P_ID] in used_targets:
                    continue

                # Use predicted position (79% of the time in data)
                tx, ty = self._predict(tgt, sx, sy, send_base, turn)
                if path_hits_sun(sx, sy, tx, ty):
                    continue

                d   = dist2(sx, sy, tx, ty)
                eta = travel_time(d, send_base)

                # Garrison at arrival
                extra    = tgt[P_PROD] * eta if tgt[P_OWNER] != NEUTRAL else 0
                garrison = tgt[P_SHIPS] + extra

                # Score — replicates observed behaviour:
                # production matters but not overwhelmingly (30% go for prod-1)
                # enemy preferred 2:1 over neutral
                # closer is better
                score = tgt[P_PROD] * 80
                score -= d * 1.5
                score -= garrison * 2
                if tgt[P_OWNER] != NEUTRAL:
                    score += 120   # enemy bonus — matches 66% enemy targeting

                if score > best_score:
                    best_score  = score
                    best_action = [src[P_ID], angle_to(sx, sy, tx, ty), send_base]
                    best_tgt_id = tgt[P_ID]

            if best_action:
                results.append(best_action)
                used_sources.add(src[P_ID])
                used_targets.add(best_tgt_id)

        return results

    # ----------------------------------------------------------
    # REINFORCEMENT
    # ----------------------------------------------------------
    def _find_reinforcer(self, planets, target_id, needed,
                         my_id, turn, used):
        tgt  = planets[target_id]
        tx, ty = self._pos(tgt, turn)
        best, best_d = None, 1e9
        for pid, p in planets.items():
            if p[P_OWNER] != my_id or pid == target_id or pid in used:
                continue
            avail = p[P_SHIPS] - GARRISON_MIN
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
