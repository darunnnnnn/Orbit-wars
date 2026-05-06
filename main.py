import math
from utils import (
    fleet_speed, travel_time, dist2, planet_pos,
    predict_pos, angle_to, path_hits_sun, comet_future_pos
)

P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = 0, 1, 2, 3, 4, 5, 6
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 0, 1, 2, 3, 4, 5, 6
NEUTRAL = -1

# Fleet size scaling — exact from 38-game replay data
def scaled_fleet_size(turn):
    if turn <= 20:  return 17
    if turn <= 50:  return 31
    if turn <= 100: return 43
    if turn <= 200: return 73
    if turn <= 300: return 86
    return 126

SEND_RATIO       = 0.90   # 84% of time sends 81-100% of available — kept from data
MULTI_LAUNCH_MIN = 378    # avg trigger from data — kept
GARRISON_MIN     = 5      # kept

# ------------------------------------------------------------------
# UNKNOWN 1 — SCORING WEIGHTS
# Previous: prod*80 - dist*1.5 - garrison*2 + enemy_bonus 120
# Problem:  flat weights ignore game phase — early dist matters more,
#           late prod matters more. Tuned per phase below.
# ------------------------------------------------------------------
def score_target(tgt, d, garrison, turn):
    prod = tgt[P_PROD]
    owner = tgt[P_OWNER]

    if turn <= 50:
        # Early: grab close cheap planets fast to snowball production
        s = prod * 60 - d * 3.0 - garrison * 3
    elif turn <= 150:
        # Mid: balance production vs distance
        s = prod * 100 - d * 1.5 - garrison * 2
    else:
        # Late: production is king, distance less important
        s = prod * 160 - d * 0.8 - garrison * 1.5

    # Enemy bonus — matches observed 66% enemy targeting from data
    if owner != NEUTRAL:
        s += 120

    return s


# ------------------------------------------------------------------
# UNKNOWN 2 — DEFENSE THRESHOLD
# Previous: reinforce if incoming > garrison (any threat)
# Problem:  top bot likely ignores small threats and only defends
#           when the planet is actually worth defending
# Tuned:    only defend if planet prod >= 2 AND shortfall > 10
# ------------------------------------------------------------------
def should_defend(planet, incoming, eta):
    garrison = planet[P_SHIPS] + planet[P_PROD] * eta
    shortfall = incoming - garrison
    # Not worth defending low-prod planets — abandon them
    if planet[P_PROD] < 2 and shortfall < 30:
        return False
    return shortfall > 10


# ------------------------------------------------------------------
# UNKNOWN 3 — LATE GAME DUMP MODE
# Data showed fleets up to 951 ships — a "dump everything" mode
# Trigger: turn >= 350 OR we have 2x enemy ships
# Action:  send ALL available ships (not 90%) at highest-prod target
# ------------------------------------------------------------------
def is_dump_mode(turn, my_total, enemy_total):
    if turn >= 350:
        return True
    if enemy_total > 0 and my_total >= enemy_total * 2:
        return True
    return False


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

        turn  = self._turn
        self._turn += 1

        my_id        = obs.get('player', 0)
        planets      = {p[P_ID]: p for p in obs.get('planets', [])}
        fleets       = obs.get('fleets', [])
        comet_ids    = set(obs.get('comet_planet_ids', []))
        comet_groups = obs.get('comets', [])

        my_planets = [p for p in planets.values() if p[P_OWNER] == my_id]
        my_total   = sum(p[P_SHIPS] for p in my_planets)
        en_total   = sum(p[P_SHIPS] for p in planets.values()
                         if p[P_OWNER] not in (my_id, NEUTRAL))

        dump_mode = is_dump_mode(turn, my_total, en_total)

        # How many launches this turn — from data
        max_launches = 1
        if my_total >= MULTI_LAUNCH_MIN:
            max_launches = 2
        if my_total >= MULTI_LAUNCH_MIN * 2:
            max_launches = 3
        if dump_mode:
            max_launches = len(my_planets)  # dump: every planet fires

        actions      = []
        used_sources = set()
        used_targets = set()

        # Track planets already targeted by our in-transit fleets
        for f in fleets:
            if f[F_OWNER] == my_id:
                tgt = self._guess_target(f, planets, turn)
                if tgt:
                    used_targets.add(tgt)

        # --- DEFENSE ---
        threats = self._detect_threats(planets, fleets, my_id, turn)
        for pid, (incoming, eta) in threats.items():
            if len(actions) >= max_launches:
                break
            p = planets[pid]
            if not should_defend(p, incoming, eta):
                continue
            shortfall = incoming - p[P_SHIPS]
            helper = self._find_reinforcer(
                planets, pid, shortfall + GARRISON_MIN, my_id, turn, used_sources)
            if helper:
                actions.append(helper)
                used_sources.add(helper[0])

        # --- ATTACK ---
        remaining = max_launches - len(actions)
        if remaining > 0:
            attacks = self._pick_attacks(
                planets, my_id, turn, comet_ids,
                used_sources, used_targets, remaining, dump_mode)
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
    # THREAT DETECTION — returns {pid: (incoming_ships, eta)}
    # ----------------------------------------------------------
    def _detect_threats(self, planets, fleets, my_id, turn):
        threats = {}
        my_pids = {pid for pid, p in planets.items() if p[P_OWNER] == my_id}
        for f in fleets:
            if f[F_OWNER] == my_id:
                continue
            tgt = self._guess_target(f, planets, turn)
            if tgt not in my_pids:
                continue
            d   = dist2(f[F_X], f[F_Y], *self._pos(planets[tgt], turn))
            eta = travel_time(d, f[F_SHIPS])
            prev_incoming, _ = threats.get(tgt, (0, eta))
            threats[tgt] = (prev_incoming + f[F_SHIPS], eta)
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

    # ----------------------------------------------------------
    # ATTACK SELECTION
    # ----------------------------------------------------------
    def _pick_attacks(self, planets, my_id, turn, comet_ids,
                      used_sources, used_targets, max_count, dump_mode):
        results = []

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

            sx, sy    = self._pos(src, turn)
            available = src[P_SHIPS] - GARRISON_MIN
            if available <= 0:
                continue

            if dump_mode:
                # UNKNOWN 3: dump everything at highest-prod target
                send = available
            else:
                send = max(int(available * SEND_RATIO), scaled_fleet_size(turn))
                send = min(send, available)
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

                # UNKNOWN 1: phase-aware scoring
                s = score_target(tgt, d, garrison, turn)

                if s > best_score:
                    best_score  = s
                    best_action = [src[P_ID], angle_to(sx, sy, tx, ty), send]
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
        tgt    = planets[target_id]
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
