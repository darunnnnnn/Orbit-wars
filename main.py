import math
from utils import (
    fleet_speed, travel_time, dist2, is_orbiting,
    planet_pos, predict_pos, angle_to, path_hits_sun, comet_future_pos
)

# Planet list indices
P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = 0, 1, 2, 3, 4, 5, 6
# Fleet list indices
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 0, 1, 2, 3, 4, 5, 6

NEUTRAL = -1
MIN_GARRISON_FACTOR = 10
REAR_MAX_FACTOR = 20
SAFETY_MARGIN = 1.20
OBS_WINDOW = 50
FFA_GARRISON_FACTOR = 15  # higher buffer in 4p — more threats


class Agent:
    def __init__(self):
        self._turn = 0
        self._initial_planets = {}   # id -> planet list from initial_planets
        self._angular_velocity = 0.0
        self._history = []
        self._opponent_type = "D"
        self._num_players = 2        # detected on first turn
        self._primary_target = None  # in 4p: the one enemy we focus on

    # ------------------------------------------------------------------
    # ENTRY POINT — called by Kaggle as agent(obs)
    # ------------------------------------------------------------------
    def __call__(self, obs):
        # Bootstrap initial state once
        if self._turn == 0:
            self._angular_velocity = obs.get("angular_velocity", 0.0)
            for p in obs.get("initial_planets", []):
                self._initial_planets[p[P_ID]] = p

        turn = self._turn
        self._turn += 1

        my_id = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
        comet_ids = set(obs.get("comet_planet_ids", []))
        comet_groups = obs.get("comets", [])

        planets = {p[P_ID]: p for p in raw_planets}
        av = self._angular_velocity

        # Detect number of players from unique non-neutral owners
        all_owners = {p[P_OWNER] for p in planets.values() if p[P_OWNER] != NEUTRAL}
        self._num_players = len(all_owners)
        is_ffa = self._num_players > 2

        # In 4p: pick one primary enemy to focus on
        if is_ffa:
            self._primary_target = self._pick_primary_enemy(planets, raw_fleets, my_id, turn)

        targeted = set()
        actions = []

        # Use higher garrison in FFA
        garrison_factor = FFA_GARRISON_FACTOR if is_ffa else MIN_GARRISON_FACTOR

        # Phase 2 — threat detection
        threats = self._detect_threats(planets, raw_fleets, my_id, turn, av)

        # Phase 3 — classify opponent
        self._update_history(planets, my_id)
        if turn > 0 and turn % OBS_WINDOW == 0:
            self._classify_opponent()

        # Defend critical threats
        for pid, incoming in threats.items():
            p = planets[pid]
            needed = incoming - p[P_SHIPS]
            if needed > 0:
                action = self._find_reinforcement(planets, pid, needed, my_id, turn, av, garrison_factor)
                if action:
                    actions.append(action)

        # Phase 6 — comet interception
        comet_action = self._comet_intercept(planets, comet_ids, comet_groups, my_id, turn, av)
        if comet_action:
            actions.append(comet_action)
            targeted.add(comet_action[0])  # mark source used

        # Phase 4+5 — best attack
        attack = self._best_attack(planets, my_id, turn, av, comet_ids, targeted,
                                   garrison_factor, self._primary_target if is_ffa else None)
        if attack:
            actions.append(attack)

        # Phase 7 — flow rear ships to frontline
        flow = self._flow_rear_to_front(planets, my_id, turn, av, threats, garrison_factor)
        actions.extend(flow)

        return actions

    # ------------------------------------------------------------------
    # HELPERS — position
    # ------------------------------------------------------------------
    def _pos(self, p, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return planet_pos(p, ip, self._angular_velocity, turn)

    def _predict(self, p, src_x, src_y, ships, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return predict_pos(p, ip, self._angular_velocity, turn, ships, src_x, src_y)

    # ------------------------------------------------------------------
    # PHASE 2 — THREAT DETECTION
    # ------------------------------------------------------------------
    def _detect_threats(self, planets, fleets, my_id, turn, av):
        threats = {}
        my_pids = {pid for pid, p in planets.items() if p[P_OWNER] == my_id}
        for f in fleets:
            if f[F_OWNER] == my_id:
                continue
            target = self._fleet_target(f, planets, turn)
            if target not in my_pids:
                continue
            tgt_p = planets[target]
            d = dist2(f[F_X], f[F_Y], *self._pos(tgt_p, turn))
            eta = travel_time(d, f[F_SHIPS])
            if eta > 20:
                continue
            garrison = tgt_p[P_SHIPS] + tgt_p[P_PROD] * eta
            if f[F_SHIPS] > garrison:
                threats[target] = threats.get(target, 0) + f[F_SHIPS]
        return threats

    def _fleet_target(self, fleet, planets, turn):
        fx, fy, fa = fleet[F_X], fleet[F_Y], fleet[F_ANGLE]
        best, best_diff = None, math.pi
        for pid, p in planets.items():
            px, py = self._pos(p, turn)
            a = angle_to(fx, fy, px, py)
            diff = abs((a - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff, best = diff, pid
        return best if best_diff < 0.3 else None

    # ------------------------------------------------------------------
    # PHASE 3 — OPPONENT CLASSIFICATION
    # ------------------------------------------------------------------
    def _update_history(self, planets, my_id):
        my_s = sum(p[P_SHIPS] for p in planets.values() if p[P_OWNER] == my_id)
        en_s = sum(p[P_SHIPS] for p in planets.values() if p[P_OWNER] not in (my_id, NEUTRAL))
        my_pr = sum(p[P_PROD] for p in planets.values() if p[P_OWNER] == my_id)
        en_pr = sum(p[P_PROD] for p in planets.values() if p[P_OWNER] not in (my_id, NEUTRAL))
        self._history.append((my_s, en_s, my_pr, en_pr))

    def _classify_opponent(self):
        if len(self._history) < 10:
            return
        recent = self._history[-OBS_WINDOW:]
        en_ships = [h[1] for h in recent]
        drops = sum(1 for i in range(1, len(en_ships)) if en_ships[i] < en_ships[i-1] * 0.7)
        en_prod_growth = recent[-1][3] - recent[0][3]
        if drops >= 3:
            self._opponent_type = "B"
        elif en_prod_growth >= 8:
            self._opponent_type = "A"
        elif recent[-1][1] > recent[0][1] * 1.5 and en_prod_growth < 3:
            self._opponent_type = "C"
        else:
            self._opponent_type = "D"

    # ------------------------------------------------------------------
    # 4P HELPER — pick the one enemy to focus on this turn
    # ------------------------------------------------------------------
    def _pick_primary_enemy(self, planets, fleets, my_id, turn):
        # Ships per enemy player on planets
        enemy_strength = {}
        for p in planets.values():
            o = p[P_OWNER]
            if o == my_id or o == NEUTRAL:
                continue
            enemy_strength[o] = enemy_strength.get(o, 0) + p[P_SHIPS]

        if not enemy_strength:
            return None

        # Check if two enemies are actively fighting each other —
        # if so, stay out and grab neutrals (primary = None signals this)
        enemy_fleets = [f for f in fleets if f[F_OWNER] != my_id and f[F_OWNER] != NEUTRAL]
        fleet_owners = {f[F_OWNER] for f in enemy_fleets}
        enemies_fighting = len(fleet_owners) >= 2

        if enemies_fighting and self._turn < 100:
            # Let them bleed — return the weakest one so we still have a target
            # but scoring will prefer neutrals over them
            return min(enemy_strength, key=enemy_strength.get)

        # Otherwise focus on the weakest enemy — easiest to eliminate
        return min(enemy_strength, key=enemy_strength.get)

    # ------------------------------------------------------------------
    # PHASE 4+5 — ATTACK
    # ------------------------------------------------------------------
    def _best_attack(self, planets, my_id, turn, av, comet_ids, targeted,
                     garrison_factor=MIN_GARRISON_FACTOR, primary_enemy=None):
        my_planets = [(pid, p) for pid, p in planets.items() if p[P_OWNER] == my_id]
        candidates = [p for p in planets.values()
                      if p[P_OWNER] != my_id and p[P_ID] not in targeted]

        # In 4p: check if enemies are fighting — if yes, prefer neutrals
        enemy_fleets_active = primary_enemy is not None and self._turn < 100

        best_action, best_score = None, -1e9

        for src_id, src in my_planets:
            if src_id in targeted:
                continue
            sx, sy = self._pos(src, turn)
            min_garrison = src[P_PROD] * garrison_factor
            available = src[P_SHIPS] - min_garrison
            if available <= 0:
                continue

            for tgt in candidates:
                tx, ty = self._predict(tgt, sx, sy, available, turn)
                if path_hits_sun(sx, sy, tx, ty):
                    continue

                d = dist2(sx, sy, tx, ty)
                eta = travel_time(d, available)
                prod_gain = tgt[P_PROD] * eta if tgt[P_OWNER] != NEUTRAL else 0
                garrison_at_arrival = tgt[P_SHIPS] + prod_gain
                needed = math.ceil(garrison_at_arrival * SAFETY_MARGIN) + 1
                if available < needed:
                    continue

                score = tgt[P_PROD] * 100 - d * 2 - tgt[P_SHIPS] * 3
                if tgt[P_ID] in comet_ids:
                    score -= 50

                # 4p adjustments
                if primary_enemy is not None:
                    if tgt[P_OWNER] == NEUTRAL:
                        score += 80   # grab neutrals while enemies fight
                    elif tgt[P_OWNER] == primary_enemy:
                        score += 150  # bonus for hitting our focused target
                    else:
                        score -= 200  # heavily penalise attacking non-primary enemy

                # 1v1 adjustments
                if self._opponent_type == "C" and primary_enemy is None:
                    score += tgt[P_PROD] * 50
                elif self._opponent_type == "B" and tgt[P_OWNER] == my_id:
                    continue

                if score > best_score:
                    best_score = score
                    best_action = [src_id, angle_to(sx, sy, tx, ty), needed]
                    targeted.add(tgt[P_ID])

        return best_action

    # ------------------------------------------------------------------
    # PHASE 6 — COMET INTERCEPTION
    # ------------------------------------------------------------------
    def _comet_intercept(self, planets, comet_ids, comet_groups, my_id, turn, av):
        if not comet_ids:
            return None
        my_planets = [(pid, p) for pid, p in planets.items() if p[P_OWNER] == my_id]

        for group in comet_groups:
            for cid in group.get("planet_ids", []):
                if cid not in planets:
                    continue
                comet = planets[cid]
                # Use path to find where comet will be in ~5 turns
                future = comet_future_pos(group, cid, 5)
                if future is None:
                    cx, cy = comet[P_X], comet[P_Y]
                else:
                    cx, cy = future

                # Find closest owned planet with spare ships
                for src_id, src in sorted(my_planets,
                        key=lambda x: dist2(*self._pos(x[1], turn), cx, cy)):
                    sx, sy = self._pos(src, turn)
                    send = min(10, src[P_SHIPS] - src[P_PROD] * MIN_GARRISON_FACTOR)
                    if send <= 0:
                        continue
                    if path_hits_sun(sx, sy, cx, cy):
                        continue
                    return [src_id, angle_to(sx, sy, cx, cy), send]
        return None

    # ------------------------------------------------------------------
    # PHASE 7 — REAR TO FRONT FLOW
    # ------------------------------------------------------------------
    def _flow_rear_to_front(self, planets, my_id, turn, av, threats, garrison_factor=MIN_GARRISON_FACTOR):
        actions = []
        my_planets = {pid: p for pid, p in planets.items() if p[P_OWNER] == my_id}
        if len(my_planets) < 2:
            return actions

        enemy_pos = [(p[P_X], p[P_Y]) for p in planets.values()
                     if p[P_OWNER] not in (my_id, NEUTRAL)]
        if not enemy_pos:
            return actions

        def min_en_dist(pos):
            return min(dist2(pos[0], pos[1], ex, ey) for ex, ey in enemy_pos)

        sorted_p = sorted(my_planets.items(),
                          key=lambda x: min_en_dist(self._pos(x[1], turn)))
        front_pos = self._pos(sorted_p[0][1], turn)

        for pid, p in sorted_p[1:]:
            if pid in threats:
                continue
            excess = p[P_SHIPS] - p[P_PROD] * max(REAR_MAX_FACTOR, garrison_factor * 2)
            if excess <= 5:
                continue
            sx, sy = self._pos(p, turn)
            if path_hits_sun(sx, sy, front_pos[0], front_pos[1]):
                continue
            actions.append([pid, angle_to(sx, sy, front_pos[0], front_pos[1]), excess])

        return actions

    # ------------------------------------------------------------------
    # REINFORCEMENT
    # ------------------------------------------------------------------
    def _find_reinforcement(self, planets, target_id, needed, my_id, turn, av,
                             garrison_factor=MIN_GARRISON_FACTOR):
        tgt = planets[target_id]
        tx, ty = self._pos(tgt, turn)
        best, best_d = None, 1e9
        for pid, p in planets.items():
            if p[P_OWNER] != my_id or pid == target_id:
                continue
            available = p[P_SHIPS] - p[P_PROD] * garrison_factor
            if available < needed:
                continue
            sx, sy = self._pos(p, turn)
            d = dist2(sx, sy, tx, ty)
            if d < best_d and not path_hits_sun(sx, sy, tx, ty):
                best_d = d
                best = [pid, angle_to(sx, sy, tx, ty), needed]
        return best


# Kaggle calls agent(obs) — module-level function
_agent = Agent()

def agent(obs):
    return _agent(obs)
