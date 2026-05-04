import math
from utils import (
    fleet_speed, travel_time, dist2, is_orbiting,
    planet_pos, predict_pos, angle_to, path_hits_sun, comet_future_pos
)

P_ID, P_OWNER, P_X, P_Y, P_RADIUS, P_SHIPS, P_PROD = 0, 1, 2, 3, 4, 5, 6
F_ID, F_OWNER, F_X, F_Y, F_ANGLE, F_FROM, F_SHIPS = 0, 1, 2, 3, 4, 5, 6

NEUTRAL = -1
# Keep only this many ships on a planet as buffer — very low so we always attack
BUFFER = 5
# How much extra to send on top of garrison to guarantee capture
OVERKILL = 5


class Agent:
    def __init__(self):
        self._turn = 0
        self._initial_planets = {}
        self._angular_velocity = 0.0
        # track which planets we already sent a fleet to this turn
        self._in_transit_targets = set()

    def __call__(self, obs):
        if self._turn == 0:
            self._angular_velocity = obs.get("angular_velocity", 0.0)
            for p in obs.get("initial_planets", []):
                self._initial_planets[p[P_ID]] = p

        turn = self._turn
        self._turn += 1

        my_id = obs.get("player", 0)
        planets = {p[P_ID]: p for p in obs.get("planets", [])}
        fleets = obs.get("fleets", [])
        comet_ids = set(obs.get("comet_planet_ids", []))
        comet_groups = obs.get("comets", [])

        # Update in-transit targets from our own fleets
        self._in_transit_targets = set()
        for f in fleets:
            if f[F_OWNER] == my_id:
                self._in_transit_targets.add(f[F_FROM])  # source already launched

        # Incoming threats: planet_id -> total incoming enemy ships
        threats = self._incoming_threats(planets, fleets, my_id, turn)

        actions = []
        used_sources = set()  # planets we already launched from this turn

        # --- STEP 1: Emergency defense — reinforce planets about to fall ---
        for pid, incoming in threats.items():
            p = planets[pid]
            px, py = self._pos(p, turn)
            shortfall = incoming - p[P_SHIPS]
            if shortfall <= 0:
                continue
            helper = self._best_reinforcer(planets, pid, shortfall + BUFFER, my_id, turn, used_sources)
            if helper:
                actions.append(helper)
                used_sources.add(helper[0])

        # --- STEP 2: Attack from EVERY owned planet that has ships to spare ---
        my_planets = [p for p in planets.values() if p[P_OWNER] == my_id]
        targets_taken = set()  # don't send two fleets to same target this turn

        # Sort my planets: most ships first so strongest planets attack first
        for src in sorted(my_planets, key=lambda p: -p[P_SHIPS]):
            if src[P_ID] in used_sources:
                continue
            sx, sy = self._pos(src, turn)
            available = src[P_SHIPS] - BUFFER
            if available <= 1:
                continue

            best = self._pick_target(planets, src, sx, sy, available, my_id,
                                     turn, comet_ids, targets_taken, threats)
            if best is None:
                continue

            tgt, tx, ty, send = best
            actions.append([src[P_ID], angle_to(sx, sy, tx, ty), send])
            targets_taken.add(tgt[P_ID])
            used_sources.add(src[P_ID])

        # --- STEP 3: Grab comets with tiny fleets ---
        comet_actions = self._grab_comets(planets, comet_ids, comet_groups,
                                          my_id, turn, used_sources)
        actions.extend(comet_actions)

        return actions

    # ------------------------------------------------------------------
    # POSITION HELPERS
    # ------------------------------------------------------------------
    def _pos(self, p, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return planet_pos(p, ip, self._angular_velocity, turn)

    def _predict(self, p, sx, sy, ships, turn):
        ip = self._initial_planets.get(p[P_ID], p)
        return predict_pos(p, ip, self._angular_velocity, turn, ships, sx, sy)

    # ------------------------------------------------------------------
    # THREAT DETECTION
    # ------------------------------------------------------------------
    def _incoming_threats(self, planets, fleets, my_id, turn):
        threats = {}
        my_pids = {pid for pid, p in planets.items() if p[P_OWNER] == my_id}
        for f in fleets:
            if f[F_OWNER] == my_id:
                continue
            # figure out which planet this fleet is heading to
            target = self._guess_target(f, planets, turn)
            if target not in my_pids:
                continue
            threats[target] = threats.get(target, 0) + f[F_SHIPS]
        return threats

    def _guess_target(self, fleet, planets, turn):
        fx, fy, fa = fleet[F_X], fleet[F_Y], fleet[F_ANGLE]
        best, best_diff = None, math.pi
        for pid, p in planets.items():
            px, py = self._pos(p, turn)
            a = angle_to(fx, fy, px, py)
            diff = abs((a - fa + math.pi) % (2 * math.pi) - math.pi)
            if diff < best_diff:
                best_diff, best = diff, pid
        return best if best_diff < 0.25 else None

    # ------------------------------------------------------------------
    # TARGET SELECTION — called per source planet
    # ------------------------------------------------------------------
    def _pick_target(self, planets, src, sx, sy, available,
                     my_id, turn, comet_ids, taken, threats):
        best_score = -1e9
        best = None

        for tgt in planets.values():
            if tgt[P_OWNER] == my_id:
                continue
            if tgt[P_ID] in taken:
                continue
            # skip comets in main attack loop — handled separately
            if tgt[P_ID] in comet_ids:
                continue

            tx, ty = self._predict(tgt, sx, sy, available, turn)
            if path_hits_sun(sx, sy, tx, ty):
                continue

            d = dist2(sx, sy, tx, ty)
            eta = travel_time(d, available)

            # garrison at arrival: neutral stays flat, enemy grows
            extra = tgt[P_PROD] * eta if tgt[P_OWNER] != NEUTRAL else 0
            garrison = tgt[P_SHIPS] + extra
            send = garrison + OVERKILL

            if send > available:
                # can't afford full overkill — try with everything we have
                # only skip if we'd lose
                if available <= garrison:
                    continue
                send = available

            # Score: production value is king, penalise distance and cost
            score = tgt[P_PROD] * 200 - d * 1.5 - garrison * 2
            # Bonus for attacking enemy (not neutral) — denies their production
            if tgt[P_OWNER] != NEUTRAL:
                score += 100
            # Bonus for planets already under our fleet pressure
            if tgt[P_ID] in self._in_transit_targets:
                score -= 500  # don't double-send

            if score > best_score:
                best_score = score
                best = (tgt, tx, ty, send)

        return best

    # ------------------------------------------------------------------
    # EMERGENCY REINFORCEMENT
    # ------------------------------------------------------------------
    def _best_reinforcer(self, planets, target_id, needed, my_id, turn, used):
        tgt = planets[target_id]
        tx, ty = self._pos(tgt, turn)
        best, best_d = None, 1e9
        for pid, p in planets.items():
            if p[P_OWNER] != my_id or pid == target_id or pid in used:
                continue
            avail = p[P_SHIPS] - BUFFER
            if avail < needed:
                continue
            sx, sy = self._pos(p, turn)
            d = dist2(sx, sy, tx, ty)
            if d < best_d and not path_hits_sun(sx, sy, tx, ty):
                best_d = d
                best = [pid, angle_to(sx, sy, tx, ty), needed]
        return best

    # ------------------------------------------------------------------
    # COMET GRABBING
    # ------------------------------------------------------------------
    def _grab_comets(self, planets, comet_ids, comet_groups, my_id, turn, used):
        actions = []
        my_planets = [p for p in planets.values() if p[P_OWNER] == my_id]

        for group in comet_groups:
            for cid in group.get("planet_ids", []):
                if cid not in planets:
                    continue
                comet = planets[cid]
                if comet[P_OWNER] == my_id:
                    continue  # already ours

                future = comet_future_pos(group, cid, 3)
                cx, cy = future if future else (comet[P_X], comet[P_Y])

                for src in sorted(my_planets, key=lambda p: dist2(*self._pos(p, turn), cx, cy)):
                    if src[P_ID] in used:
                        continue
                    sx, sy = self._pos(src, turn)
                    send = comet[P_SHIPS] + OVERKILL
                    avail = src[P_SHIPS] - BUFFER
                    if avail < send:
                        continue
                    if path_hits_sun(sx, sy, cx, cy):
                        continue
                    actions.append([src[P_ID], angle_to(sx, sy, cx, cy), send])
                    used.add(src[P_ID])
                    break

        return actions


_agent = Agent()

def agent(obs):
    return _agent(obs)
