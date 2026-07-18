from __future__ import annotations

import dataclasses
from typing import Any, Generic, TypedDict, TypeVar

from reil.datatypes.feature import FeatureSet
from reil.logger import Logger
from reil.utils.functions import dist, in_range, interpolate, square_dist

# NOTE:
# `retrospective` was meant to be used to indicate regular reward computation
# vs lookahead. However, lookahead should be done by an `Environment`. So,
# I removed `retrospective` to simplify the implementation and improve
# performance.

reil_func_logger = Logger('reil_functions')

TypeY = TypeVar('TypeY')
TypeX = TypeVar('TypeX')


@dataclasses.dataclass
class ReilFunction(Generic[TypeY, TypeX]):
    name: str
    y_var_name: str
    x_var_name: str | None = None
    length: int = -1
    multiplier: float = 1.0
    constant: float = 0.0
    interpolate: bool = True

    def __post_init__(self):
        self._fn = self._inter if self.interpolate else self._no_inter

    def __call__(self, args: FeatureSet) -> float:
        temp = args.value
        fn_args: dict[str, Any] = {'y': temp[self.y_var_name]}
        if self.x_var_name:
            fn_args['x'] = temp[self.x_var_name]

        try:
            result = self.multiplier * self._fn(**fn_args)
        except NotImplementedError:
            result = self.multiplier * self._default_function(**fn_args)

        return result + self.constant

    # just for the compatibility with old saved models. Should not be used.
    def _retro_inter(self, y: list[TypeY], x: list[TypeX]) -> float:
        raise NotImplementedError

    # just for the compatibility with old saved models. Should not be used.
    def _retro_no_inter(self, y: list[TypeY], x: list[TypeX]) -> float:
        raise NotImplementedError

    def _inter(self, y: list[TypeY], x: list[TypeX]) -> float:
        raise NotImplementedError

    def _no_inter(self, y: list[TypeY]) -> float:
        raise NotImplementedError

    def _default_function(
            self, y: list[TypeY], x: list[TypeX] | None = None) -> float:
        raise NotImplementedError


class CompoundReilFunctionComponent(TypedDict):
    reil_function: ReilFunction
    weight: float


@dataclasses.dataclass
class CompoundReilFunction(ReilFunction[TypeY, TypeX]):
    rail_function_list: list[CompoundReilFunctionComponent] = dataclasses.field(
        default_factory=list)

    def __call__(self, args: FeatureSet) -> float:
        return sum(
            fn['weight'] * fn['reil_function'](args)
            for fn in self.rail_function_list
        )


@dataclasses.dataclass
class NormalizedSquareDistance(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    amplifying_factor: float = 1.0
    exclude_first: bool = False
    average: bool = False
    # monitoring_coef (kappa): a flat per-DECISION monitoring cost subtracted
    #   once per reward call (i.e. once per blood draw / retest), independent of
    #   the interval length. Paper-3 EB (doc 220 §9): with the summed control
    #   term (average=False) under the adherence sim a longer interval already
    #   accrues more out-of-range days, so kappa supplies the burden side of the
    #   trade-off and the interval gains a genuine control-vs-visits optimum;
    #   sweeping kappa traces the PTTR-vs-visits Pareto curve (EC). Applied after
    #   multiplier/constant and inherited by every subclass. Default 0.0 is
    #   byte-identical to the pre-kappa reward (parity).
    monitoring_coef: float = 0.0

    def __call__(self, args: FeatureSet) -> float:
        return super().__call__(args) - self.monitoring_coef

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        result = sum(
            (self.amplifying_factor ** i) * square_dist(
                self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        if self.average:
            result /= len_y

        return result


@dataclasses.dataclass
class DurationIncentiveSquareDistance(NormalizedSquareDistance):
    '''Additive dose-duration reward (Paper 3, plan 200 §3).

        r = duration_coef * tau  -  c * sum_t (mu_t - center)^2

    The square-distance term is exactly `NormalizedSquareDistance` (a *cost*,
    i.e. configured with multiplier=-1 and normalized by band_width so
    c = (2 / band_width)**2 = 4 at band_width=1). On top of that cost we add a
    flat `+ duration_coef * tau` reward for a longer monitoring interval tau,
    so the policy extends tau only while the patient stays near `center` and
    shortens it when INR drifts. With c fixed by the band-width normalization,
    `duration_coef` (= lambda) is the single dose-duration trade-off dial;
    lambda = 0 recovers `sq_dist` exactly and the margin condition gives the
    "indifference deviation" delta* = sqrt(lambda / c).

    tau is taken as the length of the per-decision daily-INR window `y`
    (the days elapsed since the previous decision). The `+ duration_coef * tau`
    term is added *after* the parent's `multiplier`/`constant` are applied, so
    it stays a positive interval bonus rather than being flipped by the
    cost's multiplier=-1.
    '''
    duration_coef: float = 0.0

    def __call__(self, args: FeatureSet) -> float:
        base = super().__call__(args)
        tau = len(args.value[self.y_var_name])  # type: ignore
        return base + self.duration_coef * tau


@dataclasses.dataclass
class DoseDurationSafetyReward(DurationIncentiveSquareDistance):
    '''Duration reward with a state-dependent SAFETY CEILING (Paper 3, 2026-07-07).

        r = duration_coef * tau
            - c * sum_t (mu_t - center)^2                      (control)
            - safety_coef * max(0, tau - tau_safe(Delta, mu0))^2   (safety)

    The first two terms are exactly `DurationIncentiveSquareDistance`
    (lambda*tau minus the normalized square-distance cost). The third term
    penalises scheduling the next test LATER than is clinically prudent given
    the decision state, encoding the clinical rule that a blood draw is a
    burden but is needed sooner when the patient is unstable. `tau_safe` is the
    longest prudent interval:

      * out of range OR big dose change      -> tau_short   (quick revision)
      * in range AND small dose change       -> tau_med     (~ weekly)
      * in range AND no dose change (stable)  -> tau_long    (long interval)

    This gives an interior optimum at tau_safe (unlike the pure lambda*tau term,
    which is linear and bangs to an extreme), so the medium interval (tau_med,
    e.g. 7) becomes reward-optimal for the small-adjustment regime. It is a
    smooth, learnable generalisation of Aurora's stable-days -> retest ladder.

    Inputs beyond `y_var_name` (= daily_INR_history): the decision INR is taken
    as the first day of the window (mu0 = y[0]); the dose change Delta is read
    from `dose_var_name` (per-decision dose_history, length >= 2) as
    |dose[-1]/dose[-2] - 1|. The reward's state definition must therefore
    include `dose_history` (see `recent_daily_INR_w_dose` in warfarin.py).
    '''
    safety_coef: float = 0.0                 # rho
    dose_var_name: str = 'dose_history'
    tau_short: float = 2.0
    tau_med: float = 7.0
    tau_long: float = 28.0
    delta_zero: float = 0.05                  # |Delta| below this = "no change"
    delta_big: float = 0.30                   # |Delta| at/above this = "big change"
    range_lo: float = 2.0
    range_hi: float = 3.0
    # symmetric=False (default): penalise only OVER-shooting tau_safe (the
    #   burden-vs-safety "Form A"). This makes tau=1 a risk-free safe-harbor and
    #   empirically collapses the policy to tau=1 (2026-07-07). symmetric=True
    #   ("Form B"): penalise (tau - tau_safe)^2 in BOTH directions so the policy
    #   tracks tau_safe (tau=1 is penalised when the state warrants 7). Use with
    #   duration_coef=0 (tau_safe already encodes "long when stable") and a small
    #   safety_coef so the penalty is on the scale of the control term.
    symmetric: bool = False
    # ramp=True: Aurora/Intermountain-informed tau_safe. Instead of the tiered
    #   Form-A/B logic (stable->tau_long, small-change->tau_med, else tau_short),
    #   tau_safe is a zone-graded reset + an IN-RANGE STABILITY RAMP that climbs
    #   `ladder` by `s` = consecutive in-range days (read from `stab_var_name`):
    #     big change or far off (< far_lo / > far_hi)  -> tau_short
    #     mildly off (in [far_lo, range_lo)/(range_hi, far_hi])  -> tau_mild
    #     in range [range_lo, range_hi]  -> largest ladder rung <= s
    #   This reproduces Aurora's graded 7-centred distribution (the tiered form
    #   gave a bimodal 2/28 target on real data). Use with symmetric=True.
    ramp: bool = False
    stab_var_name: str = 'consecutive_in_range'
    tau_mild: float = 5.0
    far_lo: float = 1.5
    far_hi: float = 4.0
    ladder: tuple = (1, 3, 7, 14, 28)
    # include_control=True (default): keep the parent's -c*sum dev^2 control
    #   term (r = lambda*tau - c*sum dev^2 - safety). include_control=False
    #   (Paper-3 per-head split, 2026-07-09): DROP the control term so the
    #   duration head's reward is PURELY burden + safety (r = lambda*tau -
    #   safety) -- control is the dose head's job, and dropping the sum-over-days
    #   term also removes its hidden short-tau pull.
    include_control: bool = True
    # penalty_shape: how the safety term penalises the gap g = tau - tau_safe.
    #   'quadratic' (default): rho * g^2 -- gradient 2*rho*g grows without bound,
    #       so a far-off exploration guess yields a catastrophic gradient that
    #       collapses the policy to the tau=1 safe boundary (2026-07-09 diagnosis).
    #   'linear': rho * |g| -- constant gradient rho; kills the variance but has
    #       a kink at tau_safe and (since gradient is constant) bang-bangs to
    #       tau_max whenever lambda > rho. Baseline only.
    #   'huber': rho * g^2 for |g| <= huber_delta, else rho*(2*delta*|g| -
    #       delta^2). Quadratic bowl near tau_safe (smooth, tunable interior
    #       optimum) + LINEAR arms beyond delta (gradient capped at 2*rho*delta),
    #       so rare far guesses cost a bounded amount. Tolerates lambda up to
    #       2*rho*delta before banging. Paper-3 fix for the residual tau=1 pull.
    penalty_shape: str = 'quadratic'
    huber_delta: float = 3.0

    def _tau_safe(self, delta: float, mu0: float, s: float) -> float:
        if not self.ramp:
            # tiered (Form A/B): stable->long, small change->med, else short
            in_range = self.range_lo <= mu0 <= self.range_hi
            if (not in_range) or (delta >= self.delta_big):
                return self.tau_short
            if delta >= self.delta_zero:
                return self.tau_med
            return self.tau_long
        # RAMP: zone-graded reset + in-range stability ramp over `s`.
        if delta >= self.delta_big or mu0 < self.far_lo or mu0 > self.far_hi:
            return self.tau_short                     # big correction / far off
        if mu0 < self.range_lo or mu0 > self.range_hi:
            return self.tau_mild                      # mildly off
        ts = float(self.ladder[0])                    # in range: climb by s
        for rung in self.ladder:
            if rung <= s:
                ts = float(rung)
            else:
                break
        return ts

    def __call__(self, args: FeatureSet) -> float:
        val = args.value
        inr = val[self.y_var_name]           # daily INR over the window
        tau = len(inr)                       # type: ignore
        if self.include_control:
            base = super().__call__(args)    # lambda*tau - c*sum dev^2 - kappa
        else:
            # lambda*tau only (no control term). This branch bypasses
            # super().__call__, so re-apply the per-visit monitoring cost
            # (kappa) here explicitly; otherwise it would be silently dropped
            # for the per-head duration reward (Paper-3 EB, doc 220 §9).
            base = self.duration_coef * tau - self.monitoring_coef
        if not tau or self.safety_coef == 0.0:
            return base
        mu0 = float(inr[0])                   # type: ignore  decision-day INR
        doses = val.get(self.dose_var_name)
        delta = 0.0
        if doses is not None and len(doses) >= 2 and doses[-2]:
            delta = abs(float(doses[-1]) / float(doses[-2]) - 1.0)
        s = 0.0
        if self.ramp:
            sv = val.get(self.stab_var_name)
            if sv is not None:
                s = float(sv[0] if hasattr(sv, '__len__') else sv)
        gap = tau - self._tau_safe(delta, mu0, s)
        if not self.symmetric:
            gap = max(0.0, gap)
        g = abs(gap)
        if self.penalty_shape == 'linear':
            penalty = g
        elif self.penalty_shape == 'huber':
            d = self.huber_delta
            penalty = g * g if g <= d else (2.0 * d * g - d * d)
        else:  # 'quadratic'
            penalty = g * g
        return base - self.safety_coef * penalty


@dataclasses.dataclass
class TimeInRangeReward(ReilFunction[float, int]):
    '''Opportunity-cost / lookahead DURATION reward (Paper 3, 2026-07-14).

        r = sum_t [ +1                      if lo <= INR_t <= hi
                    -overshoot_penalty      otherwise ]

    over the per-decision daily-INR window `y` (`daily_INR_history`). Every day the
    held dose keeps INR in range scores +1 (rewarding a LONGER safe interval), and
    every out-of-range day costs `-overshoot_penalty`. Its optimum interval is
    exactly tau* = the longest interval that stays in range ("go as long as you can,
    overshoot is costly") — the lookahead objective, realised from the OBSERVED
    window with no separate forward roll. Pair on the DURATION head of a per-head
    tandem (dose head keeps `sq_dist_avg`). `overshoot_penalty` >> 1 makes drifting
    out costlier than stopping a day early (asymmetry alpha >> beta=1).
    '''
    lo: float = 2.0
    hi: float = 3.0
    overshoot_penalty: float = 4.0

    def _no_inter(self, y: list[float]) -> float:
        return sum(1.0 if self.lo <= v <= self.hi else -self.overshoot_penalty
                   for v in y)

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        return self._no_inter(y)


@dataclasses.dataclass
class DeadbandSquareDistance(ReilFunction[float, int]):
    '''Squared distance from `center` with an epsilon-insensitive deadband.

    Identical to `NormalizedSquareDistance` except that no penalty accrues
    while the value is within `tolerance` of `center`. The per-point penalty
    is ``(max(0, |center - v| - tolerance))**2``, so ``tolerance = 0``
    recovers `NormalizedSquareDistance` exactly, while ``tolerance = 0.5``
    (with ``center = 2.5``) makes the reward flat across the [2, 3]
    therapeutic band and quadratic only outside it. Used to test whether the
    inward conservatism of the distilled cut-offs is driven by penalising
    deviation from the midpoint vs. from the range (EXP-C2-RW1, Paper 2).
    '''
    center: float = 0.0
    band_width: float = 1.0
    amplifying_factor: float = 1.0
    tolerance: float = 0.0
    exclude_first: bool = False
    average: bool = False

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        result = sum(
            (self.amplifying_factor ** i) * sum(
                max(0.0, abs(self.center - v) - self.tolerance) ** 2
                for v in interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        if self.average:
            result /= len_y

        return result


@dataclasses.dataclass
class AsymmetricSquareDistance(ReilFunction[float, int]):
    '''Squared distance from `center` with side-dependent weights.

    Per-point penalty is ``w * (center - v)**2`` with ``w = over_weight``
    when ``v > center`` (supratherapeutic) and ``w = under_weight``
    otherwise. ``over_weight = under_weight = 1`` recovers
    `NormalizedSquareDistance`. A heavier `over_weight` encodes the clinical
    asymmetry that a high INR (bleeding risk) is worse than a low INR
    (clotting risk); it is expected to pull the policy's effective target —
    and hence the distilled cut-offs — downward (EXP-C2-RW1, Paper 2).
    '''
    center: float = 0.0
    band_width: float = 1.0
    amplifying_factor: float = 1.0
    under_weight: float = 1.0
    over_weight: float = 1.0
    exclude_first: bool = False
    average: bool = False

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        result = sum(
            (self.amplifying_factor ** i) * sum(
                (self.over_weight if v > self.center else self.under_weight)
                * (self.center - v) ** 2
                for v in interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        if self.average:
            result /= len_y

        return result


@dataclasses.dataclass
class NormalizedDistance(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    amplifying_factor: float = 1.0
    exclude_first: bool = False

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        result = sum(
            (self.amplifying_factor ** i) * dist(
                self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        # normalize
        result *= (2.0 / self.band_width) ** 2

        return result


@dataclasses.dataclass
class PercentInRange(ReilFunction[float, int]):
    acceptable_range: tuple[float, float] = (0.0, 1.0)
    exclude_first: bool = False

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)
        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        result = sum(
            in_range(
                self.acceptable_range,
                interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x))

        total_durations = sum(_x)

        return result / total_durations


@dataclasses.dataclass
class NotEqual(ReilFunction[float, int]):
    interpolate: bool = False

    def _no_inter(
            self, y: list[float], x: list[int] | None = None) -> float:
        if x:
            reil_func_logger.info(
                'x is provided, but is not used in `NotEqual` function.')

        try:
            result = sum(
                y1 != y2
                for y1, y2 in zip(y[:-1], y[1:])
            ) / (len(y) - 1)
        except ZeroDivisionError:  # NotEqual for one observation is 0.
            result = 0

        return result


@dataclasses.dataclass
class CustomDistance(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False
    average: bool = True

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        l1_distance_list = tuple(
            dist(self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)
        )

        distance_penalty = (2.0 / self.band_width) ** 2 * sum(
            dis ** 2
            for dis, x_i in zip(l1_distance_list, _x))

        if self.average:
            distance_penalty /= len_y

        average_l1_distance = sum(l1_distance_list) / len_y

        duration_penalty = (
            len_y / 14 * average_l1_distance if average_l1_distance > 0.5 else
            (len_y / 14 - 2) * (average_l1_distance - 0.5))

        return distance_penalty + duration_penalty


@dataclasses.dataclass
class CustomDistance2(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False
    average: bool = True

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        l1_distance_list = tuple(
            dist(self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)
        )

        distance_penalty = (2.0 / self.band_width) ** 2 * sum(
            dis ** 2
            for dis, x_i in zip(l1_distance_list, _x))

        if self.average:
            distance_penalty /= len_y

        last_l1_distance = l1_distance_list[-1]

        duration_penalty = (
            len_y / 14 * last_l1_distance if last_l1_distance > 0.5 else
            (len_y / 14 - 2) * (last_l1_distance - 0.5))

        return distance_penalty + duration_penalty


@dataclasses.dataclass
class CustomDistance3(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False
    average: bool = True

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        l1_distance_list = tuple(
            dist(self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)
        )

        distance_penalty = (2.0 / self.band_width) ** 2 * sum(
            dis ** 2
            for dis, x_i in zip(l1_distance_list, _x))

        if self.average:
            distance_penalty /= len_y

        first_l1_distance = l1_distance_list[0]

        duration_penalty = (
            len_y / 14 * first_l1_distance if first_l1_distance > 0.5 else
            (len_y / 14 - 2) * (first_l1_distance - 0.5))

        return distance_penalty + duration_penalty


@dataclasses.dataclass
class CustomDistance4(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False
    average: bool = True
    c_i: float = 1.
    c_o: float = 0.3
    kappa: float = -0.3
    delta: float = -20.2

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        l1_distance_list = tuple(
            dist(self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)
        )

        distance_penalty = (2.0 / self.band_width) ** 2 * sum(
            dis ** 2
            for dis, x_i in zip(l1_distance_list, _x))

        if self.average:
            distance_penalty /= len_y

        first_l1_distance = l1_distance_list[0]

        # duration_penalty = (
        #     len_y / 14 * first_l1_distance + 0.5 if first_l1_distance > 0.5 else
        #     (len_y / 3.5 - 2) * (first_l1_distance - 1.5))
        duration_penalty = (
            len_y * self.c_o * first_l1_distance + self.delta if first_l1_distance > 0.5 else
            self.c_i * (len_y - 2) * (self.kappa - first_l1_distance))

        return distance_penalty + duration_penalty


@dataclasses.dataclass
class CustomDistance4b(ReilFunction[float, int]):
    center: float = 0.0
    band_width: float = 1.0
    exclude_first: bool = False
    average: bool = True
    c_i: float = 0.1
    c_o: float = 0.3
    kappa: float = 2.5
    delta: float = 4.

    def _default_function(
            self, y: list[float], x: list[int] | None = None) -> float:
        len_y = len(y)
        _x = x or [1] * (len_y - 1)

        if len_y != len(_x) + 1:
            raise ValueError(
                'y should have exactly one item more than x.')

        if not self.exclude_first:
            _x = [1] + _x
            _y = [0.0, *y]
        else:
            _y = y

        l1_distance_list = tuple(
            dist(self.center, interpolate(_y[i], _y[i + 1], x_i))
            for i, x_i in enumerate(_x)
        )

        distance_penalty = (2.0 / self.band_width) ** 2 * sum(
            dis ** 2
            for dis, x_i in zip(l1_distance_list, _x))

        if self.average:
            distance_penalty /= len_y

        first_l1_distance = l1_distance_list[0]

        duration_penalty = (
            len_y * self.c_o * first_l1_distance + self.delta if first_l1_distance > 0.5 else
            self.c_i * (len_y - 2) * (first_l1_distance - self.kappa))

        return distance_penalty + duration_penalty


# TODO: not implemented yet!
# @dataclasses.dataclass
# class Delta(ReilFunction):
#     '''
#     Get changes in the series.

#     available `op`s:
#         count: counts the number of change points in y.
#         sum: sum of value changes
#         average: average value change

#     available `interpolation_method`s:
#         linear
#         post: y = y[i] at x[i]
#         pre: y = y[i] at x[i-1]
#     '''
#     exclude_first: bool = False
#     op: str = 'count'
#     interpolation_method: str = 'linear'

# def _default_function(
#         self, y: list[Any], x: list[Any] | None = None) -> float:
#     if self.op == 'count':
#         result = sum(yi != y[i+1]
#                     for i, yi in enumerate(y[:-1]))

#     return result


# class Functions:
#     @staticmethod
#     def dose_change_count(dose_list: list[float],
#                           durations: list[int] | None = None) -> int:
#         # assuming dose is fixed during each duration
#         return sum(x != dose_list[i+1]
#                    for i, x in enumerate(dose_list[:-1]))

#     @staticmethod
#     def delta_dose(dose_list: list[float],
#                    durations: list[int] | None = None) -> float:
#         # assuming dose is fixed during each duration
#         return sum(abs(x-dose_list[i+1])
#                    for i, x in enumerate(dose_list[:-1]))

#     @staticmethod
#     def total_dose(dose_list: list[float],
#                    durations: list[int] | None = None) -> float:
#         if durations is None:
#             result = sum(dose_list)
#         else:
#             if len(dose_list) != len(durations):
#                 raise ValueError(
#                     'dose_list and durations should '
#                     'have the same number of items.')

#             result = sum(dose*duration
#                          for dose, duration in zip(dose_list, durations))

#         return result

#     @staticmethod
#     def average_dose(dose_list: list[float],
#                      durations: list[int] | None = None) -> float:
#         total_dose = Functions.total_dose(dose_list, durations)
#         total_duration = len(
#             dose_list) if durations is None else sum(durations)

#         return total_dose / total_duration
