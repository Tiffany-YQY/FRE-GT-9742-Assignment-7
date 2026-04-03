import math
from enum import Enum
from typing import Optional, Dict
from scipy.stats import norm


class CallOrPut(Enum):

    CALL = "call"
    PUT = "put"
    INVALID = "invalid"

    @classmethod
    def from_string(cls, value: str) -> "CallOrPut":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class SimpleMetrics(Enum):

    ## valuations
    PV = "pv"
    ## vol
    IMPLIED_NORMAL_VOL = "implied_normal_vol"
    IMPLIED_LOG_NORMAL_VOL = "implied_log_normal_vol"
    ## pv sensitivities
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    TTE_RISK = "tte_risk"
    STRIKE_RISK = "strike_risk"
    STRIKE_RISK_2 = "strike_risk_2"
    THETA = "theta"

    ## vol sensitivities
    # nv = f(ln_vol, f, k, tte)
    D_N_VOL_D_LN_VOL = "d_n_vol_d_ln_vol"
    D_N_VOL_D_FORWARD = "d_n_vol_d_forward"
    D_N_VOL_D_TTE = "d_n_vol_d_tte"
    D_N_VOL_D_STRIKE = "d_n_vol_d_strike"
    # ln_vol = f^-1(nv, f, k, tte)
    D_LN_VOL_D_N_VOL = "d_ln_vol_d_n_vol"
    D_LN_VOL_D_FORWARD = "d_ln_vol_d_forward"
    D_LN_VOL_D_TTE = "d_ln_vol_d_tte"
    D_LN_VOL_D_STRIKE = "d_ln_vol_d_strike"

    @classmethod
    def from_string(cls, value: str) -> "SimpleMetrics":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


import math
from scipy.stats import norm
from typing import Optional, Dict

class EuropeanOptionAnalytics:

    @staticmethod
    def european_option_log_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:

        if time_to_expiry <= 0 or log_normal_sigma <= 0:
            raise ValueError("Time to expiry and implied log-normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}

        sigma_sqrt_t = log_normal_sigma * math.sqrt(time_to_expiry)
        d1 = (math.log(forward / strike) + 0.5 * log_normal_sigma ** 2 * time_to_expiry) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        phi = 1.0 if option_type == CallOrPut.CALL else -1.0

        Nd1 = norm.cdf(phi * d1)
        Nd2 = norm.cdf(phi * d2)
        nd1 = norm.pdf(d1)

        pv = phi * (forward * Nd1 - strike * Nd2)
        res[SimpleMetrics.PV] = pv

        if calc_risk:
            delta = phi * Nd1
            gamma = nd1 / (forward * sigma_sqrt_t)
            vega = forward * nd1 * math.sqrt(time_to_expiry)
            theta = -forward * nd1 * log_normal_sigma / (2 * math.sqrt(time_to_expiry))
            tte_risk = theta
            strike_risk = -phi * Nd2

            res[SimpleMetrics.DELTA] = delta
            res[SimpleMetrics.GAMMA] = gamma
            res[SimpleMetrics.VEGA] = vega
            res[SimpleMetrics.THETA] = theta
            res[SimpleMetrics.TTE_RISK] = tte_risk
            res[SimpleMetrics.STRIKE_RISK] = strike_risk

        return res

    @staticmethod
    def european_option_normal(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
    ) -> Dict[SimpleMetrics, float]:

        if time_to_expiry <= 0 or normal_sigma <= 0:
            raise ValueError("Time to expiry and implied normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}

        sigma_sqrt_t = normal_sigma * math.sqrt(time_to_expiry)
        d = (forward - strike) / sigma_sqrt_t

        phi = 1.0 if option_type == CallOrPut.CALL else -1.0

        Nd = norm.cdf(phi * d)
        nd = norm.pdf(d)

        pv = phi * ((forward - strike) * Nd) + sigma_sqrt_t * nd
        # Bachelier formula: phi * [ (F-K)*N(phi*d) ] + sigma*sqrt(T)*n(d)
        # More precisely: phi*(F-K)*N(phi*d) + sigma_sqrt_t * n(d)
        pv = phi * (forward - strike) * norm.cdf(phi * d) + sigma_sqrt_t * norm.pdf(d)

        res[SimpleMetrics.PV] = pv

        if calc_risk:
            delta = phi * norm.cdf(phi * d)
            gamma = norm.pdf(d) / sigma_sqrt_t
            vega = norm.pdf(d) * math.sqrt(time_to_expiry)
            theta = -normal_sigma * norm.pdf(d) / (2 * math.sqrt(time_to_expiry))
            tte_risk = theta
            strike_risk = -phi * norm.cdf(phi * d)

            res[SimpleMetrics.DELTA] = delta
            res[SimpleMetrics.GAMMA] = gamma
            res[SimpleMetrics.VEGA] = vega
            res[SimpleMetrics.THETA] = theta
            res[SimpleMetrics.TTE_RISK] = tte_risk
            res[SimpleMetrics.STRIKE_RISK] = strike_risk

        return res

    @staticmethod
    def implied_lognormal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:

        res: Dict[SimpleMetrics, float] = {}

        sigma = EuropeanOptionAnalytics._implied_lognormal_vol_black(
            pv, forward, strike, time_to_expiry, option_type, tol
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = sigma

        if calc_risk:
            greeks = EuropeanOptionAnalytics.european_option_log_normal(
                forward, strike, time_to_expiry, sigma, option_type, calc_risk=True
            )
            vega = greeks[SimpleMetrics.VEGA]
            delta = greeks[SimpleMetrics.DELTA]
            theta = greeks[SimpleMetrics.THETA]
            strike_risk = greeks[SimpleMetrics.STRIKE_RISK]

            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = -delta / vega
            res[SimpleMetrics.D_LN_VOL_D_TTE] = -theta / vega
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = -strike_risk / vega

        return res

    @staticmethod
    def implied_normal_vol_sensitivities(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        calc_risk: Optional[bool] = False,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:

        res = {}

        sigma = EuropeanOptionAnalytics._implied_normal_vol_bachelier(
            pv, forward, strike, time_to_expiry, option_type, tol
        )
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = sigma

        if calc_risk:
            greeks = EuropeanOptionAnalytics.european_option_normal(
                forward, strike, time_to_expiry, sigma, option_type, calc_risk=True
            )
            vega = greeks[SimpleMetrics.VEGA]
            delta = greeks[SimpleMetrics.DELTA]
            theta = greeks[SimpleMetrics.THETA]
            strike_risk = greeks[SimpleMetrics.STRIKE_RISK]

            res[SimpleMetrics.D_N_VOL_D_FORWARD] = -delta / vega
            res[SimpleMetrics.D_N_VOL_D_TTE] = -theta / vega
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = -strike_risk / vega

        return res

    @staticmethod
    def lognormal_vol_to_normal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        log_normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        black_res = EuropeanOptionAnalytics.european_option_log_normal(
            forward + shift, strike + shift, time_to_expiry, log_normal_sigma, option_type, calc_risk=True
        )
        pv = black_res[SimpleMetrics.PV]
        black_vega = black_res[SimpleMetrics.VEGA]
        black_delta = black_res[SimpleMetrics.DELTA]
        black_theta = black_res[SimpleMetrics.THETA]
        black_strike_risk = black_res[SimpleMetrics.STRIKE_RISK]

        n_sigma = EuropeanOptionAnalytics._implied_normal_vol_bachelier(
            pv, forward + shift, strike + shift, time_to_expiry, option_type, tol
        )
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = n_sigma

        if calc_risk:
            bach_res = EuropeanOptionAnalytics.european_option_normal(
                forward + shift, strike + shift, time_to_expiry, n_sigma, option_type, calc_risk=True
            )
            bach_vega = bach_res[SimpleMetrics.VEGA]
            bach_delta = bach_res[SimpleMetrics.DELTA]
            bach_theta = bach_res[SimpleMetrics.THETA]
            bach_strike_risk = bach_res[SimpleMetrics.STRIKE_RISK]

            d_n_vol_d_V = 1.0 / bach_vega
            d_V_d_ln_vol = black_vega
            d_V_d_forward = black_delta
            d_V_d_strike = black_strike_risk
            d_V_d_tte = black_theta

            res[SimpleMetrics.D_N_VOL_D_LN_VOL] = d_n_vol_d_V * d_V_d_ln_vol
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = d_n_vol_d_V * d_V_d_forward - bach_delta / bach_vega
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = d_n_vol_d_V * d_V_d_strike - bach_strike_risk / bach_vega
            res[SimpleMetrics.D_N_VOL_D_TTE] = d_n_vol_d_V * d_V_d_tte - bach_theta / bach_vega

        return res

    @staticmethod
    def normal_vol_to_lognormal_vol(
        forward: float,
        strike: float,
        time_to_expiry: float,
        normal_sigma: float,
        calc_risk: Optional[bool] = False,
        shift: Optional[float] = 0.0,
        tol: Optional[float] = 1e-8,
    ) -> Dict[SimpleMetrics, float]:

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        bach_res = EuropeanOptionAnalytics.european_option_normal(
            forward + shift, strike + shift, time_to_expiry, normal_sigma, option_type, calc_risk=True
        )
        pv = bach_res[SimpleMetrics.PV]
        bach_vega = bach_res[SimpleMetrics.VEGA]
        bach_delta = bach_res[SimpleMetrics.DELTA]
        bach_theta = bach_res[SimpleMetrics.THETA]
        bach_strike_risk = bach_res[SimpleMetrics.STRIKE_RISK]

        ln_sigma = EuropeanOptionAnalytics._implied_lognormal_vol_black(
            pv, forward + shift, strike + shift, time_to_expiry, option_type, tol
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = ln_sigma

        if calc_risk:
            black_res = EuropeanOptionAnalytics.european_option_log_normal(
                forward + shift, strike + shift, time_to_expiry, ln_sigma, option_type, calc_risk=True
            )
            black_vega = black_res[SimpleMetrics.VEGA]
            black_delta = black_res[SimpleMetrics.DELTA]
            black_theta = black_res[SimpleMetrics.THETA]
            black_strike_risk = black_res[SimpleMetrics.STRIKE_RISK]

            d_ln_vol_d_V = 1.0 / black_vega
            d_V_d_n_vol = bach_vega
            d_V_d_forward = bach_delta
            d_V_d_strike = bach_strike_risk
            d_V_d_tte = bach_theta

            res[SimpleMetrics.D_LN_VOL_D_N_VOL] = d_ln_vol_d_V * d_V_d_n_vol
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = d_ln_vol_d_V * d_V_d_forward - black_delta / black_vega
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = d_ln_vol_d_V * d_V_d_strike - black_strike_risk / black_vega
            res[SimpleMetrics.D_LN_VOL_D_TTE] = d_ln_vol_d_V * d_V_d_tte - black_theta / black_vega

        return res

    @staticmethod
    def _implied_lognormal_vol_black(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 0.0,
        vol_max: Optional[float] = 10.0,
        max_iter: Optional[int] = 1000,
    ) -> float:

        sigma = EuropeanOptionAnalytics._initial_log_normal_implied_vol_guess(forward, time_to_expiry, pv)
        sigma = max(vol_min, min(vol_max, sigma))

        lo, hi = vol_min, vol_max

        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_log_normal(
                forward, strike, time_to_expiry, sigma if sigma > 1e-12 else 1e-12,
                option_type, calc_risk=True
            )
            f_val = res[SimpleMetrics.PV] - pv
            vega = res[SimpleMetrics.VEGA]

            if abs(f_val) < tol:
                break

            if f_val < 0:
                lo = sigma
            else:
                hi = sigma

            if vega > 1e-14:
                sigma_new = sigma - f_val / vega
                if lo < sigma_new < hi:
                    sigma = sigma_new
                    continue

            sigma = 0.5 * (lo + hi)

        return sigma

    @staticmethod
    def _implied_normal_vol_bachelier(
        pv: float,
        forward: float,
        strike: float,
        time_to_expiry: float,
        option_type: Optional[CallOrPut] = CallOrPut.CALL,
        tol: Optional[float] = 1e-8,
        vol_min: Optional[float] = 1e-8,
        vol_max: Optional[float] = 0.1,
        max_iter: Optional[int] = 100,
    ) -> float:

        sigma = EuropeanOptionAnalytics._initial_normal_implied_vol_guess(time_to_expiry, pv)
        sigma = max(vol_min, min(vol_max, sigma))

        lo, hi = vol_min, vol_max

        for _ in range(max_iter):
            res = EuropeanOptionAnalytics.european_option_normal(
                forward, strike, time_to_expiry, sigma, option_type, calc_risk=True
            )
            f_val = res[SimpleMetrics.PV] - pv
            vega = res[SimpleMetrics.VEGA]

            if abs(f_val) < tol:
                break

            if f_val < 0:
                lo = sigma
            else:
                hi = sigma

            if vega > 1e-14:
                sigma_new = sigma - f_val / vega
                if lo < sigma_new < hi:
                    sigma = sigma_new
                    continue

            sigma = 0.5 * (lo + hi)

        return sigma

    @staticmethod
    def _initial_log_normal_implied_vol_guess(forward: float, time_to_expiry: float, pv: float):
        return math.sqrt(2 * math.pi / time_to_expiry) * pv / forward

    @staticmethod
    def _initial_normal_implied_vol_guess(time_to_expiry: float, pv: float):
        return pv * math.sqrt(2 * math.pi / time_to_expiry)