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
        """
        Computes the Black-76 price and analytic Greeks of a European call or put option
        in the forward measure, using lognormal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or log_normal_sigma <= 0:
            raise ValueError("Time to expiry and implied log-normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}

        # pricing
        sqrt_t = math.sqrt(time_to_expiry)
        vol_sqrt_t = log_normal_sigma * sqrt_t
        d1 = (math.log(forward / strike) + 0.5 * log_normal_sigma**2 * time_to_expiry) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        if option_type == CallOrPut.CALL:
            res[SimpleMetrics.PV] = forward * norm.cdf(d1) - strike * norm.cdf(d2)
        else:
            res[SimpleMetrics.PV] = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)

        # risk
        if calc_risk:
            pdf_d1 = norm.pdf(d1)
            if option_type == CallOrPut.CALL:
                delta = norm.cdf(d1)
                strike_risk = -norm.cdf(d2)
            else:
                delta = norm.cdf(d1) - 1.0
                strike_risk = norm.cdf(-d2)

            gamma = pdf_d1 / (forward * vol_sqrt_t)
            vega = forward * pdf_d1 * sqrt_t
            tte_risk = forward * pdf_d1 * log_normal_sigma / (2.0 * sqrt_t)
            theta = -tte_risk

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
        """
        Computes the Bachelier (normal) price and analytic Greeks of a European call or put option
        in the forward measure, using normal implied volatility.

        res should include
        - SimpleMetrics.PV: present value
        - SimpleMetrics.DELTA: delta
        - SimpleMetrics.GAMMA: gamma
        - SimpleMetrics.VEGA: vega
        - SimpleMetrics.THETA: theta
        - SimpleMetrics.TTE_RISK: time to expiry risk
        - SimpleMetrics.STRIKE_RISK: strike risk

        use calc_risk to control whether to compute the risk metrics or not
        """

        if time_to_expiry <= 0 or normal_sigma <= 0:
            raise ValueError("Time to expiry and implied normal sigma must be positive")

        res: Dict[SimpleMetrics, float] = {}

        # pricing
        sqrt_t = math.sqrt(time_to_expiry)
        sigma_sqrt_t = normal_sigma * sqrt_t
        d = (forward - strike) / sigma_sqrt_t
        if option_type == CallOrPut.CALL:
            res[SimpleMetrics.PV] = (forward - strike) * norm.cdf(d) + sigma_sqrt_t * norm.pdf(d)
        else:
            res[SimpleMetrics.PV] = (strike - forward) * norm.cdf(-d) + sigma_sqrt_t * norm.pdf(d)

        # risk
        if calc_risk:
            pdf_d = norm.pdf(d)
            if option_type == CallOrPut.CALL:
                delta = norm.cdf(d)
                strike_risk = -norm.cdf(d)
            else:
                delta = norm.cdf(d) - 1.0
                strike_risk = norm.cdf(-d)

            gamma = pdf_d / sigma_sqrt_t
            vega = sqrt_t * pdf_d
            tte_risk = normal_sigma * pdf_d / (2.0 * sqrt_t)
            theta = -tte_risk

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
        """
        Computes the implied lognormal volatility from option PV under the Black-76 model and its sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: implied lognormal volatility
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of implied lognormal volatility to forward
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of implied lognormal volatility to time to expiry
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of implied lognormal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not

        """
        res: Dict[SimpleMetrics, float] = {}

        # 1) compute implied vol
        sigma_imp = EuropeanOptionAnalytics._implied_lognormal_vol_black(
            pv=pv,
            forward=forward,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            tol=tol,
        )
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = sigma_imp

        # 2) compute greeks at implied vol
        if calc_risk:
            greeks = EuropeanOptionAnalytics.european_option_log_normal(
                forward=forward,
                strike=strike,
                time_to_expiry=time_to_expiry,
                log_normal_sigma=sigma_imp,
                option_type=option_type,
                calc_risk=True,
            )

            # 3) compute sensitivities of implied vol using implicit function theorem
            # G(\sigma_imp(f, k, tte, pv), f, k, tte) = pv, where G is the pricing function
            # For instance, for f risk, we have
            # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma
            vega = greeks[SimpleMetrics.VEGA]
            delta = greeks[SimpleMetrics.DELTA]
            tte_risk = greeks[SimpleMetrics.TTE_RISK]
            strike_risk = greeks[SimpleMetrics.STRIKE_RISK]

            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = -delta / vega
            res[SimpleMetrics.D_LN_VOL_D_TTE] = -tte_risk / vega
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
        """
        Computes the implied normal volatility from option PV under the Bachelier model and,
        optionally, its sensitivities using the implicit function theorem.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: implied normal volatility
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of implied normal volatility to forward
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of implied normal volatility to time to expiry
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of implied normal volatility to strike

        use calc_risk to control whether to compute the risk metrics or not
        """

        res = {}

        # 1) Compute implied normal vol
        sigma_imp = EuropeanOptionAnalytics._implied_normal_vol_bachelier(
            pv=pv,
            forward=forward,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            tol=tol,
        )
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = sigma_imp

        # 2) Compute Greeks at implied vol
        if calc_risk:
            greeks = EuropeanOptionAnalytics.european_option_normal(
                forward=forward,
                strike=strike,
                time_to_expiry=time_to_expiry,
                normal_sigma=sigma_imp,
                option_type=option_type,
                calc_risk=True,
            )

            # 3) Compute sensitivities of implied vol
            # G(\sigma_imp(f, k, tte), f, k, tte) = pv, where G is the pricing function
            # For instance, for f risk, we have
            # dG/dsigma * dsigma / df = - dG/df => - dG/df / dG/dsigma
            vega = greeks[SimpleMetrics.VEGA]
            delta = greeks[SimpleMetrics.DELTA]
            tte_risk = greeks[SimpleMetrics.TTE_RISK]
            strike_risk = greeks[SimpleMetrics.STRIKE_RISK]

            res[SimpleMetrics.D_N_VOL_D_FORWARD] = -delta / vega
            res[SimpleMetrics.D_N_VOL_D_TTE] = -tte_risk / vega
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
        """
        Converts lognormal implied volatility into normal (Bachelier) implied volatility
        via price equivalence, and compute sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_NORMAL_VOL: equivalent normal implied volatility
        - SimpleMetrics.D_N_VOL_D_LN_VOL: sensitivity of normal vol to lognormal vol
        - SimpleMetrics.D_N_VOL_D_FORWARD: sensitivity of normal vol to forward
        - SimpleMetrics.D_N_VOL_D_STRIKE: sensitivity of normal vol to strike
        - SimpleMetrics.D_N_VOL_D_TTE: sensitivity of normal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) black price (BS'76)
        # V = BS(f, k, tte, log_normal_sigma)
        black = EuropeanOptionAnalytics.european_option_log_normal(
            forward=forward + shift,
            strike=strike + shift,
            time_to_expiry=time_to_expiry,
            log_normal_sigma=log_normal_sigma,
            option_type=option_type,
            calc_risk=calc_risk,
        )
        pv = black[SimpleMetrics.PV]

        # 2) implied normal vol (Bachelier)
        # nv = Imp(f, k, tte, V)
        # notice dnv/dV = 1 / vega
        normal_res = EuropeanOptionAnalytics.implied_normal_vol_sensitivities(
            pv=pv,
            forward=forward + shift,
            strike=strike + shift,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            calc_risk=calc_risk,
            tol=tol,
        )
        sigma_n = normal_res[SimpleMetrics.IMPLIED_NORMAL_VOL]
        res[SimpleMetrics.IMPLIED_NORMAL_VOL] = sigma_n

        if calc_risk:
            normal_greeks = EuropeanOptionAnalytics.european_option_normal(
                forward=forward + shift,
                strike=strike + shift,
                time_to_expiry=time_to_expiry,
                normal_sigma=sigma_n,
                option_type=option_type,
                calc_risk=True,
            )
            vega_n = normal_greeks[SimpleMetrics.VEGA]
            res[SimpleMetrics.D_N_VOL_D_LN_VOL] = black[SimpleMetrics.VEGA] / vega_n
            res[SimpleMetrics.D_N_VOL_D_FORWARD] = (black[SimpleMetrics.DELTA] - normal_greeks[SimpleMetrics.DELTA]) / vega_n
            res[SimpleMetrics.D_N_VOL_D_STRIKE] = (black[SimpleMetrics.STRIKE_RISK] - normal_greeks[SimpleMetrics.STRIKE_RISK]) / vega_n
            res[SimpleMetrics.D_N_VOL_D_TTE] = (black[SimpleMetrics.TTE_RISK] - normal_greeks[SimpleMetrics.TTE_RISK]) / vega_n

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
        """
        Converts normal implied volatility into lognormal implied volatility
        via price equivalence, and computes sensitivities.

        res should include
        - SimpleMetrics.IMPLIED_LOG_NORMAL_VOL: equivalent lognormal implied volatility
        - SimpleMetrics.D_LN_VOL_D_N_VOL: sensitivity of lognormal vol to normal vol
        - SimpleMetrics.D_LN_VOL_D_FORWARD: sensitivity of lognormal vol to forward
        - SimpleMetrics.D_LN_VOL_D_STRIKE: sensitivity of lognormal vol to strike
        - SimpleMetrics.D_LN_VOL_D_TTE: sensitivity of lognormal vol to time to expiry
        """

        res: Dict[SimpleMetrics, float] = {}

        option_type = CallOrPut.PUT if forward > strike else CallOrPut.CALL

        # 1) bachelier
        # V = Bachelier(f, k, tte, normal_sigma)
        normal_res = EuropeanOptionAnalytics.european_option_normal(
            forward=forward + shift,
            strike=strike + shift,
            time_to_expiry=time_to_expiry,
            normal_sigma=normal_sigma,
            option_type=option_type,
            calc_risk=calc_risk,
        )
        pv = normal_res[SimpleMetrics.PV]

        # 2) implied log normal vol (BS'76)
        # ln_nv = Imp(f, k, tte, V)
        # notice dln_nv/dV = 1 / vega
        lognormal_res = EuropeanOptionAnalytics.implied_lognormal_vol_sensitivities(
            pv=pv,
            forward=forward + shift,
            strike=strike + shift,
            time_to_expiry=time_to_expiry,
            option_type=option_type,
            calc_risk=calc_risk,
            tol=tol,
        )
        sigma_ln = lognormal_res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL]
        res[SimpleMetrics.IMPLIED_LOG_NORMAL_VOL] = sigma_ln

        if calc_risk:
            lognormal_greeks = EuropeanOptionAnalytics.european_option_log_normal(
                forward=forward + shift,
                strike=strike + shift,
                time_to_expiry=time_to_expiry,
                log_normal_sigma=sigma_ln,
                option_type=option_type,
                calc_risk=True,
            )

            vega_ln = lognormal_greeks[SimpleMetrics.VEGA]

            res[SimpleMetrics.D_LN_VOL_D_N_VOL] = normal_res[SimpleMetrics.VEGA] / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_FORWARD] = (
                normal_res[SimpleMetrics.DELTA] - lognormal_greeks[SimpleMetrics.DELTA]
            ) / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_STRIKE] = (
                normal_res[SimpleMetrics.STRIKE_RISK] - lognormal_greeks[SimpleMetrics.STRIKE_RISK]
            ) / vega_ln
            res[SimpleMetrics.D_LN_VOL_D_TTE] = (
                normal_res[SimpleMetrics.TTE_RISK] - lognormal_greeks[SimpleMetrics.TTE_RISK]
            ) / vega_ln

        return res

    ### utilities below

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
        """
        Solves for the Black-76 implied lognormal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility
        """
        intrinsic = max(forward - strike, 0.0) if option_type == CallOrPut.CALL else max(strike - forward, 0.0)
        upper_bound = forward if option_type == CallOrPut.CALL else strike

        if pv < intrinsic - tol or pv > upper_bound + tol:
            raise ValueError("Option price is outside arbitrage bounds for Black-76")

        if abs(pv - intrinsic) < tol:
            return vol_min

        sigma = max(
            EuropeanOptionAnalytics._initial_log_normal_implied_vol_guess(forward, time_to_expiry, pv),
            max(vol_min, 1e-8),
        )
        sigma = min(sigma, vol_max)

        low = vol_min
        high = vol_max

        for _ in range(max_iter):
            price_res = EuropeanOptionAnalytics.european_option_log_normal(
                forward=forward,
                strike=strike,
                time_to_expiry=time_to_expiry,
                log_normal_sigma=sigma,
                option_type=option_type,
                calc_risk=True,
            )
            price = price_res[SimpleMetrics.PV]
            vega = price_res[SimpleMetrics.VEGA]
            diff = price - pv

            if abs(diff) < tol:
                return sigma

            if diff > 0:
                high = sigma
            else:
                low = sigma

            if abs(vega) > 1e-14:
                sigma_newton = sigma - diff / vega
            else:
                sigma_newton = 0.5 * (low + high)

            if sigma_newton <= low or sigma_newton >= high:
                sigma = 0.5 * (low + high)
            else:
                sigma = sigma_newton

        raise RuntimeError("Black implied lognormal volatility did not converge")

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
        """
        Solves for the Bachelier implied normal volatility from a European option price using a
        hybrid Newton-Raphson and bisection method, subject to arbitrage bounds and convergence
        controls.

        Return "sigma" implied lognormal volatility
        """
        intrinsic = max(forward - strike, 0.0) if option_type == CallOrPut.CALL else max(strike - forward, 0.0)
        if pv < intrinsic - tol:
            raise ValueError("Option price is outside arbitrage bounds for Bachelier")

        if abs(pv - intrinsic) < tol:
            return vol_min

        sigma = max(
            EuropeanOptionAnalytics._initial_normal_implied_vol_guess(time_to_expiry, pv),
            vol_min,
        )
        sigma = min(sigma, vol_max)

        low = vol_min
        high = vol_max

        for _ in range(max_iter):
            price_res = EuropeanOptionAnalytics.european_option_normal(
                forward=forward,
                strike=strike,
                time_to_expiry=time_to_expiry,
                normal_sigma=sigma,
                option_type=option_type,
                calc_risk=True,
            )
            price = price_res[SimpleMetrics.PV]
            vega = price_res[SimpleMetrics.VEGA]
            diff = price - pv

            if abs(diff) < tol:
                return sigma

            if diff > 0:
                high = sigma
            else:
                low = sigma

            if abs(vega) > 1e-14:
                sigma_newton = sigma - diff / vega
            else:
                sigma_newton = 0.5 * (low + high)

            if sigma_newton <= low or sigma_newton >= high:
                sigma = 0.5 * (low + high)
            else:
                sigma = sigma_newton

        raise RuntimeError("Bachelier implied normal volatility did not converge")

    @staticmethod
    def _initial_log_normal_implied_vol_guess(forward: float, time_to_expiry: float, pv: float):
        return math.sqrt(2 * math.pi / time_to_expiry) * pv / forward

    @staticmethod
    def _initial_normal_implied_vol_guess(time_to_expiry: float, pv: float):
        return pv * math.sqrt(2 * math.pi / time_to_expiry)
