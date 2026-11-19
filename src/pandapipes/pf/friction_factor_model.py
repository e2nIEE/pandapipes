r"""
The friction factor :math:`\lambda` appears in the Darcy‑Weisbach type
pressure drop equations used by pandapipes. The exact form of the pressure
drop equation depends on the fluid model (incompressible or compressible).

pandapipes provides several built-in models. All friction factor models implement
the :class:`~FrictionFactorModel` protocol and provide both :math:`\lambda` and
its derivative :math:`d\lambda / dRe`,
which is essential for :ref:`constructing the Jacobian matrix <jacobian>`
in the Newton‑Raphson solver.

Since :math:`\lambda` decreases as the Reynolds number increases
(as seen in the Moody chart), :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}`
must be negative in **individual models**.
Inside the transition range of :class:`RegimeAwareFrictionModel`,
:math:`d\lambda/dRe` is instead positive -- the blend shifts
toward the higher-:math:`\lambda` turbulent model as Re increases,
so the transitional :math:`\lambda` rises even though each endpoint model
individually decreases.
"""

# needed to preserve typealiases in the docs
from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

Float64_1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FrictionFactorResult: TypeAlias = tuple[
    Float64_1D,
    Float64_1D,
]


@runtime_checkable
class FrictionFactorModel(Protocol):
    r"""Protocol for computing the Darcy-Weisbach friction factor
    :math:`\lambda` and its derivative with respect to the Reynolds number
    :math:`Re`, namely :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}`.

    The conversion to :math:`\mathrm{d}\lambda / \mathrm{d}\dot{m}`, which the
    solver requires, is applied at the call site.

    Examples:
        To use custom :class:`FrictionFactorModel` for computation,
        create one and pass it into :func:`~.pipeflow`:

        .. code-block:: python

            class CustomFrictionFactor:
                def compute_lambda_and_dlambda_dre(
                    self,
                    k_over_D: Float64_1D,
                    re: Float64_1D,
                 ) -> FrictionFactorResult:
                     lambda_ = ...
                     dlambda_dre = ...
                     return lambda_, dlambda_dre
            pp.pipeflow(net, friction_model=CustomFrictionFactor())
    """

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Compute the friction factor and its derivative.

        :param k_over_D: Relative roughness :math:`k/D`.
        :param re: Reynolds number :math:`Re`.
        :return: Tuple of :math:`\lambda` and
            :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}`.
        """
        ...


@dataclass(slots=True)
class Laminar(FrictionFactorModel):
    """Implementation of the Laminar friction factor equation."""

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Laminar friction factor:

        .. math::
            \lambda &= \frac{64}{Re} \\
            \frac{\mathrm{d}\lambda}{\mathrm{d}Re} &= -\frac{64}{Re^2}.
        """
        lambda_ = 64 / re
        dlambda_dre = -64 / re**2
        return lambda_, dlambda_dre


@dataclass(slots=True)
class Nikuradse(FrictionFactorModel):
    """Implementation of the Nikuradse friction factor equation."""

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""The model computes :math:`\lambda` as the sum of a laminar term and the fully
        rough Nikuradse term:

        .. math::
           \lambda &= \frac{64}{Re} \;+\; \frac{1}{\left( -2\log_{10} \left( \frac{k/D}{3.71} \right) \right)^2} \\
           \frac{\mathrm{d}\lambda}{\mathrm{d}Re} &= -\frac{64}{Re^2}.
        """
        laminar = 64 / re
        nikuradse = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
        lambda_ = laminar + nikuradse

        dlambda_dre = -64 / re**2
        return lambda_, dlambda_dre


@dataclass(slots=True)
class SwameeJain(FrictionFactorModel):
    """Implementation of the Swamee‑Jain explicit friction factor equation.

    This is a direct (non‑iterative) approximation valid for turbulent flow.
    """

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Swamee-Jain friction factor:

        .. math::
            \lambda = \frac{0.25}{\left[\log_{10}(x)\right]^2}
                    = \frac{a}{(\ln x)^2},

        where:

        .. math::
            a = 0.25(\ln 10)^2, \quad
            x = \frac{k/D}{3.7} + \frac{5.74}{Re^{0.9}}.

        :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}` is obtained as follows:

        .. math::
            \begin{aligned}
                \frac{d\lambda}{dRe}
                &= \frac{d\lambda}{dx} \cdot \frac{dx}{dRe} \\
                &= \left( -\frac{2a}{x(\ln x)^3} \right)
                \left( -5.74 \cdot 0.9 \, Re^{-1.9} \right) \\
                &= b \, \frac{Re^{-1.9}}{x(\ln x)^3},
            \end{aligned}

        where:

            .. math::
                b = 2a \cdot 5.74 \cdot 0.9 = 2.583 (\ln 10)^2 \approx 13.6948028193657.
        """
        inv_re_09 = 1 / re**0.9
        inner_log_term = k_over_D / 3.7 + 5.74 * inv_re_09
        log_term = np.log(inner_log_term)
        log_squared = log_term * log_term
        log_cubed = log_squared * log_term

        # a = 0.25 * ln(10)**2
        a = 1.325474527619599502640416597148504422899
        lambda_ = a / log_squared

        # b = 0.25 * ln(10)**2 * (-2) * 5.74 * (-0.9)
        b = 13.69480281936570206128078428173834769740
        dlambda_dre = b / (log_cubed * inner_log_term * re**1.9)
        return lambda_, dlambda_dre


LambdaEstimator: TypeAlias = Callable[[Float64_1D, Float64_1D], Float64_1D]


def _default_initial_estimator(k_over_D: Float64_1D, re: Float64_1D) -> Float64_1D:
    """Default lambda estimator used for Colebrook first iteration."""
    return 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2


@dataclass(slots=True)
class Colebrook(FrictionFactorModel):
    r"""Implementation of the Colebrook‑White friction factor equation.

    The Colebrook equation is implicit and solved iteratively using the
    Newton‑Raphson method.

    :param initial_estimator:
        A callable that takes :math:`k/D` and :math:`Re`
        and returns an initial estimate of :math:`\lambda`. This allows
        users to customize the first guess used in the iterative process.
        If ``None``, a default estimator based on the fully rough
        Nikuradse equation is used.
    :param tolerance:
        Convergence tolerance for :math:`\lambda`. The iteration
        stops when the absolute change between successive estimates falls
        below this value.
    :param max_iter:
        Maximum number of Newton‑Raphson iterations allowed.

    Examples:
        Use a custom ``initial_estimator`` and pass the model to the pipeflow:

        .. code-block:: python

            import pandapipes as pp
            colebrook = pp.Colebrook(initial_estimator=lambda k_over_D, re: 64 / re)
            pp.pipeflow(net, friction_model=colebrook)
    """

    initial_estimator: LambdaEstimator | None = None
    tolerance: float = 1e-4
    max_iter: int = 100

    def __post_init__(self):
        if self.initial_estimator is None:
            self.initial_estimator = _default_initial_estimator
        if not self.max_iter > 0:
            msg = "'max_iter' should be > 0"
            raise ValueError(msg)
        if not self.tolerance > 0:
            msg = "'tolerance' should be > 0"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""The Colebrook equation is solved iteratively:

        .. math::
            \frac{1}{\sqrt{\lambda}} = -2 \log_{10}(x),
            \qquad
            x = \frac{k/D}{3.71} + \frac{2.51}{Re\,\sqrt{\lambda}}.

        To derive :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}`, define the implicit function

        .. math::
            F(\lambda, Re) = \lambda^{-1/2} + 2\log_{10}(x) = 0.

        The partial derivatives are:

        .. math::
            \begin{aligned}
            \frac{\partial F}{\partial Re}
            &= - \frac{5.02}{\ln(10)\,Re^2\,x\,\sqrt{\lambda}} , \\[6pt]
            \frac{\partial F}{\partial \lambda}
            &= - \frac{1}{2\lambda^{3/2}}
               - \frac{2.51}{\ln(10)\,Re\,x\,\lambda^{3/2}}.
            \end{aligned}

        Implicit differentiation gives

        .. math::
            \frac{\mathrm{d}\lambda}{\mathrm{d}Re} =
                -\left( \frac{\partial F}{\partial Re} \right) / \left( \frac{\partial F}{\partial \lambda} \right).

        Substituting and simplifying (the :math:`\sqrt{\lambda}` and :math:`x` terms cancel) yields:

        .. math::
            \frac{d\lambda}{dRe} =
            - \frac{10.04\,\lambda}{\left(\ln(10)\,Re\,x + 5.02\right) Re}.
        """
        # TODO: move this import to top level if possible
        from pandapipes.pf.pipeflow_setup import PipeflowNotConverged

        lambda_ = self.initial_estimator(k_over_D, re)
        a = k_over_D / 3.71
        b = 2.51 / re
        # 1 / ln(10)
        inv_ln10 = 0.4342944819032518276511289189166050822944
        for _ in range(self.max_iter):
            inv_lambda_sqrt = 1 / np.sqrt(lambda_)
            inner_log_term = a + b * inv_lambda_sqrt
            cubed_inv_lambda_sqrt = inv_lambda_sqrt * inv_lambda_sqrt * inv_lambda_sqrt

            f = inv_lambda_sqrt + 2 * np.log10(inner_log_term)
            df = (
                -0.5 * cubed_inv_lambda_sqrt
                - b * cubed_inv_lambda_sqrt * inv_ln10 / inner_log_term
            )
            step = f / df
            lambda_ -= step
            if np.all(np.abs(step) < self.tolerance):
                break
        else:
            msg = (
                "The Colebrook algorithm did not converge. "
                "There might be model inconsistencies. The maximum iterations "
                "can be given as 'max_iter_colebrook' argument to the pipeflow."
            )
            raise PipeflowNotConverged(msg)
        ln10 = 2.302585092994045684017991454684364207601
        inner_log_term = a + b / np.sqrt(lambda_)
        dlambda_dre = -10.04 * lambda_ / ((ln10 * inner_log_term * re + 5.02) * re)
        return lambda_, dlambda_dre


class TransitionStrategy(Protocol):
    r"""Protocol for a laminar-turbulent transition blending strategy.

    .. note::
        pandapipes guarantees that
        :math:`0 < Re_{\mathrm{lam}} < Re_{\mathrm{turb}}` are passed to
        this method.

    Implementations must be :math:`C^1`-continuous: both
    :math:`\lambda` and :math:`d\lambda/dRe` must match the
    corresponding laminar and turbulent endpoint values, so the
    friction factor and its mass-flow derivative are continuous across
    the regime boundaries.

    A custom callable does not need to inherit from this protocol;
    structural subtyping ensures that any callable with a compatible
    signature is accepted by type checkers.
    """

    def __call__(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        *,
        re_lam: float,
        re_turb: float,
        f_lam: FrictionFactorModel,
        f_turb: FrictionFactorModel,
    ) -> FrictionFactorResult:
        r"""Blend the laminar and turbulent friction factors.

        :param k_over_D: Relative roughness :math:`k/D`.
        :param re: Reynolds number :math:`Re`.
        :param re_lam: Upper Reynolds number of the laminar regime.
        :param re_turb: Lower Reynolds number of the turbulent regime.
        :param f_lam: Laminar friction factor model.
        :param f_turb: Turbulent friction factor model.
        :return: Tuple of :math:`\lambda` and :math:`d\lambda/dRe`.
        """


def hermite_log(
    k_over_D: Float64_1D,
    re: Float64_1D,
    *,
    re_lam: float,
    re_turb: float,
    f_lam: FrictionFactorModel,
    f_turb: FrictionFactorModel,
) -> FrictionFactorResult:
    r"""Cubic Hermite blending strategy for the laminar-turbulent transition.

    The blend matches both :math:`\lambda` and :math:`d\lambda/dRe` at the endpoints
    :math:`Re_{\mathrm{lam}}` and :math:`Re_{\mathrm{turb}}`, so the friction
    factor and its derivative with respect to Reynolds number are continuous
    across the regime boundaries.

    The interpolation is carried out in the log-log coordinates

    .. math::
        X = \ln Re, \qquad Y = \ln\left(\lambda\,Re^2\right).

    In these coordinates the laminar regime (:math:`\lambda = 64/Re`) is a
    line of slope 1 (:math:`Y = \ln 64 + X`), and the fully rough regime
    (:math:`\lambda \approx \mathrm{const}`) is a line of slope 2
    (:math:`Y = \ln\lambda + 2X`), so the transitional curve is a smooth bend
    between two straight asymptotes.

    The endpoint data for the cubic Hermite interpolant in
    :math:`T = (X - X_{\mathrm{lam}})/(X_{\mathrm{turb}} - X_{\mathrm{lam}})` are

    .. math::
        Y_i = \ln\lambda_i + 2\ln Re_i,
        \qquad
        M_i = \left.\frac{dY}{dX}\right|_{X_i}.

    The substitution

    .. math::
        X &= \ln Re \;\Longrightarrow\; \frac{dRe}{dX} = Re,

    gives the slope

    .. math::
        \frac{dY}{dX}
        &= \frac{dY}{dRe}\frac{dRe}{dX}
         = \left(\frac{1}{\lambda}\frac{d\lambda}{dRe} + \frac{2}{Re}\right) Re \\
        &= \frac{Re}{\lambda}\frac{d\lambda}{dRe} + 2 \\
         \Longrightarrow\quad
         M_i &= \frac{Re_i}{\lambda_i}\frac{d\lambda_i}{dRe} + 2.


    The back-transform is

    .. math::
        \lambda = \frac{e^Y}{Re^2},
        \qquad
        \frac{d\lambda}{dRe}
        = \frac{\lambda}{Re}\left(\frac{dY}{dX} - 2\right).
    """
    # same-shape arrays for the protocol
    re_lam_arr = np.full_like(re, re_lam)
    re_turb_arr = np.full_like(re, re_turb)
    lambda_lam, dlambda_dre_lam = f_lam.compute_lambda_and_dlambda_dre(
        k_over_D,
        re_lam_arr,
    )
    lambda_turb, dlambda_dre_turb = f_turb.compute_lambda_and_dlambda_dre(
        k_over_D,
        re_turb_arr,
    )

    X_lam = math.log(re_lam)
    X_turb = math.log(re_turb)
    Y_lam = np.log(lambda_lam) + 2.0 * X_lam
    Y_turb = np.log(lambda_turb) + 2.0 * X_turb
    M_lam = re_lam_arr / lambda_lam * dlambda_dre_lam + 2.0
    M_turb = re_turb_arr / lambda_turb * dlambda_dre_turb + 2.0

    span = X_turb - X_lam
    inv_span = 1.0 / span

    # normalise coordinate: T = 0 at X_lam, T = 1 at X_turb.
    T = (np.log(re) - X_lam) * inv_span

    span_M_lam = span * M_lam
    span_M_turb = span * M_turb
    # Cubic Hermite in T, expanded and collected into powers of T
    c3 = 2.0 * (Y_lam - Y_turb) + span_M_lam + span_M_turb
    c2 = -3.0 * (Y_lam - Y_turb) - 2.0 * span_M_lam - span_M_turb
    c1 = span_M_lam
    c0 = Y_lam
    Y = ((c3 * T + c2) * T + c1) * T + c0
    dY_dX = ((3.0 * c3 * T + 2.0 * c2) * T + c1) * inv_span

    lambda_ = np.exp(Y) / re**2
    dlambda_re = (lambda_ / re) * (dY_dX - 2.0)
    return lambda_, dlambda_re


def smoothstep(
    k_over_D: Float64_1D,
    re: Float64_1D,
    *,
    re_lam: float,
    re_turb: float,
    f_lam: FrictionFactorModel,
    f_turb: FrictionFactorModel,
) -> FrictionFactorResult:
    r"""Smoothstep blending strategy for the laminar-turbulent transition.

    A blend between a laminar and a turbulent friction factor model using the
    cubic smoothstep weight. In the normalized coordinate

    .. math::
        t = \frac{Re - Re_{\mathrm{lam}}}{Re_{\mathrm{turb}} - Re_{\mathrm{lam}}},
        \qquad
        s(t) = 3t^2 - 2t^3,

    the friction factor is

    .. math::
        \lambda = (1 - s)\,\lambda_{\mathrm{lam}} + s\,\lambda_{\mathrm{turb}}.

    Because :math:`s(0) = 0`, :math:`s(1) = 1`, and :math:`s'(0) = s'(1) = 0`,
    both :math:`\lambda` and :math:`d\lambda/dRe` are continuous at the
    regime boundaries.

    Its derivative with respect to Reynolds number is

    .. math::
        \frac{d\lambda}{dRe}
         = \frac{d\lambda_{\mathrm{lam}}}{dRe}
         + s\left(\frac{d\lambda_{\mathrm{turb}}}{dRe} - \frac{d\lambda_{\mathrm{lam}}}{dRe}\right)
         + \frac{ds}{dRe}\left(\lambda_{\mathrm{turb}} - \lambda_{\mathrm{lam}}\right).

    The weight derivative follows from the chain rule:

    .. math::
        \frac{ds}{dRe}
        = \frac{ds}{dt}\frac{dt}{dRe}
        = \frac{6t(1 - t)}{Re_{\mathrm{turb}} - Re_{\mathrm{lam}}}.
    """
    t = (re - re_lam) / (re_turb - re_lam)
    t_sq = t * t
    s = -2 * t_sq * t + 3 * t_sq

    lambda_lam, dlambda_dre_lam = f_lam.compute_lambda_and_dlambda_dre(k_over_D, re)
    lambda_turb, dlambda_dre_turb = f_turb.compute_lambda_and_dlambda_dre(k_over_D, re)
    ds_dre = (1 / (re_turb - re_lam)) * 6 * (-t_sq + t)
    lambda_ = (1 - s) * lambda_lam + s * lambda_turb

    dlambda_dre = (
        dlambda_dre_lam
        + s * (dlambda_dre_turb - dlambda_dre_lam)
        + ds_dre * (lambda_turb - lambda_lam)
    )
    return lambda_, dlambda_dre


@dataclass(slots=True)
class RegimeAwareFrictionModel(FrictionFactorModel):
    r"""Friction factor model spanning the full Reynolds number range.

    Dispatches each element of :math:`Re` to one of three branches:

    .. math::
        \lambda(Re) =
        \begin{cases}
            \lambda_{\mathrm{lam}}(Re), & Re \le Re_{\mathrm{lam}}, \\
            \lambda_{\mathrm{trans}}(Re), &
                Re_{\mathrm{lam}} < Re < Re_{\mathrm{turb}}, \\
            \lambda_{\mathrm{turb}}(Re), & Re_{\mathrm{turb}} \le Re,
        \end{cases}

    where :math:`\lambda_{\mathrm{trans}}` is computed by
    ``transition_strategy``.

    The composite function is :math:`C^1`-continuous
    provided ``transition_strategy`` is, as required by
    :class:`TransitionStrategy`.

    :param laminar: Model used for :math:`Re \le Re_{\mathrm{lam}}`.
    :param turbulent: Model used for :math:`Re_{\mathrm{turb}} \le Re`.
    :param transition_strategy: Strategy that blends the two models in
        :math:`Re_{\mathrm{lam}} < Re < Re_{\mathrm{turb}}`. Must be
        :math:`C^1`-continuous (see :class:`TransitionStrategy`).
    :param re_lam: Upper Reynolds number of the laminar regime. Must be
        positive.
    :param re_turb: Lower Reynolds number of the turbulent regime. Must
        be greater than ``re_lam``.
    """

    laminar: FrictionFactorModel = dataclasses.field(default_factory=Laminar)
    turbulent: FrictionFactorModel = dataclasses.field(default_factory=Colebrook)
    transition_strategy: TransitionStrategy = dataclasses.field(default=hermite_log)
    re_lam: float = 2000
    re_turb: float = 4000

    def __post_init__(self):
        if self.re_lam <= 0:
            msg = f"re_lam = {self.re_lam}, must be > 0"
            raise ValueError(msg)
        if self.re_turb <= self.re_lam:
            msg = f"re_turb = {self.re_turb}, must be > re_lam = {self.re_lam}"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dre(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Compute :math:`\lambda` and :math:`d\lambda/dRe` by regime.

        Dispatches each element of :math:`Re` to the corresponding branch --
        laminar, transition, or turbulent -- based on its regime.
        """
        lambda_ = np.empty_like(re, dtype=np.float64)
        dlambda_dre = np.empty_like(re, dtype=np.float64)

        lam = re <= self.re_lam
        turb = self.re_turb <= re
        trans = ~lam & ~turb

        for mask, friction_model in zip((lam, turb), (self.laminar, self.turbulent)):
            lambda_[mask], dlambda_dre[mask] = (
                friction_model.compute_lambda_and_dlambda_dre(
                    k_over_D[mask],
                    re[mask],
                )
            )
        lambda_[trans], dlambda_dre[trans] = self.transition_strategy(
            k_over_D[trans],
            re[trans],
            re_lam=self.re_lam,
            re_turb=self.re_turb,
            f_lam=self.laminar,
            f_turb=self.turbulent,
        )
        return lambda_, dlambda_dre
