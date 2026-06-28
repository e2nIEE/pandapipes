r"""
.. _friction-derivations:

Friction factor models for pipe flow
------------------------------------

This module provides implementations of the Darcy‑Weisbach friction factor,
including explicit models (Swamee‑Jain, Nikuradse), an iterative model
(Colebrook), and a regime‑aware model that switches between them
based on the Reynolds number. All models conform to the
:class:`FrictionFactorModel` protocol.

Each model also computes the derivative

.. math::
    \frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}

of the friction factor with respect to mass flow. This derivative is essential
for building the Jacobian matrix in the Newton‑Raphson pipe flow solver.
"""

# needed to preserve typealiases in the docs
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import numpy as np

Float64_1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
FrictionFactorResult: TypeAlias = tuple[
    Float64_1D,
    Float64_1D,
]


@runtime_checkable
class FrictionFactorModel(Protocol):
    r"""Protocol for computing the Darcy‑Weisbach friction factor
    :math:`\lambda` and its derivative with respect to mass flow
    :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`.
    """

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Compute the friction factor and its derivative.

        The friction factor :math:`\lambda` is an even function of the mass flow
        :math:`\dot{m}`:

        .. math::
            \lambda(-\dot{m}) = \lambda(\dot{m}).

        Its derivative :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}` is
        therefore an odd function:

        .. math::
            \frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}(-\dot{m})
            = - \frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}(\dot{m}).

        These symmetry properties must be respected by any implementation.

        .. note::
                pandapipes guarantees that :math:`Re > 0` and :math:`\dot{m} \neq 0`
                are passed to this method.

        :param k_over_D: Relative roughness :math:`k/D`.
        :param re: Reynolds number :math:`Re`.
        :param m: Mass flow :math:`\dot{m}`.
        :return: Tuple of :math:`\lambda` and
            :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`.
        """
        ...


@dataclass(slots=True)
class SwameeJain(FrictionFactorModel):
    """Implementation of the Swamee‑Jain explicit friction factor equation.

    This is a direct (non‑iterative) approximation valid for turbulent flow.
    """

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Swamee-Jain friction factor:

        .. math::
            \lambda = \frac{0.25}{\left[\log_{10}(x)\right]^2}
                    = \frac{a}{(\ln x)^2},

        where:

        .. math::
            a = 0.25(\ln 10)^2, \quad
            x = \frac{k/D}{3.7} + \frac{5.74}{Re^{0.9}}.

        :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}` is obtained as follows:

        .. math::
            \begin{aligned}
                \frac{d\lambda}{dm}
                &= \frac{d\lambda}{dx} \cdot \frac{dx}{dRe} \cdot \frac{dRe}{dm} \\
                &= \left( -\frac{2a}{x(\ln x)^3} \right)
                \left( -5.74 \cdot 0.9 \, Re^{-1.9} \right)
                \left( \frac{Re}{m} \right) \\
                &= b \, \frac{Re^{-0.9}}{x(\ln x)^3 \, m},
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

        # a = 0.25 * ln(10)**2 * (-2) * 5.74 * (-0.9)
        b = 13.69480281936570206128078428173834769740
        dlambda_dm = b * inv_re_09 / (log_cubed * inner_log_term * m)
        return lambda_, dlambda_dm


@dataclass(slots=True)
class Nikuradse(FrictionFactorModel):
    """Implementation of the Nikuradse friction factor equation."""

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        r"""The model computes :math:`\lambda` as the sum of a laminar term and the fully
        rough Nikuradse term:

        .. math::
           \lambda = \frac{64}{Re} \;+\; \frac{1}{\left( -2\log_{10} \left( \frac{k/D}{3.71} \right) \right)^2}.

        :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}` is obtained as follows:

        .. math::
           \lambda = \frac{64}{\mathrm{Re}} + \text{const}
           \;\Longrightarrow\;
           \frac{d\lambda}{d\dot{m}} = -\frac{64}{\mathrm{Re}^2}\frac{d\mathrm{Re}}{d\dot{m}}.

        With :math:`\mathrm{Re} = C|\dot{m}|`, we have

        .. math::
           \frac{d\mathrm{Re}}{d\dot{m}} = C \,\mathrm{sgn}(\dot{m}),
           \qquad \mathrm{sgn}(\dot{m}) = \frac{\dot{m}}{|\dot{m}|}.

        Therefore,

        .. math::
           \frac{d\lambda}{d\dot{m}}
           &= -\frac{64}{(C|\dot{m}|)^2} \cdot C \frac{\dot{m}}{|\dot{m}|} \\
           &= -\frac{64}{C |\dot{m}| \dot{m}} \\
           &= -\frac{64}{\mathrm{Re} \cdot \dot{m}}.
        """
        laminar = 64 / re
        nikuradse = 1 / (-2 * np.log10(k_over_D / 3.71)) ** 2
        lambda_ = laminar + nikuradse

        # FIXME?: mathematically, dlambda / dm should be an odd function,
        # but with m**2 the function is even
        # return -64 / (re * m)
        dlambda_dm = -64 / (re * np.abs(m))
        return lambda_, dlambda_dm


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
        Use custom ``initial_estimator``:

        >>> import pandapipes as pp
        >>> colebrook = pp.Colebrook(initial_estimator=lambda k_over_D, re: 64 / re)
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

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        r"""The Colebrook equation is solved iteratively:

        .. math::
            \frac{1}{\sqrt{\lambda}} = -2 \log_{10}(x),
            \qquad
            x = \frac{k/D}{3.71} + \frac{2.51}{Re\,\sqrt{\lambda}}.

        To derive :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}m}`, define the implicit function

        .. math::
            F(\lambda, m) = \lambda^{-1/2} + 2\log_{10}(x) = 0.

        Using :math:`Re = C|m|` (so :math:`dRe/dm = Re/m`), the partial derivatives are:

        .. math::
            \begin{aligned}
            \frac{\partial F}{\partial m}
            &= - \frac{5.02}{\ln(10)\,Re\,x\,\sqrt{\lambda}} \frac{1}{m}, \\[6pt]
            \frac{\partial F}{\partial \lambda}
            &= - \frac{1}{2\lambda^{3/2}}
               - \frac{2.51}{\ln(10)\,Re\,x\,\lambda^{3/2}}.
            \end{aligned}

        Implicit differentiation gives

        .. math::
            \frac{\mathrm{d}\lambda}{\mathrm{d}m} =
                -\left( \frac{\partial F}{\partial m} \right) / \left( \frac{\partial F}{\partial \lambda} \right).

        Substituting and simplifying (the :math:`\sqrt{\lambda}` and :math:`x` terms cancel) yields:

        .. math::
            \frac{d\lambda}{dm} =
            - \frac{10.04\,\lambda}{m\left(\ln(10)\,Re\,x + 5.02\right)}.
        """
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged

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
        dlambda_dm = -10.04 * lambda_ / ((ln10 * inner_log_term * re + 5.02) * m)
        return lambda_, dlambda_dm


@dataclass(slots=True)
class RegimeAwareFrictionFactorModel(FrictionFactorModel):
    r"""Friction factor that respects flow regimes.

    Uses appropriate friction factor model for a specified flow regime.
    Laminar, transient and turbulent flow regimes are supported.

    - laminar:   :math:`0 < Re \le \text{re\_laminar}`
    - transient: :math:`\text{re\_laminar} < Re \le \text{re\_turbulent}`
    - turbulent: :math:`\text{re\_turbulent} < Re`

    The derivative is computed by the same sub‑model, ensuring consistency
    across the regime transition.

    :param laminar:
        Friction factor model used for laminar flow.
    :param transient:
        Friction factor model used for transient flow.
    :param turbulent:
        Friction factor model used for turbulent flow.
    :param re_laminar:
        Upper Reynolds number bound for the laminar regime.
    :param re_turbulent:
        Lower Reynolds number bound for the turbulent regime.
    """

    laminar: FrictionFactorModel
    transient: FrictionFactorModel
    turbulent: FrictionFactorModel
    re_laminar: float = 2300
    re_turbulent: float = 4000

    def __post_init__(self):
        if not (0 < self.re_laminar < self.re_turbulent):
            msg = "Must have 0 < re_laminar < re_turbulent"
            raise ValueError(msg)

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D: Float64_1D,
        re: Float64_1D,
        m: Float64_1D,
    ) -> FrictionFactorResult:
        r"""Compute :math:`\lambda` and :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`.

        The appropriate sub‑model is selected based on the Reynolds number
        according to the regimes defined in the class docstring.
        """
        lam = re <= self.re_laminar
        turb = re > self.re_turbulent
        trans = ~lam & ~turb
        ranges = [
            (lam, self.laminar),
            (trans, self.transient),
            (turb, self.turbulent),
        ]

        lambda_ = np.empty_like(re, dtype=np.float64)
        dlambda_dm = np.empty_like(lambda_)
        for mask, model in ranges:
            if mask.any():
                lambda_[mask], dlambda_dm[mask] = model.compute_lambda_and_dlambda_dm(
                    k_over_D[mask],
                    re[mask],
                    m[mask],
                )

        return lambda_, dlambda_dm
