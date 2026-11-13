import dataclasses
from dataclasses import dataclass
from functools import partial
from unittest.mock import Mock

import numpy as np
import pytest

import pandapipes as pp
from pandapipes.pf import friction_factor_model as fm
from pandapipes.pf.derivative_calculation import calc_moody


@pytest.fixture
def model_payload():
    """Typical input arrays (k_over_D, Re) for a friction factor model."""
    dtype = np.float64
    return {
        "k_over_D": np.array([0.05], dtype=dtype),
        "re": np.array([2000], dtype=dtype),
    }


@pytest.mark.parametrize(
    "model_class, expected_lambda, expected_dlambda_dre",
    (
        (fm.Laminar, 0.032, -1.6e-5),
        (fm.Nikuradse, 0.103461, -1.6e-5),
        (fm.SwameeJain, 0.0858357, -6.1399e-6),
        (fm.Colebrook, 0.0818189, -4.7054e-6),
    ),
)
def test_compute_lambda_and_dlambda_dre(
    model_payload,
    model_class,
    expected_lambda,
    expected_dlambda_dre,
):
    """Verify that each friction factor model gives the known correct
    output (and derivative) for a standard input -- a regression test.
    """
    model = model_class()
    lambda_, dlambda_dre = model.compute_lambda_and_dlambda_dre(**model_payload)
    np.testing.assert_allclose(lambda_, expected_lambda, rtol=1e-6)
    np.testing.assert_allclose(dlambda_dre, expected_dlambda_dre, rtol=1e-5)


_INDIVIDUAL_MODELS = (
    fm.Laminar,
    fm.Nikuradse,
    fm.SwameeJain,
    fm.Colebrook,
)

_REGIME_AWARE_MODELS = (
    pytest.param(
        partial(fm.RegimeAwareFrictionModel, transition_strategy=fm.hermite_log),
        id="Regime_Aware_Hermite",
    ),
    pytest.param(
        partial(fm.RegimeAwareFrictionModel, transition_strategy=fm.smoothstep),
        id="Regime_Aware_Smoothstep",
    ),
)


@pytest.fixture(params=[*_INDIVIDUAL_MODELS, *_REGIME_AWARE_MODELS])
def model_class(request):
    """Fixture that yields each friction‑factor model class."""
    return request.param


def test_friction_result_shape_and_dtype(
    model_class,
    model_payload,
):
    """Check lambda and dlambda / dre are returned as 1D float64 arrays.

    Even if the input values are of int type (where possible).
    """
    model = model_class()
    model_payload["re"] = model_payload["re"].astype(np.int64)
    lambda_, dlambda_dre = model.compute_lambda_and_dlambda_dre(**model_payload)

    re = model_payload["re"]
    for param in (lambda_, dlambda_dre):
        assert isinstance(param, np.ndarray)
        assert param.dtype == np.float64
        assert param.size == re.size
        assert param.shape == re.shape


def test_empty_inputs(model_class):
    """Models must handle empty arrays gracefully (e.g., empty regime masks)."""
    model = model_class()
    re = np.array([])
    k_over_D = np.array([])
    lambda_, dlambda_dre = model.compute_lambda_and_dlambda_dre(k_over_D, re)
    assert lambda_.shape == (0,)
    assert dlambda_dre.shape == (0,)


@pytest.mark.parametrize("model_class", _INDIVIDUAL_MODELS)
def test_lambda_decreases_as_Re_increases(model_class):
    """lambda decreases as Re increases, as observed experimentally (Moody chart)."""
    model = model_class()
    re = np.array((5_000, 10_000, 100_000, 1_000_000), dtype=np.float64)
    k_over_D = np.full_like(re, 0.05)
    lambdas, _ = model.compute_lambda_and_dlambda_dre(k_over_D, re)
    assert all(np.diff(lambdas) < 0), lambdas


@pytest.mark.parametrize("model_class", _REGIME_AWARE_MODELS)
def test_regime_aware_derivative_positive_in_transition(model_class):
    r"""lambda increases as Re increases in the transition regime.

    Note that :math:`d\lambda/dRe` may still be negative near the
    boundaries, so we test only in the middle of the regime.
    """
    model = model_class()
    # interior points only
    re = np.linspace(model.re_lam, model.re_turb, 5)[1:-1]
    k_over_D = np.full_like(re, 0.1)
    _, dlambda_dre = model.compute_lambda_and_dlambda_dre(k_over_D, re)
    assert np.all(dlambda_dre > 0)


def test_colebrook_convergence_failure(model_payload):
    """Verify that the Colebrook model raises a PipeflowNotConverged error
    when the iterative solution fails.
    """
    from pandapipes.pf.pipeflow_setup import PipeflowNotConverged

    model = fm.Colebrook(max_iter=1, tolerance=1e-12)
    with pytest.raises(PipeflowNotConverged):
        model.compute_lambda_and_dlambda_dre(**model_payload)


def test_colebrook_estimator_called_once(model_payload):
    """Verify that the initial estimator is called
    exactly once per computation (not once per iteration).
    """
    estimators = (
        Mock(side_effect=fm._default_initial_estimator),
        Mock(side_effect=fm._default_initial_estimator),
    )
    for estimator in estimators:
        model = fm.Colebrook(initial_estimator=estimator)
        model.compute_lambda_and_dlambda_dre(**model_payload)
        estimator.assert_called_once()

    model = fm.Colebrook()
    for estimator in estimators:
        estimator.reset_mock()
        model.initial_estimator = estimator
        model.compute_lambda_and_dlambda_dre(**model_payload)
        estimator.assert_called_once()


@pytest.fixture(params=[fm.hermite_log, fm.smoothstep])
def transition_strategy(request):
    """Fixture that yields each transition strategy function."""
    return request.param


def test_transition_strategy_c1_continuity(transition_strategy):
    """Check the transition strategies are C1-continuous at re_lam / re_turb.

    At either boundary the blended result (and its derivative) must
    exactly reproduce the laminar or turbulent model it joins up with.
    """
    re_lam = 2000
    re_turb = 4000
    re = np.array([re_lam, re_turb], dtype=np.float64)
    k_over_D = np.full_like(re, 0.1)
    f_lam = fm.Laminar()
    f_turb = fm.Colebrook()
    lambda_, dlambda_dre = transition_strategy(
        k_over_D,
        re,
        re_lam=re_lam,
        re_turb=re_turb,
        f_lam=f_lam,
        f_turb=f_turb,
    )
    lambda_lam, dlambda_dre_lam = f_lam.compute_lambda_and_dlambda_dre(k_over_D, re)
    lambda_turb, dlambda_dre_turb = f_turb.compute_lambda_and_dlambda_dre(k_over_D, re)
    np.testing.assert_allclose(lambda_[0], lambda_lam[0])
    np.testing.assert_allclose(dlambda_dre[0], dlambda_dre_lam[0])
    np.testing.assert_allclose(lambda_[1], lambda_turb[1])
    np.testing.assert_allclose(dlambda_dre[1], dlambda_dre_turb[1])


@dataclass(slots=True)
class SpyFrictionModel:
    """A stub friction model returning constant values and recording
    the Re values it was called with -- used to spy on regime dispatch.

    Implements both the sub-model interface (`compute_lambda_and_dlambda_dre`)
    and the transition-strategy call interface (`__call__`).
    """

    lambda_: float
    dlambda_dre: float
    re_seen: list = dataclasses.field(default_factory=list)

    def __call__(self, k_over_D, re, *, re_lam, re_turb, f_lam, f_turb):
        self.re_seen.extend(re.tolist())
        return np.full_like(re, self.lambda_), np.full_like(re, self.dlambda_dre)

    def compute_lambda_and_dlambda_dre(self, k_over_D, re):
        self.re_seen.extend(re.tolist())
        return np.full_like(re, self.lambda_), np.full_like(re, self.dlambda_dre)


def test_regime_aware_respects_regimes():
    """Verify that each sub-model only sees Re values from its own regime.

    Each Re is dispatched to exactly one of {laminar, transition, turbulent}
    based on re_lam / re_turb, and the stitched result reflects that.
    """
    laminar = SpyFrictionModel(1, 2)
    turbulent = SpyFrictionModel(3, 4)
    transition_strategy = SpyFrictionModel(5, 6)
    model = fm.RegimeAwareFrictionModel(
        laminar=laminar,
        turbulent=turbulent,
        transition_strategy=transition_strategy,
        re_lam=2000,
        re_turb=4000,
    )

    re = np.array([1999, 2000, 2001, 3999, 4000, 4001], dtype=np.float64)
    k_over_D = np.full_like(re, 0.1)

    expected_lambda = np.repeat(
        [
            laminar.lambda_,
            transition_strategy.lambda_,
            turbulent.lambda_,
        ],
        repeats=2,
    )
    expected_dlambda_dre = np.repeat(
        [
            laminar.dlambda_dre,
            transition_strategy.dlambda_dre,
            turbulent.dlambda_dre,
        ],
        repeats=2,
    )
    lambda_, dlambda_dre = model.compute_lambda_and_dlambda_dre(k_over_D, re)

    np.testing.assert_allclose(lambda_, expected_lambda)
    np.testing.assert_allclose(dlambda_dre, expected_dlambda_dre)
    assert laminar.re_seen == [1999, 2000]
    assert transition_strategy.re_seen == [2001, 3999]
    assert turbulent.re_seen == [4000, 4001]


def test_regime_aware_validates_re_thresholds():
    """Verify that invalid re_lam / re_turb configurations are rejected."""
    payload = {
        "laminar": SpyFrictionModel(1, 2),
        "turbulent": SpyFrictionModel(5, 6),
        "transition_strategy": SpyFrictionModel(3, 4),
    }
    with pytest.raises(ValueError, match="^re_lam"):
        fm.RegimeAwareFrictionModel(**payload, re_lam=-1, re_turb=1)
    with pytest.raises(ValueError, match="^re_turb"):
        fm.RegimeAwareFrictionModel(**payload, re_lam=2, re_turb=1)
    with pytest.raises(ValueError, match="^re_turb"):
        fm.RegimeAwareFrictionModel(**payload, re_lam=2, re_turb=2)


@pytest.fixture
def one_pipe_net():
    """A simple one-pipe gas network."""
    net = pp.create_empty_network("", "lgas")
    pp.create_junctions(net, nr_junctions=2, pn_bar=1, tfluid_k=273.15)
    pp.create_ext_grid(net, junction=0, p_bar=1, t_k=273.15)
    pp.create_sink(net, junction=1, mdot_kg_per_s=1)
    pp.create_pipe_from_parameters(
        net,
        from_junction=0,
        to_junction=1,
        k=0.1,
        inner_diameter_mm=100,
        outer_diameter_mm=120,
        length_km=0.001,
    )
    return net


def test_one_pipe_net(one_pipe_net, model_class, use_numba):
    """Integration test: pipeflow converges when using
    each friction factor model (via the model_class fixture).
    """
    model = model_class()
    pp.pipeflow(one_pipe_net, friction_model=model, use_numba=use_numba)


@pytest.mark.parametrize(
    "model_name, expected_model_class",
    (
        ("colebrook", fm.Colebrook),
        ("swamee-jain", fm.SwameeJain),
        ("nikuradse", fm.Nikuradse),
    ),
)
def test_friction_factor_model_as_string_still_works(
    one_pipe_net,
    model_name,
    expected_model_class,
):
    """Backward‑compatibility check: passing the friction model as a string
    (e.g. 'colebrook') still works.
    """
    pp.pipeflow(one_pipe_net, friction_model=model_name, use_numba=False)
    assert isinstance(one_pipe_net["_options"]["friction_model"], expected_model_class)


@pytest.fixture
def moody_payload():
    """Three branches: a normal one, one at standstill and one of zero length."""
    d = np.full(3, 0.1)
    return {
        "m": np.array([2., 0., 2.]),
        "eta": np.full(3, 1e-3),
        "d": d,
        "k": np.full(3, 0.1e-3),
        "area": np.pi * (d / 2) ** 2,
        "lengths": np.array([1., 1., 0.]),
    }


def test_calc_moody_matches_the_model_it_was_given(moody_payload, model_class):
    """calc_moody does no friction physics of its own - it only decides which rows the model is
    asked about, and hands back what the model returned for them.
    """
    model = model_class()
    re, lambda_, der_lambda = calc_moody(friction_model=model, **moody_payload)

    expected_re = (moody_payload["m"] * moody_payload["d"]
                   / (moody_payload["eta"] * moody_payload["area"]))
    np.testing.assert_allclose(re, expected_re)

    expected_lambda, expected_der = model.compute_lambda_and_dlambda_dre(
        moody_payload["k"][:1] / moody_payload["d"][:1], re[:1])
    expected_der *= expected_re[:1] / moody_payload["m"][:1]
    np.testing.assert_allclose(lambda_[:1], expected_lambda)
    np.testing.assert_allclose(der_lambda[:1], expected_der)


def test_calc_moody_leaves_out_branches_without_a_friction_term(moody_payload, model_class):
    """A branch at standstill has no Reynolds number to look a lambda up for, and a branch of
    zero length has no friction term in its momentum equation at all - every model would divide
    by zero on the first of the two if it were evaluated anyway.
    """
    re, lambda_, der_lambda = calc_moody(friction_model=model_class(), **moody_payload)

    np.testing.assert_array_equal(lambda_[1:], 0.)
    np.testing.assert_array_equal(der_lambda[1:], 0.)
    # the zero-length branch does carry flow, so its Reynolds number is reported either way
    assert re[1] == 0.
    assert re[2] > 0.


@pytest.mark.parametrize("fluid", ["water", "lgas"])
@pytest.mark.parametrize("k_mm", [0.1, 0.])
def test_parallel_stagnant_pipes_converge(use_numba, k_mm, fluid):
    """The quadratic friction loss has no slope at zero flow. Parallel pipes without loss
    coefficient that carry no flow rely on the laminar slope to keep the Jacobian regular.
    """
    net = pp.create_empty_network(fluid=fluid)
    j = pp.create_junctions(net, 4, pn_bar=5, tfluid_k=293.15)
    pp.create_ext_grid(net, j[0], p_bar=5, t_k=293.15)
    pp.create_pipes_from_parameters(net, [j[0], j[1], j[1], j[2]], [j[1], j[2], j[2], j[3]],
                                    length_km=0.1, inner_diameter_mm=50, k_mm=k_mm)
    pp.create_sink(net, j[1], mdot_kg_per_s=0.05)

    pp.pipeflow(net, use_numba=use_numba)

    np.testing.assert_allclose(net.res_pipe.mdot_from_kg_per_s.values[1:], 0., atol=1e-8)


def test_calc_moody_accepts_a_model_name(moody_payload):
    """Same backwards compatibility as the pipeflow option itself."""
    by_name = calc_moody(friction_model="swamee-jain", **moody_payload)
    by_instance = calc_moody(friction_model=fm.SwameeJain(), **moody_payload)

    for from_name, from_instance in zip(by_name, by_instance):
        np.testing.assert_allclose(from_name, from_instance)
