from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

import pandapipes as pp
from pandapipes.pf import friction_factor_model as fm


@pytest.fixture
def model_payload():
    """Typical input arrays (k_over_D, Re, mass flow) for a friction factor model."""
    dtype = np.float64
    return {
        "k_over_D": np.array([0.05], dtype=dtype),
        "re": np.array([2000], dtype=dtype),
        "m": np.array([1], dtype=dtype),
    }


@pytest.mark.parametrize(
    "model_class, expected_lambda, expected_dlambda_dm",
    (
        (fm.Nikuradse, 0.103461, -0.032),
        (fm.SwameeJain, 0.085836, -0.012279),
        (fm.Colebrook, 0.081818, -0.009411),
    ),
)
def test_compute_lambda_and_dlambda_dm(
    model_payload,
    model_class,
    expected_lambda,
    expected_dlambda_dm,
):
    """Verify that each friction factor model gives the known correct
    output (and derivative) for a standard input -- a regression test.
    """
    model = model_class()
    lambda_, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    np.testing.assert_allclose(lambda_, expected_lambda, atol=1e-6)
    np.testing.assert_allclose(dlambda_dm, expected_dlambda_dm, atol=1e-6)


def MockRegimeAwareFrictionFactorModel(
    re_laminar=2300,
    re_turbulent=4000,
):
    """Factory for a RegimeAwareFrictionFactorModel with typical sub‑models,
    to reduce duplication in tests.
    """
    return fm.RegimeAwareFrictionFactorModel(
        laminar=fm.Nikuradse(),
        transient=fm.SwameeJain(),
        turbulent=fm.Colebrook(),
        re_laminar=re_laminar,
        re_turbulent=re_turbulent,
    )


@pytest.fixture(
    params=[
        fm.Nikuradse,
        fm.SwameeJain,
        fm.Colebrook,
        MockRegimeAwareFrictionFactorModel,
    ]
)
def model_class(request):
    """Fixture that yields each friction‑factor model class."""
    return request.param


def test_friction_result_shape_and_dtype(
    model_class,
    model_payload,
):
    """Check lambda and dlambda / dm are returned as 1D float64 arrays.

    Even if the input values are of int type (where possible).
    """
    model = model_class()
    model_payload["re"] = model_payload["re"].astype(np.int64)
    model_payload["m"] = model_payload["m"].astype(np.int64)
    lambda_, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)

    re = model_payload["re"]
    for param in (lambda_, dlambda_dm):
        assert isinstance(param, np.ndarray)
        assert param.dtype == np.float64
        assert param.size == re.size
        assert param.shape == re.shape


@pytest.mark.parametrize(
    "model_class",
    (
        fm.Nikuradse,
        fm.SwameeJain,
        fm.Colebrook,
    ),
)
def test_lambda_independent_of_explicit_m(model_class, model_payload):
    """Test lambda(m) depends only on Re and k_over_D.

    lambda(m) should be not depend on the explicitly passed m:
    it's used only for derivative computation.
    """
    model = model_class()
    model_payload["m"] = np.array([0.1, 1, 10, 100, 1000])
    lambdas, _ = model.compute_lambda_and_dlambda_dm(**model_payload)

    lambda0 = lambdas[0]
    for lambda_ in lambdas[1:]:
        np.testing.assert_allclose(lambda0, lambda_)


@pytest.mark.parametrize(
    "model_class",
    (
        pytest.param(
            fm.Nikuradse,
            marks=pytest.mark.skip(reason="dlambda / dm is not odd yet"),
        ),
        fm.SwameeJain,
        fm.Colebrook,
    ),
)
def test_dlambda_dm_is_odd(model_class, model_payload):
    """Test dlambda / dm is an odd function w.r.t. m.

    Since lambda(m) is an even function, its derivative should be
    an odd one: -f(m) = f(-m).
    """
    model = model_class()
    _, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    model_payload["m"] *= -1
    _, dlambda_dm2 = model.compute_lambda_and_dlambda_dm(**model_payload)
    np.testing.assert_allclose(-dlambda_dm, dlambda_dm2)

@pytest.mark.parametrize(
    "model_class",
    (
        pytest.param(
            fm.Nikuradse,
            marks=pytest.mark.skip(reason="dlambda / dm is not correct yet"),
        ),
        fm.SwameeJain,
        fm.Colebrook,
    ),
)
def test_lambda_decreases_as_Re_increases(model_class, model_payload):
    """Test lambda(m) decreases as Re increases.

    Experiments show (e.g. Moody chart), that with increasing Re, lambda decreases.

    lambda(m) = lambda(Re), with Re proportional to |m|, therefore
    dlambda_dm = dlambda_dRe * dRe_dm.

    "lambda decreases as Re increases" means, that dlambda_dRe < 0 (by definition).
    For m > 0, dRe_dm > 0, therefore dlambda_dm = dlambda_dRe * dRe_dm < 0.

    Since dlambda_dm is an odd function, for m < 0 it should be > 0.
    """
    model = model_class()
    model_payload["m"] = 1
    _, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    np.testing.assert_allclose(dlambda_dm < 0, True)

    model_payload["m"] = -1
    _, dlambda_dm = model.compute_lambda_and_dlambda_dm(**model_payload)
    np.testing.assert_allclose(dlambda_dm > 0, True)


@dataclass(slots=True)
class MockFrictionFactorModel(fm.FrictionFactorModel):
    """A utility class, that returns predictable lambda and dlambda / dm."""

    res_value: float = 1

    def compute_lambda_and_dlambda_dm(
        self,
        k_over_D,
        re,
        m,
    ) -> fm.FrictionFactorResult:
        res = np.full_like(re, self.res_value)
        return res, res


def test_regime_aware_friction_factor_model_respects_regimes_ranges(
    model_payload,
):
    """Verify that the regime‑aware model delegates to the right sub‑model
    depending on the Reynolds number, including boundary values.
    """
    lam_value = 1
    trans_value = 2
    turb_value = 3
    re_lam = 2300
    re_turb = 4000
    model = fm.RegimeAwareFrictionFactorModel(
        re_laminar=re_lam,
        re_turbulent=re_turb,
        laminar=MockFrictionFactorModel(lam_value),
        transient=MockFrictionFactorModel(trans_value),
        turbulent=MockFrictionFactorModel(turb_value),
    )
    model_payload.pop("re")

    def _assert_lambda_and_dlambda_dm(re, expected_val):
        re = np.atleast_1d(re).astype(np.float64)
        lambda_, dlambda_dm = model.compute_lambda_and_dlambda_dm(
            **model_payload,
            re=re,
        )
        np.testing.assert_allclose(lambda_, expected_val)
        np.testing.assert_allclose(dlambda_dm, expected_val)

    # test laminar re range: 0 < re <= re_lam
    _assert_lambda_and_dlambda_dm(re=re_lam * 0.8, expected_val=lam_value)
    _assert_lambda_and_dlambda_dm(re=re_lam, expected_val=lam_value)

    # test transient re range: re_lam < re <= re_turb
    _assert_lambda_and_dlambda_dm(re=re_lam + 1, expected_val=trans_value)
    _assert_lambda_and_dlambda_dm(re=re_turb, expected_val=trans_value)

    # test turbulent re range: re_turb < re
    _assert_lambda_and_dlambda_dm(re=re_turb + 1, expected_val=turb_value)

    re = np.array([2000, 3000, 5000])
    model_payload = {k: np.ones_like(re) for k in model_payload}
    expected_val = np.array([lam_value, trans_value, turb_value])
    _assert_lambda_and_dlambda_dm(re=re, expected_val=expected_val)


def test_regime_aware_friction_factor_model_incorrect_re_ranges():
    """Ensure that constructing the model with invalid Re boundaries
    (negative or reversed) raises a ValueError.
    """
    mock = MockFrictionFactorModel(42)
    payload = {
        "laminar": mock,
        "transient": mock,
        "turbulent": mock,
    }

    with pytest.raises(ValueError, match="Must have 0 < re_laminar < re_turbulent"):
        fm.RegimeAwareFrictionFactorModel(re_laminar=-1, re_turbulent=4000, **payload)

    with pytest.raises(ValueError, match="Must have 0 < re_laminar < re_turbulent"):
        fm.RegimeAwareFrictionFactorModel(re_laminar=4000, re_turbulent=2000, **payload)


def test_colebrook_convergence_failure(model_payload):
    """Verify that the Colebrook model raises a PipeflowNotConverged error
    when the iterative solution fails.
    """
    from pandapipes.pipeflow import PipeflowNotConverged

    model = fm.Colebrook(max_iter=1, tolerance=1e-12)
    with pytest.raises(PipeflowNotConverged):
        model.compute_lambda_and_dlambda_dm(**model_payload)


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
        model.compute_lambda_and_dlambda_dm(**model_payload)
        estimator.assert_called_once()

    model = fm.Colebrook()
    for estimator in estimators:
        estimator.reset_mock()
        model.initial_estimator = estimator
        model.compute_lambda_and_dlambda_dm(**model_payload)
        estimator.assert_called_once()


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


def test_one_pipe_net(one_pipe_net, model_class):
    """Integration test: pipeflow converges when using
    each friction factor model (via the model_class fixture).
    """
    model = model_class()
    pp.pipeflow(one_pipe_net, friction_model=model)


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
    pp.pipeflow(one_pipe_net, friction_model=model_name)
    assert isinstance(one_pipe_net["_options"]["friction_model"], expected_model_class)
