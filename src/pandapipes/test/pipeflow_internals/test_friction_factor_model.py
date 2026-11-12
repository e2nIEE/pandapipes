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


@pytest.fixture(
    params=[
        fm.Nikuradse,
        fm.SwameeJain,
        fm.Colebrook,
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


def test_lambda_decreases_as_Re_increases(model_class, model_payload):
    """lambda decreases as Re increases, as observed experimentally (Moody chart)."""
    model = model_class()
    re = np.array((5_000, 10_000, 100_000, 1_000_000), dtype=np.float64)
    payload = {
        "re": re,
        "k_over_D": np.full_like(re, model_payload["k_over_D"]),
        "m": np.full_like(re, model_payload["m"]),
    }
    lambdas, _ = model.compute_lambda_and_dlambda_dm(**payload)
    assert all(np.diff(lambdas) < 0), lambdas


def test_colebrook_convergence_failure(model_payload):
    """Verify that the Colebrook model raises a PipeflowNotConverged error
    when the iterative solution fails.
    """
    from pandapipes.pf.pipeflow_setup import PipeflowNotConverged

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
