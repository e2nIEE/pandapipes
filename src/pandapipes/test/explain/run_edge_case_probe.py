import sys

import pandapipes as pp

from pandapipes.explain.core.config import get_config
from pandapipes.explain.knowledge.loader import load_relevant
from pandapipes.explain.llm import client as llm_client
from pandapipes.explain.llm import prompt as llm_prompt
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged


def base_net():
    net = pp.create_empty_network(fluid="lgas")
    junctions = [
        pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=0)
        for _ in range(6)
    ]
    pp.create_ext_grid(net, junction=junctions[0], p_bar=1.1, t_k=293.15)
    for from_j, to_j in zip(junctions[:-1], junctions[1:]):
        pp.create_pipe_from_parameters(
            net,
            from_j,
            to_j,
            length_km=0.1,
            inner_diameter_mm=300,
            k_mm=0.1,
        )
    pp.create_sink(net, junction=junctions[-1], mdot_kg_per_s=0.1)
    return net


def build_messages(net):
    try:
        pp.pipeflow(net)
    except Exception as exc:
        tb = exc.__traceback__
        error_type = type(exc).__name__
        error_msg = str(exc)
    else:
        try:
            raise PipeflowNotConverged("The hydraulic calculation did not converge to a solution.")
        except PipeflowNotConverged as exc:
            tb = exc.__traceback__
            error_type = type(exc).__name__
            error_msg = str(exc)

    stats = llm_prompt.extract_net_stats(tb)
    diagnostics = llm_prompt.extract_diagnostics(tb, error_type, error_msg)
    knowledge = load_relevant(error_type, error_msg, context=f"{stats}\n{diagnostics}")
    return diagnostics, llm_prompt.build(
        error_type=error_type,
        error_msg=error_msg,
        user_contexts=[],
        lib_contexts=[],
        knowledge=knowledge,
        net_stats=stats,
        diagnostics=diagnostics,
    )


def main():
    get_config().model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    scenarios = []

    net = base_net()
    net.junction.loc[3, "height_m"] = 1e6
    scenarios.append(("unrealistic elevation", "height_m", diagnostics_and_messages(net)))

    net = base_net()
    net.sink.loc[0, "mdot_kg_per_s"] = 10000.0
    scenarios.append(("unrealistic sink demand", "mdot_kg_per_s", diagnostics_and_messages(net)))

    net = base_net()
    net.pipe.loc[0, "inner_diameter_mm"] = 0.001
    scenarios.append(("unrealistically small pipe diameter", "inner_diameter_mm", diagnostics_and_messages(net)))

    net = base_net()
    net.pipe.loc[0, "to_junction"] = 999
    scenarios.append(("invalid pipe junction reference", "to_junction", diagnostics_and_messages(net)))

    net = base_net()
    net.pipe.loc[0, "length_km"] = 10000.0
    scenarios.append(("unrealistically large pipe length", "length_km", diagnostics_and_messages(net)))

    for title, expected_term, payload in scenarios:
        diagnostics, messages = payload
        print(f"\n=== {title} ===")
        print("DIAGNOSTICS:")
        print(diagnostics)
        response = llm_client.chat(messages)
        print("\nOLLAMA RESPONSE:")
        print(response or "<no response>")
        ok = bool(response and expected_term.lower() in response.lower())
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'} expected term: {expected_term}")


def diagnostics_and_messages(net):
    return build_messages(net)


if __name__ == "__main__":
    main()
