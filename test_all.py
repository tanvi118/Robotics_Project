"""Member 5 : Meet Grover -- run with `python -m pytest tests` or `python tests/test_all.py`"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controllers import ACCConstantHeadway, ACCAdaptiveHeadway, CACC
from platoon_sim import simulate
from metrics import compute_metrics


def test_no_collision_all_methods():
    for c in (ACCConstantHeadway(), ACCAdaptiveHeadway(), CACC()):
        for sc in ("brake", "sine"):
            assert not compute_metrics(simulate(c, scenario=sc))["collision"]


def test_equilibrium_is_steady():
    r = simulate(CACC(), scenario="sine", t_end=4.0)
    assert abs(r["err"][1:, :50]).max() < 1e-6


def test_cacc_not_worse_than_acc():
    a = compute_metrics(simulate(ACCConstantHeadway(), scenario="sine"))
    c = compute_metrics(simulate(CACC(), scenario="sine"))
    assert c["rms_spacing_err_m"] <= a["rms_spacing_err_m"]


if __name__ == "__main__":
    test_no_collision_all_methods(); test_equilibrium_is_steady(); test_cacc_not_worse_than_acc()
    print("all tests passed")
