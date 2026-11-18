"""
Member 3 : Krishna Mittal (SAP 500124263)
Module   : platoon_sim.py -- platoon simulation engine (with V2V delay & noise)
"""
import numpy as np
from vehicle_dynamics import VehicleParams, step_vehicle, lead_acceleration


def simulate(controller, n_followers=5, scenario="brake", t_end=45.0, dt=0.05,
             v0=20.0, comm_delay=0.2, noise_std=0.0, seed=0, params=None):
    """Simulate a platoon: vehicle 0 = leader, 1..n = followers.
    Returns dict with t, x, v, a, gap, err arrays (vehicle x time)."""
    p = params or VehicleParams()
    rng = np.random.default_rng(seed)
    n = n_followers + 1
    T = int(round(t_end / dt)) + 1
    t = np.arange(T) * dt
    x = np.zeros((n, T)); v = np.zeros((n, T)); a = np.zeros((n, T))
    gap = np.zeros((n, T)); err = np.zeros((n, T))
    d_eq = controller.d0 + controller.h * v0
    for i in range(n):
        x[i, 0] = -i * (d_eq + p.length)
        v[i, 0] = v0
    delay_steps = int(round(comm_delay / dt))

    for k in range(T - 1):
        # leader: ideal acceleration command
        al = lead_acceleration(t[k], v[0, k], scenario, v0)
        a[0, k] = al
        v[0, k + 1] = max(0.0, v[0, k] + dt * al)
        x[0, k + 1] = x[0, k] + dt * v[0, k] + 0.5 * dt ** 2 * al
        for i in range(1, n):
            g = x[i - 1, k] - x[i, k] - p.length
            gap[i, k] = g
            g_meas = g + noise_std * rng.standard_normal()
            vp_meas = v[i - 1, k] + noise_std * rng.standard_normal() * 0.2
            kd_ = max(0, k - delay_steps)
            a_prev = a[i - 1, kd_] if controller.uses_v2v else 0.0
            u = controller.compute(g_meas, v[i, k], vp_meas, a_prev)
            err[i, k] = g - (controller.d0 + controller.headway(
                v[i, k], v[i - 1, k] - v[i, k]) * v[i, k])
            x[i, k + 1], v[i, k + 1], a[i, k + 1] = step_vehicle(
                x[i, k], v[i, k], a[i, k], u, dt, p)
    # final-sample bookkeeping
    for i in range(1, n):
        gap[i, -1] = x[i - 1, -1] - x[i, -1] - p.length
        err[i, -1] = err[i, -2]
    a[0, -1] = a[0, -2]
    return dict(t=t, x=x, v=v, a=a, gap=gap, err=err, dt=dt,
                controller=controller.name, scenario=scenario)


if __name__ == "__main__":
    from controllers import ACCConstantHeadway
    r = simulate(ACCConstantHeadway())
    print(f"[platoon_sim] self-test OK: {r['v'].shape[0]} vehicles, "
          f"final speeds = {np.round(r['v'][:, -1], 2)}")
