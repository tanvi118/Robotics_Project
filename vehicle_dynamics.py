"""
Member 1 : Vitankar Tanvi Ashish (SAP 500120658)
Module   : vehicle_dynamics.py  -- vehicle model + lead-vehicle scenarios
"""
import numpy as np

VEHICLE_LENGTH = 4.5  # m


class VehicleParams:
    """Physical limits of every follower vehicle."""
    def __init__(self, tau=0.5, a_min=-6.0, a_max=3.0, length=VEHICLE_LENGTH):
        self.tau = tau          # engine/brake actuator time constant [s]
        self.a_min = a_min      # max braking [m/s^2]
        self.a_max = a_max      # max acceleration [m/s^2]
        self.length = length    # body length [m]


def step_vehicle(x, v, a, u, dt, p):
    """Advance one vehicle by dt. Model: a_dot = (u - a)/tau (first-order lag)."""
    u = float(np.clip(u, p.a_min, p.a_max))
    a_new = a + dt * (u - a) / p.tau
    v_new = max(0.0, v + dt * a)
    x_new = x + dt * v + 0.5 * dt ** 2 * a
    return x_new, v_new, a_new


def lead_acceleration(t, v, scenario="brake", v_cruise=20.0):
    """Acceleration command of the (ideal) lead vehicle for a scenario."""
    if scenario == "brake":      # hard brake 20 -> 10 m/s, then recover
        if 10.0 <= t < 14.0:
            return -2.5
        if t >= 24.0 and v < v_cruise:
            return 1.5
        return 0.0
    if scenario == "sine":       # stop-and-go style speed oscillation
        if 5.0 <= t <= 35.0:
            return 1.0 * np.sin(2 * np.pi * (t - 5.0) / 10.0)
        return 0.0
    raise ValueError(f"unknown scenario '{scenario}'")


if __name__ == "__main__":
    p = VehicleParams()
    x, v, a = 0.0, 20.0, 0.0
    for _ in range(100):
        x, v, a = step_vehicle(x, v, a, 1.0, 0.05, p)
    print(f"[vehicle_dynamics] self-test OK: v={v:.2f} m/s, a={a:.2f} m/s^2")
