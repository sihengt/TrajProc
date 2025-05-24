from dataclasses import dataclass
import numpy as np

@dataclass
class MpcAction:
    com_acc: float
    steer_f: float
    steer_r: float

    def numpy(self):
        return np.array([self.com_acc, self.steer_f, self.steer_r])

@dataclass
class SimAction:
    wheel_speed: float
    steer_f: float
    steer_r: float

    def numpy(self):
        return np.array([self.wheel_speed, self.steer_f, self.steer_r])

@dataclass
class MpcState:
    com_x: float
    com_y: float
    com_v: float
    yaw: float

    def numpy(self):
        return np.array([self.com_x, self.com_y, self.com_v, self.yaw])
    
@dataclass
class SimState:
    com_x: float
    com_y: float
    yaw: float
    com_v: float

    def numpy(self):
        return np.array([self.com_x, self.com_y, self.yaw, self.com_v])

def SimToMpcState(x: SimState):
    return MpcState(x.com_x, x.com_y, x.com_v, x.yaw)

def MpcToSimAction(wheel_r, dt, v_com_curr, u: MpcAction):
    v_next = v_com_curr + u.com_acc * dt
    return SimAction(v_next / wheel_r, u.steer_f, u.steer_r)

# class MpcToSimAction:
#     """Converts (a, δ_f, δ_r) ↔ (ω, δ_f, δ_r)."""
#     def __init__(self, wheel_radius: float, dt: float):
#         self.r = wheel_radius
#         self.dt = dt

#     def to_vehicle_cmd(self, v_curr: float, act: MpcAction) -> SimAction:
#         v_next = v_curr + act.com_acc * self.dt
#         return SimAction(v_next / self.r, act.steer_f, act.steer_r)