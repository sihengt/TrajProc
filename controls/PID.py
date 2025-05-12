import numpy as np
import matplotlib.pyplot as plt

class PID:
    def __init__(self, k_p, k_i, k_d):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.prev_error = 0
        self.integral = 0

    def get_control(self, error, dt):
        derivative = (error - self.prev_error) / dt

        # Update previous error
        self.prev_error = error

        # Accumulates error
        self.integral += error * dt
        
        return self.k_p * error + self.k_i * self.integral + self.k_d * derivative