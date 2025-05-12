import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms

from collections import namedtuple

WheelDims = namedtuple("wheel_dims", ["l", "w"])


class DoubleSteerKBM:
    def __init__(self, L, l_f, l_r, wheel_dims):
        """
        Params:
            wheel_dims: named_tuple containing length x width of wheels
        """
        self.L = L
        self.l_f = l_f
        self.l_r = l_r
        
        # x_cg, y_cg, v_cg, theta
        self.state = np.array([0, 0, 0, 0]).T

        # v_cg, delta_f, delta_r
        self.controls = np.array([0, 0, 0]).T
        self.wheel_dims = wheel_dims
        
        # Positions of front / rear axle (for plotting)
        self.f_axle_s = None
        self.r_axle_s = None
    
    def draw_car(self, ax):
        # 0. Extract car coordinates
        x, y, v, yaw = self.state.flatten()

        T_sb = np.array([
            [np.cos(yaw) , -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]
        ])

        R_sb = np.array([
            [np.cos(yaw) , np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        
        # Calculating location of each wheel (bottom left of rectangle of wheel) in world coordinates
        f_wheel_b = np.array([-self.l_r - self.wheel_dims.l/2, -self.wheel_dims.w/2, 1])
        r_wheel_b = np.array([self.l_f - self.wheel_dims.l/2, -self.wheel_dims.w/2, 1])
        f_wheel_s = T_sb @ f_wheel_b
        r_wheel_s = T_sb @ r_wheel_b

        # Calculating location of each axle
        l_axle_b = np.array([-self.l_r, 0, 1])
        r_axle_b = np.array([self.l_f, 0, 1])
        l_axle_s = T_sb @ l_axle_b
        r_axle_s = T_sb @ r_axle_b

        self.l_axle_s = l_axle_s
        self.r_axle_s = r_axle_s

        # 1. Draw a rectangle corresponding to both wheels
        t_l = transforms.Affine2D().rotate_around(f_wheel_s[0], f_wheel_s[1], yaw) + ax.transData
        t_r = transforms.Affine2D().rotate_around(r_wheel_s[0], r_wheel_s[1], yaw) + ax.transData
        ax.add_patch(mpatches.Rectangle(f_wheel_s[0:2], self.wheel_dims.l, self.wheel_dims.w, color="red", alpha=0.5, transform=t_l))
        ax.add_patch(mpatches.Rectangle(r_wheel_s[0:2], self.wheel_dims.l, self.wheel_dims.w, color="green", alpha=0.5, transform=t_r))

        # 2. Draw center point of car
        ax.scatter(x, y, color='black', label="Center")

        # 3. Draw a line between both top and bottom of car
        ax.plot([l_axle_s[0], r_axle_s[0]], [l_axle_s[1], r_axle_s[1]], color='black')

    def draw_velocity_vectors(self, v):
        """
        params:
            v: velocity = [v_l, v_r]
        """
        # These velocity vectors are pointing straight ahead according to the yaw.
        x, y, v, yaw = self.state
        v_l, v_r = v

        # Compute body-frame vectors
        v_l_b = np.array([v_l, 0])
        v_r_b = np.array([v_r, 0])

        # Rotate to world frame using yaw
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])

        T = np.array([
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw),  np.cos(yaw), y],
            [0, 0, 1]
        ])

        v_l_w = R @ v_l_b
        v_r_w = R @ v_r_b

        print(v_l_w)

        r = (v_l + v_r)/(v_r - v_l) * (self.wheelbase/2)
        omega = (v_r - v_l) / self.wheelbase
        ax.quiver(
            self.l_axle_s[0], self.l_axle_s[1],
            v_l_w[0], v_l_w[1],
            angles='xy', scale_units='xy', scale=1,
            color='red'
        )
        ax.quiver(self.r_axle_s[0], self.r_axle_s[1], v_r_w[0], v_r_w[1], angles='xy', scale_units='xy', scale=1,color='green')

        # Draw ICR
        icr = [0, (v_l + v_r)/(v_r - v_l) * (self.wheelbase/2), 1]
        icr_s = T @ icr 
        ax.scatter(icr_s[0], icr_s[1], label="ICR")
        ax.legend()
        plt.show()

if __name__== "__main__":
    wheel_dims = WheelDims(l=0.1, w=0.3)
    dsk = DoubleSteerKBM(1.0, 0.3, 0.7, wheel_dims)
    fig, ax = plt.subplots()
    dsk.draw_car(ax)
    plt.show()