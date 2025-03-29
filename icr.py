import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import namedtuple

WheelDims = namedtuple("wheel_dims", ["l", "w"])

wheelbase = 1.0

# Global frame for now
v_L = np.array([-0.3, 0]).T
v_R = np.array([-0.5, 0]).T

# velocity vector
v = (v_L + v_R)/2

# robot position is initially at (0,0)
# robot state is x, y, psi=yaw
car_X = np.array([0, 0, 0]).T
wheel_dims = WheelDims(l=0.3, w=0.1)

class DiffDrive:
    def __init__(self, wheelbase, wheel_dims):
        """
        Params:
            wheel_dims: named_tuple containing length x width of wheels
        """
        self.wheelbase = wheelbase
        self.state = np.array([0, 0, 0]).T
        self.controls = np.array([0, 0]).T # left_wheel velocity, right_wheel velocity.
        self.wheel_dims = wheel_dims
        self.l_axle_s = None
        self.r_axle_s = None
    
    def draw_car(self, ax):
        # 0. Extract car coordinates
        x, y, yaw = self.state.flatten()

        T_sb = np.array([
            [np.cos(yaw) , -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]
        ])

        R_sb = np.array([
            [np.cos(yaw) , np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]
        ])
        
        # Calculating location of each wheel (bottom left of rectangle) in world coordinates
        l_wheel_b = np.array([-self.wheel_dims.l/2, self.wheelbase/2 - self.wheel_dims.w/2, 1])
        r_wheel_b = np.array([-self.wheel_dims.l/2, -self.wheelbase/2 - self.wheel_dims.w/2, 1])
        l_wheel_s = T_sb @ l_wheel_b
        r_wheel_s = T_sb @ r_wheel_b

        # Calculating location of each axle
        l_axle_b = np.array([0, self.wheelbase/2, 1])
        r_axle_b = np.array([0, -self.wheelbase/2, 1])
        l_axle_s = T_sb @ l_axle_b
        r_axle_s = T_sb @ r_axle_b

        self.l_axle_s = l_axle_s
        self.r_axle_s = r_axle_s

        # 1. Draw a rectangle corresponding to both wheels
        import matplotlib.transforms as transforms
        t_l = transforms.Affine2D().rotate_around(l_wheel_s[0], l_wheel_s[1], yaw) + ax.transData
        t_r = transforms.Affine2D().rotate_around(r_wheel_s[0], r_wheel_s[1], yaw) + ax.transData
        ax.add_patch(mpatches.Rectangle(l_wheel_s[0:2], self.wheel_dims.l, self.wheel_dims.w, color="red", alpha=0.5, transform=t_l))
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
        x, y, yaw = self.state
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


dd = DiffDrive(wheelbase, wheel_dims)

dd.state = np.array([2, 4, np.pi/6]).T

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
dd.draw_car(ax)
dd.draw_velocity_vectors(np.array([1.0, 1.0]))

# Three things to make a diff-drive robot
# 1. left wheel / right wheel, a rectangle at position wheelbase/2 away from origin. If yaw is 0, this position is
# directly up / down (0, +wheelbase/2) (0, -wheelbase/2). If the yaw is nonzero we have some geometry to do.
# 2. wheel orientation in this case exactly the same as yaw
# 3. a line with its center at car_X[0:2] connecting both wheels together.

# a rectangle corresponding to left wheel
