import numpy as np
import matplotlib.pyplot as plt


class Simulation:
    
    def __init__(self, title, t, dt, mean_gps, std_gps, mean_enc, std_enc, L, r):
        self.title = title
        self.dt = dt        
        self.mean_gps = mean_gps
        self.std_gps = std_gps
        self.mean_enc = mean_enc
        self.std_enc = std_enc
        self.L = L
        self.r = r
        
        self.epochs = int(t / dt)
        self.t = np.arange(0, t, dt).reshape(-1, 1)
        self.v = np.zeros((self.epochs, 3, 1), dtype=np.float64)
        self.x = np.zeros((self.epochs, 3, 1), dtype=np.float64)
        self.x_gps = np.zeros((self.epochs, 3, 1), dtype=np.float64)
        self.x_enc = np.zeros((self.epochs, 3, 1), dtype=np.float64)
        self.enc_noise = np.zeros((2, 1))
        
    def R(self, theta):
        theta = theta[0]
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta),  np.cos(theta), 0],
                         [0, 0, 1]], dtype=np.float64)
        
    def get_wheels_velocity(self, v):
        vx = v[0]
        wz = v[2]
        wl = (vx - wz * self.L / 2) / self.r
        wr = (vx + wz * self.L / 2) / self.r
        return np.array([wl, wr]).reshape(-1, 1)
    
    def get_wheels_pose(self, x):
        vx = x[0]
        wz = x[2]
        wl = (vx - wz * self.L / 2) / self.r
        wr = (vx + wz * self.L / 2) / self.r
        return np.array([wl, wr]).reshape(-1, 1)
    
    def get_robot_velocity(self, omega):
        v = omega * self.r
        v_global = (v[1][0] + v[0][0]) / 2
        w_global = (v[1][0] - v[0][0]) / self.L
        return np.array([v_global, 0, w_global]).reshape(-1, 1)
        
    def set_velocity(self, vx, wz, t):
        v = np.array([vx, 0, wz]).reshape(-1, 1)
        self.v[round(t/self.dt):] = v
        
    def set_velocities(self, vels):
        for vel in vels:
            self.set_velocity(*vel)
        
    def spin(self, t):
        if t == 0:
            return
                
        self.x[t] = self.x[t-1] + self.R(self.x[t-1][2]) @ self.v[t-1] * self.dt
        
        self.x_gps[t] = self.x[t] + np.vstack([np.random.normal(self.mean_gps, self.std_gps, (2, 1)), 0]).reshape(-1, 1)
        
        self.enc_noise += np.random.normal(self.mean_enc, self.std_enc, 2).reshape(-1, 1)
        x = self.get_wheels_pose(self.x[t])
        omega = self.get_wheels_velocity(self.v[t-1])# + self.enc_noise
        v = self.get_robot_velocity(omega)
        self.x_enc[t] = self.x_enc[t-1] + self.R(self.x_enc[t-1][2]) @ v * self.dt + np.random.normal(self.mean_enc, self.std_enc, 3).reshape(-1, 1)
        
    def run(self):
        for t in range(self.epochs):
            self.spin(t)
            
    def draw(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(self.title)
        
        axs[0][0].set_title('x')
        axs[0][0].plot(self.t, self.x[:, 0], label='x true', color='blue')
        axs[0][0].plot(self.t, self.x_gps[:, 0], label='x gps', color='red')
        axs[0][0].plot(self.t, self.x_enc[:, 0], label='x enc', color='green')
        
        axs[0][1].set_title('y')
        axs[0][1].plot(self.t, self.x[:, 1], label='y true', color='blue')
        axs[0][1].plot(self.t, self.x_gps[:, 1], label='y gps', color='red')
        axs[0][1].plot(self.t, self.x_enc[:, 1], label='y enc', color='green')
        
        axs[1][0].set_title('$\\theta$')
        axs[1][0].plot(self.t, self.x[:, 2], label='$\\theta$ true', color='blue')
        axs[1][0].plot(self.t, self.x_enc[:, 2], label='$\\theta$ enc', color='green')
        
        axs[1][1].set_title('path')
        axs[1][1].plot(self.x[:, 0], self.x[:, 1], label='x true', color='blue')
        axs[1][1].plot(self.x_gps[:, 0], self.x_gps[:, 1], label='x gps', color='red')
        axs[1][1].plot(self.x_enc[:, 0], self.x_enc[:, 1], label='x enc', color='green')
        
        for ax in axs.flatten():
            ax.legend(loc='upper left')
            ax.grid(True)
            
    def get_data(self):
        return self.t, self.x, self.x_gps, self.x_enc, self.v


def generate_vels_straight():
    return [[0, np.pi / 6, 0], [1, 0, 1]]


def generate_vels_circle():
    return [[1, 1, 0]]


def generate_vels_turn():
    return [[1, 0, 0], [0, np.pi / 2, 10], [1, 0, 11]]


if __name__ == '__main__':
    t = 40
    dt = 0.1
    mean_gps = 0
    std_gps = 1
    mean_enc = 0
    std_enc = 0.02
    
    robot_base = 0.2
    wheel_radius = 0.04
    
    sim_straight = Simulation('Straight', t, dt, mean_gps, std_gps, mean_enc, std_enc, robot_base, wheel_radius)
    sim_circle = Simulation('Circle', t, dt, mean_gps, std_gps, mean_enc, std_enc, robot_base, wheel_radius)
    sim_turn = Simulation('Turn', t, dt, mean_gps, std_gps, mean_enc, std_enc, robot_base, wheel_radius)
    
    sim_straight.set_velocities(generate_vels_straight())
    sim_circle.set_velocities(generate_vels_circle())
    sim_turn.set_velocities(generate_vels_turn())
    
    sim_straight.run()
    sim_circle.run()
    sim_turn.run()
    
    sim_straight.draw()
    sim_circle.draw()
    sim_turn.draw()
    
    plt.show()
    