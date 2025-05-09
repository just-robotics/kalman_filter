import numpy as np

from scipy.spatial.transform import Rotation


class ExtendedKalmanFilter:
    
    def __init__(self, X0: np.array, std_enc : float, std_uwb : float):
        self.std_enc = std_enc
        
        self.state = np.array([
            [X0[0, 0]],  # x
            [X0[1, 0]],  # y
            [X0[2, 0]],  # theta
            [0.0],       # v
            [0.0],       # w
            [0.0],       # e_v
            [0.0],       # e_w
        ])
        
        self.P = np.eye(self.state.shape[0]) * 0.01 # system covariance matrix
        
        self.z_enc = np.array([  # encoder measurements
            [0.0],  # z_v
            [0.0],  # z_w
        ])
        
        self.z_uwb = np.array([  # uwb measurements
            [0.0],  # z_x
            [0.0],  # z_y
        ])
        
        self.R = np.eye(self.z_uwb.shape[0]) * std_uwb**2  # uwb noise covariance matrix
        
        self.H = np.array([  # uwb measurements matrix
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ])
        
    def g_enc(self, dt):
        return np.array([
            [self.x + self.v * np.cos(self.theta) * dt],
            [self.y + self.v * np.sin(self.theta) * dt],
            [self.theta + self.w * dt],
            [self.z_v - self.e_v],
            [self.z_w - self.e_w],
            [self.e_v],
            [self.e_w],
        ])
        
    def G_enc(self, dt):
        return np.array([
            [1, 0, -self.v * np.sin(self.theta) * dt, np.cos(self.theta) * dt, 0, 0, 0],
            [0, 1,  self.v * np.cos(self.theta) * dt, np.sin(self.theta) * dt, 0, 0, 0],
            [0, 0, 1, 0, dt, 0, 0],
            [0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        
    def g_uwb(self, dt):
        return np.array([
            [self.x + self.v * np.cos(self.theta) * dt],
            [self.y + self.v * np.sin(self.theta) * dt],
            [self.theta + self.w * dt],
            [self.v],
            [self.w],
            [self.e_v],
            [self.e_w],
        ])
        
    def G_uwb(self, dt):
        return np.array([
            [1, 0, -self.v * np.sin(self.theta) * dt, np.cos(self.theta) * dt, 0, 0, 0],
            [0, 1,  self.v * np.cos(self.theta) * dt, np.sin(self.theta) * dt, 0, 0, 0],
            [0, 0, 1, 0, dt, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        
    def Q(self, dt):  # encoder noise covariance matrix
        Q = np.zeros(self.P.shape)
        Q[-2, -2] = self.std_enc**2 * dt
        Q[-1, -1] = self.std_enc**2 * dt
        return Q
    
    def h(self, state):
        return np.array([
            [state[0, 0]],
            [state[1, 0]],
        ])
        
    def update_measurements_enc(self, v, w):
        self.z_enc = np.array([
            [v],
            [w],
        ])
        
    def update_measurements_uwb(self, x, y):
        self.z_uwb = np.array([
            [x],
            [y],
        ])
        
    def encoder_measurement_callback(self, dt):
        self.state = self.g_enc(dt)
        G = self.G_enc(dt)
        self.P = G @ self.P @ G.T + self.Q(dt)
        
    def uwb_measurement_callback(self, dt):
        state_pred = self.g_uwb(dt)
        G = self.G_uwb(dt)
        P_pred = G @ self.P @ G.T + self.Q(dt)
        
        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)
        self.state = state_pred + K @ (self.z_uwb - self.h(state_pred))
        self.P = (np.eye(P_pred.shape[0]) - K @ self.H) @ P_pred
        
    @property
    def x(self):
        return self.state[0][0]
    
    @property
    def y(self):
        return self.state[1][0]
    
    @property
    def theta(self):
        return self.state[2][0]
    
    @property
    def v(self):
        return self.state[3][0]
    
    @property
    def w(self):
        return self.state[4][0]
    
    @property
    def e_v(self):
        return self.state[5][0]
    
    @property
    def e_w(self):
        return self.state[6][0]
    
    @property
    def z_v(self):
        return self.z_enc[0][0]
    
    @property
    def z_w(self):
        return self.z_enc[1][0]
        
    def get_T(self):
        return tf(Rz(self.theta), np.array([self.x, self.y, 0]).reshape(-1, 1))
    
    def get_velocities(self):
        return self.state[3:5]
    
    def get_errors(self):
        return self.state[5:]
    
        
def Rz(theta):
    return Rotation.from_euler('z', [theta], degrees=False).as_matrix()[0]


def tf(R, t):
    return np.vstack([np.hstack([R, t]),
                      np.hstack([np.zeros((1, 3)), np.ones((1, 1))])])
    
    
def Rz2theta(Rz):
    return Rotation.from_matrix(Rz).as_euler('zyx', degrees=False)[0]
