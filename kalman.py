import numpy as np
from typing import NamedTuple, Dict


class RobotParams ( NamedTuple ) :
    r: float
    L: float

class KalmanFilter :
    def __init__(self, X0: np.ndarray, Dx: np.ndarray,
                 sigma_n: float, sigma_ksi: float, params: RobotParams) :
        self.X = X0.copy ()
        self.Dx = Dx.copy ()
        self.sigma_n = sigma_n
        self.sigma_ksi = sigma_ksi
        self.params = params

        self.H = np.array ( [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0]
        ] )

        self.G = np.array ( [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1]
        ] )

    def predict(self, meas_enc: np.ndarray, dt: float) -> None :
        self.X = self._make_F ( meas_enc, dt )
        dF = self._make_dF ( dt )
        Dksi = np.diag ( [self.sigma_ksi ** 2, self.sigma_ksi ** 2] )
        self.Dx = dF @ self.Dx @ dF.T + self.G @ Dksi @ self.G.T

    def update(self, meas_gps: np.ndarray) -> None :
        Dn = np.diag ( [self.sigma_n ** 2, self.sigma_n ** 2] )
        S = self.H @ self.Dx @ self.H.T + Dn
        K = self.Dx @ self.H.T @ np.linalg.inv ( S )
        y = meas_gps - self.H @ self.X
        self.X = self.X + K @ y
        self.Dx = (np.eye ( 7 ) - K @ self.H) @ self.Dx

    def _make_F(self, meas_enc: np.ndarray, dt: float) -> np.ndarray :

        X = self.X.reshape ( -1 )
        r, L = self.params.r, self.params.L
        wR, wL = meas_enc.flatten ()

        F = np.zeros ( (7, 1) )

        F[0, 0] = X[0] + X[2] * np.cos ( X[3] ) * dt  # x
        F[1, 0] = X[1] + X[2] * np.sin ( X[3] ) * dt  # y
        F[2, 0] = (wR - X[5] + wL - X[6]) * r / 2  # v
        F[3, 0] = X[3] + X[4] * dt  # theta
        F[4, 0] = (wR - X[5] - wL + X[6]) * r / L  # omega
        F[5, 0] = X[5]  # bias_wR
        F[6, 0] = X[6]  # bias_wL

        return F

    def _make_dF(self, dt: float) -> np.ndarray :

        X = self.X.flatten ()
        r, L = self.params.r, self.params.L

        dF = np.eye ( 7 )

        theta = X[3].item ()
        velocity = X[2].item ()

        dF[0, 2] = np.cos ( theta ) * dt
        dF[0, 3] = -velocity * np.sin ( theta ) * dt
        dF[1, 2] = np.sin ( theta ) * dt
        dF[1, 3] = velocity * np.cos ( theta ) * dt
        dF[2, 5] = -r / 2
        dF[2, 6] = -r / 2
        dF[3, 4] = dt
        dF[4, 5] = -r / L
        dF[4, 6] = r / L

        return dF

def fusion_kalman_mode2(meas: list, X0: np.ndarray, Dx: np.ndarray,
                        sigma_n: float, sigma_ksi: float, sim_params: RobotParams) -> dict :

    t = np.array ( [m['t'] for m in meas] )
    meas_gps = np.array ( [[m['gps_x'], m['gps_y']] for m in meas] ).T
    meas_enc = np.array ( [[m['wR'], m['wL']] for m in meas] ).T

    X = np.zeros ( (7, len ( t )) )
    X[:, 0] = X0.reshape ( -1 )

    Dx_hist = np.zeros ( (7, len ( t )) )
    Dx_hist[:, 0] = np.diag ( Dx )

    ekf = KalmanFilter ( X0, Dx, sigma_n, sigma_ksi, sim_params )

    for i in range ( 1, len ( t ) ) :
        dt = t[i] - t[i - 1]
        ekf.predict ( meas_enc[:, i], dt )
        ekf.update ( meas_gps[:, i] )

        X[:, i] = ekf.X.reshape ( -1 )[:7]
        Dx_hist[:, i] = np.diag ( ekf.Dx )

    return {'X' : X, 'Dx' : Dx_hist}

if __name__ == "__main__" :

    params = RobotParams ( r=0.1, L=0.5 )

    X0 = np.array ( [[0], [0], [0], [0], [0], [0], [0]] )

    Dx = np.diag ( [0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01] )

    sigma_n = 0.5
    sigma_ksi = 0.01

    meas = [
        {'t' : 0.0, 'gps_x' : 0.0, 'gps_y' : 0.0, 'wR' : 0.0, 'wL' : 0.0},
        {'t' : 0.1, 'gps_x' : 0.1, 'gps_y' : 0.0, 'wR' : 1.0, 'wL' : 0.9},
        {'t' : 0.2, 'gps_x' : 0.2, 'gps_y' : 0.0, 'wR' : 1.0, 'wL' : 1.0},
    ]

    result = fusion_kalman_mode2 ( meas, X0, Dx, sigma_n, sigma_ksi, params )

    print ( "Оценки состояния:" )
    print ( result['X'] )

