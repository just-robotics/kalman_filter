import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from kalman_filter import ExtendedKalmanFilter, Rz, Rz2theta, tf


TPR = 330
r = 0.0323
l = 0.2437

std_enc = 0.1
std_uwb = 10

zero_vec = np.zeros((3, 1))
T_zero = tf(Rz(0), zero_vec)

X0 = np.array([2.03, 1.07, np.pi/2]).reshape(-1, 1)
T_enc_uwb = tf(Rz(X0[2, 0]), np.array([X0[0, 0], X0[1, 0], 0]).reshape(-1, 1))  # enc frame tf in uwb frame


def main():
    data = np.genfromtxt('data/ekf_log_18_26_29.log', delimiter=',')
    
    data[:, 1] -= data[0, 1]  # time normalization
    
    # n = 5500
    # n = 10000
    n = data.shape[0]
    data = data[1:n]
    
    ekf = ExtendedKalmanFilter(X0, std_enc, std_uwb)
    
    t_prev = 0
    
    ticks_l_prev = 0
    ticks_r_prev = 0
    
    P_prev = [0.0, 0.0]
    
    Xarr_ekf = np.zeros((n, 4, 4))
    Xarr_ekf[0] = T_enc_uwb
    
    Xarr_uwb = np.zeros((n, 4, 4))
    Xarr_uwb[0] = T_enc_uwb
    
    X_prev_enc = T_enc_uwb
    Xarr_enc = np.zeros((n, 4, 4))
    Xarr_enc[0] = X_prev_enc
    
    X_prev_odom = T_enc_uwb
    Xarr_odom = np.zeros((n, 4, 4))
    Xarr_odom[0] = X_prev_odom
    
    tarr = np.zeros((n, 1))
    
    errarr = np.zeros((n, 2, 1))
    velarr = np.zeros((n, 2, 1))
    thetaarr = np.zeros((n, 1, 1))
    
    for i, d in enumerate(data):
        t = d[1]
        dt = t - t_prev
        
        if int(d[0]) == 0:  # enc
            ticks_l = d[2]
            ticks_r = d[3]
            
            d_ticks_l = ticks_l - ticks_l_prev
            d_ticks_r = ticks_r - ticks_r_prev
            
            w_l = 2.0 * np.pi * d_ticks_l / TPR / dt
            w_r = 2.0 * np.pi * d_ticks_r / TPR / dt

            vl = w_l * r
            vr = w_r * r
            
            v = (vl + vr) / 2
            w = (vr - vl) / l
            
            ekf.update_measurements_enc(v, w)
            ekf.encoder_measurement_callback(dt)
            
            Xarr_odom[i+1], P_prev = odom(ticks_l, ticks_r, X_prev_odom, P_prev)
            X_prev_odom = Xarr_odom[i+1]
            
            theta_prev = Rz2theta(X_prev_enc[:3, :3])
            x_enc = X_prev_enc[0, -1] + v*np.cos(theta_prev)*dt
            y_enc = X_prev_enc[1, -1] + v*np.sin(theta_prev)*dt
            theta_enc = theta_prev + w*dt
            
            Xarr_enc[i+1] = tf(Rz(theta_enc), np.array([x_enc, y_enc, 0]).reshape(-1, 1))
            X_prev_enc = Xarr_enc[i+1]

            ticks_l_prev = ticks_l
            ticks_r_prev = ticks_r
            
        if int(d[0]) == 1:  # uwb
            x, y = d[2], d[3]
            ekf.update_measurements_uwb(x, y)
            ekf.uwb_measurement_callback(dt)
            Xarr_uwb[i+1] = np.array([x, y, 0, 1]).reshape(-1, 1)
            
        t_prev = t
            
        Xarr_ekf[i+1] = ekf.get_T()
        thetaarr[i+1] = ekf.theta
        errarr[i+1] = ekf.get_errors()
        velarr[i+1] = ekf.get_velocities()
        tarr[i+1] = t
    
    # refactor for plotting only
    Xarr_ekf = Xarr_ekf[Xarr_ekf[:, 3, 3] == 1]
    Xarr_uwb = Xarr_uwb[Xarr_uwb[:, 3, 3] == 1]
    Xarr_enc = Xarr_enc[Xarr_enc[:, 3, 3] == 1]
    Xarr_odom = Xarr_odom[Xarr_odom[:, 3, 3] == 1]

    # xy(t)
    plt.plot(Xarr_ekf[:, 0, -1], Xarr_ekf[:, 1, -1], label='ekf', color='blue')
    # plt.plot(Xarr_uwb[:, 0, -1], Xarr_uwb[:, 1, -1], label='uwb', color='red')
    # plt.plot(Xarr_enc[:, 0, -1], Xarr_enc[:, 1, -1], label='enc', color='green')
    # plt.plot(Xarr_odom[:, 0, -1], Xarr_odom[:, 1, -1], label='odom', color='orange')
    plt.title('Odometry')
    plt.xlabel('y, meters')
    plt.ylabel('x, meters'),
    plt.legend()
    plt.grid(False)
    
    # x(t)
    plt.figure()
    plt.plot(tarr[:, 0], Xarr_ekf[:, 0, -1], label='x', color='blue')
    plt.title('x-axis')
    plt.xlabel('time, s')
    plt.ylabel('x, meters'),
    plt.legend()
    plt.grid(True)
    
    # y(t)
    plt.figure()
    plt.plot(tarr[:, 0], Xarr_ekf[:, 1, -1], label='y', color='blue')
    plt.title('y-axis')
    plt.xlabel('time, s')
    plt.ylabel('y, meters'),
    plt.legend()
    plt.grid(True)
    
    # theta(t)
    plt.figure()
    plt.plot(tarr[:, 0], thetaarr[:, 0, 0], label='theta', color='blue')
    plt.title('theta')
    plt.xlabel('time, s')
    plt.ylabel('theta, rad'),
    plt.legend()
    plt.grid(True)
    
    # e_v(t), e_w(t)
    plt.figure()
    plt.plot(tarr[:, 0], errarr[:, 0, 0], label='err_v', color='blue')
    plt.plot(tarr[:, 0], errarr[:, 1, 0], label='err_w', color='red')
    plt.title('Errors')
    plt.xlabel('time, s')
    plt.ylabel('err'),
    plt.legend()
    plt.grid(True)
    
    # v(t)
    plt.figure()
    plt.plot(tarr[:, 0], velarr[:, 0, 0], label='v', color='blue')
    plt.title('Linear velocity')
    plt.xlabel('time, s')
    plt.ylabel('v, m/s'),
    plt.legend()
    plt.grid(True)
    
    # w(t)
    plt.figure()
    plt.plot(tarr[:, 0], velarr[:, 1, 0], label='w', color='blue')
    plt.title('Angular velocity')
    plt.xlabel('time, s')
    plt.ylabel('w, rad/s'),
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # idx = 200
    # animation(Xarr_ekf[idx:])
    
    plt.show()
    
    
def odom(ticks_l, ticks_r, X_prev, P_prev):
    
    rads_l = 2.0 * np.pi * ticks_l / TPR
    rads_r = 2.0 * np.pi * ticks_r / TPR

    dpl = rads_l - P_prev[0]
    dpr = rads_r - P_prev[1]

    ds = (dpr + dpl) * r / 2.0
    dth = (dpr - dpl) * r / l

    dx = ds * np.cos(dth)
    dy = ds * np.sin(dth)
    dY = dth
    
    theta_prev = Rz2theta(X_prev[:3, :3])

    x = X_prev[0, -1] + dx * np.cos(theta_prev) - dy * np.sin(theta_prev)
    y = X_prev[1, -1] + dx * np.sin(theta_prev) + dy * np.cos(theta_prev)
    theta = theta_prev + dY

    P_prev = [rads_l, rads_r]

    return tf(Rz(theta), np.array([x, y, 0]).reshape(-1, 1)), P_prev
    
    
def animation(Xarr):
    
    x = Xarr[:, 0, -1].reshape(-1)
    y = Xarr[:, 1, -1].reshape(-1)

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(np.min(x)-1, np.max(x)+1)
    ax.set_ylim(np.min(y)-1, np.max(y)+1)

    robot_arrow = None

    def update(frame):
        global robot_arrow
        ax.clear()
        ax.set_xlim(np.min(x)-1, np.max(x)+1)
        ax.set_ylim(np.min(y)-1, np.max(y)+1)
        ax.set_title(f"Frame {frame}")
        
        ax.plot(x[:frame+1], y[:frame+1], 'b-', lw=2, label='Trajectory')

        theta = Rz2theta(Xarr[frame, :3, :3])
        dx = np.cos(theta) * 0.5
        dy = np.sin(theta) * 0.5
        robot_arrow = ax.arrow(x[frame], y[frame], dx, dy, head_width=0.2, head_length=0.3, fc='r', ec='r')
        return robot_arrow

    ani = FuncAnimation(fig, update, frames=len(x), interval=0.1, blit=False)
    plt.show()


if __name__ == '__main__':
    main()
    