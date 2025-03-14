import numpy as np
import matplotlib.pyplot as plt


def kalman_filter(sigma_ksi, sigma_n, y_enc, y_gps, est_e_last):
    est_e = est_e_last + sigma_ksi / (2 * sigma_n**2) *\
    (np.sqrt(sigma_ksi**2 + 4 * sigma_n**2) - sigma_ksi) *\
    ((y_enc - y_gps) - est_e_last)
    return est_e


def const(_, c):
    return c


def linear(t, _):
    return t


def quadratic(t, _):
    return t**2


def simulation(gps_mean, gps_std, enc_mean, enc_std, size, func, c=0):
    data = []
    data_enc = []
    data_gps = []
    enc_pose_est = []
    kalman_err = []
    est_err = []
    e = 0
    noise_enc = 0
    
    for t in range(size):
        data.append(func(t, c))

        noise_enc = noise_enc + np.random.normal(loc=enc_mean, scale=enc_std)
        data_enc.append(data[t] + noise_enc)

        noise_gps = np.random.normal(loc=gps_mean, scale=gps_std)
        data_gps.append(data[t] + noise_gps)

        e = kalman_filter(enc_std, gps_std, data_enc[t], data_gps[t], e)
        est_err.append(e)
        
        enc_pose_est.append(data_enc[t] - e)
        kalman_err.append(data[t] - enc_pose_est[t])

    return np.std(kalman_err)


def change_enc_std(std_enc_arrange, gps_std_array):
    const_data = []
    linear_data = []
    quadratic_data = []
    
    for i in range(std_enc_arrange.shape[0]):
        gps_data_const = []
        gps_data_linear = []
        gps_data_quadro = []
        for gps_std in gps_std_array:
            gps_data_const.append(simulation(0, gps_std, 0, std_enc_arrange[i], 1000, const, 10))
            gps_data_linear.append(simulation(0, gps_std, 0, std_enc_arrange[i], 1000, linear))
            gps_data_quadro.append(simulation(0, gps_std, 0, std_enc_arrange[i], 1000, quadratic))
        const_data.append(gps_data_const)
        linear_data.append(gps_data_linear)
        quadratic_data.append(gps_data_quadro)
    
    return const_data, linear_data, quadratic_data
     
    
if __name__ == '__main__':
    # mean = 0        
    # std_dev_enc = 10
    # std_dev_gps = 5
    # size = 1000
    # time = np.arange(0, size, 1)
    
    # data, data_enc, data_gps, est_err, enc_pose_est, kalman_err = simulation(mean, std_dev_gps, mean, std_dev_enc, size, linear)
    
    # print(np.std(kalman_err))
    
    # plt.plot(time, data, label='true pose', color='blue')
    # plt.plot(time, data_enc, label='enc', color='green')
    # plt.plot(time, data_gps, label='gps', color='red')
    # plt.plot(time, est_err, label='error', color='black')
    # plt.plot(time, kalman_err, label='Kalman error', color='orange')
    # plt.xlabel("time")
    # plt.ylabel("pose")
    # plt.legend()
    # plt.show()
    
    std_enc_arrange = np.arange(0, 10.1, 0.1)
    std_gps_array = [1, 5, 10]
    
    const_data, linear_data, quadratic_data = change_enc_std(std_enc_arrange, std_gps_array)
    
    const_data = np.array(const_data)
    linear_data = np.array(linear_data)
    quadratic_data = np.array(quadratic_data)
    
    plt.figure(1)
    plt.plot(std_enc_arrange, const_data.T[0], label='gps_std = 1', color='blue')
    plt.plot(std_enc_arrange, const_data.T[1], label='gps_std = 5', color='green')
    plt.plot(std_enc_arrange, const_data.T[2], label='gps_std = 10', color='red')
    plt.title('Const motion')
    plt.xlabel('enc_std_dev')
    plt.ylabel('std_KF_error')
    plt.legend()
    plt.grid()
    
    plt.figure(2)
    plt.plot(std_enc_arrange, linear_data.T[0], label='gps_std = 1', color='blue')
    plt.plot(std_enc_arrange, linear_data.T[1], label='gps_std = 5', color='green')
    plt.plot(std_enc_arrange, linear_data.T[2], label='gps_std = 10', color='red')
    plt.title('Linear motion')
    plt.xlabel('enc_std_dev')
    plt.ylabel('std_KF_error')
    plt.legend()
    plt.grid()
    
    plt.figure(3)
    plt.plot(std_enc_arrange, quadratic_data.T[0], label='gps_std = 1', color='blue')
    plt.plot(std_enc_arrange, quadratic_data.T[1], label='gps_std = 5', color='green')
    plt.plot(std_enc_arrange, quadratic_data.T[2], label='gps_std = 10', color='red')
    plt.title('Quadratic motion')
    plt.xlabel('enc_std_dev')
    plt.ylabel('std_KF_error')
    plt.legend()
    plt.grid()
    
    plt.show()
    