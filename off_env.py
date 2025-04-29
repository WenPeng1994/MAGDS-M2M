"""
input:channel, data queue length
output:
"""

# 没有加相邻卫星，等运行正常再加
import random
import numpy as np
import torch


M_n = 3   #number of LEOs
N_n = 5  #number of IoTDs
U_data_size = 1000  # KB,the input computation task data size
U_required_CPU = 4  # Gcycles/s,the required CPU cycles

bandwidth_nm = 20e6  # bandwidth of the wireless channel between IoTD and LEO:20MHz
noise_power = -174  ##(dBm/Hz)the additive white Gaussian noise(AWGN) sigma的平方
transmission_power_n = 23  # transmission power of IoTD 23dBm
path_loss_exponent = 2.0  # 路径损耗指数
h_leo = 784  # LEO的高度:784km,一般在[400,2000]公里之间
antenna_gain_dBi = 14  # IoTD的天线增益
N_capacity = 0.1  # computation capacity of the IoTD 0.1Gcycles/s
M_capacity = 5  # computation capacity of the LEO 5Gcycles/s


class LeoEnv(object):
    def __init__(self, M_n, N_n, U_data_size, U_required_CPU, bandwidth_nm, noise_power,
                 transmission_power_n, path_loss_exponent, h_leo, antenna_gain_dBi, N_capacity, M_capacity):
        self.M_n = M_n  # number of LEOs
        self.N_n = N_n
        self.U_data_size = U_data_size
        self.U_required_CPU = U_required_CPU
        self.bandwidth_nm = bandwidth_nm
        self.noise_power = noise_power  ##(dBm/Hz)the additive white Gaussian noise(AWGN) sigma的平方
        self.transmission_power_n = transmission_power_n  # transmission power of IoTD 23dBm
        self.path_loss_exponent = path_loss_exponent  # 路径损耗指数
        self.h_leo = h_leo  # LEO的高度:784km,一般在[400,2000]公里之间
        self.antenna_gain_dBi = antenna_gain_dBi #IoTD的天线增益
        self.N_capacity = N_capacity    #computation capacity of the IoTD 0.1Gcycles/s
        self.M_capacity = M_capacity      #computation capacity of the LEO 5Gcycles/s
        self.IoTD_positions = np.random.uniform(low=0, high=10000, size=(N_n, 2))  # 假设IoTD的位置在二维平面上
        self.LEO_positions = np.array([[0, 0, h_leo]])  # 假设LEO的位置在三维空间中，高度为h_leo

        #self.Q = np.zeros((100, 5))   # 100个时间步，5个IoTD
        self.Q = None
        self.A = None
        self.n_steps = 0


    def calculate_channel(self, LEO_index=0):
        # 计算每个IoTD与指定LEO之间的距离
        IoTD_to_LEO_distances = np.linalg.norm(self.IoTD_positions[:, :2] - self.LEO_positions[LEO_index, :2], axis=1)
        IoTD_to_LEO_distances += np.sqrt(IoTD_to_LEO_distances ** 2 + self.h_leo ** 2)  # 考虑高度差

        rayleigh_fading = (np.random.randn(self.N_n) + 1j * np.random.randn(self.N_n)) / np.sqrt(2)
        shadowing_fading = 0.1 * np.random.randn(self.N_n)
        rain_fading = 0.05 * np.random.randn(self.N_n)
        total_fading = rayleigh_fading + shadowing_fading + rain_fading
        channel_coefficient = total_fading * (IoTD_to_LEO_distances ** -self.path_loss_exponent)
        # 计算增益
        channel_gain = 10 ** (self.antenna_gain_dBi / 10) * abs(channel_coefficient) ** 2
        return channel_gain

    def calculate_rate_nm(self):
        self.channel_gain = self.calculate_channel()
        # Convert dBm to mW for transmission power
        transmission_power_mW = 10 ** (self.transmission_power_n / 10)
        # Convert dBm/Hz to mW for noise power spectral density
        noise_power_mW = self.bandwidth_nm * (10 ** (self.noise_power / 10))
        # Calculate Signal-to-Noise Ratio (SNR)
        snr_nm = transmission_power_mW * self.channel_gain / noise_power_mW
        rate_nm = self.bandwidth_nm * np.log2(1 + snr_nm)
        return rate_nm

# 上面有了状态信道，应该还有队列，然后时延作为reward，再有step


    def calculate_local(self, local_ratio):
        local_latency = local_ratio * self.U_required_CPU * self.U_data_size / self.N_capacity
        return local_latency

    def calculate_leo(self, leo_ratio):
        self.rate_nm = self.calculate_rate_nm()
        leo_com_latency = leo_ratio * self.U_required_CPU * self.U_data_size / self.M_capacity
        leo_trans_latency = leo_ratio * self.U_data_size / self.rate_nm
        leo_latency = leo_com_latency + leo_trans_latency
        return leo_latency

    def step(self, action):
        local_ratio, leo_ratio = action
        leo_ratio = 1 - local_ratio
        # 检查local_ratio和leo_ratio的和是否为1
        if not (0 <= local_ratio <= 1 and 0 <= leo_ratio <= 1):
            raise ValueError("local_ratio and leo_ratio must be between 0 and 1.")

        if abs(local_ratio + leo_ratio - 1) > 1e-6:  # 使用一个小的阈值来容忍浮点数误差
            raise ValueError("The sum of local_ratio and leo_ratio must be equal to 1.")

        local_latency = self.calculate_local(local_ratio)
        leo_latency = self.calculate_leo(leo_ratio)
        total_latency = local_latency + leo_latency

        self.n_steps += 1
        steps = self.n_steps

        # 更新IoTD队列
        self.Q[steps] = self.Q[
                            steps - 1] + local_ratio * self.U_data_size - self.N_capacity / self.U_required_CPU * steps

        # self.Q[step] = torch.where(self.Q[step] >= 0, self.Q[step], torch.tensor(0.0))


        # 更新LEO队列
        self.A[steps] = self.A[steps - 1] + leo_ratio * self.U_data_size - self.M_capacity / self.U_required_CPU * steps

        # self.A[step] = torch.where(self.A[step] >= 0, self.A[step], torch.tensor(0.0))


        self.state = (self.Q[steps], self.A[steps], np.array(self.channel_gain))

        reward = -total_latency

        return self.state, reward

    def reset(self):
        # self.Q = torch.zeros(self.N_n, dtype=torch.float32)
        # self.A = torch.zeros(self.M_n, dtype=torch.float32)
        self.Q = np.random.rand(51)  # 确保数组至少有51个元素
        self.A = np.random.rand(51)  # 确保数组至少有51个元素
        self.channel_gain = self.calculate_channel()
        self.n_steps = 0
        self.state = (self.Q[0], self.A[0], np.array(self.channel_gain))
        return self.state

if __name__ == "__main__":
    env = LeoEnv(M_n, N_n, U_data_size, U_required_CPU, bandwidth_nm, noise_power,
                 transmission_power_n, path_loss_exponent, h_leo, antenna_gain_dBi, N_capacity, M_capacity)
    n_steps = 50
    state = env.reset()
    total_reward = 0
    for step in range(n_steps):
        action = (np.random.uniform(low=0.0, high=1.0), np.random.uniform(low=0.0, high=1.0))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step: {step}, State: {state}, Action: {action}, Reward: {reward}")

        if done:
            print(f"Episode finished after {step+1} steps with total reward {total_reward}")
            break

        state = next_state


