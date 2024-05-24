########################################################################
# Author(s):    Ashwin Kanhere
# Date:         21 September 2021
# Desc:         Constants required for calculations with GNSS measures 
########################################################################
import numpy as np

WE = 7.2921151467e-5  # 地球自转角速度
LIGHTSPEED = 2.99792458e8  # 光速
WEEKSEC = 604800  # 一周秒数
GRAVITY = -9.7803267714  # 修改为本地重力加速度

class gpsconsts:
    def __init__(self):

        self.a = 6378137.                       # semi-major axis of the earth [m]地球长半轴，赤道半径
        self.b = 6356752.3145                   # semi-minor axis of the earth [m]地球短半轴，地球极半径
        self.e = np.sqrt(1-(self.b**2)/(self.a**2))            # eccentricity of the earth = 0.08181919035596地球离心率
        self.lat_accuracy_thresh = 1.57e-6      # 10 meter latitude accuracy纬度精度阈值为10m
        self.muearth = 398600.5e9               # G*Me, the "gravitational constant" for orbital
                                                # motion about the Earth [m^3/s^2] 地球引力常数
        self.OmegaEDot = 7.2921151467e-5        # the sidereal rotation rate of the Earth
                                                # (WGS-84) [rad/s] 地球自转角速度
        self.c = 299792458.                     # speed of light [m/s] 光速
        self.F = -4.442807633e-10               # Relativistic correction term [s/m^(1/2)] 相对论修正项
        self.f1 = 1.57542e9                     # GPS L1 frequency [Hz] GPS L1
        self.f2 = 1.22760e9                     # GPS L2 frequency [Hz] GPS L2
        self.pi = 3.1415926535898               # pi
        self.t_trans = 70*0.001                 ## 70 ms is the average time taken for signal transmission from GPS sats 
# GPS卫星信号传输时间平均值，设置为70ms
