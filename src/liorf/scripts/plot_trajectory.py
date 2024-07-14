#!/usr/bin/env python

import rospy
from sensor_driver_msgs.msg import GpswithHeading
import matplotlib.pyplot as plt
from sensor_driver_msgs.msg import GpswithHeading
import rospy
import matplotlib.pyplot as plt
from math import *
import pandas as pd
import numpy as np
import os
import re
import time
import math
import sys
import threading

# 存储经纬度转换后的位置数据
converted_positions = {'GPSmsg': [], 'gpsdata': [], 'liorf/gpsdata': [], 'sensor_fusion_output': []}
def positionGPStoMeters( longitude, latitude):
    WGS84_ECCENTRICITY = 0.0818192
    WGS84_EQUATORIAL_RADIUS = 6378.137
    k0 = 0.9996

    Zone = (int)(longitude / 6) + 1
    lonBase = Zone * 6 - 3

    vPhi = (float)(1 / sqrt(1 - pow(WGS84_ECCENTRICITY * sin(latitude * pi / 180.0), 2)))
    A = (float)((longitude - lonBase) * pi / 180.0 * cos(latitude * pi / 180.0))
    sPhi = (float)((1 - pow(WGS84_ECCENTRICITY, 2) / 4.0 - 3 * pow(WGS84_ECCENTRICITY, 4) / 64.0
                    - 5 * pow(WGS84_ECCENTRICITY, 6) / 256.0) * latitude * pi / 180.0
                - (3 * pow(WGS84_ECCENTRICITY, 2) / 8.0 + 3 * pow(WGS84_ECCENTRICITY, 4) / 32.0
                    + 45 * pow(WGS84_ECCENTRICITY, 6) / 1024.0) * sin(2 * latitude * pi / 180.0)
                + (15 * pow(WGS84_ECCENTRICITY, 4) / 256.0 + 45 * pow(WGS84_ECCENTRICITY, 6) / 256.0)
                * sin(4 * latitude * pi / 180.0)
                - (35 * pow(WGS84_ECCENTRICITY, 6) / 3072.0) * sin(6 * latitude * pi / 180.0))
    T = (float)(pow(tan(latitude * pi / 180.0), 2))
    C = (float)((pow(WGS84_ECCENTRICITY, 2) / (1 - pow(WGS84_ECCENTRICITY, 2)))
                * pow(cos(latitude * pi / 180.0), 2))

    pose_x = (float)((k0 * WGS84_EQUATORIAL_RADIUS * vPhi * (A + (1 - T + C) * pow(A, 3) / 6.0
                                                            + (5 - 18 * T + pow(T, 2)) * pow(A, 5) / 120.0)) * 1000)
    pose_y = (float)((k0 * WGS84_EQUATORIAL_RADIUS * (sPhi + vPhi * tan(latitude * pi / 180.0) * (pow(A, 2) / 2
                                                                                                + (
                                                                                                            5 - T + 9 * C + 4 * C * C) * pow(
                A, 4) / 24.0 + (61 - 58 * T + T * T) * pow(A, 6) / 720.0))) * 1000)
    global e0, n0
    
    if (0 == e0 and 0 == n0):
        e0 = int(pose_x)
        n0 = int(pose_y)

    pose_x -= e0
    pose_y -= n0

    return pose_x, pose_y
# # 经纬度转米数函数
# def lat_lon_to_meters(lat1, lon1, lat2, lon2):
#     # 使用WGS84椭球体的半长轴和扁率
#     a = 6378137.0  # 半长轴（单位：米）
#     f = 1.0 / 298.257223563  # 扁率

#     # 计算纬度差和经度差的弧度
#     d_lat = math.radians(lat2 - lat1)
#     d_lon = math.radians(lon2 - lon1)

#     # 计算纬度差和经度差的弧度平均值
#     lat_avg = 0.5 * (math.radians(lat1) + math.radians(lat2))

#     # 计算子午圈半径和横向半径
#     R = a / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(lat_avg) * math.sin(lat_avg))
#     Rx = R * math.cos(lat_avg)

#     # 计算东北天坐标系中的距离
#     dx = Rx * d_lon
#     dy = R * d_lat

#     return dx, dy

def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    # 每纬度单位对应的米数
    meters_per_degree_latitude = 111111.0  # 在大致纬度范围内使用这个值

    # 计算经纬度差值
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # 将经纬度差值转换为米制坐标差值
    dx = d_lon * meters_per_degree_latitude * math.cos(math.radians(lat1))
    dy = d_lat * meters_per_degree_latitude

    return dx, dy
    
# GPS消息回调函数
def gps_callback(msg, topic):
    global converted_positions
    global a, b
    a, b =positionGPStoMeters(msg.gps.longitude, msg.gps.latitude)
    converted_positions[topic].append((a, b))
        # converted_positions[topic].append((msg.gps.latitude, msg.gps.longitude))

def main():
    rospy.init_node('gps_trajectory_plotter', anonymous=True)
    topics = ['GPSmsg', 'gpsdata', 'liorf/gpsdata', 'sensor_fusion_output']
    global e0, n0
    global a, b
    a = 0
    b = 0
    e0 = 0
    n0 = 0
    for topic in topics:
        
        rospy.Subscriber(topic, GpswithHeading, gps_callback, callback_args=topic)

    plt.ion()  # 开启交互模式，实时更新绘图

    try:
        while not rospy.is_shutdown():
            plt.clf()  # 清除上一次的绘图内容

            for topic, positions in converted_positions.items():
                if positions:
                    x = [pos[0] for pos in positions]
                    y = [pos[1] for pos in positions]
                    label = topic  # 从话题中提取最后一部分作为标签
                    plt.plot(x, y, label=label)

            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.xlim((a-50, a+50))
            plt.ylim((b-50, b+50))
            plt.title('Converted GPS Trajectories')
            plt.legend()
            plt.grid()

            plt.pause(0.1)  # 暂停一段时间，等待绘图更新

    except rospy.ROSInterruptException:
        pass

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()

