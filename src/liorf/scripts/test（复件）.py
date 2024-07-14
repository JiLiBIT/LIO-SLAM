#!/usr/bin/env python

import rospy
from sensor_driver_msgs.msg import GpswithHeading
import matplotlib.pyplot as plt
import math

# 存储经纬度转换后的位置数据
converted_positions = {'GPSmsg': [], 'gpsdata': [], 'liorf/gpsdata': []}

# 经纬度转米数函数
def lat_lon_to_meters(lat1, lon1, lat2, lon2):
    # 使用WGS84椭球体的半长轴和扁率
    a = 6378137.0  # 半长轴（单位：米）
    f = 1.0 / 298.257223563  # 扁率

    # 计算纬度差和经度差的弧度
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)

    # 计算纬度差和经度差的弧度平均值
    lat_avg = 0.5 * (math.radians(lat1) + math.radians(lat2))

    # 计算子午圈半径和横向半径
    R = a / math.sqrt(1.0 - (2.0 * f - f * f) * math.sin(lat_avg) * math.sin(lat_avg))
    Rx = R * math.cos(lat_avg)

    # 计算东北天坐标系中的距离
    dx = Rx * d_lon
    dy = R * d_lat

    return dx, dy
    
# GPS消息回调函数
def gps_callback(msg, topic):
    global converted_positions

    if len(converted_positions[topic]) > 0:
        prev_lat, prev_lon = converted_positions[topic][-1]
        dx, dy = lat_lon_to_meters(prev_lat, prev_lon, msg.gps.latitude, msg.gps.longitude)
        converted_positions[topic].append((msg.gps.latitude, msg.gps.longitude))
    else:
        converted_positions[topic].append((msg.gps.latitude, msg.gps.longitude))

def main():
    rospy.init_node('gps_trajectory_plotter', anonymous=True)

    topics = ['GPSmsg', 'gpsdata', 'liorf/gpsdata']

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
                    label = topic.split('/')[-1]  # 从话题中提取最后一部分作为标签
                    plt.plot(x, y, label=label)

            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
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

