# -*- coding: UTF-8 -*-

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# 基于lio-loam的slam结果进行绘制，绘制三维图像，并输出对应的误差值

class plot_3D():
    def __init__(self, path_gps, path_odom):
        self.path_gps = path_gps
        self.path_odom = path_odom

    def process(self):
        gps_data = pd.read_csv(self.path_gps)
        odom_data = pd.read_csv(self.path_odom)
        # gpsx, gpsy, gpsz = gps_data['field.pose.pose.position.x'], gps_data['field.pose.pose.position.y'], gps_data[
        #     'field.pose.pose.position.z']
        # gpsx, gpsy, gpsz = gps_data['field.gps.latitude'], gps_data['field.gps.longitude'], gps_data[
        #     'field.gps.altitude']
        odomx, odomy, odomz = odom_data['field.pose.pose.position.x'], odom_data['field.pose.pose.position.y'], \
            odom_data['field.pose.pose.position.z']
        ax = plt.axes(projection='3d')
        # ax.plot3D(gpsx, gpsy, gpsz, 'gray')
        ax.plot3D(odomx, odomy, odomz, marker='+', color='red')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.set_zlim([-1, 3])
        plt.savefig("test.png")
        ax.view_init(0, 90)
        plt.savefig("test2.png")

        fig = plt.figure(figsize=(6, 4))
        # plt.plot(gpsx, gpsy, 'red')
        plt.plot(odomx, odomy, marker='+', color= 'green')
        plt.savefig("test1.png")

        fig = plt.figure(figsize=(6, 4))
        # plt.plot(gpsx, gpsy, 'red')
        plt.plot(odomx, odomy, marker='+', color='green')
        plt.savefig("test1.svg")
        # loss_average =
        # loss_average_height =
        return 1


if __name__ == '__main__':
    file_path_gps = "/home/liji/6t/SLAM_test/fusionGPS/20230507driving01/gpsodom.csv"
    file_path_odom = "/media/liji/T71/cloudwithGps/wave/odom.csv"
    plot = plot_3D(file_path_gps, file_path_odom)
    # plot = plot_3D(file_path_gps)
    plot.process()
    # rostopic echo -b ${bag_name} -p /gpsdata > gpsdata.csv
