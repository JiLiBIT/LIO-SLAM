#include "utility.h"
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_driver_msgs/GpswithHeading.h>



class fusionLocalization : public ParamServer
{

public:
    ros::Subscriber subGpsReset;
    ros::Subscriber subOdometry;
    ros::Subscriber subGPS;
    ros::Subscriber subGpsOdometry;

    ros::Publisher pubFusionLocalizaiton;
    ros::Publisher pubImuPath;

    Eigen::Affine3f fusionOdomAffine;

    std::deque<nav_msgs::Odometry> fusionOdomQueue;

    sensor_driver_msgs::GpswithHeading fusionGpsData;
    sensor_driver_msgs::GpswithHeading gpsData;

    double fusionOdomTime = -1;
    double latitude_reset_;
    double longitude_reset_;
    double altitude_reset_;
        


    fusionLocalization()
    {
        subGpsReset             = nh.subscribe<sensor_msgs::NavSatFix> ("liorf/mapping/gps_reset", 50, &fusionLocalization::resetInfoHandler, this, ros::TransportHints().tcpNoDelay());
        
        // subOdometry             = nh.subscribe<nav_msgs::Odometry> ("liorf/mapping/odometry", 50, &fusionLocalization::odomInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // subGpsOdometry          = nh.advertise<nav_msgs::Odometry> ("liorf/mapping/gps_odom", 1, &fusionLocalization::gpsInfoHandler, this, ros::TransportHints().tcpNoDelay());

        subGPS   = nh.subscribe<sensor_driver_msgs::GpswithHeading> (gpsTopic, 200, &fusionLocalization::gpsInfoHandler, this, ros::TransportHints().tcpNoDelay());

        pubFusionLocalizaiton   = nh.advertise<sensor_driver_msgs::GpswithHeading>("liorf/gpsdata", 1);

        // allocateMemory();
    }


    // Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    // {
    //     double x, y, z, roll, pitch, yaw;
    //     x = odom.pose.pose.position.x;
    //     y = odom.pose.pose.position.y;
    //     z = odom.pose.pose.position.z;
    //     tf::Quaternion orientation;
    //     tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
    //     tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
    //     return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    // }

    Eigen::Vector3d enu_to_wgs84(double x_east, double y_north, double z_up, double lati, double longti, double height)
    {
        double pi = std::acos(-1.0);
        double a = 6378137.0;    //earth radius in meters
        double b = 6356752.3142; //earth semiminor in meters 
        double f = (a - b) / a;
        double e_sq = f * (2-f);
    
        double lamb = lati * pi / 180.0;
        double phi = longti * pi / 180.0;
        double sl = std::sin(lamb);
        double N = a / std::sqrt(1 - e_sq * sl * sl);
        double sin_lambda = std::sin(lamb);
        double cos_lambda = std::cos(lamb);
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);
        double x0 = (height + N) * cos_lambda * cos_phi;
        double y0 = (height + N) * cos_lambda * sin_phi;
        double z0 = (height + (1 - e_sq) * N) * sin_lambda;
        double t = cos_lambda * z_up - sin_lambda * y_north;
        double zd = sin_lambda * z_up + cos_lambda * y_north;
        double xd = cos_phi * t - sin_phi * x_east;
        double yd = sin_phi * t + cos_phi * x_east;
        
        //Convert from ECEF cartesian coordinates to 
        //latitude, longitude and height.  WGS-8
        double x = xd + x0;
        double y = yd + y0;
        double z = zd + z0;
        double x2 = std::pow(x, 2);
        double y2 = std::pow(y, 2);
        double z2 = std::pow(z, 2);
        double e = std::sqrt (1-std::pow((b/a) , 2));
        double b2 = b*b;
        double e2 = e*e;
        double ep = e*(a/b); 
        double r = std::sqrt(x2+y2); 
        double r2 = r*r;
        double E2 = a*a - b*b;
        double F = 54*b2*z2;
        double G = r2 + (1-e2)*z2 - e2*E2;
        double c = (e2*e2*F*r2)/(G*G*G); 
        double s = std::pow(( 1 + c + std::sqrt(c*c + 2*c) ) ,(1/3)); 
        double P = F / (3 * std::pow((s+1/s+1), 2) * G*G); 
        double Q = std::sqrt(1+2*e2*e2*P);
        double ro = -(P*e2*r)/(1+Q) + std::sqrt((a*a/2)*(1+1/Q) - (P*(1-e2)*z2)/(Q*(1+Q)) - P*r2/2);
        double tmp = std::pow((r - e2*ro), 2); 
        double U = std::sqrt( tmp + z2 ); 
        double V = std::sqrt( tmp + (1-e2)*z2 ); 
        double zo = (b2*z)/(a*V); 
        double high = U*( 1 - b2/(a*V) );
        double lat = std::atan( (z + ep*ep*zo)/r );
        double temp = std::atan(y/x);
        double lon = temp - pi;
        if (x >= 0) {
            lon = temp;
        }
        else if (x < 0 && y >= 0) {
            lon = pi + temp;
        }
        Eigen::Vector3d result;
        result[0] = lat/(pi/180);
        result[1] = lon/(pi/180);
        result[2] = high;
        return result;
    }

    void resetInfoHandler(const sensor_msgs::NavSatFix::ConstPtr& resetMsg)
    {
        // std::cout << "listening to gpsreset!" << std::endl;
        latitude_reset_ = resetMsg->latitude;
        longitude_reset_ = resetMsg->longitude;
        altitude_reset_ = resetMsg->altitude;




    }


    // void odomInfoHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    // {
        
    //     // std::cout << "listening to fusionodom!" << std::endl;
        
    //     fusionOdomTime = odomMsg->header.stamp.toSec();
    //     // // static tf
    //     // static tf::TransformBroadcaster tfMap2Odom;
    //     // static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
    //     // tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

    //     // std::lock_guard<std::mutex> lock(mtx);

    //     fusionOdomQueue.push_back(*odomMsg);
    //     fusionOdomAffine = odom2affine(*odomMsg);

    //     // get latest odometry (at current Odom stamp)
    //     if (fusionOdomTime == -1)
    //         return;
    //     while (!fusionOdomQueue.empty())
    //     {
    //         if (1)
    //         // if (fusionOdomQueue.front().header.stamp.toSec() <= fusionOdomTime)
    //             fusionOdomQueue.pop_front();
    //         else
    //             break;
    //     }
    //     Eigen::Affine3f fusionOdomAffineFront = odom2affine(fusionOdomQueue.front());
    //     std::cout << "debug11" << std::endl;
    //     Eigen::Affine3f fusionOdomAffineBack = odom2affine(fusionOdomQueue.back());
    //     std::cout << "debug22" << std::endl;
    //     Eigen::Affine3f fusionOdomAffineIncre = fusionOdomAffineFront.inverse() * fusionOdomAffineBack;
    //     std::cout << "debug33" << std::endl;
    //     Eigen::Affine3f fusionOdomAffineLast = fusionOdomAffine * fusionOdomAffineIncre;
        
    //     float x, y, z, roll, pitch, yaw;
    //     pcl::getTranslationAndEulerAngles(fusionOdomAffineLast, x, y, z, roll, pitch, yaw);

        
    //     // std::cout.precision(16);
    //     // std::cout << x << std::endl;
    //     // std::cout << y << std::endl;
    //     // std::cout << z << std::endl;  
    //     // std::cout << latitude_reset_ << std::endl;
    //     // std::cout << longitude_reset_ << std::endl;
    //     // std::cout << altitude_reset_ << std::endl;  


    //     // caculate gps and heading 
    //     Eigen::Vector3d lla;
    //     lla = fusionLocalization::enu_to_wgs84(x, y, z, latitude_reset_, longitude_reset_, altitude_reset_);

    //     // std::cout << lla[0] << std::endl;
    //     // std::cout << lla[1] << std::endl;
    //     // std::cout << lla[2] << std::endl;  
    //     fusionGpsData.header = odomMsg->header;
    //     fusionGpsData.header.frame_id = "gps_frame";
    //     fusionGpsData.gps.header = odomMsg->header;
    //     fusionGpsData.gps.header.frame_id = "gps_frame";
    //     fusionGpsData.gps.latitude = lla[0];
    //     fusionGpsData.gps.longitude = lla[1];
    //     fusionGpsData.gps.altitude = lla[2];
    //     float yawAngle;
    //     yawAngle = 180*yaw/std::acos(-1.0);
    //     if(yawAngle < -180)
	// 	    yawAngle += 360;
    //     if(yawAngle > 180)
	// 	    yawAngle -= 360;
    //     fusionGpsData.heading = yawAngle;


    // }

    void gpsInfoHandler(const sensor_driver_msgs::GpswithHeadingConstPtr& gpsMsg)
    {
        
        gpsData.header = gpsMsg->header;
        gpsData.header.frame_id = "gps_frame";
        gpsData.gps.header = gpsMsg->header;
        gpsData.gps.header.frame_id = "gps_frame";
        gpsData.gps.latitude = gpsMsg->gps.latitude;
        gpsData.gps.longitude = gpsMsg->gps.altitude;
        gpsData.gps.altitude = gpsMsg->gps.altitude;
        gpsData.heading = gpsMsg->heading;
    }

    void logicLocalizaionThread()
    {

    //     ros::Rate rate(pubLocalizationFrequency);  
    //     // std::cout << "publishing  fusion Gps!" << std::endl;
    //     while (ros::ok())
    //     {
    //         rate.sleep();
    //         if(1)
    //         {
    //             pubFusionLocalizaiton.publish(fusionGpsData);
    //         }
    //         else
    //         {
    //             pubFusionLocalizaiton.publish(gpsData);
    //         }
    //     }
        
    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "liorf");

    // extern Eigen::Vector3d gpsReset;

    fusionLocalization FL;

    ROS_INFO("\033[1;32m----> Fusion Localization Started.\033[0m");
    
    std::thread pubthread(&fusionLocalization::logicLocalizaionThread, &FL);

    ros::spin();

    pubthread.join();

    return 0;
}
