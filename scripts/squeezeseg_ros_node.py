#! /usr/bin/env python
import rospy
import os.path
import sys
import numpy as np
import tensorflow as tf

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

from squeezeseg.config import *
from squeezeseg.imdb import kitti
from squeezeseg.utils.util import *
from squeezeseg.nets import *

def _normalize(x):
  return (x - x.min())/(x.max() - x.min())

def squeezeseg_tf_init():
    global sess, model, mc
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.Graph().as_default():
        mc = kitti_squeezeSeg_config()
        mc.LOAD_PRETRAINED_MODEL = False
        mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
        model = SqueezeSeg(mc, device_id)

        saver = tf.train.Saver(model.model_params)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, FLAGS.checkpoint)

def predict_class(lidar):
    '''
    function: predict point-wise class using pre-trained model
    input: lidar(64, 512 ,5) (x,y,z,i,r)
           i: intesity range must be (0, 1), not (0, 255)
    output:pred_cls(1, 64, 512):point-wise class
           0: 'unknown'
           1: 'car'
           2: 'pedestrian'
           3: 'cyclist'
    '''
    global sess, model, mc
    lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )
    lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD
    # print("lidar data ", lidar.shape)

    pred_cls = sess.run(
        model.pred_cls,
        feed_dict={
            model.lidar_input:[lidar],
            model.keep_prob: 1.0,
            model.lidar_mask:[lidar_mask]
        }
    )
    # print("pred_cls ", pred_cls.shape)
    return pred_cls

def addColor2LabelPoints(lidar, pre_cls):
    '''
    function: add color to labeled points
            0: 'unknown'    --> white
            1: 'car'        --> red    
            2: 'pedestrian' --> green
            3: 'cyclist'    --> blue
    input:
        lidar(64, 512, 5), 5-->(x,y,z,intnesity,range)
        pre_cls(1, 64, 512)
    output:
        cloud_xyzirgb(64*512, 7)-->(x,y,z,intensity,r,g,b)
    '''
    lidar = lidar[:, :, :4] #(x,y,z,intnesity)
    rgb = np.zeros_like(lidar[:, :, :3])
    lidar = np.concatenate((lidar, rgb), axis = 2)

    cls_mask = np.squeeze(pre_cls)
    cls_mask = cls_mask[:, :, np.newaxis]

    #(x, y, z, intnesity, r, g, b, class)
    lidar_cls = np.concatenate((lidar, cls_mask), axis=2)

    white_rgb = (1.0, 1.0, 1.0)
    red_rgb   = (1.0, 0.0, 0.0)
    green_rgb = (0.0, 1.0, 0.0)
    blue_rgb  = (0.0, 0.0, 1.0)

    lidar_unkown     = lidar_cls[np.where(lidar_cls[:, :, 7] == 0)]
    lidar_car        = lidar_cls[np.where(lidar_cls[:, :, 7] == 1)]
    lidar_pedestrian = lidar_cls[np.where(lidar_cls[:, :, 5] == 2)]
    lidar_cyclist    = lidar_cls[np.where(lidar_cls[:, :, 5] == 3)]

    lidar_unkown[:, 4:7]     = white_rgb 
    lidar_car[:, 4:7]        = red_rgb 
    lidar_pedestrian[:, 4:7] = green_rgb 
    lidar_cyclist[:, 4:7]    = blue_rgb 

    print("[points] unkown:%d, car:%d, pedestrian:%d, cyclist:%d"
        %(lidar_unkown.shape[0],lidar_car.shape[0],
        lidar_pedestrian.shape[0],lidar_cyclist.shape[0]))

    cloud = np.concatenate((lidar_unkown, lidar_car, lidar_pedestrian, lidar_cyclist), axis=0)
    cloud_crop = cloud[:, :7]#(x, y, z, intnesity, r, g, b)

    # print("cloud_crop.shape : ", cloud_crop.shape)
    return cloud_crop


def crop_cloud(cloud, fov=(-45, 45)):
    '''
    function: crop cloud by fov of (-45, 45) at azimuth 
    input: cloud: (n, 4)-->(x, y, z, i)
    ouput: cloud: (m, 4)-->(x, y, z, i) (m ~= n/4)
    ''' 
    x = cloud[:, 0]
    y = cloud[:, 1]
    azi = np.arctan2(y, x).reshape(len(x), 1) * 180.0 / np.pi
    cloud_xyzia = np.concatenate((cloud, azi), axis=1)#(x, y, z, i, azimuth)
    crop_index = (cloud_xyzia[:, 4] > fov[0]) &  (cloud_xyzia[:, 4] < fov[1])#fov
    crop_points = cloud_xyzia[crop_index]
    return crop_points[:, :4] # (x, y, z, i)

def projectCloud2Image(cloud, width=512, height=64, fov=(-45, 45)):
    '''
    function: project cloud to spherical image
    input:  cloud:           (n, 4)      --> (x, y, z, i)
    output: spherical image: (64, 512, 5)--> (x, y, z, i, r)
    '''
    n = cloud.shape[0]
    x = cloud[:, 0].reshape(n, 1)
    y = cloud[:, 1].reshape(n, 1)
    z = cloud[:, 2].reshape(n, 1)
    r = np.sqrt(x**2 + y**2 + z**2).reshape(n, 1)

    yaw = np.arctan2(y, x).reshape(n, 1)
    pitch = np.arcsin(z/r).reshape(n, 1)

    #compute resolution of the cloud at each direction
    resolution_w = (yaw.max() - yaw.min()) / (width - 1)
    resolution_h = (pitch.max() - pitch.min()) / (height - 1)

    #compute each point's grid index in the image
    index_w = np.floor(((yaw - yaw.min()) / resolution_w)).astype(np.int)
    index_h = np.floor((pitch - pitch.min()) / resolution_h).astype(np.int)

    cloud_xyzir = np.concatenate((cloud, r), axis=1) #(x,y,z,i,r)
    
    spherical_image = np.zeros((height, width, 5))
    spherical_image[index_h, index_w, :] = cloud_xyzir[:, np.newaxis, :] #broadcast
    spherical_image = spherical_image[::-1, ::-1, :] #reverse image
    return spherical_image

def m_create_cloud(points_input):
    '''
    points_input = (x, y, z, intensity)
    '''
    header = Header()
    header.stamp = rospy.Time().now()
    # header.frame_id = "velodyne"
    # header.frame_id = "velo_link"
    header.frame_id = frame_id

    # add r, g, b fields
    fields = []
    fields.append( PointField( 'x', 0, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'y', 4, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'z', 8, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'intensity', 12, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'r', 16, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'g', 20, PointField.FLOAT32, 1 ) )
    fields.append( PointField( 'b', 24, PointField.FLOAT32, 1 ) )

    out_cloud = pc2.create_cloud(header, fields, points_input)
    return out_cloud

def velo_callback(msg):
    pcl_msg = pc2.read_points(msg, skip_nans=False, field_names=(
    x_channel, y_channel, z_channel, i_channel))
    cloud_input = np.array(list(pcl_msg), dtype=np.float32)
    print("\ncloud_input.shape = ", cloud_input.shape)
    
    #intensity must be(0, 1), normalize intensity in order for (0, 255)
    _normalize(cloud_input[:, 3])

    #crop and project cloud to spherical image
    crop_points = crop_cloud(cloud_input) #[-45, 45], (x,y,z,i)
    print("crop_points.shape = ", crop_points.shape)
    feature_image = projectCloud2Image(crop_points)#(64, 512, 5), 5->(x,y,z,i,range)
    print("feature_image.shape = ", feature_image.shape)

    #compute point-wise label  
    pre_cls = predict_class(feature_image)
    print("pre_cls.shape = ", pre_cls.shape)

    #add color to each point for ros visualization
    cloud_xyzirgb = addColor2LabelPoints(feature_image, pre_cls)
    # print("cloud_xyzirgb.shape = ", cloud_xyzirgb.shape)

    #create sensor_msg/pointcloud2 for publish
    cloud = m_create_cloud(cloud_xyzirgb)

    pub_velo_.publish(cloud)


if __name__ == '__main__':
    print("[+] squeezeseg_ros_node has started!")
    rospy.init_node('squeezeseg_ros_node')

    #read parameters from launch file
    checkpoint = rospy.get_param('checkpoint')
    pub_topic = rospy.get_param('pub_topic')
    sub_topic = rospy.get_param('sub_topic')
    frame_id = rospy.get_param('frame_id')
    gpu = rospy.get_param('gpu')
    x_channel = rospy.get_param('x_channel')
    y_channel = rospy.get_param('y_channel')
    z_channel = rospy.get_param('z_channel')
    i_channel = rospy.get_param('i_channel')

    device_id = rospy.get_param('device_id')

    #tensorflow setting
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'checkpoint', checkpoint,
        """Path to the model paramter file.""")
    tf.app.flags.DEFINE_string('gpu', gpu, """gpu id.""")
    squeezeseg_tf_init()

    # publish and subscribe
    sub_velo_ = rospy.Subscriber(sub_topic, PointCloud2, velo_callback, queue_size=10)
    pub_velo_ = rospy.Publisher(pub_topic, PointCloud2, queue_size=10)
                            
    rospy.spin()
