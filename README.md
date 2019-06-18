# **ROS for SqueezeSeg**

## Introduction
This is a ROS implementation of [SqueezeSeg](https://arxiv.org/abs/1710.07368) written in python, this project does not train the network, just runs the demo using the pre-trained model. 
![][image0] 

## Dependencies
- `python3.5+`
- `TensorFlow`(CPU or GPU version will be both OK, but GPU is much faster)
- `ROS`
- `numpy`
- `easydict`
- `joblib`

## 3D Lidar Data
The input data must be from Velodyne HDL-64E. I put the rosbag-data at "Baidu Cloud Disk":
- Link address : [kitti_2011_09_26_drive_0001_synced.bag](https://pan.baidu.com/s/1r3hQy6oGUsuRNJxqHMtkmw)
- Extracted code : 6mc2
- Play the data :
```bash
$ roscore
$ rosbag play -l kitti_2011_09_26_drive_0001_synced.bag
```

##  Run the code
1. Download the repository to your ROS workspace: catkin_ws/src
2. make: 
```bash
$ catkin_make
```
3. source:
```bash
$ source devel/setup.bash
```
4. add authority to script file
```bash
$ chmod 755 src/squeezeseg_ros/scripts/squeezeseg_ros_node.py
```
5. run: 
- For GPU :
```bash
$ roslaunch squeezeseg_ros run_kitti_gpu.launch
```
- For CPU
```bash
$ roslaunch squeezeseg_ros run_kitti_cpu.launch
```

##  Attention
If using your own data, you should modify the launch file for the following situations:
1. topic to subscribe the velodyne points:
    `<param name="sub_topic" type="string" value="/kitti/velo/pointcloud" />`
    defalut sub_topic is "/kitti/velo/pointcloud", you should use your own points' topic.
2. intensity  channel:
`<param name="i_channel" type="string" value="i" />`
if the intensity channel is "intensity", and you should change the value from "i" to "intensity" 

## Reference 
### code
[BichenWuUCB/SqueezeSeg](https://github.com/BichenWuUCB/SqueezeSeg)

### article
[SqueezeSeg](https://arxiv.org/abs/1710.07368)

    @inproceedings{wu2017squeezeseg,
       title={Squeezeseg: Convolutional neural nets with recurrent crf for real-time road-object segmentation from 3d lidar point cloud},
       author={Wu, Bichen and Wan, Alvin and Yue, Xiangyu and Keutzer, Kurt},
       booktitle={ICRA}, 
       year={2018}
     }


[//]: # "Image References"
[image0]: ./data/samples_out/squeezeseg.png

