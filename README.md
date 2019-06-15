# **ROS for SqueezeSeg**

## Introduction
This is a ROS implementation of [SqueezeSeg](https://arxiv.org/abs/1710.07368).
![][image0] 

## Dependencies
- `python3.5+`
- `TensorFlow`(CPU or GPU version will be both OK)
- `ROS`
- `numpy`
- `easydict`
- `joblib`

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
4. run: 
```bash
$ roslaunch squeezeseg_ros run.launch
```

##  Attention
You should modify the launch/run.launch for the following situations:
1. topic to subscribe the velodyne points:
    `<param name="sub_topic" type="string" value="/kitti/velo/pointcloud" />`
    defalut sub_topic is "/kitti/velo/pointcloud", you should change your own points' topic.

2. CPU or GPU
    `<param name="device_id" type="int" value="-1" /> <!--(cpu:-1)(gpu:0)-->`
    defalut value is "-1" which means cpu, if you want to use gpu, you should change the value to "0" 

3. intensity  channel:
`<param name="i_channel" type="string" value="i" />`
sometimes the intensity channel is "intensity", and you should change the value from "i" to "intensity" 

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

