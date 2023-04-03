#!/bin/bash

echo Post workspace/gril.jpg
wget --post-file=workspace/gril.jpg --read-timeout=1200 "http://127.0.0.1:8090/api/detect_seg" -O-
#wget -O /media/wangch/PortableSSD/tensorRT/learning-cuda-trt-main/tensorrt-integrate-1.22-resful-http/workspace/caught.jpg "http://127.0.0.1:9090/api/getFile" 

# echo Post workspace/vedio/test005.mp4
# wget --post-file=workspace/video/test005.mp4 --read-timeout=1200 "http://127.0.0.1:9090/api/detect_seg_ai" -O-