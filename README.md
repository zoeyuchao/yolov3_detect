# **yolo_v3前向简化代码**

本代码适用于python2和python3

- python2

```
pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

- python3

```
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

安装完之后执行：

```
git clone https://github.com/MoynaChen/detect_obj
python detect.py
```

**Tips**

1.在data文件夹中添加xxx.data,xxx.names(示例为球类检测，ball.data和ball.name已经添加到data文件夹中)

2.在weight文件夹中添加训练好的best.pt

3.在cfg文件夹中添加训练网络时修改过的yolov3.cfg（球类检测的yolov3已经添加至cfg文件夹中）

4.可能遇到no model named xxx，解决方法 pip install xxx

5.detect.py程序中

    if __name__ == '__main__':
    img = cv2.imread('data/samples/three_balls_2.jpg')#可修改为自己的图片
    with torch.no_grad():
     	result_obj=detect(img)
     	print(result_obj)
     	print(result_obj.shape)

**result_obj**

返回值为（n,7）维tensor矩阵，其中n为所识别图片中的目标物体数，如出现一个足球，一个篮球，则n为2

每行包含7列，前4列为xyxy,即矩形框角点的左边，第5列，第6类分别为不同指标下该类别的置信度，可以第6列为准

第7列为识别的物体类别，0,1,2,3分别对应于ball.name中的类别顺序。
