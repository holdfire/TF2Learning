#### 1. 运行tensorflow代码时，报错：  
Tensorflow: device CUDA:0 not supported by XLA service while setting up XLA_GPU_JIT device number 0   
错误原因：显卡被占用了，要指定显卡！os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#### 2.使用keras.applications.Mobilenet_V2时，报错：
UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [Op:Conv2D]
错误原因：指定GUP时，要使用语句：os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:1"

#### 3.使用tensorflow2.0在做卷积运算时，报错：
Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. 
这是因为cuDNN的版本和tensorflow的版本不兼容。
+ 尝试解决方法1，在程序前面加上代码：
import os  
os.environ["CUDA_VISIBLE_DEVICES"] = '1'    
from tensorflow.compat.v1 import ConfigProto    
from tensorflow.compat.v1 import InteractiveSession    
config = ConfigProto()  
config.gpu_options.allow_growth = True  
session = InteractiveSession(config=config)  
+ 尝试解决方法2，使用tensorflow官方提供的docker镜像，切记建立容器后，不要apt-get upgrade，会更新cuda和cudnn的版本。
