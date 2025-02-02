# import logging  # 引入logging模块
# import os.path
# import time
# # 第一步，创建一个logger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)  # Log等级总开关
# # 第二步，创建一个handler，用于写入日志文件
# rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
# print(os.getcwd())
# log_path = os.path.dirname(os.getcwd()) + '/Logs/'
# print(log_path)
# if not os.path.exists(log_path):
#     os.mkdir(log_path)
# log_name = log_path + rq + '.log'
# logfile = log_name
# fh = logging.FileHandler(logfile, mode='w')
# fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# # 第三步，定义handler的输出格式
# formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# fh.setFormatter(formatter)
# # 第四步，将logger添加到handler里面
# logger.addHandler(fh)
# # 日志
# logger.debug('this is a logger debug message')
# logger.info('this is a logger info message')
# logger.warning('this is a logger warning message')
# logger.error('this is a logger error message')
# logger.critical('this is a logger critical message')
import cv2
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()
img1 = cv2.imread("/media/raid/UIED/PCQI/contrast_changed.png")
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# img1 = eng.rgb2gray(matlab.uint8(img1.tolist()))

img2 = cv2.imread("/media/raid/UIED/PCQI/ref.png")
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# img2 = eng.rgb2gray(matlab.uint8(img2.tolist()))

res = eng.PCQI(matlab.double(img2.tolist()),matlab.double(img1.tolist()))

print(res)

