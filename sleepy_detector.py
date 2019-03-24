# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 22:20:37 2018

@author: James Wu
"""

import dlib
import cv2
import numpy as np

#==============================================================================
#   1.landmarks格式转换函数 
#       输入：dlib格式的landmarks
#       输出：numpy格式的landmarks
#==============================================================================          
def landmarks_to_np(landmarks, dtype="int"):
    # 获取landmarks的数量
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

#==============================================================================
#   **************************主函数入口***********************************
#==============================================================================

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"#人脸关键点训练数据路径
detector = dlib.get_frontal_face_detector()#人脸检测器detector
predictor = dlib.shape_predictor(predictor_path)#人脸关键点检测器predictor

cap = cv2.VideoCapture(0)

# 初始化时间序列queue
queue = np.zeros(30,dtype=int)
queue = queue.tolist()

while(cap.isOpened()):
    #读取视频帧
    _, img = cap.read()
    
    #转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 人脸检测
    rects = detector(gray, 1)
    
    # 对每个检测到的人脸进行操作
    for i, rect in enumerate(rects):
        # 得到坐标
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        
        # 绘制边框，加文字标注
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 检测landmarks        
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)
        # 标注landmarks
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
     
        #计算欧氏距离
        d1 =  np.linalg.norm(landmarks[37]-landmarks[41])
        d2 =  np.linalg.norm(landmarks[38]-landmarks[40])
        d3 =  np.linalg.norm(landmarks[43]-landmarks[47])
        d4 =  np.linalg.norm(landmarks[44]-landmarks[46])
        d_mean = (d1+d2+d3+d4)/4
        d5 =np.linalg.norm(landmarks[36]-landmarks[39])
        d6 =np.linalg.norm(landmarks[42]-landmarks[45])
        d_reference = (d5+d6)/2
        d_judge = d_mean/d_reference
        print(d_judge)
        
        flag = int(d_judge<0.25)# 睁/闭眼判定标志:根据阈值判断是否闭眼,闭眼flag=1,睁眼flag=0 (阈值可调)
        
        # flag入队
        queue = queue[1:len(queue)] + [flag]
        
        # 判断是否疲劳：根据时间序列中低于阈值的元素个数是否超过一半
        if sum(queue) > len(queue)/2 :
            cv2.putText(img, "WARNING !", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "SAFE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # 显示结果
    cv2.imshow("Result", img)
    
    k = cv2.waitKey(5) & 0xFF
    if k==27:   #按“Esc”退出
        break

cap.release()
cv2.destroyAllWindows()
