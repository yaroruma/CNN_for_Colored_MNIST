#!/usr/bin/env python
# coding: utf-8

# # Image data augmentation

# In[1]:


import torch
import numpy as np
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt


# In[2]:


# 배경 패턴 추가시 300*300 이미지를 imread로 읽어서 pts리스트에 추가할 것
pt1 = plt.imread('patterns/pt1.jpg')
pt2 = plt.imread('patterns/pt2.jpg')
pt3 = plt.imread('patterns/pt3.jpg')
pts = [pt1, pt2, pt3]
train_data = datasets.MNIST('./', train = True, download = False, transform = torchvision.transforms.ToTensor())


# In[3]:


def augmentation(img, mode, pt, pt2 = None): 
    img = img.repeat(3, 1, 1)
    img = (img*255).numpy().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    r = np.random.randint(272)
    #mode 0 : 배경 입히기(테두리 있음)
    if mode == 0:
        img = np.where(img > 0, img, img + pt[r:r+28,r:r+28])

    #mode 1 : 숫자부분을 패턴 이미지로 대체(색칠)
    elif mode == 1:
        img = np.where(img > 0, pt[r:r+28,r:r+28], img)

    #mode 2 : 숫자부분을 페턴이미지로 대체하고 배경을 다른 패턴이미지로 사용(테두리 없음)
    elif mode == 2:
        r2 = np.random.randint(272)
        img = np.where(img > 0, pt[r:r+28,r:r+28] , pt2[r2:r2+28,r2:r2+28])

    #mode 3 : 중앙부분에 흰색 네모 박스 생성(가림)
    else:
        r = np.random.randint(7) # 박스 크기
        r2 = np.random.randint(14) #박스 시작점 위치
        box = np.zeros_like(img)
        box[7+r2:r2+r+7, 7+r2:r2+r+7] = 255
        img = np.where(box > 0, box, img)

    return img


# In[5]:


results = []
for i, data in enumerate(train_data, 0):
    inputs, labels = data
    mode = np.random.randint(4) #한 가지 모드만 사용할 때는 변경
    ptns = list(range(len(pts)))
    ptn1 = ptns.pop(np.random.randint(len(ptns)))
    ptn2 = ptns.pop(np.random.randint(len(ptns)))
    outputs = augmentation(inputs, mode, pts[ptn1], pts[ptn2])
    path = 'images/' + format(i, '06') + '.png'
    plt.imsave(path, outputs)
    results.append(labels)
    #작업 내역을 보고싶을 때 사용
    '''if i % 2000 == 0:
        plt.imshow(outputs)
        plt.show()
    '''
np.savetxt("label.txt", np.array(results), fmt='%d', delimiter='\n')


# In[ ]:


#########################
#데이터를 불러올 때 사용#
#########################

#이미지 데이터는 템플릿의 run.py 참고

label = numpy.loadtxt("label.txt", delimiter='\n')

