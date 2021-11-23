# CNN for Colored MNIST

## Brief 
딥러닝 과목 기말 프로젝트 코드입니다. 다양한 컬러 숫자이미지를 인식할 수 있는 CNN모델 코드와 MNIST를 이용해 패턴 컬러 이미지를 만드는 data augmentation 코드로 이루어져 있습니다. 

## Mission
28*28 사이즈의 사진, 그림 등으로 이루어진 컬러 숫자이미지를 인식하는 CNN 모델 제작. general model을 위한 training data augmentation. 모델에 대한 분석.
![tests](https://user-images.githubusercontent.com/72549957/143024998-36a5dcd7-0aa3-4009-bc2b-0b54f15c582d.JPG)

## Environment
- Anaconda 3
- python 3.9
- pytorch
- torchvision

## Data augmentation
패턴 이미지와 MNIST 데이터셋을 4가지 방법으로 조합하여 training data 제작. 각 이미지별 label data 기록.
![data](https://user-images.githubusercontent.com/72549957/143024995-bee5539c-665e-4f0c-86cd-54f3a15aabd7.JPG)

## Model
(3*3 conv2d + ReLU + MaxPooling) x2, (Fully connected + ReLU) x3
![model](https://user-images.githubusercontent.com/72549957/143024997-1c1ccf18-3562-4e95-8076-01bc3c7ba7f4.JPG)
