# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse #argparse 사용, 명령행 옵션, 인자를 쉽게 파싱할 수 있게 해주는 모듈
import sys #sys 사용, 파이썬 인터프리터 관련 정보와 기능을 제공하는 모듈
from copy import deepcopy #copy 모듈 사용, 객체를 복사하는 모듈
from pathlib import Path #pathlib 모듈 사용, 파일시스템 경로를 문자열로 다루기 쉽게 해주는 모듈

FILE = Path(__file__).resolve() #파일 경로, 절대 경로로 변환
ROOT = FILE.parents[1]  # YOLOv5 root directory, 경로의 상위 디렉토리 반환
if str(ROOT) not in sys.path: #ROOT가 sys.path에 없으면
    sys.path.append(str(ROOT))  # add ROOT to PATH, sys.path에 ROOT 추가
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np #numpy 사용, 수치 데이터를 다루는 파이썬 패키지
import tensorflow as tf #tensorflow 사용, 딥러닝 모델을 만들고 학습시키는 라이브러리
import torch #torch 사용, 딥러닝 라이브러리
import torch.nn as nn #torch.nn 사용, 신경망 모듈
from tensorflow import keras #keras 사용, 딥러닝 모델을 만들고 학습시키는 라이브러리

from models.common import ( #models.common 모듈 사용, 공통 모듈
    C3, #  convolutional layer 타입 or a model architecture, 3x3 convolutional layer
    SPP, # Spatial Pyramid Pooling, a technique used in convolutional neural networks에서 사용되는 scale levels 나타내는 pool features.
    SPPF, # Spatial Pyramid Pooling에 연관된 함수, SPP와 비슷하지만 더 빠르게 동작
    Bottleneck, #차원을 줄이고, 특징을 추출하는 레이어, ResNet에서 사용
    BottleneckCSP, # Cross Stage Partial connections에서 사용, 모델 향상을 위해 사용, ResNet에서 사용
    C3x, #cross convolutions 사용하는 C3 layer, cross convolutions은 두 레이어 사이의 정보를 교환하는 방법
    Concat, #axis와 tensor list를 합침, axis는 합칠 방향을 나타냄
    Conv, # Convolutional neural network에서 사용되는 레이어, 입력 데이터에 필터를 적용하여 출력 데이터를 만드는 레이어
    CrossConv, # a cross or criss-cross pattern 연결을 사용하는 convolutional layer, 두 레이어 사이의 정보를 교환하는 방법
    DWConv, # Depthwise Convolutional layer, 분리된 커널을 사용하여 입력 채널마다 따로따로 컨볼루션 연산을 수행, 모델의 파라미터 수를 줄이고, 모델의 크기를 줄이는 효과
    DWConvTranspose2d, # Depthwise Convolutional Transpose layer, 입력 채널마다 따로따로 컨볼루션 연산을 수행, 모델의 파라미터 수를 줄이고, 모델의 크기를 줄이는 효과
    Focus, # 집중화된 wh 정보를 c-space로, 입력 이미지의 중심을 기준으로 4개의 이미지를 만들어서 합치는 방법
    autopad, #padding을 자동으로 설정, 입력 이미지의 크기와 커널 사이즈에 따라 padding을 자동으로 설정
)
from models.experimental import MixConv2d, attempt_load #실험적 모델, MixConv2d, attempt_load 모듈 사용
from models.yolo import Detect, Segment #yolo 모델, Detect, Segment 모듈 사용
from utils.activations import SiLU #활성화 함수, SiLU 모듈 사용
from utils.general import LOGGER, make_divisible, print_args #일반적인 유틸리티, LOGGER, make_divisible, print_args 모듈 사용


class TFBN(keras.layers.Layer): #keras.layers.Layer 상속, BN 레이어
    # TensorFlow BatchNormalization wrapper, keras.layers.Layer 상속
    def __init__(self, w=None): #초기화, w는 None
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        self.bn = keras.layers.BatchNormalization( #BatchNormalization 레이어, 배치 정규화
            beta_initializer=keras.initializers.Constant(w.bias.numpy()), #beta 초기화, bias 초기화
            gamma_initializer=keras.initializers.Constant(w.weight.numpy()), #gamma 초기화, weight 초기화
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.numpy()), #moving_mean 초기화, running_mean 초기화
            moving_variance_initializer=keras.initializers.Constant(w.running_var.numpy()), #moving_variance 초기화, running_var 초기화
            epsilon=w.eps, #epsilon,
        )

    def call(self, inputs):#호출, inputs를 받아서
        return self.bn(inputs) #bn 레이어에 inputs 전달, 결과 반환


class TFPad(keras.layers.Layer): #keras.layers.Layer 상속, pad 레이어
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad): #초기화, pad를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        if isinstance(pad, int): #pad가 int형이면, 즉, pad가 정수형이면
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]) #pad 설정, 4x2 텐서
        else:  # tuple/list #그렇지 않으면, 즉, pad가 튜플이나 리스트이면
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]) #pad 설정, 4x2 텐서

    def call(self, inputs): #호출, inputs를 받아서
        return tf.pad(inputs, self.pad, mode="constant", constant_values=0) #inputs에 pad 적용, 결과 반환


class TFConv(keras.layers.Layer): #keras.layers.Layer 상속, conv 레이어
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None): #초기화, c1, c2, k, s, p, g, act, w를 받아서
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()#상속받은 클래스의 __init__ 메소드 호출
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument" #g가 1이 아니면 에러, 즉, g가 1이 아니면 에러
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
        conv = keras.layers.Conv2D( #Conv2D 레이어, 2차원 컨볼루션 레이어
            filters=c2, #필터, 출력 공간의 차원(필터 수)
            kernel_size=k, #커널 사이즈, 정수 혹은 튜플/리스트
            strides=s, #스트라이드, 정수 혹은 튜플/리스트
            padding="SAME" if s == 1 else "VALID", #패딩이 SAME이면 SAME, 아니면 VALID, 즉, s가 1이면 SAME, 아니면 VALID
            use_bias=not hasattr(w, "bn"), #w에 bn이 없으면 bias 사용, 즉, w에 bn이 없으면 bias 사용
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()), #kernel 초기화, weight 초기화
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()),#bias 초기화, bias 초기화
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv]) # 셀프 conv에 conv 레이어 저장, 즉, s가 1이면 conv 저장, 아니면 Padd와 conv 저장
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity #bn 레이어 저장, w에 bn이 있으면 bn 레이어 저장, 아니면 tf.identity 저장
        self.act = activations(w.act) if act else tf.identity #활성화 함수 저장, act가 True이면 활성화 함수 저장, 아니면 tf.identity 저장

    def call(self, inputs): #호출, inputs를 받아서
        return self.act(self.bn(self.conv(inputs))) #conv, bn, act 순서로 inputs에 적용, 결과 반환


class TFDWConv(keras.layers.Layer): #keras.layers.Layer 상속, DWConv 레이어
    # Depthwise convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True, w=None):# 초기화, c1, c2, k, s, p, act, w를 받아서
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        assert c2 % c1 == 0, f"TFDWConv() output={c2} must be a multiple of input={c1} channels" #c2가 c1의 배수가 아니면 에러, 즉, c2가 c1의 배수가 아니면 에러
        conv = keras.layers.DepthwiseConv2D( #DepthwiseConv2D 레이어, Depthwise convolution layer
            kernel_size=k, #커널 사이즈, 정수 혹은 튜플/리스트
            depth_multiplier=c2 // c1,#depth_multiplier, 출력 채널 수를 입력 채널 수의 몇 배로 할 것인지
            strides=s,#스트라이드, 정수 혹은 튜플/리스트
            padding="SAME" if s == 1 else "VALID", #패딩이 SAME이면 SAME, 아니면 VALID, 즉, s가 1이면 SAME, 아니면 VALID
            use_bias=not hasattr(w, "bn"), #w에 bn이 없으면 bias 사용, 즉, w에 bn이 없으면 bias 사용
            depthwise_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).numpy()),#depthwise 초기화, weight 초기화
            bias_initializer="zeros" if hasattr(w, "bn") else keras.initializers.Constant(w.conv.bias.numpy()), #bias 초기화, bias 초기화
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv]) #셀프 conv에 conv 레이어 저장, 즉, s가 1이면 conv 저장, 아니면 Padd와 conv 저장 
        self.bn = TFBN(w.bn) if hasattr(w, "bn") else tf.identity #bn 레이어 저장, w에 bn이 있으면 bn 레이어 저장, 아니면 tf.identity 저장
        self.act = activations(w.act) if act else tf.identity #활성화 함수 저장, act가 True이면 활성화 함수 저장, 아니면 tf.identity 저장

    def call(self, inputs): #호출, inputs를 받아서
        return self.act(self.bn(self.conv(inputs))) #conv, bn, act 순서로 inputs에 적용, 결과 반환


class TFDWConvTranspose2d(keras.layers.Layer):#keras.layers.Layer 상속, DWConvTranspose2d 레이어
    # Depthwise ConvTranspose2d
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0, w=None): #초기화, c1, c2, k, s, p1, p2, w를 받아서
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super().__init__()#상속받은 클래스의 __init__ 메소드 호출, 초기화
        assert c1 == c2, f"TFDWConv() output={c2} must be equal to input={c1} channels" #c1과 c2가 같지 않으면 에러, 즉, c1과 c2가 같지 않으면 에러
        assert k == 4 and p1 == 1, "TFDWConv() only valid for k=4 and p1=1" #k가 4이고 p1이 1이 아니면 에러, 즉, k가 4이고 p1이 1이 아니면 에러
        weight, bias = w.weight.permute(2, 3, 1, 0).numpy(), w.bias.numpy() #weight, bias 초기화, weight, bias 초기화
        self.c1 = c1 #c1 저장, 입력 채널 수
        self.conv = [ #conv 레이어 , DWConvTranspose2d 레이어
            keras.layers.Conv2DTranspose( #Conv2DTranspose 레이어, 2차원 컨볼루션 트랜스포즈 레이어
                filters=1, #필터, 출력 공간의 차원(필터 수)
                kernel_size=k, #커널 사이즈, 정수 혹은 튜플/리스트
                strides=s, #스트라이드, 정수 혹은 튜플/리스트
                padding="VALID", #패딩, VALID
                output_padding=p2, #output_padding, 출력 패딩
                use_bias=True, #bias 사용, True
                kernel_initializer=keras.initializers.Constant(weight[..., i : i + 1]), #kernel 초기화, weight 초기화
                bias_initializer=keras.initializers.Constant(bias[i]), #bias 초기화
            )
            for i in range(c1) #c1만큼 반복, i는 0부터 c1-1까지
        ]

    def call(self, inputs):#호출, inputs를 받아서
        return tf.concat([m(x) for m, x in zip(self.conv, tf.split(inputs, self.c1, 3))], 3)[:, 1:-1, 1:-1]#conv에 inputs 적용, 결과 반환


class TFFocus(keras.layers.Layer): #keras.layers.Layer 상속, Focus 레이어
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None): #초기화, c1, c2, k, s, p, g, act, w를 받아서
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv) #conv 레이어, c1*4, c2, k, s, p, g, act, w.conv를 받아서

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c), Focus wh information into c-space
        # inputs = inputs / 255  # normalize 0-255 to 0-1
        inputs = [inputs[:, ::2, ::2, :], inputs[:, 1::2, ::2, :], inputs[:, ::2, 1::2, :], inputs[:, 1::2, 1::2, :]]#inputs를 4개로 나눔
        return self.conv(tf.concat(inputs, 3))#conv에 inputs 적용, 결과 반환


class TFBottleneck(keras.layers.Layer): #keras.layers.Layer 상속, Bottleneck 레이어
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = int(c2 * e)  # hidden channels, c_는 c2*e
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2) #cv2 레이어, c_, c2, 3, 1, g, w.cv2를 받아서
        self.add = shortcut and c1 == c2 #add 저장, shortcut이 True이고 c1과 c2가 같으면 True, 아니면 False

    def call(self, inputs): #호출, inputs를 받아서
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs)) #add가 True이면 inputs + self.cv2(self.cv1(inputs)), 아니면 self.cv2(self.cv1(inputs)) 반환


class TFCrossConv(keras.layers.Layer): #keras.layers.Layer 상속, CrossConv 레이어
    # Cross Convolution
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False, w=None): #초기화, c1, c2, k, s, g, e, shortcut, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = int(c2 * e)  # hidden channels #c_는 c2*e
        self.cv1 = TFConv(c1, c_, (1, k), (1, s), w=w.cv1) #cv1 레이어, c1, c_, (1, k), (1, s), w.cv1를 받아서
        self.cv2 = TFConv(c_, c2, (k, 1), (s, 1), g=g, w=w.cv2) #cv2 레이어, c_, c2, (k, 1), (s, 1), g, w.cv2를 받아서
        self.add = shortcut and c1 == c2 #add 저장, shortcut이 True이고 c1과 c2가 같으면 True, 아니면 False

    def call(self, inputs): #호출, inputs를 받아서
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs)) #add가 True이면 inputs + self.cv2(self.cv1(inputs)), 아니면 self.cv2(self.cv1(inputs)) 반환


class TFConv2d(keras.layers.Layer): #keras.layers.Layer 상속, Conv2d 레이어
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None): #초기화, c1, c2, k, s, g, bias, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument" #g가 1이 아니면 에러, 즉, g가 1이 아니면 에러
        self.conv = keras.layers.Conv2D( #Conv2D 레이어, 2차원 컨볼루션 레이어
            filters=c2, #필터, 출력 공간의 차원(필터 수)
            kernel_size=k,#커널 사이즈, 정수 혹은 튜플/리스트
            strides=s, #스트라이드, 정수 혹은 튜플/리스트
            padding="VALID", #패딩, VALID
            use_bias=bias, #bias 사용, bias
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).numpy()), #kernel 초기화, weight 초기화
            bias_initializer=keras.initializers.Constant(w.bias.numpy()) if bias else None, #bias 초기화, bias 초기화
        )

    def call(self, inputs): #호출, inputs를 받아서
        return self.conv(inputs) #conv에 inputs 적용, 결과 반환


class TFBottleneckCSP(keras.layers.Layer): #keras.layers.Layer 상속, BottleneckCSP 레이어
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None): #초기화, c1, c2, n, shortcut, g, e, w를 받아서
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = int(c2 * e)  # hidden channels, #c_는 c2*e
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2) #cv2 레이어, c1, c_, 1, 1, bias=False, w.cv2를 받아서
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3) #cv3 레이어, c_, c_, 1, 1, bias=False, w.cv3를 받아서
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4) #cv4 레이어, 2*c_, c2, 1, 1, w.cv4를 받아서
        self.bn = TFBN(w.bn) #bn 레이어, w.bn을 받아서
        self.act = lambda x: keras.activations.swish(x) #활성화 함수, swish 함수
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)]) #m 레이어, TFBottleneck 레이어

    def call(self, inputs): #호출, inputs를 받아서
        y1 = self.cv3(self.m(self.cv1(inputs))) #y1 저장, cv3, m, cv1에 inputs 적용
        y2 = self.cv2(inputs) #y2 저장, cv2에 inputs 적용
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3)))) #cv4에 act, bn, y1과 y2를 concat한 것 적용, 결과 반환


class TFC3(keras.layers.Layer): #keras.layers.Layer 상속, C3 레이어
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None): #초기화, c1, c2, n, shortcut, g, e, w를 받아서
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = int(c2 * e)  # hidden channels, #c_는 c2*e
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2) #cv2 레이어, c1, c_, 1, 1, w.cv2를 받아서
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3) #cv3 레이어, 2*c_, c2, 1, 1, w.cv3를 받아서
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)]) #m 레이어, TFBottleneck 레이어

    def call(self, inputs): #호출, inputs를 받아서
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3)) #cv3에 m, cv1, cv2에 inputs 적용, 결과 반환


class TFC3x(keras.layers.Layer): #keras.layers.Layer 상속, C3x 레이어
    # 3 module with cross-convolutions 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None): #초기화, c1, c2, n, shortcut, g, e, w를 받아서
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = int(c2 * e)  # hidden channels, #c_는 c2*e
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2) #cv2 레이어, c1, c_, 1, 1, w.cv2를 받아서
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3) #cv3 레이어, 2*c_, c2, 1, 1, w.cv3를 받아서
        self.m = keras.Sequential( #m 레이어, TFCrossConv 레이어
            [TFCrossConv(c_, c_, k=3, s=1, g=g, e=1.0, shortcut=shortcut, w=w.m[j]) for j in range(n)] #TFCrossConv 레이어
        )

    def call(self, inputs): #호출, inputs를 받아서
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3)) #cv3에 m, cv1, cv2에 inputs 적용, 결과 반환


class TFSPP(keras.layers.Layer): #keras.layers.Layer 상속, SPP 레이어
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None): #초기화, c1, c2, k, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = c1 // 2  # hidden channels, #c_는 c1//2
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2) #cv2 레이어, c_*(len(k)+1), c2, 1, 1, w.cv2를 받아서
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding="SAME") for x in k] #m 레이어, MaxPool2D 레이어

    def call(self, inputs): #호출, inputs를 받아서
        x = self.cv1(inputs) #x 저장, cv1에 inputs 적용
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3)) #cv2에 x와 m에 x를 적용한 것을 concat한 것 적용, 결과 반환


class TFSPPF(keras.layers.Layer): #keras.layers.Layer 상속, SPPF 레이어
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None): #초기화, c1, c2, k, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        c_ = c1 // 2  # hidden channels, #c_는 c1//2
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1) #cv1 레이어, c1, c_, 1, 1, w.cv1를 받아서
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2) #cv2 레이어, c_ * 4, c2, 1, 1, w.cv2를 받아서
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding="SAME") #m 레이어, MaxPool2D 레이어

    def call(self, inputs): #호출, inputs를 받아서
        x = self.cv1(inputs) #x 저장, cv1에 inputs 적용
        y1 = self.m(x) #y1 저장, m에 x 적용
        y2 = self.m(y1) #y2 저장, m에 y1 적용
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3)) #cv2에 x, y1, y2, m에 y2를 적용한 것을 concat한 것 적용, 결과 반환


class TFDetect(keras.layers.Layer): #keras.layers.Layer 상속, Detect 레이어
    # TF YOLOv5 Detect layer
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer, #초기화, nc, anchors, ch, imgsz, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        self.stride = tf.convert_to_tensor(w.stride.numpy(), dtype=tf.float32) #stride 저장, w.stride를 텐서로 변환
        self.nc = nc  # number of classes, #nc 저장, 클래스 수
        self.no = nc + 5  # number of outputs per anchor, #no 저장, 각 앵커당 출력 수
        self.nl = len(anchors)  # number of detection layers, #nl 저장, 검출 레이어 수
        self.na = len(anchors[0]) // 2  # number of anchors, #na 저장, 앵커 수
        self.grid = [tf.zeros(1)] * self.nl  # init grid, #grid 초기화
        self.anchors = tf.convert_to_tensor(w.anchors.numpy(), dtype=tf.float32), #anchors 저장, w.anchors를 텐서로 변환
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]), [self.nl, 1, -1, 1, 2]) #anchor_grid 저장, anchors와 stride를 이용하여 계산
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)] #m 레이어, TFConv2d 레이어
        self.training = False  # set to False after building model, #training 저장, 모델 빌드 후 False로 설정
        self.imgsz = imgsz #imgsz 저장, 이미지 사이즈
        for i in range(self.nl): #nl만큼 반복
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i] #ny, nx 계산
            self.grid[i] = self._make_grid(nx, ny) #grid[i] 저장, _make_grid(nx, ny) 호출

    def call(self, inputs): #호출, inputs를 받아서
        z = []  # inference output, #z 초기화, 추론 출력
        x = [] #training output, #x 초기화, 트레이닝 출력
        for i in range(self.nl): #nl만큼 반복, i는 0부터 nl-1까지
            x.append(self.m[i](inputs[i])) #x에 m[i]에 inputs[i] 적용한 것 추가
            # x(bs,20,20,255) to x(bs,3,20,20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i] #ny, nx 계산, 이미지 사이즈를 stride로 나눈 것
            x[i] = tf.reshape(x[i], [-1, ny * nx, self.na, self.no]) #x[i]를 reshape, [-1, ny * nx, self.na, self.no]

            if not self.training:  # inference, #training이 False이면
                y = x[i], #y에 x[i] 추가
                grid = tf.transpose(self.grid[i], [0, 2, 1, 3]) - 0.5, #grid 저장, grid[i]를 transpose한 것에서 0.5를 뺌
                anchor_grid = tf.transpose(self.anchor_grid[i], [0, 2, 1, 3]) * 4, #anchor_grid 저장, anchor_grid[i]를 transpose한 것에 4를 곱함
                xy = (tf.sigmoid(y[..., 0:2]) * 2 + grid) * self.stride[i]  # xy, #xy 저장, (y[..., 0:2]에 sigmoid를 취한 것에 2를 곱하고 grid를 더한 것에 stride[i]를 곱함
                wh = tf.sigmoid(y[..., 2:4]) ** 2 * anchor_grid #wh 저장, y[..., 2:4]에 sigmoid를 취한 것에 2를 제곱하고 anchor_grid를 곱함
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32) #xy를 이미지 사이즈로 나눔
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32) #wh를 이미지 사이즈로 나눔
                y = tf.concat([xy, wh, tf.sigmoid(y[..., 4 : 5 + self.nc]), y[..., 5 + self.nc :]], -1) #y 저장, xy, wh, y[..., 4 : 5 + self.nc], y[..., 5 + self.nc :]를 concat한 것
                z.append(tf.reshape(y, [-1, self.na * ny * nx, self.no])) #z에 y를 reshape한 것 추가

        return tf.transpose(x, [0, 2, 1, 3]) if self.training else (tf.concat(z, 1),) #training이 True이면 x를 transpose한 것 반환, 아니면 z를 concat한 것 반환

    @staticmethod #정적 메소드
    def _make_grid(nx=20, ny=20): #_make_grid 메소드, nx=20, ny=20
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny)) #xv, yv 계산
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32) #xv, yv를 stack한 것을 reshape한 것을 cast한 것 반환


class TFSegment(TFDetect): #keras.layers.Layer 상속, Segment 레이어
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None): #초기화, nc, anchors, nm, npr, ch, imgsz, w를 받아서
        super().__init__(nc, anchors, ch, imgsz, w) #상속받은 클래스의 __init__ 메소드 호출, 초기화
        self.nm = nm  # number of masks, #nm 저장, 마스크 수 
        self.npr = npr  # number of protos, #npr 저장, 프로토 수
        self.no = 5 + nc + self.nm  # number of outputs per anchor, #no 저장, 각 앵커당 출력 수
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv, #m 레이어, TFConv2d 레이어
        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos, #proto 레이어, TFProto 레이어
        self.detect = TFDetect.call #detect 레이어, TFDetect.call 메소드 

    def call(self, x): #호출, x를 받아서
        p = self.proto(x[0]) #p 저장, proto에 x[0] 적용
        # p = TFUpsample(None, scale_factor=4, mode='nearest')(self.proto(x[0]))  # (optional) full-size protos
        p = tf.transpose(p, [0, 3, 1, 2])  # from shape(1,160,160,32) to shape(1,32,160,160), #p를 transpose, [0, 3, 1, 2]
        x = self.detect(self, x) #x 저장, detect에 self, x 적용
        return (x, p) if self.training else (x[0], p) #training이 True이면 (x, p) 반환, 아니면 (x[0], p) 반환


class TFProto(keras.layers.Layer): #keras.layers.Layer 상속, Proto 레이어
    def __init__(self, c1, c_=256, c2=32, w=None): #초기화, c1, c_, c2, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1) #cv1 레이어, c1, c_, k=3, w.cv1를 받아서
        self.upsample = TFUpsample(None, scale_factor=2, mode="nearest") #upsample 레이어, TFUpsample 레이어
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2) #cv2 레이어, c_, c_, k=3, w.cv2를 받아서
        self.cv3 = TFConv(c_, c2, w=w.cv3) #cv3 레이어, c_, c2, w.cv3를 받아서

    def call(self, inputs): #호출, inputs를 받아서
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs)))) #cv3에 cv2에 upsample에 cv1에 inputs 적용한 것 반환, 결과 반환


class TFUpsample(keras.layers.Layer):# keras.layers.Layer 상속, Upsample 레이어
    # TF version of torch.nn.Upsample()
    def __init__(self, size, scale_factor, mode, w=None):  # warning: all arguments needed including 'w', #초기화, size, scale_factor, mode, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2" #scale_factor가 2의 배수여야 함
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode) #upsample 레이어, tf.image.resize로 구현
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs): #호출, inputs를 받아서
        return self.upsample(inputs) #upsample에 inputs 적용, 결과 반환


class TFConcat(keras.layers.Layer): #keras.layers.Layer 상속, Concat 레이어
    # TF version of torch.concat()
    def __init__(self, dimension=1, w=None): #초기화, dimension, w를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        assert dimension == 1, "convert only NCHW to NHWC concat" #dimension이 1이어야 함
        self.d = 3 #dimension 저장, 3

    def call(self, inputs): #호출, inputs를 받아서
        return tf.concat(inputs, self.d) #inputs를 concat한 것 반환, 결과 반환


def parse_model(d, ch, model, imgsz):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}") #LOGGER.info 출력
    anchors, nc, gd, gw, ch_mul = ( #anchors, nc, gd, gw, ch_mul 저장
        d["anchors"], #anchors
        d["nc"], #nc
        d["depth_multiple"], #gd
        d["width_multiple"], #gw
        d.get("channel_multiple"), #ch_mul
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors, #na 저장, anchors가 list이면 len(anchors[0])//2, 아니면 anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) , #no 저장, 출력 수 = 앵커 * (클래스 + 5)
    if not ch_mul: #ch_mul이 없으면
        ch_mul = 8 #ch_mul은 8

    layers, save, c2 = [], [], ch[-1]  # layers, save, c2 초기화
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, arguments, #d["backbone"] + d["head"]에서 반복
        m_str = m #m_str 저장, m
        m = eval(m) if isinstance(m, str) else m  # eval strings, #m이 str이면 eval
        for j, a in enumerate(args):#args에서 반복, j, a
            try: # try to evaluate string, #문자열 평가 시도
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings, #a가 str이면 eval
            except NameError:# NameError 발생 시
                pass#pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain, #n 저장, n*gd, 1 중 큰 값, n이 1보다 크면 n, 아니면 1
        if m in [ #m이 [Conv, DWConv, DWConvTranspose2d, Bottleneck, SPP, SPPF, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3x]에 있으면
            nn.Conv2d,
            Conv,
            DWConv,
            DWConvTranspose2d,
            Bottleneck,
            SPP,
            SPPF,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3x,
        ]: #Conv2d, Conv, DWConv, DWConvTranspose2d, Bottleneck, SPP, SPPF, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3x
            c1, c2 = ch[f], args[0] #c1, c2 저장, ch[f], args[0]
            c2 = make_divisible(c2 * gw, ch_mul) if c2 != no else c2 #c2 저장, c2*gw를 ch_mul로 나눈 것, c2가 no와 같지 않으면 c2, 아니면 c2

            args = [c1, c2, *args[1:]] #args 저장, [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3x]: #m이 BottleneckCSP, C3, C3x에 있으면
                args.insert(2, n) #args에 n 추가
                n = 1 #n은 1
        elif m is nn.BatchNorm2d: #m이 nn.BatchNorm2d이면
            args = [ch[f]] #args는 [ch[f]]
        elif m is Concat: #m이 Concat이면
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f) #c2 저장, ch[-1 if x == -1 else x + 1]의 합
        elif m in [Detect, Segment]: #m이 Detect, Segment에 있으면
            args.append([ch[x + 1] for x in f]) #args에 [ch[x + 1] for x in f] 추가
            if isinstance(args[1], int):  # number of anchors, #args[1]이 int이면
                args[1] = [list(range(args[1] * 2))] * len(f) #args[1]은 [list(range(args[1] * 2))] * len(f)
            if m is Segment: #m이 Segment이면
                args[3] = make_divisible(args[3] * gw, ch_mul) #args[3] 저장, args[3]*gw를 ch_mul로 나눈 것
            args.append(imgsz) # args에 imgsz 추가
        else: #그 외
            c2 = ch[f] #c2 저장, ch[f]

        tf_m = eval("TF" + m_str.replace("nn.", "")) #tf_m 저장, "TF" + m_str.replace("nn.", "")로 eval한 것
        m_ = ( #m_
            keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) #Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)])
            if n > 1 #n이 1보다 크면
            else tf_m(*args, w=model.model[i]) #아니면 tf_m(*args, w=model.model[i])
        )  # module

        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args) # pytorch module, #torch_m_ 저장, nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")  # module type, #t 저장, str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in torch_m_.parameters())  # number params, #np 저장, torch_m_의 parameters의 numel의 합
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params, #m_에 i, f, t, np 추가
        LOGGER.info(f"{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}")  # print, #LOGGER.info 출력
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist, #save에 f가 int이면 [f], 아니면 f에서 -1이 아닌 것을 i로 나눈 것 추가
        layers.append(m_)  # append to layers, #layers에 m_ 추가
        ch.append(c2) #ch에 c2 추가
    return keras.Sequential(layers), sorted(save) #Sequential(layers), sorted(save) 반환


class TFModel: #TFModel 클래스
    # TF YOLOv5 model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, model=None, imgsz=(640, 640)):  # model, channels, classes, #초기화, cfg, ch, nc, model, imgsz를 받아서
        super().__init__() #상속받은 클래스의 __init__ 메소드 호출, 초기화
        if isinstance(cfg, dict): #cfg가 dict이면
            self.yaml = cfg  # model dict, #yaml은 cfg
        else:  # is *.yaml, #그 외
            import yaml  # for torch hub, #torch hub를 위해 import

            self.yaml_file = Path(cfg).name #yaml_file 저장, cfg의 파일 이름
            with open(cfg) as f: #cfg를 열어서
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict, #yaml은 cfg를 load한 것

        # Define model
        if nc and nc != self.yaml["nc"]: #nc가 있고, nc가 self.yaml["nc"]와 다르면
            LOGGER.info(f"Overriding {cfg} nc={self.yaml['nc']} with nc={nc}") #LOGGER.info 출력
            self.yaml["nc"] = nc  # override yaml value, #yaml["nc"]는 nc
        self.model, self.savelist = parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz) #model, savelist 저장, parse_model(deepcopy(self.yaml), ch=[ch], model=model, imgsz=imgsz)

    def predict( #predict 메소드
        self, # self
        inputs, # inputs
        tf_nms=False, # tf_nms=False
        agnostic_nms=False, # agnostic_nms=False
        topk_per_class=100, # topk_per_class=100
        topk_all=100, # topk_all=100
        iou_thres=0.45, # iou_thres=0.45
        conf_thres=0.25, # conf_thres=0.25
    ):
        y = []  # outputs, #y 초기화
        x = inputs #x는 inputs
        for m in self.model.layers: #model.layers에서 반복
            if m.f != -1:  # if not from previous layer, #이전 레이어가 아니면
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers, #y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)  # run, #x에 m 적용
            y.append(x if m.i in self.savelist else None)  # save output, #y에 x 추가, m.i가 savelist에 있으면 None 추가

        # Add TensorFlow NMS
        if tf_nms:#tf_nms이면
            boxes = self._xywh2xyxy(x[0][..., :4]) #boxes 저장, _xywh2xyxy(x[0][..., :4])
            probs = x[0][:, :, 4:5] #probs 저장, x[0][:, :, 4:5]
            classes = x[0][:, :, 5:] #classes 저장, x[0][:, :, 5:]
            scores = probs * classes #scores 저장, probs*classes
            if agnostic_nms: #agnostic_nms이면
                nms = AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres) #nms 저장, AgnosticNMS()((boxes, classes, scores), topk_all, iou_thres, conf_thres)
            else: #그 외
                boxes = tf.expand_dims(boxes, 2) #boxes는 boxes의 차원을 2로 확장
                nms = tf.image.combined_non_max_suppression(
                    boxes, scores, topk_per_class, topk_all, iou_thres, conf_thres, clip_boxes=False
                )
            return (nms,)
        return x  # output [1,6300,85] = [xywh, conf, class0, class1, ...]
        # x = x[0]  # [x(1,6300,85), ...] to x(6300,85)
        # xywh = x[..., :4]  # x(6300,4) boxes
        # conf = x[..., 4:5]  # x(6300,1) confidences
        # cls = tf.reshape(tf.cast(tf.argmax(x[..., 5:], axis=1), tf.float32), (-1, 1))  # x(6300,1)  classes
        # return tf.concat([conf, cls, xywh], 1)

    @staticmethod
    def _xywh2xyxy(xywh):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        x, y, w, h = tf.split(xywh, num_or_size_splits=4, axis=-1)
        return tf.concat([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


class AgnosticNMS(keras.layers.Layer):
    # TF Agnostic NMS
    def call(self, input, topk_all, iou_thres, conf_thres):
        # wrap map_fn to avoid TypeSpec related error https://stackoverflow.com/a/65809989/3036450
        return tf.map_fn(
            lambda x: self._nms(x, topk_all, iou_thres, conf_thres),
            input,
            fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.int32),
            name="agnostic_nms",
        )

    @staticmethod
    def _nms(x, topk_all=100, iou_thres=0.45, conf_thres=0.25):  # agnostic NMS
        boxes, classes, scores = x
        class_inds = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
        scores_inp = tf.reduce_max(scores, -1)
        selected_inds = tf.image.non_max_suppression(
            boxes, scores_inp, max_output_size=topk_all, iou_threshold=iou_thres, score_threshold=conf_thres
        )
        selected_boxes = tf.gather(boxes, selected_inds)
        padded_boxes = tf.pad(
            selected_boxes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )
        selected_scores = tf.gather(scores_inp, selected_inds)
        padded_scores = tf.pad(
            selected_scores,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        selected_classes = tf.gather(class_inds, selected_inds)
        padded_classes = tf.pad(
            selected_classes,
            paddings=[[0, topk_all - tf.shape(selected_boxes)[0]]],
            mode="CONSTANT",
            constant_values=-1.0,
        )
        valid_detections = tf.shape(selected_inds)[0]
        return padded_boxes, padded_scores, padded_classes, valid_detections


def activations(act=nn.SiLU):
    # Returns TF activation from input PyTorch activation
    if isinstance(act, nn.LeakyReLU):
        return lambda x: keras.activations.relu(x, alpha=0.1)
    elif isinstance(act, nn.Hardswish):
        return lambda x: x * tf.nn.relu6(x + 3) * 0.166666667
    elif isinstance(act, (nn.SiLU, SiLU)):
        return lambda x: keras.activations.swish(x)
    else:
        raise Exception(f"no matching TensorFlow activation found for PyTorch activation {act}")


def representative_dataset_gen(dataset, ncalib=100):
    # Representative dataset generator for use with converter.representative_dataset, returns a generator of np arrays
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break


def run(
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # inference size h,w
    batch_size=1,  # batch size
    dynamic=False,  # dynamic batch size
):
    # PyTorch model
    im = torch.zeros((batch_size, 3, *imgsz))  # BCHW image
    model = attempt_load(weights, device=torch.device("cpu"), inplace=True, fuse=False)
    _ = model(im)  # inference
    model.info()

    # TensorFlow model
    im = tf.zeros((batch_size, *imgsz, 3))  # BHWC image
    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    _ = tf_model.predict(im)  # inference

    # Keras model
    im = keras.Input(shape=(*imgsz, 3), batch_size=None if dynamic else batch_size)
    keras_model = keras.Model(inputs=im, outputs=tf_model.predict(im))
    keras_model.summary()

    LOGGER.info("PyTorch, TensorFlow and Keras models successfully verified.\nUse export.py for TF model export.")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch size")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
