# YOLOv8_Openvino
## 0 内容
具体包括：基于CPU的Openvino和ONNXRuntime下的YOLOv5/v6/v7/v8/v9的检测、跟踪、分割、关键点检测。

## 1 环境：
CPU：i5-12500

Python：3.8.18

VS2019

注：Bytetrack中的lap和cython_bbox库需要编译安装，直接安装报错，故下载VS2019。
## 2 安装Openvino和ONNXRuntime
### 2.1 Openvino简介
Openvino是由Intel开发的专门用于优化和部署人工智能推理的半开源的工具包，主要用于对深度推理做优化。

Openvino内部集成了Opencv、TensorFlow模块，除此之外它还具有强大的Plugin开发框架，允许开发者在Openvino之上对推理过程做优化。

Openvino整体框架为：Openvino前端→ Plugin中间层→ Backend后端
Openvino的优点在于它屏蔽了后端接口，提供了统一操作的前端API，开发者可以无需关心后端的实现，例如后端可以是TensorFlow、Keras、ARM-NN，通过Plugin提供给前端接口调用，也就意味着一套代码在Openvino之上可以运行在多个推理引擎之上，Openvino像是类似聚合一样的开发包。

### 2.2 ONNXRuntime简介
ONNXRuntime是微软推出的一款推理框架，用户可以非常便利的用其运行一个onnx模型。ONNXRuntime支持多种运行后端包括CPU，GPU，TensorRT，DML等。可以说ONNXRuntime是对ONNX模型最原生的支持。

虽然大家用ONNX时更多的是作为一个中间表示，从pytorch转到onnx后直接喂到TensorRT或MNN等各种后端框架，但这并不能否认ONNXRuntime是一款非常优秀的推理框架。而且由于其自身只包含推理功能（最新的ONNXRuntime甚至已经可以训练），通过阅读其源码可以解深度学习框架的一些核心功能原理（op注册，内存管理，运行逻辑等）
总体来看，整个ONNXRuntime的运行可以分为三个阶段，Session构造，模型加载与初始化和运行。和其他所有主流框架相同，ONNXRuntime最常用的语言是python，而实际负责执行框架运行的则是C++。

### 2.3 安装
pip install openvino -i  https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxruntime -i  https://pypi.tuna.tsinghua.edu.cn/simple



