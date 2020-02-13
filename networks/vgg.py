# -*- coding: utf-8 -*-
"""
This module's purpose is to give access to the VGG* networks trained on ImageNet
and distributed with Keras (as 'application').
"""
from keras.applications import vgg16, vgg19

from .base import KerasNetwork


class VGG16(KerasNetwork):
    def __init__(self):
        super().__init__(vgg16.VGG16())

    @staticmethod
    def bounds():
        return super().bounds()

    def name():
        return 'VGG16'

class VGG19(KerasNetwork):
    def __init__(self):
        super().__init__(vgg19.VGG19())

    @staticmethod
    def bounds():
        return super().bounds()

    @staticmethod
    def name():
        return 'VGG19'
