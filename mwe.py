# -*- coding: utf-8 -*-
"""
This module consists of minimal working examples (MWEs) presenting how to use
the network interface and the implemented defense mechanisms.
"""
import defense_mechanisms as dms
import networks

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Let this path point to an example image.
SAMPLE_IMAGE_PATH = "dog.jpg"

def sample_image(path):
    img = image.load_img(path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    return img

def vgg19_l1():
    """
    Setup the VGG19 neural network, protect it using Gu and Rigazio's L1 defense
    mechanism and perform an inference on an example input.
    """
    model = networks.VGG19()
    dm = dms.GuRigazio(
        keras_model = model.wrapped_model(),
        noise_stddev = 2.46,
        how = 'L1'
    )
    sample_img = sample_image(SAMPLE_IMAGE_PATH)
    votes = dm.predict_n(sample_img)
    prediction = dms.aggregate_predict_n_by(['count', 'mean'], votes)
    return prediction

def vgg19_lstar():
    """
    Setup the VGG19 neural network, protect it using Gu and Rigazio's L* defense
    mechanism and perform an inference on an example input.
    """
    model = networks.VGG19()
    dm = dms.GuRigazio(
        keras_model = model.wrapped_model(),
        noise_stddev = 4.71e-4,
        how = 'L*',
        interpretation = 'weights'
    )
    sample_img = sample_image(SAMPLE_IMAGE_PATH)
    votes = dm.predict_n(sample_img)
    prediction = dms.aggregate_predict_n_by(['count', 'mean'], votes)
    return prediction

def vgg19_lplus():
    """
    Setup the VGG19 neural network, protect it using our L+ adaptation of Gu and
    Rigazio's defense mechanisms and perform an inference on an example input.
    """
    model = networks.VGG19()
    dm = dms.GuRigazio(
        keras_model = model.wrapped_model(),
        noise_stddev = 4.71e-4,
        how = 'L+',
        interpretation = 'weights'
    )
    sample_img = sample_image(SAMPLE_IMAGE_PATH)
    votes = dm.predict_n(sample_img)
    prediction = dms.aggregate_predict_n_by(['count', 'mean'], votes)
    return prediction

def vgg19_rpenn():
    """
    Setup the VGG19 neural network, protect it using our RPENN defense mechanism
    and perform an inference on an example input.
    """
    model = networks.VGG19()
    dm = dms.RPENN(
        keras_model = model.wrapped_model(),
    )
    sample_img = sample_image(SAMPLE_IMAGE_PATH)
    votes = dm.predict_n(sample_img)
    prediction = dms.aggregate_predict_n_by(['count', 'mean'], votes)
    return prediction

def main():
    for prediction in [vgg19_l1(), vgg19_lstar(), vgg19_lplus(), vgg19_rpenn()]:
        print(prediction)

if __name__ == '__main__':
    main()
