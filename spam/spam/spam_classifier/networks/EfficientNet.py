from keras.layers import Input, Flatten, Dense
from keras import Model
from efficientnet.keras import EfficientNetB2
from os import path


def frozen_efficientnet(input_size, n_classes):
    model_ = EfficientNetB2(
            include_top=False,
            input_tensor=Input(shape=input_size))
    for layer in model_.layers:
        layer.trainable = False
    x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)
    frozen_model = Model(model_.input, x)

    return frozen_model
