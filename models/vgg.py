from tensorflow import keras as K


def vgg16_layers(layer_names):
    """ Creates a vgg16 model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = K.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = K.Model(inputs=vgg.input, outputs=outputs)
    return model


layers = ['block2_pool', 'block5_pool']
# layers = ['block2_conv2', 'block5_conv3']
vgg16 = vgg16_layers(layers)
