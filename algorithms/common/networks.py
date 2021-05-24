import tensorflow as tf

def _regularizer(weights_decay):
    """l2 regularizer"""
    return tf.keras.regularizers.l2(weights_decay)

def _kernel_init(scale=1.0, seed=None):
    """He normal initializer"""
    return tf.keras.initializers.he_normal()

def ConvBlock(x, f, k, s, wd):
    conv = tf.keras.layers.Conv2D(filters=f, kernel_size=k, strides=s,
                                  use_bias=False, padding='same',
                                  kernel_initializer=_kernel_init(),
                                  kernel_regularizer=_regularizer(wd))
    bn = tf.keras.layers.BatchNormalization()
    act = tf.keras.layers.ReLU()
    return act(bn(conv(x))) 

def getFeatureExtractor(input_shape, name='extractor'):
    extractor = tf.keras.applications.MobileNetV2(input_shape=input_shape, alpha=0.35, include_top=False, weights='imagenet')
    inputs = [extractor.input]
    outputs = [extractor.layers[54].output,
               extractor.layers[116].output,
               extractor.layers[143].output,]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def FPN(inputs, wd):
    output1 = ConvBlock(x=inputs[0], f=64, k=1, s=1, wd=wd)
    output2 = ConvBlock(x=inputs[1], f=64, k=1, s=1, wd=wd)
    output3 = ConvBlock(x=inputs[2], f=64, k=1, s=1, wd=wd)
    
    up3 = tf.image.resize(output3, tf.shape(output2)[1:3], method='nearest')
    output2 = ConvBlock(x=output2+up3, f=64, k=3, s=1, wd=wd)
    
    up2 = tf.image.resize(output2, tf.shape(output1)[1:3], method='nearest')
    output1 = ConvBlock(x=output1+up2, f=64, k=3, s=1, wd=wd)
    
    return output1, output2, output3

def BBoxHead(x, wd):
    out = ConvBlock(x=x, f=32, k=3, s=1, wd=wd)
    out = ConvBlock(x=out, f=32, k=3, s=1, wd=wd)
    out = tf.keras.layers.Conv2D(filters=4, kernel_size=1, strides=1) (out)
    return out

def ClassHead(x, wd):
    out = ConvBlock(x=x, f=32, k=3, s=1, wd=wd)
    out = ConvBlock(x=out, f=32, k=3, s=1, wd=wd)
    out = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1) (out)
    out = tf.keras.layers.Softmax(axis=-1)(out)
    return out

def OutputHead(x, wd):
    bbox_pred = BBoxHead(x, wd)
    class_pred = ClassHead(x, wd)
    return tf.keras.layers.concatenate([bbox_pred, class_pred], axis=-1)