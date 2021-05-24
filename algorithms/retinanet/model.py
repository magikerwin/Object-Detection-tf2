from algorithms.common.networks import *

def getRetinaNetModel(input_shape, training=False, iou_th=.4, score_th=.02, name='FCOSModel'):
    """FCOS detector"""
    
    extractor = getFeatureExtractor(input_shape) # MobileNetV2
    
    # define model
    outputs = inputs = tf.keras.Input(input_shape, name='input_tensor')
    outputs = extractor(outputs)
    outputs = FPN(inputs=outputs, wd=5e-4)
    outputs = [OutputHead(out, 0.01) for idx, out in enumerate(outputs)]
    return tf.keras.Model(inputs, outputs=outputs, name=name)