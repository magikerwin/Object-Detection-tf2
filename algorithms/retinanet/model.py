from algorithms.common.networks import *

def getRetinaNetModel(input_shape, num_anchors, num_classes, weights_decay, training=False, iou_th=.4, score_th=.02, name='FCOSModel'):
    """FCOS detector"""
    
    extractor = getFeatureExtractor(input_shape) # MobileNetV2
    
    # define model
    outputs = inputs = tf.keras.Input(input_shape, name='input_tensor')
    outputs = extractor(outputs)
    outputs = FPN(inputs=outputs, wd=weights_decay)
    outputs = [OutputHead(out, num_anchors, num_classes, 0.01) for idx, out in enumerate(outputs)]
    return tf.keras.Model(inputs, outputs=outputs, name=name)