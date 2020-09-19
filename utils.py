#Define swish here
import tensorflow as tf
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'
        
def swish_act(x, beta=1):
    return (x * sigmoid(beta * x))


def limit_gpu_memory(gpu_memory):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)]
            )
        except RuntimeError as e:
            print(e)
            
label_map = {
    'buildings': 0,
    'forest': 1,
    'glacier': 2,
    'mountain': 3,
    'sea': 4,
    'street': 5
}
labels = list(label_map.keys())

get_custom_objects().update({'swish_act': SwishActivation(swish_act)}) 
