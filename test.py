import tensorflow as tf
print("TensorFlow-DirectML version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))