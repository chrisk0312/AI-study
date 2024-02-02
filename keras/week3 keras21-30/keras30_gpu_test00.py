import tensorflow as tf

print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
if gpus:
    print("GPU RUNNGING...\nGPU: ",gpus)
else:
    print("GPU not exist")