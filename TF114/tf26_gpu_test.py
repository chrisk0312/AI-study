import tensorflow as tf

# tf.compat.v1.enable_eager_execution()
tf.compat.v1.disable_eager_execution()

print("tensorflow version: ",tf.__version__)
print("즉시실행모드 ",tf.executing_eagerly())

gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    try:
        tf.config.experimental.set_visible_devices(gpu[0],'GPU')
        print(gpu[0])
    except RuntimeError as e:
        print('error', e)
else:
    print('gpu is none')