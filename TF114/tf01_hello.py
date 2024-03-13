import tensorflow as tf
print(tf.__version__) #1.14.0

print("Hello, TensorFlow!")

hello =tf.constant("Hello, TensorFlow!!!")
print(hello)

sess = tf.Session()
print(sess.run(hello))



