import tensorflow as tf
print(tf.__version__) #1.14.0
print(tf.executing_eagerly()) #False, 즉 즉시 실행 모드가 아니다. 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행된다.

tf.compat.v1.disable_eager_execution() #즉시 실행모드를 비활성화

node1 = tf.constant(30.0, tf.float32)
node2 = tf.constant(4.0) 
node3 = tf.add(node1, node2)

sess = tf.compat.v1.Session()

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add_node = a + b

print(sess.run(add_node, feed_dict={a:3, b:4}))
print(sess.run(add_node, feed_dict={a:30, b:4.5}))

