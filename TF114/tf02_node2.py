import tensorflow as tf
node1=tf.constant(2.0)
node2=tf.constant(3.0)

# add :node3
# deduct :node4
# multiply :node5
# divide :node6

node3=tf.add(node1,node2)
node4=tf.subtract(node1,node2)
node5=tf.multiply(node1,node2)
node6=tf.divide(node1,node2)

sess = tf.Session()
print(sess.run([node3,node4,node5,node6]))
# [5.0, -1.0, 6.0, 0.6666666666666666]