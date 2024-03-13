import tensorflow as tf
print(tf.__version__) #1.14.0
print(tf.executing_eagerly()) #False, 즉 즉시 실행 모드가 아니다. 텐서1의 그래프형태의 구성없이 자연스러운 파이썬 문법으로 실행된다.

tf.compat.v1.disable_eager_execution() #즉시 실행모드를 비활성화
tf.compat.v1.enable_eager_execution() # 즉시 실행모드로 전환한다.

print(tf.executing_eagerly()) #True, 즉 즉시 실행 모드이다. 텐서2의 즉시 실행모드이다.

hello = tf.constant("Hello, TensorFlow")

sess = tf.compat.v1.Session()
print(sess.run(hello)) #b'Hello, TensorFlow'

# 가상환경  즉시실행모드        사용가능
# 1.14.0    ensable             에러
# 1.14.0    disable (default)   가능
# 2.9.0     enable (default)    에러    
# 2.9.0    disable              가능 