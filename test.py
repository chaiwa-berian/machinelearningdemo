import tensorflow as tf

#hello = tf.constant('Hello, Tensorflow')
#sess = tf.Session()
#print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly
print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))
#we can combine Tensor nodes with operations
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))
#A graph can be parameterized to accept external inputs, known as placeholders
#A placeholder is a promise to provide a value later
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b #+provides a shortcut for tf.add(a,b)
#We can evaluate this graph with multiple inputs by using
#the feed_dict parameter to specify Tensors that provide
#concrete values to these placeholders:
print(sess.run(adder_node,{a:3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b:[2,4]}))
#We can add another operation to the computational graph
add_and_triple = adder_node*3.
print(sess.run(add_and_triple, {a:3, b:4.5}))
#To make the model trainable, we need to be able to modify
#the graph to get new outputs with the same input
#Variables allow us to add trainable parameters to a graph
#They are constructed with a type and initial Value
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b
#To initialize all variables in a TensorFlow program,
#you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)
#Since x is a placeholder, we can evaluate linear_model for
#several values of x simultaneously as follows:
print(sess.run(linear_model, {x:[1,2,3,4]}))
#To evaluate the model on training data, we need a y placeholder
#to provide the desired values, and we need to write a 
#loss function
#Loss Function measures how far apart the current model is from 
#from the provided data
#linear_model-y creates a vector where each element is the
#corresponding example's error delta
#We call tf.square to square that error
#Then we sum all the squared errors to create a single scalar
#that abstracts the erros of all examples using tf.reduce_sum:

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model-y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))
#A variable is initialized to the value provided to tf.Variable
#but can be changed using operations like tf.assign
#E.g W=-1 and b=1 are the optimal parameters for our model
#We can change W and b accordingly
fixW = tf.assign(W,[-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss,{x: [1,2,3,4], y:[0,-1,-2,-3]}))
#We guessed the "perfect" values of W and b, but the whole point of 
#machine learning is to find the correct model parameters
#automatically
#TensorFlow provides optimizers that slowly change each variable
#in order to minimize the loss function
#The simplest optimizer is gradient descent
#It modifies each variable according to the magnitude of the
#derivative of loss with respect to that variable
#Example:
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) #reset values to incorrect defaults
for i in range(1000):
    sess.run(train,{x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b]))




















