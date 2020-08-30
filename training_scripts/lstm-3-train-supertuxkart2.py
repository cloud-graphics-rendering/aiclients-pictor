import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.contrib import rnn

#frequently modified parameters
training_file_dir = '../training_data/clean-data/'
n_input = 1 		# sequential input vector numbers for lstm
tensor_size = 2         # length of each input vector
n_classes = 3           # number of classes for the output
input_index = 0
output_index = 1
# Target log path
logs_path = '../training_data/logs/'
writer = tf.summary.FileWriter(logs_path)

# number of units in RNN cell
n_hidden = 512

#parameters for training
learning_rate = 0.000001
training_iters = 5000000
display_step = 1000
checkpoint_step = 50000

# tf Graph input
x = tf.placeholder("float", [None, n_input, tensor_size])
y = tf.placeholder("float", [None, n_classes])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden,forget_bias=1.0),rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)])
    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    #rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

training_all = np.zeros(shape=(0, output_index + 1))
for filename in os.listdir(training_file_dir):
    filename = training_file_dir + filename
    d = np.load(filename)
    training_all = np.concatenate((training_all, d))

data = list(training_all)
print("Loaded training data...")
print(len(data))

#generate saver
saver = tf.train.Saver() #generate saver

# Launch the graph
with tf.Session() as session:
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    session.run(init)

    writer.add_graph(session.graph)
    if os.path.isfile(logs_path+"checkpoint"):
        saver.restore(session,logs_path+"lstm-model")
   
    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(data)-end_offset):
            offset = random.randint(0, n_input+1)

        symbols_in_keys = [data[i][input_index] for i in range(offset, offset+n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, tensor_size])
        leftright_out_onehot = np.reshape(data[offset+n_input-1][output_index],[1,-1])
        #print(leftright_out_onehot)

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: leftright_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", AvgLoss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", AvgAcc= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [data[i][input_index] for i in range(offset, offset + n_input)]
            symbols_out = data[offset + n_input -1][output_index]
            print(onehot_pred)
            symbols_out_pred = int(tf.argmax(onehot_pred, 1).eval())
            print(symbols_in,symbols_out,symbols_out_pred)
            if acc_total > 0.98:
                break
        if(step+1) % checkpoint_step == 0:
            saver.save(session, logs_path+"lstm-model")

        step += 1
        offset += (n_input+1)
    saver.save(session, logs_path+"lstm-model")
    print("Optimization Finished!")
    #print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
