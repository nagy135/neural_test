import tensorflow as tf
from data_creator import create_data

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_bits = 10

n_classes = 10
batch_size = 100

place_x = tf.placeholder('float64', [None, n_bits])
place_y = tf.placeholder('float64', [None, n_bits])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_bits, n_nodes_hl1], dtype='float64')),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1], dtype='float64'))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2], dtype='float64')),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2], dtype='float64'))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3], dtype='float64')),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3], dtype='float64'))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_bits], dtype='float64')),
                    'biases':tf.Variable(tf.random_normal([n_bits], dtype='float64')),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(place_x):
    prediction = neural_network_model(place_x)
    print('neural network created')
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=place_y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            feed_dict_temp = dict()
            generated_data = create_data(70000, n_bits)
            for i, item in enumerate(generated_data['X']):
                print(item)
                print(generated_data['y'][i])
                print()
            _, c = sess.run([optimizer, cost], feed_dict={place_x: generated_data['X'], place_y: generated_data['y']})
            epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(place_y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        generated_data_test = create_data(3000, n_bits)
        print('Accuracy:',accuracy.eval({place_x:generated_data_test['X'], place_y:generated_data_test['y']}))

train_neural_network(place_x)
