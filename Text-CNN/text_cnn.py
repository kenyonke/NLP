import tensorflow as tf

class textCNN(object):
    
    def __init__(self, seq_length, classes_number, word_number, embedding_size, kernel_number, kernel_size, dropout_prob):
        self.embedding_size = embedding_size
        self.kernel_number = kernel_number
        self.kernel_size = kernel_size
        self.seq_length = seq_length
        self.dropout_prob = dropout_prob
        self.classes_number = classes_number
        
        self.x = tf.placeholder(tf.int32,shape=[None,seq_length])
        self.y = tf.placeholder(tf.float32,shape=[None,classes_number])
        self.embedding = tf.Variable(tf.random_uniform([word_number, embedding_size], -1.0, 1.0))
        
        self.prediction = None
        self.loss = None
        
        # Create Model
        # ------------------
        # Embedding
        seq_vectors = tf.nn.embedding_lookup(self.embedding,self.x)
            #reshape input to (batch_size * seq_length * embedimg_size * channel)
        input_vector = tf.reshape(seq_vectors,[-1, self.seq_length, self.embedding_size, 1]) 
        
        pooled_output = []
        # Convolution layer
        for i,size in enumerate(self.kernel_size):
            
            W = tf.Variable(tf.random_uniform([size, self.embedding_size, 1, self.kernel_number], -1.0, 1.0), name='w'+str(i))
            
            # Convolution
            conv = tf.nn.conv2d(
                    input_vector,
                    W,
                    strides = [1,1,1,1],
                    padding='VALID'
                    )
            
            # Activation function
            act_conv = tf.nn.relu(conv) 
            
            # Max Pooling
            pool_output = tf.nn.max_pool(
                    act_conv,
                    ksize = [1, self.seq_length - size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',                    
                    )
            
            # Collect all pooled output
            pooled_output.append(pool_output)
        
        # Convert list into tensor
        pooled_outputs = tf.concat(pooled_output, 3)  # the shape of CNN output with max_pooing is[1,1,1,1]
        reshped_pooled_outputs = tf.reshape(pooled_outputs,shape=[-1, self.kernel_number*len(self.kernel_size)])
        
        # Dropout to avoid overfitting
        outputs = tf.nn.dropout(reshped_pooled_outputs, self.dropout_prob)
        
        # Softmax for output
        softmax_w = tf.Variable(tf.zeros([self.kernel_number*len(self.kernel_size), self.classes_number]))
        softmax_b = tf.Variable(tf.zeros([1]))
        
        # Model final prediction
        self.prediction = tf.nn.softmax(tf.add(tf.matmul(outputs,softmax_w), softmax_b))
        
        # Cross-entropy Losses
        self.losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, 
                                    labels=self.y)) + 0.01 * tf.nn.l2_loss(softmax_w) + 0.01 * tf.nn.l2_loss(softmax_b)
        
        # ACC caculation
        self.accuracy = tf.metrics.accuracy(labels = tf.argmax(self.y, axis=1),
                          predictions = tf.argmax(self.prediction, axis=1),)[1]
        