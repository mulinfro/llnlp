
from tensorflow import keras

class TextCnn:

    def __init__(self, input_length, embedding_size, vocab_size,
                    filters, num_channels, num_classes, dropout_rate, regularizers_lambda):
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.filters = filters
        sefl.num_channels = num_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.regularizers_lambda = regularizers_lambda


    pool_outputs = []
    def create_model(self):
        # input embedding
        inputs = keras.Input(shape=(self.input_length,), name = "input_data")
        embedding = keras.layers.embedding(shape=(self.vocab_size, self.embedding_size),
                                inititalizer = keras.layer.Uniform(-1, 1))

        input_embed = embedding(inputs)
        input_embed = keras.reshape((self.vocab_size, self.embedding_size, 1), name = "add_channel")(input_embed)


        # apply filter && pool
        for filter_size in filters:
            filter_shape = (filter_size, self.embedding_size)
            conv = keras.layers.Conv2D(filter_size, self.num_channels,
                                    strides=(1,1), padding = "valid",
                                    kernel_initializer = "glorot_normal",
                                    data_format = "channels_last",
                                    bias_initializer = keras.initializers.constant(0.1),
                                    name = "conv_{:d}".format(filter_size)
                                    )(input_embed)

            max_pool_shape = (self.input_length - filter_size + 1, 1)
            pool = keras.layers.Maxpool2D(pool_size=max_pool_shape, 
                                           strides=(1,1), padding = "valid",
                                           data_format = "channels_last"
                                           name = "max_pooling_{:d}".format(filter_size)
                                )(conv)

            pool_outputs.append(pool)
    
        # concate && flat
        outputs = keras.layers.concatenate(pool_outputs, axis = -1, name = "concatenate")
        outputs = keras.layers.Flatten(data_format="channels_last", name = "flatten")(outputs)

        # dropout
        outputs = keras.layers.Dropout(rate=self.dropout_rate)(outputs)

        outputs = keras.layers.Dense(self.num_classes,
                                    kernel_initializer = "glorot_normal",
                                    bias_initializer = keras.initializers.constant(0.1),
                                    kernel_regularizer=keras.regularizers.l2(self.regularizers_lambda),
                                    bias_regularizer=keras.regularizers.l2(self.regularizers_lambda),
                                    name = "full connect"
                                )(outputs)

        model = keras.Model(input = inputs, output = outputs)
        return model
