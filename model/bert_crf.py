
import tensorflow as tf
from bert import modeling
from layers import CrfLayer
from bert import optimization

class BertCrf:

    def __init__(self, bert_config, layer_type, number_tags):
        self.bert_layer_type = layer_type.lower()
        self.bert_config = bert_config
        self.use_one_hot_embeddings = False
        self.num_tags = number_tags

    def create_model(self, inputs, training_mode, dropout_rate,
                        learning_rate, num_train_steps, num_warmup_steps):
        input_ids, tag_ids = inputs
        input_mask = tf.math.equal(input_ids, 0)
        model = modeling.BertModel(
            config = self.bert_config,
            is_training = training_mode,
            input_ids =  input_ids,
            input_mask = input_mask,
            token_type_ids = None,
            use_one_hot_embeddings = self.use_one_hot_embeddings
        )

        if self.bert_layer_type   == "pooled":
            bert_output_layer = model.get_pooled_output()
        elif self.bert_layer_type == "last2":
            bert_output_layer = model.get_all_encoder_layers()[-2]
        else:
            bert_output_layer = model.get_sequence_output()

        initializer = tf.contrib.layers.xavier_initializer()
        crf_layer = CrfLayer(self.num_tags, hp.max_seq_length, initializer)

        bert_output_layer = tf.layers.dropout(bert_output_layer, rate=dropout_rate, training=training_mode)
        linear = tf.keras.layers.Dense(self.num_tags, activation = None)
        emission = linear(bert_output_layer)

        log_likehood, trans, out_tags = crf_layer.call((emission, tag_ids), input_mask)
        loss = tf.math.negative(tf.math.reduce_mean(log_likehood))
        # optimizer = tf.train.AdamOptimizer()
        # train_op = optimizer.minimize(loss, global_step = global_step)
        train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, False)

        return loss, train_op, out_tags, trans
