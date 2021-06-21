
import tensorflow as tf
from bert_ner.bert_modeling import BertModel, BertConfig


from tensorflow_addons.text import crf_log_likelihood, crf_decode

class CrfLayer(tf.keras.layers.Layer):

    def __init__(self, num_tags, initializer, name = "crf", **kargs) -> None:
        super().__init__(name=name, **kargs)
        self.num_tags = num_tags
        # self.max_seq_length = max_seq_length
        self.initializer = initializer
        self.transition_weight = self.add_weight(
            shape=(self.num_tags, self.num_tags),
            dtype = tf.float32,
            initializer = self.initializer,
            name = "transition_weight"
        )

    def call(self, inputs, mask):
        emissions, tag_ids = inputs 
        mask = tf.cast(mask, tf.int32)
        seq_lengths = tf.math.reduce_sum(mask, axis=1)
        log_likehood, trans = crf_log_likelihood(
            inputs = emissions,
            tag_indices = tag_ids,
            sequence_lengths = seq_lengths,
            transition_params = self.transition_weight
        )
        predict, viterbi_score = tf.contrib.crf.crf_decode(emissions, self.transition_weight , seq_lengths)
        # self.add_loss(tf.math.negative(tf.math.reduce_mean()))

        return log_likehood, trans, predict


class BertCrf(tf.keras.Model):

    def __init__(self, bert_config_file, num_tags, max_seq_length, **kwargs):
        super().__init__(**kwargs)
        bert_conf = BertConfig.from_json_file(bert_config_file)

        input_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
        input_type_ids = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_type_ids")

        self.bert_layer = BertModel(bert_conf)
        pooled, sequence_output = self.bert_layer(input_ids, input_mask, input_type_ids)
        # 直接project
        self.linear = tf.keras.layers.Dense(num_tags, initializer = tf.keras.initializer.xavier,
                                       activation=None, name="linear_project")
        sequence_output()

        initializer = tf.keras.initializers.glorot_uniform
        self.crf = CrfLayer(self.num_tags, initializer)



    def call(self, input_ids, input_masks = None, input_type_ids = None):

        self.bert_model.call(input_ids, input_masks, input_type_ids)
        