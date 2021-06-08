
import tensorflow as tf
from tensorflow_addons.text import crf_log_likelihood, crf_decode



class CrfLayer(tf.keras.layers.Layer):

    def __init__(self, num_tags, max_seq_length, initializer, name = "crf", **kargs) -> None:
        super().__init__(name=name, **kargs)
        self.num_tags = num_tags
        self.max_seq_length = max_seq_length
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

