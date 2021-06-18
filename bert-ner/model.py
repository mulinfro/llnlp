
import tensorflow as tf

class BertCrf(tf.keras.Model):

    def __init__(self, *args, **kwargs):
	super().__init__(*args, **kwargs)


    def call(self, input_ids, input_masks = None, input_type_ids = None):