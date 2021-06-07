
import bert
import tensorflow as tf

from hparams import Hparams
from model.layers import CrfLayer
import utils
import model.bert_processer as bproc
from data_load import get_batch

import logging
logging.basicConfig(level=logging.INFO)

hparams = Hparams()
hp = hparams.parse.parse_args()

tag2idx, idx2tag, num_tags = utils.load_crf_tags()

train_batches, num_train_batches, num_sample = get_batch(hp.train_path, hp.batch_size, hp.max_seq_length, hp.vocab_file, hp.tag_mapping_file, do_lower_case = False)
eval_batches,  num_eval_batches,  num_sample = get_batch(hp.test_path, hp.batch_size, hp.max_seq_length, hp.vocab_file, hp.tag_mapping_file, do_lower_case = False)

iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, ori_chars, ori_tags = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op  = iter.make_initializer(eval_batches)

initializer = tf.contrib.layers.xavier_initializer()
max_seq_length = hp.max_seq_length


train_seq = (hp.train_path)

bproc.create_bert_model()

crf_layer = CrfLayer(num_tags, max_seq_length, initializer)

with tf.Session() as sess:
