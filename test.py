
import bert
import tensorflow as tf

from hparams import Hparams
from model.layers import CrfLayer
import utils
import os
from model.bert_crf import BertCrf
from data_load import get_batch, idx2token
from vocab_manager import Vocab

import logging
logging.basicConfig(level=logging.INFO)

hparams = Hparams()
hp = hparams.parse.parse_args()

tag2idx, idx2tag, num_tags = utils.load_crf_tags()

test_batches,  num_test_batches,  num_sample = get_batch(hp.test_path, hp.batch_size,
                                                hp.max_seq_length, hp.vocab_file,
                                                hp.tag_mapping_file, do_lower_case = False)

iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys, ori_chars, ori_tags = iter.get_next()

test_init_op  = iter.make_initializer(test_batches)

bert_config = utils.load_json(hp.bert_config)

all_steps = test_batches

training_mode = tf.placeholder_with_default(True, shape=())
bert_crf_fn = BertCrf(bert_config, hp.layer_type, num_tags)
loss, train_op, out_tags, transition_weight = bert_crf_fn.create_model(xs, False, hp.dropout_rate, hp.lr, all_steps, 100)

global_step = tf.train.get_or_create_global_step()
train_summaries = None

logging.info("# Session")
saver = tf.train.Saver(max_to_keep = hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("initial from scrtch")
        sess.run(tf.global_variables_initializer())
        utils.save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    sess.run(test_init_op)
        
    logging.info("# write result")
    import time
    for i in range(num_test_batches):
        [_out_tag, _ori_chrs, _ori_tags] = sess.run([out_tags, _ori_chrs, _ori_tags], feed_dict={training_mode: False})
        eval_tags = idx2token(_out_tag)

        model_name = "bert_crf_test_" + time.strftime("%y-%m-%D", time.localtime())
        evaled_tags_file = os.path.join(hp.testdir, model_name)
        with open(evaled_tags_file, "w") as f:
            for tags in eval_tags:
                f.write(" ".join(tags) + "\n")