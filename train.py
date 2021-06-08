
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

train_batches, num_train_batches, num_sample = get_batch(hp.train_path, hp.batch_size,
                                                hp.max_seq_length, hp.vocab_file,
                                                hp.tag_mapping_file, do_lower_case = False)
eval_batches,  num_eval_batches,  num_sample = get_batch(hp.test_path, hp.batch_size,
                                                hp.max_seq_length, hp.vocab_file,
                                                hp.tag_mapping_file, do_lower_case = False)

iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, ori_chars, ori_tags = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op  = iter.make_initializer(eval_batches)

bert_config = utils.load_json(hp.bert_config)


all_steps = train_batches * hp.num_epochs
num_warmup_steps = int(all_steps * hp.warmup_prop)

training_mode = tf.placeholder_with_default(True, shape=())
bert_crf_fn = BertCrf(bert_config, hp.layer_type, num_tags)
loss, train_op, out_tags, transition_weight = bert_crf_fn.create_model(xs, training_mode, hp.dropout_rate, hp.lr, all_steps, num_warmup_steps)

global_step = tf.train.get_or_create_global_step()
train_summaries = None

logging.info("# Session")
saver = tf.train.Saver(max_to_keep = hp.num_epochs)
sess.run(train_init_op)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("initial from scrtch")
        sess.run(tf.global_variables_initializer())
        utils.save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

        
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, all_steps + 1)):

        _, _gs, _summary = sess.run([train_op, global_step, train_summaries], feed_dict={training_mode: True})

        if _gs and _gs % num_train_batches == 0:
            epochs = _gs / num_train_batches
            logging.info("Epoch %d is done"%epochs)

            _loss = sess.run(loss, feed_dict={training_mode: False})  # train loss
            sess.run(eval_init_op)
            sess.run([out_tags, transition_weight], feed_dict={training_mode: False})


            sess.run(train_init_op)


        sess.run(loss)
