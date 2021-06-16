
import bert
import tensorflow as tf

from hparams import Hparams
from model.layers import CrfLayer
import utils
import tqdm, os
from model.bert_crf import BertCrf
from data_load import get_batch, idx2token
from vocab_manager import Vocab

import logging
logging.basicConfig(level=logging.INFO)

hparams = Hparams()
hp = hparams.parser.parse_args()


tag_vocab = Vocab(hp.tag_mapping_file)
num_tags = len(tag_vocab.vocab_dict)

print("####### get batch")
train_batches, num_train_batches, num_sample = get_batch(hp.train_path, hp.batch_size,
                                                hp.max_seq_length, hp.vocab_file,
                                                hp.tag_mapping_file, do_lower_case = False)
eval_batches,  num_eval_batches,  num_sample = get_batch(hp.test_path, hp.batch_size,
                                                hp.max_seq_length, hp.vocab_file,
                                                hp.tag_mapping_file, do_lower_case = False)


print("####### init0")
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys, ori_chars, ori_tags = iter.get_next()

print("####### init1")

train_init_op = iter.make_initializer(train_batches)
eval_init_op  = iter.make_initializer(eval_batches)

print("####### init2")
bert_config = utils.load_json(hp.bert_config)

print("####### bert config")
all_steps = train_batches * hp.num_epochs
num_warmup_steps = int(all_steps * hp.warmup_prop)

training_mode = tf.placeholder_with_default(True, shape=())
bert_crf_fn = BertCrf(bert_config, hp.layer_type, num_tags)
loss, train_op, out_tags = bert_crf_fn.create_model(xs, training_mode, hp.dropout_rate, hp.lr, all_steps, num_warmup_steps)

global_step = tf.train.get_or_create_global_step()
train_summaries = None

tf.summary.scalar("global_step", global_step)
train_summary = tf.summary.merge_all()

logging.info("# Session")
saver = tf.train.Saver(max_to_keep = hp.num_epochs)
with tf.Session() as sess:
    sess.run(train_init_op)
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("initial from scrtch")
        sess.run(tf.global_variables_initializer())
        utils.save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
        
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, all_steps + 1)):

        _, _gs, _summary = sess.run([train_op, global_step, train_summaries], feed_dict={training_mode: True})

        if _gs and _gs % num_train_batches == 0:
            epoch = _gs / num_train_batches
            logging.info("Epoch %d is done"%epoch)

            _summary, _loss = sess.run([train_summary, loss], feed_dict={training_mode: False})  # train loss
            summary_writer.add_summary(_summary, _gs)

            sess.run(eval_init_op)
            for i in range(num_eval_batches):
                [_out_tag, _ori_chrs, _ori_tags] = sess.run([out_tags, ori_chars, ori_tags], feed_dict={training_mode: False})
                eval_tags = tag_vocab.batches2vocab(_out_tag)

            model_name = "bert_crf_E%d_%.3f"%(epoch, _loss)
            logging.info("# write result")
            evaled_tags_file = os.path.join(hp.logdir, model_name)
            with open(evaled_tags_file, "w") as f:
                for tags in eval_tags:
                    f.write(" ".join(tags) + "\n")

            logging.info("# save model")
            ckpt_name = os.path.join(hp.logdir, model_name)
            saver.save(sess, ckpt_name, global_step=_gs)

            logging.info("# back to train")
            sess.run(train_init_op)

        sess.run(loss)
