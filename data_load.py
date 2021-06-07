
import tensorflow as tf
from bert import tokenization

def load_seq_data(data_path):
    seqs = []
    text_seq, tag_seq = [], []
    with open(data_path, "r") as reader:
        for line in reader.readline():
            line = convert_to_unicode(line).strip()
            eles = line.split(" \t")
            if len(eles) < 2:
                if len(text_seq):
                    seqs.append((text_seq, tag_seq))
                text_seq, tag_seq = [], []
            else:
                text_seq.append(eles[0])
                tag_seq.append(eles[1])
    
    return seqs


def generate_fn(seq_datas):
    for seq in seq_datas:
        yield seq

def input_fn(seq_datas, batch_size, shuffle):

    shapes = ([None], [None], [None], [None])
    types = (tf.int32, tf.int32, tf.string, tf.string)

    dataset = tf.data.Dataset.from_generator(
        generate_fn,
        output_shapes = shapes,
        output_types = types,
        args = (seq_datas)
    )

    if shuffle:
        dataset = dataset.shuffle(batch_size * 128)

    padding_values = (0, 0, "", "")
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size, shapes, padding_values).prefetch(2)

    return dataset

def calc_num_batches(total_num, batch_size):
    return total_num // batch_size + int(total_num % batch_size != 0)

def token2idx(seqs, vocab_file, tag_mapping_file, do_lower_case):
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
    tag_mapping = tokenization.load_vocab(tag_mapping_file)
    seq_ids = []
    for chrs, tags in seqs:
        text = "".join(chrs)
        tokens = tokenizer.tokenize(text)
        cur = 0
        merged_tags = []
        for token in tokens:
            if token.startswith("##"):
                token = token[2:]
            merged_tags.append(tags[cur])
            cur += len(token)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        tag_ids = convert_tokens_to_ids(merged_tags, tag_mapping)
        seq_ids.append((token_ids, tag_ids, chars, tags))

    return seq_ids


def get_batch(data_file, batch_size, max_seq_len, vocab_file, tag_mapping_file, do_lower_case = True, shuffle = False):
    seqs = load_seq_data(data_path, vocab_file)
    seg_idxs = token2idx(seqs, vocab_file, tag_mapping_file, do_lower_case)
    dataset = input_fn(seg_idxs, batch_size, shuffle)

    samples = len(seqs)
    num_batches = cal_num_batches(samples, batch_size)

    return dataset, num_batches, samples
