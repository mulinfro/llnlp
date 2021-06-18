
import tensorflow as tf
from bert import tokenization
import utils
from vocab_manager import Vocab

_CLS = "[CLS]"
_PAD = "[PAD]"
_SEP = "[SEP]"

def pad_with_length(seq, max_len, pad=_PAD):
    if len(seq) > max_len:
        return seq[0:max_len]
    else:
        for i in range(max_len - len(seq)):
            seq.append(pad)
        return seq

def warp_seq(seq):
    return [_CLS] + seq + [_SEP]

def warp_then_pad(seq, max_seq_len):
    seq = warp_seq(seq)
    return pad_with_length(seq, max_seq_len)

def load_seq_data(data_path, max_seq_len):
    seqs = []
    text_seq, tag_seq = [], []
    with open(data_path, "r") as reader:
        for line in reader.readlines():
            line = tokenization.convert_to_unicode(line).strip()
            eles = line.split(" ")
            if len(eles) < 2:
                if len(text_seq):
                    text_seq = warp_then_pad(text_seq, max_seq_len)
                    tag_seq  = warp_then_pad(tag_seq,  max_seq_len)
                    seqs.append((text_seq, tag_seq))
                text_seq, tag_seq = [], []
            else:
                text_seq.append(eles[0])
                tag_seq.append(eles[1])

        if len(text_seq):
            text_seq = warp_then_pad(text_seq, max_seq_len)
            tag_seq  = warp_then_pad(tag_seq,  max_seq_len)
            seqs.append((text_seq, tag_seq))
    
    return seqs


def generate_fn(seq_datas):
    for seq in seq_datas:
        #print("SEQ",seq)
        yield seq

def input_fn(seq_datas, batch_size, shuffle):
    print("fn 1")
    dataset = tf.data.Dataset.from_tensor_slices(seq_datas)
    print("fn 2")
    if shuffle:
        print("fn 3")
        dataset = dataset.shuffle(batch_size * 128)

    print("fn 4")
    dataset = dataset.repeat()
    print("fn 5")
    dataset = dataset.batch(batch_size).prefetch(8)

    return dataset



def input_fn_generator(seq_datas, batch_size, shuffle):

    #shapes = ([None], [None],(), ())
    types = (tf.int32, tf.int32, tf.string, tf.string)
    shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))

    dataset = tf.data.Dataset.from_generator(
        generate_fn,
        types,
        shapes,
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


def convert_tokens_to_ids(tokens, vocab_dict):
    res = [vocab_dict[seq] for seq in seqs]
    return res

def idx2token(seqs, tag_mapping_file):
    tag_mapping = tokenization.load_vocab(tag_mapping_file)
    res = [tag_mapping[seq] for seq in seqs]
    return res

#def token2idx_tmp(seqs, vocab_file, tag_mapping_file, do_lower_case):
#    # tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
#    word_vocab = Vocab(vocab_file)
#    tag_mapping = Vocab(tag_mapping_file)
#    # tag_mapping_file = tokenization.load_vocab(tag_mapping_file)
#    seq_ids = []
#    for chrs, tags in seqs:
#        # text = "".join(chrs)
#        #tokens = tokenizer.tokenize(text)
#        tokens = ["[CLS]"] + chrs + ["[SEP]"]
#        tags = ["[CLS]"] + tags + ["[SEP]"]
#        cur = 0
#        merged_tags = []
#        for token in tokens:
#            if token.startswith("##"):
#                token = token[2:]
#            if cur >= len(tags):
#                print(tokens, "\n", chrs, "\n", tags, "\n", cur, token)
#            merged_tags.append(tags[cur])
#            if token not in ["[CLS]", "[SEP]"]:
#                cur += len(token)
#            else:
#                cur += 1
#
#        token_ids = tokenizer.convert_tokens_to_ids(tokens)
#        tag_ids = tokenization.convert_tokens_to_ids(tag_mapping, merged_tags)
#        assert len(token_ids) == len(tag_ids)
#        seq_ids.append(token_ids, tag_ids, tokens, tags)
#
#    return seq_ids

def token2idx(seqs, vocab_file, tag_mapping_file, do_lower_case):
    word_vocab = Vocab(vocab_file)
    tag_mapping = Vocab(tag_mapping_file)
    seq_ids, tag_ids_lst, tokens_lst, tags_lst = [], [], [], []
    for tokens, tags in seqs:
        token_ids = word_vocab.vocabs2idx(tokens, add_unk=True, do_lower_case=True)
        tag_ids = tag_mapping.vocabs2idx(tags, add_unk = False, do_lower_case=False)

        assert len(token_ids) == len(tag_ids), "%d_%d"%(len(token_ids), len(tags))
        #seq_ids.append( (token_ids, tag_ids, " ".join(tokens), " ".join(tags)) )
        seq_ids.append(token_ids)
        tag_ids_lst.append(tag_ids)
        tokens_lst.append(" ".join(tokens))
        tags_lst.append(" ".join(tags))

    return seq_ids, tag_ids_lst, tokens_lst, tags_lst


def get_batch(data_file, batch_size, max_seq_len, vocab_file, tag_mapping_file, do_lower_case = True, shuffle = False):
    seqs = load_seq_data(data_file, max_seq_len)
    seq_idxs = token2idx(seqs, vocab_file, tag_mapping_file, do_lower_case)

    print("input fn", len(seq_idxs), len(seq_idxs[0]), data_file)
    dataset = input_fn(seq_idxs, batch_size, shuffle)

    samples = len(seqs)
    num_batches = calc_num_batches(samples, batch_size)

    print("batches done", data_file)
    return dataset, num_batches, samples

