

import tensorflow as tf

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab

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



def get_batch(data_file, batch_size, max_seg_len, vocab_file, shuffle = False):
    seqs = load_seq_data(data_path)
