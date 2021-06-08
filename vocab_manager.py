import collections, six

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def reverse_vocab(vocab2idx):
    rev_vocab = {}
    for k,v in vocab2idx.items():
        rev_vocab[v] = k

    return rev_vocab


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


class Vocab:

    def __init__(self, vocab_file):
        self.vocab_dict = load_vocab(vocab_file)
        self.idx2vocab_dict = reverse_vocab(self.vocab_dict)

    def idxes2vocabs(self, seqs):
        res = [self.vocab_dict[seq] for seq in seqs]
        return res

    def batches2vocab(self, seq_of_seq):
        res = []
        for seq in seq_of_seq:
            res.append(self.idxes2vocabs(seq))
        return res