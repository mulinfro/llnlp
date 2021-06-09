import collections, six

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    for line in reader.readlines():
      token = convert_to_unicode(line)
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
        res = [self.idx2vocab_dict[seq] for seq in seqs]
        return res

    def vocabs2idx(self, seqs, add_unk=False, do_lower_case=False):
        res = []
        for seq in seqs:
            if do_lower_case and seq[0]!= "[" and seq[-1] != "]":
                seq = seq.lower()

            if seq not in self.vocab_dict:
                seq = "##" + seq

            if add_unk and seq not in self.vocab_dict:
                seq = "[UNK]"
            res.append(self.vocab_dict[seq])

        return res

    def batches2vocab(self, seq_of_seq):
        res = []
        for seq in seq_of_seq:
            res.append(self.idxes2vocabs(seq))
        return res
