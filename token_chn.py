
from utils import load_vocab

class BasicTokenizer(object):

    def __init__(self, vocab_file):
        vocabs = load_vocab(vocab_file)
    
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
          cp = ord(char)
          if self._is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
          else:
            output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
           # This defines a "chinese character" as anything in the CJK Unicode block:
           #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
           #
           # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
           # despite its name. The modern Korean Hangul alphabet is a different block,
           # as is Japanese Hiragana and Katakana. Those alphabets are used to write
           # space-separated words, so they are not treated specially and handled
           # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
          return True

 
    def whitespace_tokenize(text):
        """Runs basic whitespace cleaning and splitting on a piece of text."""
        text = text.strip()
        if not text:
          return []
        tokens = text.split()
        return tokens


    def mask_num_alpha(token):
        if len(token) == 1:
            return token
        else:
          m = re.match("^[0-9_-]+$", init_checkpoint)
          if m is not None:
              return "00"
          else:
              return "A"
     
     
    def tokenize(self, text):
        text = self._tokenize_chinese_chars(text)
        tokens = self.whitespace_tokenize(text)
        
