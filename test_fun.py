
import data_load


dataset, num_batches, samples = data_load.get_batch(
                            "train.conll",
                            2,
                            32,
                            "word_vocab.txt",
                            "tags2.txt"
                            )

print(num_batches, samples)

i = 0
for d in dataset:
    i += 1
    if i > 1:
        break
    print(d[0], d[1], d[2], d[3])
