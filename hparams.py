
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # training scheme
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate")
    parser.add_argument("--warmup_prop", default=0.5, type=float, help="warmup_prop")
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--layer_type", default="last2", type=str)
    parser.add_argument("--dropout_rate", default=0.3, type=float, help="dropout rate")

    parser.add_argument("--train_path", default="", type=str)
    parser.add_argument("--test_path", default="", type=str)
    parser.add_argument("--vocab_file", default="", type=str)
    parser.add_argument("--tag_mapping_file", default="", type=str)
    parser.add_argument("--bert_config", default="", type=str)

    parser.add_argument("--logdir", default="log", type=str)
    parser.add_argument("--evaldir", default="evaldir", type=str)
