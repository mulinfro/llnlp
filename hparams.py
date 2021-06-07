
import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=32000, type=int)

    # training scheme
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)