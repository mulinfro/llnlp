
import tensorflow as tf
from text_cnn import TextCNN


def train(x_train, y_train, filters, batch_size, num_epochs, save_path):
    model = TextCNN(args.padding_size, args.embed_size, vocab_size, filters, args.num_channels,
                    args.num_classes, args.dropout_rate, args.regularizers_lambda)
    model.summary()
    parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    parallel_model.compile(tf.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = tf.one_hot(y_train, args.num_classes)
    history = parallel_model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=args.fraction_validation, shuffle=True)
    keras.models.save_model(model, save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN train project.')
    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float, help='The fraction of test data.(default=0.1)')
    parser.add_argument('-p', '--padding_size', default=128, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-e', '--embed_size', default=512, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=128, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.5, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=18, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float, help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='./results/', type=str, help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    args = parser.parse_args()
    print('Parameters:', args, '\n')
