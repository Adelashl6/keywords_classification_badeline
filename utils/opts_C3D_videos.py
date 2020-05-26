import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    # ------------------------------------------------ Data Path -------------------------------------------------------
    parser.add_argument(
        '--c3d_path',
        type=str,
        default='./pre-trained/s3d_nce_pretrained.pth',
        help='Pre-trained C3D weights.')
    parser.add_argument(
        '--ann_path',
        type=str,
        default='./data/keyword_extraction',
        help='Path of HowTo100 Dataset Annotations.')

    parser.add_argument(
        '--keyword_file',
        type=str,
        default='train_recipe_3k_keywords.pkl',
        help='Caption files.')

    parser.add_argument(
        '--vocab_file',
        type=str,
        default='vocab.pkl',
        help='Vocabulary diction files.')
    parser.add_argument(
        '--splits_file',
        type=str,
        default='splits.pkl',
        help='Splits files.')

    # ------------------------------------------------ Model Setting ---------------------------------------------------

    parser.add_argument(
        '--max_length',
        type=int,
        default=143+2,
        help='Caption files.')

    parser.add_argument(
        '--feature_mode',
        type=str,
        default='ResNet',
        help='Model of feature: ResNet|BN-Inception.')

    parser.add_argument(
        '--input_dropout_p',
        type=float,
        default=0.2,
        help='strength of dropout in the Language Model RNN')

    parser.add_argument(
        '--dim_word',
        type=int,
        default=768,
        help='the encoding size of each token in the vocabulary, and the video.')

    parser.add_argument(
        '--dim_model',
        type=int,
        default=768,
        help='size of the rnn hidden layer')

    parser.add_argument(
        '--dim_language',
        type=int,
        default=768,
        help='dim of language feature from GPT')

    # 12-12 8 6 4
    parser.add_argument(
        '--num_head',
        type=int,
        default=8,
        help='Numbers of head in transformers.')

    parser.add_argument(
        '--num_layer',
        type=int,
        default=2,
        help='Numbers of layers in transformers.')

    parser.add_argument(
        '--dim_head',
        type=int,
        default=48,
        help='Dimension of the attention head.')

    parser.add_argument(
        '--dim_inner',
        type=int,
        default=1024,
        help='Dimension of inner feature in Encoder/Decoder.')

    # Optimization: General
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='number of epochs')

    parser.add_argument(
        '--warm_up_steps',
        type=int,
        default=500,
        help='Warm up steps.')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='mini-batch size')

    # -----------------------------------------------Checkpoint Setting-------------------------------------------------

    parser.add_argument(
        '--save_checkpoint_every',
        type=int,
        default=10,
        help='how often to save a model checkpoint (in epoch)?')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./ckpt',
        help='directory to store check pointed models')

    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default='./ckpt/Model_c3d_2tsfm_90.pth',
        help='directory to load check pointed models')

    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    args = parser.parse_args()

    return args
