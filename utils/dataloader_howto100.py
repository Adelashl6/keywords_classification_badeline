import os
import pickle
import torch
from torch.utils.data import Dataset


class HowTo100MDataset(Dataset):

    # Auxiliary functions
    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.word_to_ix

    def get_ix_to_word(self):
        return self.ix_to_word

    def __init__(self, opt, mode='train'):
        super(HowTo100MDataset, self).__init__()
        self.opt = opt
        self.mode = mode

        # Load the caption annotations
        self.caption_hub = pickle.load(open(os.path.join(opt['ann_path'], opt['keyword_file']), 'rb'))

        # Read the training/testing split
        self.splits = pickle.load(open(os.path.join(opt['ann_path'], opt['splits_file']), 'rb'))[mode]

        # Read the vocabulary file
        vocab = pickle.load(open(os.path.join(opt['ann_path'], opt['vocab_file']), 'rb'))

        # Load the vocabulary dictionary
        self.word_to_ix = vocab['word_to_ix']
        self.ix_to_word = vocab['ix_to_word']

    def __getitem__(self, video_id=False):
        video_id = self.splits[video_id]

        # Load the caption and time boundaries
        data = {}
        annotation = self.caption_hub[video_id]
        segment = annotation['segment']
        keywords = annotation['keywords']
        feature = annotation['feature']

        data['video_feat'] = torch.tensor(feature)
        data['segment'] = segment
        data['keywords'] = keywords
        data['video_id'] = video_id
        data['arr_length'] = data['video_feat'].shape[0]
        return data

    def __len__(self):
        length = len(self.splits)
        return length


def HowTo100_collate_fn(batch_lst):
    '''
    :param batch_lst: Raw instance level annotation from HowTo100 Dataset class.
    :return: batch annotations that include: features, captions, length of features and video ids.
    '''
    batch_lens = [_['arr_length'] for _ in batch_lst]
    max_length = max(batch_lens)
    batch_feat = torch.zeros((len(batch_lst), max_length, 1024))
    keywords = []
    video_ids = []

    for batch_id, batch_data in enumerate(batch_lst):
        batch_feat[batch_id][: batch_data['arr_length']] = batch_data['video_feat']
        keywords.append(batch_data['keywords'])
        video_ids.append(batch_data['video_id'])
    return batch_feat, keywords, torch.tensor(batch_lens), video_ids

