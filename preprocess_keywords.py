import pickle
import json
import random
import math

keywords_file = './data/keyword_extraction/train_recipes_3k_goodkey_keywords_arnav.json'
keywords_list = json.load(open(keywords_file, 'r'))['contents']

s3d_file = './data/howto100annotation/recipe_mixed_pool.pkl'
s3d_feat = pickle.load(open(s3d_file, 'rb'), encoding="utf-32")


# turn keyword list into a string
def list_to_sent(sent_list):
    re = ''
    for sent in sent_list:
        re += sent + ' '
    return re


# generate the keyword annotations in captioning format
total_ann = {}
count = 0
key_len = 0
for keywords in keywords_list:
    video_idx = keywords['id']
    if video_idx not in s3d_feat:
        continue
    key_list = keywords['keywords']
    start_list = keywords['start']
    end_list = keywords['end']
    total_time = end_list[-1]
    total_feat = s3d_feat[video_idx].shape[0]

    # Iterate the video clips
    clip_idx = 0
    length = len(key_list)
    #interval = 2
    for i in range(0, length):
        clip_ann = {}
        start = start_list[i]
        end = end_list[i]

        clip_ann['segment'] = [start, end]
        clip_ann['keywords'] = list_to_sent(key_list[i])[:-1]
        print(clip_ann['keywords'])
        clip_ann['feature'] = s3d_feat[video_idx][int((start / total_time) * total_feat):
                                                  int(math.ceil((end / total_time) * total_feat))]
        _ = video_idx + '_' + str(clip_idx)
        total_ann[_] = clip_ann
        clip_idx += 1
        count += 1
        if key_len < len(key_list[i]):
            key_len = len(key_list[i])

pickle.dump(total_ann, open('./data/keyword_extraction/train_recipe_3k_keywords.pkl', 'wb'))
print(key_len)


# generate vocabulary
word_to_ix = {}
word_to_ix["<sep>"] = 4
word_to_ix["<eos>"] = 3
word_to_ix["<sos>"] = 2
word_to_ix["<UNK>"] = 1
word_to_ix["<PAD>"] = 0

idx = 5
for keywords in keywords_list:
    video_idx = keywords['id']
    if video_idx not in s3d_feat:
        continue
    key_list = keywords['keywords']
    for keys in key_list:
        for key in keys:
            if key not in word_to_ix:
                word_to_ix[key] = idx
                idx += 1

ix_to_word = {}
for key in word_to_ix:
    ix = word_to_ix[key]
    ix_to_word[str(ix)] = key

vocab = {}
vocab['word_to_ix'] = word_to_ix
vocab['ix_to_word'] = ix_to_word

# Save the annotation file
pickle.dump(vocab, open('./data/keyword_extraction/vocab.pkl', 'wb'))

# split train and val sets
_ = './data/keyword_extraction/train_recipe_3k_keywords.pkl'
ann = pickle.load(open(_, 'rb'), encoding="utf-32")
video_idx = list(ann.keys())
random.shuffle(video_idx)

train_len = int(len(video_idx) * 0.9)
train_idx = video_idx[0:train_len]
test_idx = video_idx[train_len:]

splits = {}
splits['train'] = train_idx
splits['val'] = test_idx

pickle.dump(splits, open('./data/keyword_extraction/splits.pkl', 'wb'))
print(len(train_idx))
print(len(test_idx))

