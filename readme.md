The basic model for keywords classification

# dataset: 3k miniset from HowTo100M datasets
The keyword annotation file can be downloaded from: 
https://drive.google.com/file/d/1NB1rd9wd9h1GzyG9exz9uRx3oUcNMHZY/view?usp=sharing
And move it to ./data/keyword_extraction

# preprocess
run the preprocess_keywords.py to generate the corresponding files.

train_recipe_3k_keyword.pkl: regenerate the keyword annotation file into a format that can easily be loaded.

vovab.pkl: collect the keywords of the whole dataset as a corpus

split.pkl: split the dataset into training and validation datasets and restore the video_ids for training and validation sets

The files can be downloaded from:
https://drive.google.com/drive/folders/1h7reIb_YuJRReeLtqeTcZ1TG_QjUvYaF?usp=sharing

And move them into ./data/keyword_extraction

# train
run the file train_C3D_keywords_classification.py

# test
run the file test_C3D_keywords_classification.py

# model
The model can be downloaded from: 
https://drive.google.com/file/d/1auAoWbTInGsp9nNol3U6rxq1hor4J3Cm/view?usp=sharing
And move it to ./ckpt
