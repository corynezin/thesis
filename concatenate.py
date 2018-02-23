# Name: Cory Nezin
# Date: 12/29/2017
# Goal: Concatenate all text files in a given folder to create a corpus

import sys
import os
import re
import argparse
import preprocess

corpus = ''

traindir = './aclImdb/train/posneg'
for file_obj in os.scandir(traindir):
    f = open(traindir + '/' + file_obj.name)
    s = preprocess.simplify(f.read())
    corpus += ' ' + s

f = open('./corpus/imdb_train_corpus.txt','w')
f.write(corpus)
