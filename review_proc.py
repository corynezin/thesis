import preprocess
import numpy as np
import re

class review:
    def __init__(self,review_filename):
        self.review_rating = int(re.search('_\d+',review_filename).group()[1:])
        target_list = [0,1] if self.review_rating > 5  else [1,0]
        self.sentiment = 'pos' if self.review_rating > 5 else 'neg'
        review_file = open(review_filename).read()
        self.tokens = preprocess.tokenize(review_file)
        self.targets = np.array([target_list]);
        self.length = len(self.tokens)

    def translate(self, max_time, word_to_embedding_index, embedding_index_to_word):
        inputs = np.zeros((1,max_time),int)
        lookup = []
        self.unk_loc = []
        for index in range(min(max_time,self.length)):
            inputs[0][index] = word_to_embedding_index.get(self.tokens[index],0)
            lookup.append( embedding_index_to_word.get(inputs[0][index],'UNK') )
            if lookup[-1] == 'UNK':
                self.unk_loc.append(index)
        
        self.index_vector = inputs
        self.word_list = lookup

    def vec(self, word_embedding):
        self.vector_list = word_embedding[self.index_vector[0][0:self.length],:]
