import sys
import os
import collections
import string
from nltk import word_tokenize
from nltk import pos_tag
from operator import itemgetter


class Solution2 :
    #Implementation of IBM Model 1
    def __init__(self, source_file, target_file, pos_tagging=False):
        self.pos_tagging = pos_tagging
        s_lines = self.read_text_file(source_file)
        t_line = self.read_text_file(target_file)
        
        #Generate corpus in the form of list of tuples
        corpus = []
        for index, s_line in enumerate(s_lines):
            s_words = list(word_tokenize(s_line))
            t_words = list(word_tokenize(t_line[index]))
            if self.pos_tagging:
                s_words = pos_tag(s_words)
                t_words = pos_tag(t_words)
            corpus.append((t_words, s_words))
        
        self.model = self.train(corpus, 25)#training the model
        
    def trans(self, source_file):
       
        if self.pos_tagging:
            output_file = os.path.splitext(source_file)[0] + '_pos.translated'
        else:
            output_file = os.path.splitext(source_file)[0] + '.translated'
            
        try:
            
            output_file = open(output_file, 'w')
        except:
            print('Cannot open file' + output_file + ' for writing', file=sys.stderr)
            sys.exit(1)
        
        s_lines = self.read_text_file(source_file)
        for s_line in s_lines:
            s_words = list(word_tokenize(s_line.strip()))
            if self.pos_tagging:
                s_words = pos_tag(s_words)
                
            words_translated = []
            for word in s_words:
                if self.model[word]:
                    word_translated = max(self.model[word].items(), key=itemgetter(1))[0]
                    words_translated.append(word_translated)
            #Removing POS tags
            if self.pos_tagging:
                words_translated = [word[0] for word in words_translated]
            translated_sentence = self.word_sentence(words_translated)
      
            output_file.write(translated_sentence + '\n')
            
    @staticmethod
    def read_text_file(filename):
        try:
            file = open(filename, 'r')
        except:
            print('Cannot read file ' + filename + '. Check the path', file=sys.stderr)
            sys.exit(1)
        output = []
    
        for line in file:
            line = line.strip().lower()
            output.append(line)
        return output
    
    @staticmethod
    def word_sentence(words):
        return ''.join([word if word in string.punctuation else ' ' + word for word in words]).strip()
    
    @staticmethod
    def train(corpus, iterations=100):
        vocabulary = set()
        for (t_words, s_words) in corpus:
            vocabulary = vocabulary.union(set(s_words))
        default_probability = 1 / len(vocabulary)
        probabilities = collections.defaultdict(lambda: default_probability)
        model = collections.defaultdict(collections.defaultdict)
        
        for i in range(iterations):
            count = collections.defaultdict(lambda: 0.0)
            total_arr = collections.defaultdict(lambda: 0.0)
            s_total = collections.defaultdict(lambda: 0.0)
            
            for (t_words, s_words) in corpus:
                for target_word in t_words:
                    count[target_word] = 0.0
                    for source_word in s_words:
                        count[target_word] += probabilities[(target_word, source_word)]
                for target_word in t_words:
                    for source_word in s_words:
                        total = probabilities[(target_word, source_word)] / count[target_word]
                        total_arr[(target_word, source_word)] += total
                        s_total[source_word] += total
            for (target_word, source_word) in total_arr.keys():
                probabilities[(target_word, source_word)] = total_arr[(target_word, source_word)] / s_total[source_word]
            #Converting model to dictionary
            for target_word, source_word in probabilities:
                model[source_word][target_word] = probabilities[(target_word, source_word)]
                
        return model
solution2 = Solution2('es-en/train/europarl-v7.es-en.es', 'es-en/train/europarl-v7.es-en.en')
print('Translating dev set')
solution2.trans('es-en/dev/newstest2012.es')
print('Translating test set')
solution2.trans('es-en/test/newstest2013.es')
print('Translation with POS tagging')
solution2 = Solution2('es-en/train/europarl-v7.es-en.es', 'es-en/train/europarl-v7.es-en.en', pos_tagging=True)
print('Translating dev set')
solution2.trans('es-en/dev/newstest2012.es')
print('Translating test set')
solution2.trans('es-en/test/newstest2013.es')
