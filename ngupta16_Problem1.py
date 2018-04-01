import sys
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk import FreqDist
import string
import json
import math
import numpy.random
import itertools


class Solution1:
    def __init__(self, dictionary_file, training_file):
        self.dictionary = self.read_json_file(dictionary_file)
        training_data = self.read_text_file(training_file)
        self.uni_words = None
        self.bi_words = None
        self.uni_words_pos = None
        self.bi_words_pos = None
        self.uni_pos = None
        self.bi_pos = None
        self.train(training_data)
	
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
    def read_json_file(filename):
        try:
            file = open(filename, 'r')
        except:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            sys.exit(1)
        return json.load(file)
    @staticmethod
    def words_sentence(words):
        return ''.join([word if word in string.punctuation else ' ' + word for word in words]).strip()
        
    @staticmethod
    def print_translation(title, source, translation):
        print('%s' % title)
        print('%s' % source)
        print('%s' % translation)
        print('\n')

    def train(self, lines):
        
        uni_words = []
        bi_words = []
        tri_words = []
        uni_words_pos = []
        bi_words_pos = []
        uni_pos = []
        bi_pos = []
        
        for line in lines:
            words = word_tokenize(line)
            words_pos = pos_tag(words)
            pos = [word[1] for word in words_pos]
            uni_words = uni_words + ['<s>'] + words + ['</s>']
            uni_words_pos = uni_words_pos + words_pos
            uni_pos = uni_pos + ['<s>'] + pos + ['</s>']
            bi_words = bi_words + list(
                ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bi_words_pos = bi_words_pos + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bi_pos = bi_pos + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            
        self.uni_words = FreqDist(uni_words)
        self.bi_words = FreqDist(bi_words)
        self.tri_words = FreqDist(tri_words)
        self.uni_words_pos = FreqDist(uni_words_pos)
        self.bi_words_pos = FreqDist(bi_words_pos)
        self.uni_pos = FreqDist(uni_pos)
        self.bi_pos = FreqDist(bi_pos)
    
    def bigram_words_probability(self, words):
        probability = 0
        vocabulary_size = len(self.uni_words)
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for bigram in bigrams:
            probability += math.log(self.bi_words.freq(bigram) + 1) - math.log(
                self.uni_words.freq(bigram[1]) + vocabulary_size)
        
        return probability

    def trigram_words_probability(self, words):
        probability = 0
        vocabulary_size = len(self.uni_words)
        trigrams = list(ngrams(words, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for trigram in trigrams:
            probability += math.log(self.tri_words.freq(trigram) + 1) - math.log(
                self.bi_words.freq(trigram[1]) + vocabulary_size)
        
        return probability
    
    def bigram_pos_words_probability(self, words):
        words = pos_tag(words)
        probability = 0
        vocabulary_size = len(self.uni_words_pos)
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for bigram in bigrams:
            probability += math.log(self.bi_words_pos.freq(bigram) + 1) - math.log(
                self.uni_words_pos.freq(bigram[1]) + vocabulary_size)
            
        return probability
    
    def bigram_pos_probability(self, words):
        probability = 0
        vocabulary_size = len(self.uni_pos)
        bigrams = list(ngrams(words, 2))
        for bigram in bigrams:
            probability += math.log(self.bi_pos.freq(bigram) + 1) - math.log(
                self.uni_pos.freq(bigram[1]) + vocabulary_size)
    
        return probability
    
    def probability_permutation(self, words, method):
        max_probability = -math.inf
        selected = None
        permutation_count = math.factorial(len(words)) if len(words) < 5 else 100
        for _ in range(permutation_count):
            permutation = numpy.random.permutation(words)
            probability = getattr(self, method)(permutation)
            if probability > max_probability:
                max_probability = probability
                selected = permutation
                
        return selected
    
    def pos_model(self, words):
        words_pos = [('', '<s>')] + pos_tag(words) + [('', '</s>')]
        length = len(words_pos)
        
        for index, word in enumerate(words_pos):
            words_window = words_pos[index : index + 4]
            
            max_probability = -math.inf
            selected = None
            permutations = itertools.permutations(words_window)
            for permutation in permutations:
                pos = [word[1] for word in permutation]
                probability = self.bigram_pos_probability(pos)
                if probability > max_probability:
                    max_probability = probability
                    selected = permutation
            
            words_pos[index] = selected[0]
            words_pos[index + 1] = selected[1]
            words_pos[index + 2] = selected[2]
            words_pos[index + 3] = selected[3]
            
            if index == length - 4:
                break;
        return [word[0] for word in words_pos]
    
    def swap_pos(self, words):
        words_pos = pos_tag(words)
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            if (word[1] == 'PRP' or word[1] == 'PRP$' or word[1] == 'JJ') \
                and (words_pos[index + 1][1] == 'VB' or words_pos[index + 1][1] == 'VBD' \
                     or words_pos[index + 1][1] == 'VBG' or words_pos[index + 1][1] == 'VBN' \
                     or words_pos[index + 1][1] == 'VBP' or words_pos[index + 1][1] == 'WP'):
                temp_word = words_pos[index + 1];
                words_pos[index + 1] = words_pos[index]
                words_pos[index] = temp_word
        return [word[0] for word in words_pos]

    def swap_verb_after_noun(self, words):
        words_pos = pos_tag(words)
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            if (word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS') \
                and (words_pos[index + 1][1] == 'VB' or words_pos[index + 1][1] == 'VBD' \
                     or words_pos[index + 1][1] == 'VBG' or words_pos[index + 1][1] == 'VBN' \
                     or words_pos[index + 1][1] == 'VBP' or words_pos[index + 1][1] == 'VBZ'):
                temp_word = words_pos[index + 1];
                words_pos[index + 1] = words_pos[index]
                words_pos[index] = temp_word
        return [word[0] for word in words_pos]
    
    def translate(self, line):
        words = word_tokenize(line)
        translated_words = []
        for i, word in enumerate(words):
            if word not in string.punctuation:
                translated_words.append(self.dictionary[word])
            else:
                translated_words.append(word)
        translated_sentence = self.words_sentence(translated_words)
        self.print_translation('Translation with 0 strategy', line, translated_sentence)
        
        #Swap the nearest adjective with the word after noun
        translated_words = self.swap_pos(translated_words)
        translated_sentence = self.words_sentence(translated_words)
        self.print_translation('Translation after swapping parts of speech', line, translated_sentence)

        #Swap the nearest verb with the word after noun
        translated_words = self.swap_verb_after_noun(translated_words)
        translated_sentence = self.words_sentence(translated_words)
        self.print_translation('Translation after swapping verb with noun', line, translated_sentence)
        
        #Bigram Language Model
        selected_translation = self.probability_permutation(translated_words, 'bigram_words_probability')
        translated_sentence = self.words_sentence(selected_translation)
        self.print_translation('Translation after applying Bigram Model', line, translated_sentence)

        #Trigram Language Model
        selected_translation = self.probability_permutation(translated_words, 'trigram_words_probability')
        translated_sentence = self.words_sentence(selected_translation)
        self.print_translation('Translation after applying Trigram Model', line, translated_sentence)

        #Bigram POS Language Model
        selected_translation = self.probability_permutation(translated_words, 'bigram_pos_words_probability')
        translated_sentence = self.words_sentence(selected_translation)
        self.print_translation('Translation after applying Bigram and POS Tagging', line, translated_sentence)
        
        #Rearrangement of POS
        selected_translation = self.pos_model(translated_words)
        translated_sentence = self.words_sentence(selected_translation)
        self.print_translation('Translation after POS rearrangement', line, translated_sentence)
        
    
    def execute(self, input_file):
        lines = self.read_text_file(input_file)
        for line in lines:
            self.translate(line)

solution1 = Solution1('dictionary.json', 'Sentence_en_dev.txt')
solution1.execute('Sentence_fr_dev.txt')
#solution1.execute('Sentence_fr_test.txt')
