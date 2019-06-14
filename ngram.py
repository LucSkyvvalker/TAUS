import nltk
import numpy as np
import sys
from collections import defaultdict

# Modified version homework Jelle van den Broek and Maurice Schaasberg

def preprocess(file_path, min_count=1, char_level=False):
    """
    Returns a generator (a data stream) that yields one pre-processed sentence at a time.
    A preprocessed sentence is:
        - a list of tokens (each token a string)
            - where tokens are lowercased
                - and possibly replaced by '<unk>' if infrequent 
        
    :param file_path: path to a text corpus
    :param min_count: minimum number of occurrences 
        if a token happens less times than this value we replace it by '<unk>'
    :returns: a generator of sentences
        A generator is an object that can be used in `for` loops
    """
    count = defaultdict(int)
    # First we count the number of occurrences of each token
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # we skip empty lines
            if char_level:
                sentence = [ch for ch in line.lower()] 
            else:
                sentence = line.lower().split()
            for token in sentence:
                count[token] += 1
    # then we yield one preprocessed sentence at a time
    # making sure we map infrequent tokens to <unk>
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # we skip empty lines
            if char_level:
                sentence = [ch for ch in line.lower()] 
            else:
                sentence = line.lower().split()
            preprocessed_sentence = [token if count[token] >= min_count else '<unk>' for token in sentence]                    
            yield preprocessed_sentence

class LM:
    
    def __init__(self, order):
        self._order = order
        self._count_table = dict()
        self._prob_table = dict()
        self._vocab = set()
        
    def order(self):
        return self._order

    def vocab(self):
        return self._vocab
        
    def count_table(self):
        return self._count_table
    
    def prob_table(self):
        return self._prob_table
                
    def get_parameter(self, history, word):
        """
        This function returns the categorical parameter associated with a certain word given a certain history.
        :param history: a sequence of words (a tuple)
        :param word: a word (a str)
        :return: a float representing P(word|history)
        """
        if history not in self._prob_table:
            return 0
        else:
            if word not in self._prob_table[history]:
                return 0
            else:
                return self._prob_table[history][word]
        
    def cpd_items(self, history):
        history = self.preprocess_history(history)
        # if the history is unseen we return an empty cpd
        return self._prob_table.get(history, dict()).items()
        
    def count_ngrams(self, data_stream):
        """
        This function should populate the attribute _count_table which should be understood as 
            - a python dict 
                - whose key is a history (a tuple of words)
                - and whose value is itself a python dict (or defaultdict)
                    - which maps a word (a string) to a count (an integer)
        
        This function will add counts to whatever counts are already stored in _count_table.
        
        This function also maintains a unique set of words in the vocabulary using the attribute _vocab
        
        :param data_stream: a generator as produced by `preprocess`
        """
        for sentence in data_stream:
            sentence = ["<s>" for _ in range(self._order)] + sentence + ["</s>"]  # add end of sentence
            for n in range(self._order, len(sentence)):
                if " ".join(sentence[n - self._order:n]).lower() in self._count_table:
                    countdic = self._count_table.get(" ".join(sentence[n - self._order:n]).lower())
                    if sentence[n] in countdic:
                        countdic[sentence[n]] += 1
                        self._count_table[" ".join(sentence[n - self._order:n]).lower()] = countdic
                    else:
                        countdic[sentence[n]] = 1
                        self._count_table[" ".join(sentence[n - self._order:n]).lower()] = countdic
                else:
                    tempdic = dict()
                    tempdic[sentence[n]] = 1
                    self._count_table[" ".join(sentence[n - self._order:n]).lower()] = tempdic
        return self._count_table
                    
    def solve_mle(self):
        """
        This function should compute the attribute _prob_table which has the exact same structure as _count_table
         but stores probability values instead of counts. 
        It can be seen as the collection of cpds of our model, that is, _prob_table
            - maps a history (a tuple of words) to a dict where
                - a key is a word (that extends the history forming an ngram)
                - and the value is the probability P(word|history)                
                
        This function will replace whatever value _prob_table currently stores by the newly computed MLE solution.
        """
        
        for his in self._count_table.keys():
            total_count = sum(self._count_table[his].values())
            tempdic = dict()
            for word in self._count_table[his].keys():
                tempdic[word] = self._count_table[his][word] / total_count
            self._prob_table[his] = tempdic
        return self._prob_table
        
    def log_prob(self, sentence):
        """
        Compute the log probability of a sentence under this model. 
                
        input: 
            sentence: a sequence of tokens
        output:
            log probability
        """
        sentence_probability_sum = 0.
        sentence = ["<s>"] * self._order + sentence + ["</s>"]
        for n in range(self._order, len(sentence)):
            history = " ".join(sentence[n - self._order:n]).lower()
            word = sentence[n]
            word_probability = self.get_parameter(history, word)
            sentence_probability_sum += np.log(word_probability)
        return sentence_probability_sum
    
    def log_perplexity(self, sentences):
        """
        Calculates the perplexity of the given text.
        This is simply 2 ** cross-entropy for the text.

        This function can make use of `lm.order()`, `lm.get_parameter()`, and `lm.log_prob()` 

        :param data_stream: generator of sentences (each sentence is a list of words)
        :param lm: an instance of the class LM
        """
        plogsum = 0.
        tokens = 0
        order = self._order
        for sentence in sentences:
            plogsum += self.log_prob(sentence)
            tokens += len(sentence) + 1
        return -plogsum/tokens

class LaplaceLM(LM):
    
    def __init__(self, order, alpha=1.):
        super(LaplaceLM, self).__init__(order)
        self._alpha = alpha   
        self._vocab.add('<unk>')
        
    def get_parameter(self, history, word):
        """
        This function returns the categorical parameter associated with a certain word given a certain history.
        :param history: a sequence of words (a tuple)
        :param word: a word (a str)
        :return: a float representing P(word|history)
        """
        if history not in self._prob_table:
            return 1/len(self._vocab)
        else:
            if word not in self._prob_table[history]:
                return self._prob_table[history]['<unk>']
            else:
                return self._prob_table[history][word]

    def count_ngrams(self, data_stream):
        """
        This function should populate the attribute _count_table which should be understood as 
            - a python dict 
                - whose key is a history (a tuple of words)
                - and whose value is itself a python dict (or defaultdict)
                    - which maps a word (a string) to a count (an integer)
        
        This function will add counts to whatever counts are already stored in _count_table.
        
        :param data_stream: a generator as produced by `preprocess`
        """
        for sentence in data_stream:
            sentence = ["<s>" for _ in range(self._order)] + sentence + ["</s>"]  # add end of sentence
            for n in range(self._order, len(sentence)):
                self._vocab.add(sentence[n])
                if " ".join(sentence[n - self._order:n]).lower() in self._count_table:
                    countdic = self._count_table.get(" ".join(sentence[n - self._order:n]).lower())
                    if sentence[n] in countdic:
                        countdic[sentence[n]] += 1
                        self._count_table[" ".join(sentence[n - self._order:n]).lower()] = countdic
                    else:
                        countdic[sentence[n]] = 1 
                        self._count_table[" ".join(sentence[n - self._order:n]).lower()] = countdic
                else:
                    tempdic = dict()
                    tempdic['<unk>'] = 0
                    tempdic[sentence[n]] = 1 
                    self._count_table[" ".join(sentence[n - self._order:n]).lower()] = tempdic
        return self._count_table
                    
    def solve_mle(self):
        """
        This function should compute the attribute _prob_table which has the exact same structure as _count_table
         but stores probability values instead of counts. 
        It can be seen as the collection of cpds of our model, that is, _prob_table
            - maps a history (a tuple of words) to a dict where
                - a key is a word (that extends the history forming an ngram)
                - and the value is the probability P(word|history)                
                
        This function will replace whatever value _prob_table currently stores by the newly computed MLE solution.
        """
        for his in self._count_table.keys():
            total_count = sum(self._count_table[his].values())
            tempdic = dict()
            for word in self._count_table[his].keys():
                tempdic[word] = (self._count_table[his][word] + self._alpha) / (total_count + self._alpha * len(self._vocab))
            self._prob_table[his] = tempdic
        return self._prob_table
    
def trainLM(order, corpus, Laplace=True):
    if Laplace:
        ngram = LaplaceLM(order)
    else:
        ngram = LM(order)
    ngram.count_ngrams(preprocess(corpus))
    ngram.solve_mle()
    return ngram

def log_ppl(LM, sentence):
    sentence = [sentence.lower().split()]
    return LM.log_perplexity(sentence)

def log_prob(LM, sentence):
    sentence = sentence.lower().split()
    return LM.log_prob(sentence)

def unk_ngrams(LM, sentence):
    order = LM.order()
    sentence = sentence.lower().split()
    count = 0
    prob_table = LM.prob_table()
    if order >= len(sentence):
        return 0
    for n in range(order, len(sentence)):
        history = " ".join(sentence[n-order:n])
        word = sentence[n]
        if history not in prob_table or word not in prob_table[history]:
            count += 1
    return count/(len(sentence) - order)

def average_occurence(LM, sentence):
    order = LM.order()
    sentence = sentence.lower().split()
    count = 0
    count_table = LM.count_table()
    vocablen = len(LM.vocab())
    if order >= len(sentence):
        return 0
    for n in range(order, len(sentence)):
        history = " ".join(sentence[n-order:n])
        word = sentence[n]
        if history in count_table and word in count_table[history]:
            count += count_table[history][word]/vocablen
    return count/(len(sentence) - order)

test = trainLM(1, 'sample_corpus_pos.en')
print(log_ppl(test, "nns vbp rbr"))
print(log_prob(test, "nns vbp rbr"))
print(average_occurence(test, "nns vbp rbr"))
print(unk_ngrams(test, "nns vbp rbr"))
