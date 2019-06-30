import nltk
import numpy as np
import sys
from collections import defaultdict


# Modified version homework Jelle van den Broek and Maurice Schaasberg
# returns a generator (a data stream) that yields one pre-processed sentence at a time.
def preprocess(file_path, min_count=1, char_level=False):
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

    # This function returns the categorical parameter associated with a certain word given a certain history.
    def get_parameter(self, history, word):
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


    # this function will add counts to whatever counts are already stored in _count_table.
    def count_ngrams(self, data_stream):
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


    # this function will replace whatever value _prob_table currently stores by the newly computed MLE solution.
    def solve_mle(self):
        for his in self._count_table.keys():
            total_count = sum(self._count_table[his].values())
            tempdic = dict()
            for word in self._count_table[his].keys():
                tempdic[word] = self._count_table[his][word] / total_count
            self._prob_table[his] = tempdic
        return self._prob_table


    # compute the log probability of a sentence under this model.
    def log_prob(self, sentence):
        sentence_probability_sum = 0.
        sentence = ["<s>"] * self._order + sentence + ["</s>"]
        for n in range(self._order, len(sentence)):
            history = " ".join(sentence[n - self._order:n]).lower()
            word = sentence[n]
            word_probability = self.get_parameter(history, word)
            sentence_probability_sum += np.log(word_probability)
        return sentence_probability_sum

    # calculates the perplexity of the given text.
    def log_perplexity(self, sentences):
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

    # this function returns the categorical parameter associated with a certain word given a certain history.
    def get_parameter(self, history, word):
        if history not in self._prob_table:
            return 1/len(self._vocab)
        else:
            if word not in self._prob_table[history]:
                return self._prob_table[history]['<unk>']
            else:
                return self._prob_table[history][word]

    # this function will add counts to whatever counts are already stored in _count_table.
    def count_ngrams(self, data_stream):
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

    # this function will replace whatever value _prob_table currently stores by the newly computed MLE solution.
    def solve_mle(self):
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
