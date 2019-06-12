# Jelle van den Broek and Maurice Schaasberg Homework

from collections import defaultdict
import sys

class LM:
    
    def __init__(self, order):
        self._order = order
        self._count_table = dict()
        self._prob_table = dict()
        self._vocab = set()
        
    def order(self):
        return self._order
        
    def print_count_table(self, output_stream=sys.stdout):
        """Prints the count table for visualisation"""
        for history, ngrams in sorted(self._count_table.items(), key=lambda pair: pair[0]):
            for word, count in sorted(ngrams.items(), key=lambda pair: pair[0]):
                print('history="%s" word=%s count=%d' % (' '.join(history), word, count), file=output_stream)
                
    def print_prob_table(self, output_stream=sys.stdout):
        """Prints the tabular cpd for visualisation"""
        for history, ngrams in sorted(self._prob_table.items(), key=lambda pair: pair[0]):
            for word, prob in sorted(ngrams.items(), key=lambda pair: pair[0]):
                print('history="%s" word=%s prob=%f' % (' '.join(history), word, prob), file=output_stream)
                
    def preprocess_history(self, history):
        """
        This function pre-process an arbitrary history to match the order of this language model.
        :param history: a sequence of words
        :return: a tuple containing exactly as many elements as the order of the model
            - if the input history is too short we pad it with <s> 
        """
        if len(history) == self._order:
            return tuple(history)
        elif len(history) > self._order:
            length = len(history)            
            return tuple(history[length - self._order: length]) #NOTE: we fixed the bug!
        else:  # here the history is too short
            missing = self._order - len(history)
            return tuple(['<s>'] * missing) + tuple(history)
                
    def get_parameter(self, history, word):
        """
        This function returns the categorical parameter associated with a certain word given a certain history.
        :param history: a sequence of words (a tuple)
        :param word: a word (a str)
        :return: a float representing P(word|history)
        """
        history = self.preprocess_history(history)
        cpd = self._prob_table.get(history, None)
        if cpd is None:
            return 0.
        else:
            # we either return P(x|h)
            #  or P(unk|h) in case x is not in the support of this cpd
            #   or 0. in case neither x nor unk are in the support of this cpd
            unk_probability = cpd.get('<unk>', 0.)
            return cpd.get(word, unk_probability)
        
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
        sentence = ["<s>" for _ in range(self._order)] + sentence + ["</s>"]
        for n in range(self._order, len(sentence)):
            tempdic = self._prob_table.get(" ".join(sentence[n - self._order:n]).lower())
            word_probability = tempdic.get(sentence[n])  
            # it is a sum of log pboabilities
            # we use np.log because it knows that log(0) is float('-inf')
            sentence_probability_sum += np.log(word_probability)
        return sentence_probability_sum