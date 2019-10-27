# —*— encoding=utf-8 —*—
import lemma
import nltk


class TianGaoSmooth():
    ''' This is a new approach for measuring similarity between two texts. It is also a new text classification method.
    usage:
    >>>smoother = TianGaoSmooth(n_grams='the number you want')
    >>>smoother.tiangao_smoothing(train_path='the path of your training set', val_path='the path of your validation set',
    model_path='the path you want to save your model in, often with filename extension".pickle"')
    the next time you don't have to train it again, just use the load_pickle function
    >>>smoother = TianGaoSmooth(n_grams='the number you want')
    >>>smoother.load_pickle(model_path='the path you want to save your model in, often with filename extension".pickle"')
    to test the similarity and do text classification, use the function below:
    >>>smoother.set_perplexity(test_path='the path of your target text')
    and you can clear the model anytime you want using clear_test and clear_train function.
    tgindex can measure the similarity between two texts and 
    the extent of the test text belonging to the training set(here we mainly discuss one single tag as training set)
    the median level is acceptable. When doing classification, the median tag might be the candidate. 
    When doing similarity test, value near 50 is acceptable. 
    >>>smoother.get_tgindex()
    '''
    def __init__(self, n_grams=2):
        self.res = {}
        self.n_grams = n_grams
        self.perp = 1
    def __repr__(self):
        return 'This is the TianGao Smoothing method for {}-grams model.'.format(self.n_grams)


    def tiangao_smoothing(self, train_path, val_path, model_path):
        bigram_dict = nltk.FreqDist(lemma.word_ngram(train_path, self.n_grams))
        val_dict = nltk.FreqDist(lemma.word_ngram(val_path, self.n_grams))
        for key in val_dict.keys():
            if key not in bigram_dict.keys():
                bigram_dict[key] = 0

        inverse_dict = {}

        for e in bigram_dict.keys():
            rawn = bigram_dict[e]
            if rawn not in inverse_dict.keys():
                inverse_dict[rawn] = []
            inverse_dict[rawn].append(e)
        n = sum(bigram_dict.values())
        turing_freq = []
        for val in inverse_dict.keys():
            if val + 1 not in inverse_dict.keys():
                if val == list(inverse_dict.keys())[-1]:
                    turing_freq.append(val)
                else:
                    turing_freq.append(0)
            else:
                new_val = (val+1)*len(inverse_dict[val+1])/ (n * len(inverse_dict[val]))
                turing_freq.append(new_val)

        inverse_dict = dict(zip(turing_freq, inverse_dict.values()))
        total_freq = 0
        for val in inverse_dict.keys():
            total_freq += val * len(inverse_dict[val])
        turing_freq = [i/total_freq for i in turing_freq]
        inverse_dict = dict(zip(turing_freq, inverse_dict.values()))
        for key in inverse_dict.keys():
            for element in inverse_dict[key]:
                self.res[element] = key
        self.save_pickle(model_path)
        return self

    def save_pickle(self, model_path):
    	import pickle
    	f = open(model_path, 'wb')
    	pickle.dump(self.res, f)
    	f.close()
    	return self


    def load_pickle(self, model_path):
    	import pickle
    	f = open(model_path, 'rb')
    	self.res = pickle.load(f)
    	f.close()
    	return self


    def set_perplexity(self, test_path):
        assert self.res != {}, 'Please do the turing smoothing first!'
        lem = lemma.word_ngram(test_path, self.n_grams)
        for gram in lem:
        	if gram not in self.res.keys():
        		self.perp *= (1-max(list(self.res.values()))) ** (-1 / len(lem))
        	else:
        		self.perp *= (1-self.res[gram]) ** (-1 / len(lem))
        print('the perplexity for the {0} is {1}'.format(test_path.split('.')[0], self.perp))
        return self

    def clear_train(self):
    	self.res = {}
    	return self

    def reset_ngrams(self, ngrams):
    	assert ngrams != None, 'please type in the ngrams!'
    	self.n_grams = ngrams
    	return self

    def clear_test(self):
    	self.perp = 1
    	return self


    def get_tgindex(self):
    	print('The TianGao Index is: ', (self.perp-1) * 10000)
    	print('If this index is too high or too low, it means that the model is not suitable for the test set.')
    	print('It might be a good value to stay around 50.')
    	return self


