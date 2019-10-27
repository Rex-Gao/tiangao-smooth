def word_ngram(text_path, n_gram=2):
    '''
    Return the list of the selected n-grams.
    '''
	import nltk
	f = open(text_path, encoding='utf-8')
	text = f.read().replace('\n', ' ')
	f.close()
	word_list = [word for word in nltk.word_tokenize(text) if word \
				not in [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']]
	return ['_'.join(word_list[i:i+n_gram]) for i in range(len(word_list)-n_gram+1)]