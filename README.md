# tiangao-smooth
This is a new approach for measuring text similarity and text classification.
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
