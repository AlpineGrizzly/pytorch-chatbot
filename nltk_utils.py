import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer() # Create a word stemmer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    bag_of_words Create a bag of words from a tokenized sentence
    :tokenized_sentence: input for our bag of words function
    
    :all_words: All words that are trained into our model

    return: Return the bag of words
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence] # Stem all the token words
    
    bag = np.zeros(len(all_words), dtype=np.float32) # initialize our bag of words

    # Check for the existence of a token word in our word corpus
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1 # 

    return bag
    