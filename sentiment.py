import numpy as np

def embed_sentence(sentence, vocabulary, embedding):
    """
    Embeds the sentence using GLOVE
    :param sentence: a list of words (strings)
    :param vocabulary: hash table (dictionary) maps word to an index of word in embedding matrix
    :param embedding: matrix, size = number of words in vocabulary x dimension of embedding space (number of features
    needed to describe a single word)
    :return: embedded matrix describing the sentence. size = #word in sentence x dimension of embedding space
    """
    embedded_sentence = np.zeros(len(sentence), embedding.shape[1])
    for word, i in enumerate(sentence):
        index = vocabulary[word]
        embedded_sentence[i] = embedding[index]
    return embedded_sentence


def extract_features(embedded_sentence):
    """
    Calculates feature vector
    :param embedded_sentence: np array of size: #words in sentence x dimension of embedding space
    :return: feature_vector: np array of size: dimension of embedding space containing means (features)
    """
    feature_vector = np.mean(embedded_sentence, axis=0)
    assert (feature_vector.shape[0] == embedded_sentence.shape[1]), "Shape of feature_vector is not correct"
    assert (feature_vector.shape[1] == 1), "Shape of feature_vector is not correct"
    return feature_vector



