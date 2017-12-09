import numpy as np
import pickle
from random import shuffle
import random
random.seed = 10
import data_utils
# encoding=utf8
import sys


def load_vocabulary(directory='twitter-datasets'):
    with open(directory + '/' + 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_samples(filename, directory='twitter-datasets'):
    with open(directory + '/' + filename, encoding="utf-8") as file:
        dataset = file.readlines()
    return dataset


def parse_samples(dataset):
    """
    Split sentences into list of words
    :param dataset: list of sentences
    :return: parsed_dataset: a list of list of words
    """
    parsed_dataset = []
    for sentence in dataset:
        parsed_sentence = sentence.replace("\n", "").split(" ")
        parsed_dataset.append(parsed_sentence)
    return parsed_dataset


def load_dataset(use_all_data=False):
    """
    Loads the dataset and generate corresponding labels, then it shuffles them
    :param use_all_data (bool) : indicates if all the dataset is used
    :return: shuffled_dataset (list) : dataset containing the sentences
            shuffled_lables (np.array): labels of the dataset,
                                        0 relate to negative and 1 to positive
    """

    if use_all_data:
        filename_pos = 'train_pos_full.txt'
        filename_neg = 'train_neg_full.txt'
    else:
        filename_pos = 'train_pos.txt'
        filename_neg = 'train_neg.txt'

    dataset_pos = parse_samples(load_samples(filename_pos))
    labels_pos = np.ones(len(dataset_pos))
    dataset_neg = parse_samples(load_samples(filename_neg))
    labels_neg = np.zeros(len(dataset_neg))
    # join
    dataset = dataset_pos + dataset_neg
    labels = np.vstack((labels_pos, labels_neg)).flatten()
    # mix to train better the classifier
    indeces = np.arange(0, len(labels))
    np.random.shuffle(indeces)
    shuffled_labels = labels[indeces]
    shuffled_dataset = []
    for i in range(len(dataset)):
        shuffled_dataset.append(dataset[i])

    return shuffled_dataset, shuffled_labels

def load_params(use_all_data=False, directory='twitter-datasets'):
    """
    Load parameter necessary for training.
    :param use_all_data (bool)
    :param directory: directory of data
    :return: embeddings, vocabulary, dataset, labels
    """
    embeddings = np.load('embeddings.npy')
    with open(directory + '/' + 'vocab.pkl', 'rb') as file:
        vocabulary = pickle.load(file)
    dataset, labels = load_dataset(use_all_data)
    return embeddings, vocabulary, dataset, labels


if __name__ == '__main__':
    load_vocabulary()
    dataset = load_samples('train_neg.txt')
    parse_samples(dataset)
    embeddings, vocabulary, dataset, labels = load_params()