import numpy as np
import pickle
from random import shuffle
import random
import torch
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
    # indeces = np.arange(0, len(labels))
    # np.random.shuffle(indeces)
    # shuffled_labels = labels[indeces]
    # shuffled_dataset = []
    # for i in range(len(dataset)):
    #     shuffled_dataset.append(dataset[i])
    #
    # return shuffled_dataset, shuffled_labels
    return dataset, labels

def load_params(embedding_dim=20, use_all_data=False, directory='twitter-datasets'):
    """
    Load parameter necessary for training.
    :param use_all_data (bool)
    :param directory: directory of data
    :return: embeddings, vocabulary, dataset, labels
    """
    embeddings = np.load('embeddings_' + str(embedding_dim) + '.npy')
    with open(directory + '/' + 'vocab.pkl', 'rb') as file:
        vocabulary = pickle.load(file)
    dataset, labels = load_dataset(use_all_data)
    return embeddings, vocabulary, dataset, labels

def save_embedded_dataset(dataset, labels, filename='00',):
    directory = 'saved_models'
    with open(directory + '/' + 'dataset_' + filename + '.pkl', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    with open(directory + '/' + 'labels_' + filename + '.pkl', 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

def load_embedded_dataset(filename='00'):
    directory = 'saved_models'
    with open(directory + '/' + 'dataset_' + filename + '.pkl', 'rb') as f:
        dataset = pickle.load(f)

    with open(directory + '/' + 'labels_' + filename + '.pkl', 'rb') as f:
        labels = pickle.load(f)

    return dataset, labels


def generate_dataloader(dataset_features, labels, batch_size, shuffle=False):
    tensor_features = torch.FloatTensor(dataset_features)
    labels_tensor = torch.FloatTensor(labels)
    tensor_dataset = torch.utils.data.TensorDataset(tensor_features, labels_tensor)
    dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    load_vocabulary()
    dataset = load_samples('train_neg.txt')
    parse_samples(dataset)
    embeddings, vocabulary, dataset, labels = load_params()