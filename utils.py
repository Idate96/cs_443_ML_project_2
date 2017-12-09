import numpy as np
import pickle
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
    parsed_dataset = []
    for sentence in dataset:
        parsed_sentence = sentence.replace("\n", "").split(" ")
        parsed_dataset.append(parsed_sentence)
    return parsed_dataset


if __name__ == '__main__':
    load_vocabulary()
    dataset = load_samples('train_neg.txt')
    parse_samples(dataset)