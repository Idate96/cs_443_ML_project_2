import numpy as np
import torch
from torch import nn
import data_utils
import torch.utils.data
import torch.nn.functional as f
from torch.autograd import Variable

def embed_sentence(sentence, vocabulary, embedding):
    """
    Embeds the sentence using GLOVE
    :param sentence: a list of words (strings)
    :param vocabulary: hash table (dictionary) maps word to an index of word in embedding matrix
    :param embedding: matrix, size = number of words in vocabulary x dimension of embedding space (number of features
    needed to describe a single word)
    :return: embedded matrix describing the sentence. size = #word in sentence x dimension of embedding space
    """
    embedded_sentence = np.zeros((len(sentence), embedding.shape[1]))
    for i, word in enumerate(sentence):
        try:
            index = vocabulary[word]
        except KeyError as e:
            continue
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
    return feature_vector


def embed_dataset(dataset, vocabulary, embeddings):
    embedded_dataset = []
    for sentence in dataset:
        embedded_dataset.append(embed_sentence(sentence, vocabulary, embeddings))
    return embedded_dataset


def compute_dataset_features(dataset, vocabulary, embeddings):
    dataset_features = np.zeros((len(dataset), np.shape(embeddings)[1]))
    for i, sentence in enumerate(dataset):
        dataset_features[i] = extract_features(embed_sentence(sentence, vocabulary, embeddings))
    return dataset_features


def train(model, dataloader, epochs=10):
    iter_counter = 0
    current_loss = 0
    model.add_optimizer()

    for epoch in range(epochs):
        for x, labels in dataloader:
            if len(x) != dataloader.batch_size:
                continue

            x = Variable(x, requires_grad=False)
            labels = Variable(labels.type(torch.FloatTensor), requires_grad=False)

            model.optimizer.zero_grad()

            output = model(x)
            current_loss = model.loss(output, labels)

            current_loss.backward()
            model.optimizer.step()

            iter_counter += 1

        train_accuracy, train_loss = test(model, dataloader)
        print("epoch: {0}, loss: {1}, accuracy: {2}" .format(epoch, train_loss,
                                                             train_accuracy))


def test(model, dataloader):
    dataset = Variable(dataloader.dataset.data_tensor, requires_grad=False)
    target = Variable(dataloader.dataset.target_tensor.type(torch.FloatTensor), requires_grad=False)
    output = model(dataset)
    accuracy = compute_accuracy(predict(output.data).numpy(), target.data.numpy())
    loss = model.loss(output, target)
    return accuracy, loss.data.numpy()


def compute_accuracy(predictions, target):
    correct = predictions.ravel() == target.ravel()
    return np.sum(correct)/np.shape(target)[0]

def predict(output):
    return output > 0.5


if __name__ == '__main__':
    embeddings, vocabulary, dataset, labels = data_utils.load_params()
    # embedded_dataset = embed_dataset(dataset, vocabulary, embeddings)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    print(dataset_features.shape)
    print(labels.shape)
    dataloader = generate_dataloader(dataset_features, labels, 100)