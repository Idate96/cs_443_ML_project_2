from sentiment import *
from models import MSEModel
import data_utils

def main():
    # hyperparameters
    embedding_dim = 20
    batch_size = 500
    learning_rate = 10**-3
    epochs_num = 100

    # get features and labels of tweets
    embeddings, vocabulary, dataset, labels = data_utils.load_params(use_all_data=True)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    dataloader = data_utils.generate_dataloader(dataset_features, labels, batch_size, shuffle=True)

    # learning algo
    model = MSEModel(embedding_dim, learning_rate)
    train(model, dataloader, epochs_num)


if __name__ == "__main__":
    main()
