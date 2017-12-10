from sentiment import *
from models import BCEModel
import data_utils

def main():
    # hyperparameters
    embedding_dim = 20
    batch_size = 500
    learning_rate = 10**-3
    epochs_num = 100

    # get features and labels of tweets
    embeddings, vocabulary, dataset, labels = data_utils.load_params(embedding_dim,
                                                                     use_all_data=False)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    dataloader_train, dataloader_val = data_utils.generate_dataloader(dataset_features, labels,
                                                                    batch_size)

    # learning algo
    model = BCEModel(embedding_dim, learning_rate)
    train(model, dataloader_train, dataloader_val, epochs_num)


if __name__ == "__main__":
    main()
