from sentiment import *
from models import *
import data_utils


def main():
    # hyperparameters
    embedding_dim = 20
    batch_size = 500
    learning_rate = 10**-3
    epochs_num = 2

    # get features and labels of tweets
    embeddings, vocabulary, dataset, labels = data_utils.load_params(embedding_dim,
                                                                     use_all_data=False)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    dataloader_train, dataloader_val = data_utils.generate_dataloader(dataset_features, labels,
                                                                    batch_size)

    # learning algorithm
    model = LinearBCEModel(embedding_dim, learning_rate)
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(model, dataloader_train,
                                                                                               dataloader_val, epochs_num)

    # plots
    data_utils.plot_convergence(epochs_num, train_loss_history, val_loss_history, model.name)
    data_utils.plot_accuracy(epochs_num, train_accuracy_history, val_accuracy_history, model.name)

    # save data
    data_utils.save_data(epochs_num, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history,
                         model.name)

    # create submission
    data_utils.create_csv_submission(model, dataset_features, model.name)


if __name__ == "__main__":
    main()
