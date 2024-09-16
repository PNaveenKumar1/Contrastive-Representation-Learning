import torch
from argparse import Namespace

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import *
from LogisticRegression.model import SoftmaxRegression as LinearClassifier
from ContrastiveRepresentation.model import Encoder , Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model

import gc

import matplotlib.pyplot as plt

def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''

    gc.collect()
    torch.cuda.empty_cache()

    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y) 

    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    # Convert the data to torch tensors
    X_train = ptu.from_numpy(X_train).float()
    y_train = ptu.from_numpy(y_train).long()
    X_val = ptu.from_numpy(X_val).float()
    y_val = ptu.from_numpy(y_val).long()

    # Create the model

    encoder = Encoder(args.z_dim).to(ptu.device)
    # summary(encoder, (3, 32, 32))
    if args.mode == 'fine_tune_linear':
        classifier = LinearClassifier(args.z_dim)
    elif args.mode == 'fine_tune_nn':
        classifier = Classifier(args.z_dim)# TODO: Create the neural network classifier model
        classifier = classifier.to(ptu.device)
    
    if args.mode == 'cont_rep':

        print("Fitting contrastive model")
        losses = fit_contrastive_model(encoder, X_train, y_train, args.num_iters, args.batch_size, args.lr)
        print("Contrastive model fitted")
        
        print("Saving encoder")
        torch.save(encoder.state_dict(), './models/encoder.pth')
        print("Encoder saved")
        
        # raise NotImplementedError('Implement the contrastive representation learning')

        plt.plot(losses)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Contrastive Representation Learning Loss')
        plt.savefig(f'images/2.2.png')

        # Plot the t-SNE after fitting the encoder

        print("Plotting t-SNE")
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            X_batch = ptu.to_numpy(encoder(X_batch))
            if i==0:
                temp = X_batch
            else:
                temp = np.concatenate((temp, X_batch), axis=0)
          
        z = temp
       
        plot_tsne(z, y_train)


        print("t-SNE plotted")

        # save the encoder
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        # raise NotImplementedError('Load the encoder')
        encoder.load_state_dict(torch.load('./models/encoder.pth'))

        encoder = Encoder(args.z_dim).to(ptu.device)
        gc.collect()
        torch.cuda.empty_cache()

        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        
        # Plot the losses
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()

        # Save the predictions for the test data in a CSV file
        y_preds = []
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i+args.batch_size].to(ptu.device)
            repr_batch = encoder(X_batch)
            if 'linear' in args.mode:
                repr_batch = ptu.to_numpy(repr_batch)
            y_pred_batch = classifier(repr_batch)
            if 'nn' in args.mode:
                y_pred_batch = ptu.to_numpy(y_pred_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv',\
                y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv')
