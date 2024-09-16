import torch
import torch.nn as nn
from argparse import Namespace
from typing import Union, Tuple, List

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy

import gc
import numpy as np
import matplotlib.pyplot as plt


def calculate_loss(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        loss: float, loss of the model
    '''
    # raise NotImplementedError('Calculate negative-log-likelihood loss here')
    criterion = nn.NLLLoss()
    loss = criterion(nn.functional.log_softmax(y_logits, dim=1), y)
    return loss.item()


def calculate_accuracy(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    
    # raise NotImplementedError('Calculate accuracy here')
    y_pred = torch.argmax(y_logits, dim=1)

    acc = (y_pred == y).float().mean()
    return acc.item()



def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3
) -> None:
    '''
    Fit the contrastive model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - X: torch.Tensor, features
    - y: torch.Tensor, labels
    - num_iters: int, number of iterations for training
    - batch_size: int, batch size for training

    Returns:
    - losses: List[float], list of losses at each iteration
    '''
    # TODO: define the optimizer for the encoder only

    # TODO: define the loss function

    losses = []
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    

    encoder.train()

    for i in range(num_iters):
        # get the batch data
        y_num = ptu.to_numpy(y)
        X_num = ptu.to_numpy(X)
       

        X_a,X_p,X_n = next(get_contrastive_data_batch(X_num, y_num, batch_size))

        X_a = ptu.from_numpy(X_a).float().to(ptu.device)
        X_p = ptu.from_numpy(X_p).float().to(ptu.device)
        X_n = ptu.from_numpy(X_n).float().to(ptu.device)
            
        

        proj_a = encoder(X_a)
        proj_p = encoder(X_p)
        proj_n = encoder(X_n)

        # zero the gradients
        optim.zero_grad()

        loss = triplet_loss(proj_a, proj_p, proj_n)

        loss.backward()

        optim.step()

        if i % 10 == 0:
            # append loss
            losses.append(loss.item())
            
            print(f'Iter {i}/{num_iters} - Train Loss: {loss.item():.4f}')


        # raise NotImplementedError('Write the contrastive training loop here')
    
    return losses


def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X: torch.Tensor, images
    - y: torch.Tensor, labels
    - batch_size: int, batch size for evaluation
    - is_linear: bool, whether the classifier is linear

    Returns:
    - loss: float, loss of the model
    - acc: float, accuracy of the model
    '''
    # raise NotImplementedError('Get the embeddings from the encoder and pass it to the classifier for evaluation')

    # HINT: use calculate_loss and calculate_accuracy functions for NN classifier and calculate_linear_loss and calculate_linear_accuracy functions for linear (softmax) classifier
    encoder.eval()
    classifier.eval()

    temp = []
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size].to(ptu.device)
        X_batch = encoder(X_batch)
        temp.append(X_batch)
    
    X = torch.cat(temp)

    if(is_linear):
        X = ptu.to_numpy(X)
    
    y_preds = classifier(X)
    if(is_linear):
        return calculate_linear_loss(y_preds, y), calculate_linear_accuracy(y_preds, y)
    else:
        return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)

    # return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
    Fit the model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X_train: torch.Tensor, training images
    - y_train: torch.Tensor, training labels
    - X_val: torch.Tensor, validation images
    - y_val: torch.Tensor, validation labels
    - args: Namespace, arguments for training

    Returns:
    - train_losses: List[float], list of training losses
    - train_accs: List[float], list of training accuracies
    - val_losses: List[float], list of validation losses
    - val_accs: List[float], list of validation accuracies
    '''
    if args.mode == 'fine_tune_linear':
        gc.collect()
        torch.cuda.empty_cache()

        batch_size = 256

        ans = []
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            X_batch_projection = encoder(X_batch)
            X_batch_projection = ptu.to_numpy(X_batch_projection)
            ans.append(X_batch_projection)


       
        X_valid_projections = ptu.to_numpy(encoder(X_val))
        y_train = ptu.to_numpy(y_train)
        y_valid = ptu.to_numpy(y_val)
        X_train_projections = np.concatenate(ans)


        train_losses,train_accs,val_losses,val_accs = fit_linear_model(
        classifier, X_train_projections, y_train, X_valid_projections, y_valid, num_iters=args.num_iters,
        lr=args.lr, batch_size= batch_size, l2_lambda=args.l2_lambda,
        grad_norm_clip=args.grad_norm_clip, is_binary=args.mode == 'logistic')



        

        return train_losses, train_accs, val_losses, val_accs
        # raise NotImplementedError('Get the embeddings from the encoder and use already implemeted training method in softmax regression')
    else:
        # TODO: define the optimizer
        encoder.train()
        classifier.train()

        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []


        # use both classifier.paramters and encoder.parameters
        optim = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # batch_size = 2048 #args.batch_size
        # num_iters = = 5000#args.num_iters
        

        for i in range(args.num_iters):
            X_batch,y_batch = get_data_batch(X_train, y_train, args.batch_size)

            # zero the gradients
            X_train_batch_projections = encoder(X_batch)

            y_preds = classifier(X_train_batch_projections)

            optim.zero_grad()

            loss = criterion(y_preds, y_batch)
            acc = calculate_accuracy(y_preds, y_batch)

            loss.backward()
            optim.step()

            if i % 10 == 0:
                # append loss and accuracy

                train_losses.append(loss.item())
                train_accs.append(acc)

                val_loss, val_acc = evaluate_model(encoder, classifier, X_val, y_val, args.batch_size)

                val_losses.append(val_loss)
                val_accs.append(val_acc)


                print(f'Iter {i}/{args.num_iters} - Train Loss: {loss:.4f} - Train Acc: {acc:.4f}'
                      f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
            

        

        plt.plot(train_losses)
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.title('Fine Tuning NN Training Loss')
        plt.savefig(f'images/2.5.png')

        # raise NotImplementedError('Write the supervised training loop here')
        # return the losses and accuracies both on training and validation data

        return train_losses, train_accs, val_losses, val_accs
