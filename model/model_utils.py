import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import dydx_config
from model.model import ModelLSTM

class LinearDataset(Dataset):
    """
    Dataset for correct work with DataLoaders
    """
    def __init__(self, XData, yData=None, mode='train'):
        super().__init__()
        self.X = XData
        self.y = yData
        self.mode = mode

        self.len_ = XData.shape[0]

    def __len__(self):
        return self.len_

    def __getitem__(self, index):
        x = self.X[index]

        if self.mode == 'test':
            return x
        else:
            y = self.y[index]
            return x, y

def transformData(dataset, target, seqLen, countPred=1, indArray=[]):
    '''
    Create train/val and target data (np.array) from line of continuous measurements
    with length =seqLen for train/val data and length =countPred for target
    :param dataset: continuous measurements
    :param target: continuous measurements
    :param seqLen: int, length for train/val data
    :param countPred: int, length for target data
    :param indArray: list of first index for train/val data in dataset
    :return: train/val data, target data
    '''
    x, y = [], []

    for i in indArray:
        x_i = dataset[i : i + seqLen]
        y_i = target[i + seqLen : i + seqLen + countPred]

        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y)

def train(trainLoader, valLoader, model, criterion, optimizer, scheduler, epochs):
    '''
    Train loop for model (model) at epochs (epochs) with criterion (criterion) for calc loss
    and optimaze model parameters (optimazer) with step size changed by scheduler (scheduler)
    :param trainLoader: loader for train data
    :param valLoader: loader for val data
    :param model: trainable model
    :param criterion: criterion for calc loss
    :param optimizer: optimizer for change model parameters
    :param scheduler: scheduler for change optimizer step size
    :param epochs: count epoch for train model
    :return: None
    '''
    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0
        processedSize = 0

        for inputs, labels in trainLoader:
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = model(inputs)

            # get loss for the predicted output
            loss = criterion(outputs, labels)

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            runningLoss += loss.item() * inputs.size(0)
            processedSize += inputs.size(0)

        train_loss = runningLoss / processedSize
        scheduler.step()

        # get loss for the val_data output
        model.eval()
        runningLoss = 0.0
        processedSize = 0

        for inputs, labels in valLoader:
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            runningLoss += loss.item() * inputs.size(0)
            processedSize += inputs.size(0)

        val_loss = runningLoss / processedSize
        print('Epoch %d. \nTrain loss: %f \nVal loss: %f' % (epoch + 1, train_loss, val_loss))

def prepareData(data, seqLen, countPred) -> (DataLoader, DataLoader):
    '''
    Take data in DataFrame format, return data in two DataLoaders: train and validate
    :param data: input data in DataFrame
    :param seqLen: length of data sequence to make predict
    :param countPred: length of predict data
    :return: test DataLoader, validate DataLoader
    '''
    X = data[['low', 'high', 'open', 'close', 'baseTokenVolume', 'trades']]
    y = data['close']
    indTrain, indVal = train_test_split(range(y.shape[0] - seqLen - countPred),
                                          train_size=0.8,
                                          random_state=42)
    XTrain, yTrain = transformData(X, y, seqLen, countPred, indTrain)
    XVal, yVal = transformData(X, y, seqLen, countPred, indVal)

    if dydx_config.DATAMean == []:
        dydx_config.DATAMean = XTrain.mean(axis=0)
    if dydx_config.DATAStd == []:
        dydx_config.DATAStd = XTrain.std(axis=0)

    XTrainScaled = Variable(torch.from_numpy((XTrain - dydx_config.DATAMean ) / dydx_config.DATAStd).float())
    XValScaled = Variable(torch.from_numpy((XVal - dydx_config.DATAMean ) / dydx_config.DATAStd).float())

    yTrain = Variable(torch.from_numpy(yTrain).float())
    yVal = Variable(torch.from_numpy(yVal).float())

    trainDataset = LinearDataset(XTrainScaled, yTrain, mode='train')
    valDataset = LinearDataset(XValScaled, yVal, mode='val')

    BATCHSIZE = 8

    trainLoader = DataLoader(trainDataset, batch_size=BATCHSIZE, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=BATCHSIZE, shuffle=False)

    return trainLoader, valLoader

def initWeights(model):
    '''
    Function for initialize model parameters weigth
    :param model: input model
    :return: None
    '''
    for name, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def trainModel(data, inputDim, hidenDim, seqLen, outputDim, model=None, epochs=160, learningRate=0.001) -> ModelLSTM:
    '''
    Train or retrain model if it necessary
    :param data: input data for train model
    :param inputDim: input dimension for nn model
    :param hidenDim: hiden dimension for nn model
    :param seqLen: length of data sequence to make predict
    :param outputDim: length of predict data
    :param model: give model if necessary retrain, None if need new model
    :param epochs: count epoch to train model
    :param learningRate: starting learning rate to train model
    :return: trained model
    '''
    trainLoader, valLoader = prepareData(data, seqLen, outputDim)

    if not model:
        model = ModelLSTM(inputDim, hidenDim, seqLen, outputDim)
        model.apply(initWeights)

    criterion = torch.nn.L1Loss()  # torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
    expScheduler = lr_scheduler.StepLR(optimizer, step_size=int(epochs / 4), gamma=0.1)

    train(trainLoader, valLoader, model, criterion, optimizer, expScheduler, epochs)

    return model
