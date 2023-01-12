from dydx3 import Client
from dydx3.constants import MARKET_BTC_USD #MARKET_ETH_USD

import dydx_config
from dydx_config import HOST, MODEL_FILE, SEQLEN, COUNTPRED, DATAMEAN_FILE, DATASTD_FILE, INPUTDIM, HIDDENDIM
from dydx_utils import getCandles, mainWork
import os.path
from model.model import ModelLSTM
from model.model_utils import trainModel
import torch

import csv

CLIENT = Client(HOST)
MARKET = MARKET_BTC_USD
RESOLUTION = '5MINS'

#Try to find exist model
if os.path.isfile(MODEL_FILE):
    res = input('Load exist model? y/n: ')
if res == 'y':
    print('Loading model...')
    model = ModelLSTM(INPUTDIM, HIDDENDIM, SEQLEN, COUNTPRED)
    model.load_state_dict(torch.load(MODEL_FILE))
    print('Loading data parameters...')
    with open(DATAMEAN_FILE, 'r', newline='') as file:
        dydx_config.DATAMean = [list(map(float, row)) for row in csv.reader(file)]
    with open(DATASTD_FILE, 'r', newline='') as file:
        dydx_config.DATAStd = [list(map(float, row)) for row in csv.reader(file)]
    print('Model loaded')
else:
    print('Load actual data...')
    df = getCandles(CLIENT, MARKET, RESOLUTION, history=19)
    print('Train model...')
    model = trainModel(df, INPUTDIM, HIDDENDIM, SEQLEN, COUNTPRED, epochs=160)
    print('Saving model parameters...')
    torch.save(model.state_dict(), MODEL_FILE)
    print('Saving data parameters...')
    with open(DATAMEAN_FILE, 'w', newline='') as file:
        csv.writer(file).writerows(dydx_config.DATAMean)
    with open(DATASTD_FILE, 'w', newline='') as file:
        csv.writer(file).writerows(dydx_config.DATAStd)
    print('Model prepared')

mainWork(model, CLIENT, MARKET, RESOLUTION)