import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import time

from dydx3 import Client
import dydx3.constants

import dydx_config
from dydx_config import SEQLEN, COUNTPRED, INPUTDIM, HIDDENDIM
from model.model_utils import trainModel

def getCandles(client: Client, market: dydx3.constants, resolution: str, history=0) -> pd.DataFrame:
    """
    Get candles from dy/dx client (client) for market (market) with resolution (resolution)
    and history depth plus for first 100 items ( + 100 items * history)
    :return: DataFrame with columns ['low', 'high', 'open', 'close', 'baseTokenVolume', 'trades']
    and data in float format
    """
    #Get first 100 items
    candles = client.public.get_candles(market=market,
                                        resolution=resolution,
                                        )
    stop = candles.data['candles'][-1]['startedAt']
    df = pd.DataFrame(candles.data['candles'])

    #Add history items
    for i in range(history):
        candles = client.public.get_candles(market=market,
                                            resolution=resolution,
                                            to_iso=stop,
                                            )
        stop = candles.data['candles'][-1]['startedAt']
        df = pd.concat([df, pd.DataFrame(candles.data['candles'])])

    #Prepare results: save the required columns, convert data format
    df = df[['low', 'high', 'open', 'close', 'baseTokenVolume', 'trades']][::-1]
    df = df.astype({'low': 'float64'})
    df = df.astype({'high': 'float64'})
    df = df.astype({'open': 'float64'})
    df = df.astype({'close': 'float64'})
    df = df.astype({'baseTokenVolume': 'float64'})
    df = df.astype({'trades': 'float64'})
    df = df.reset_index().drop('index', axis=1)
    return df

def mainWork(model, client: Client, market: dydx3.constants, resolution: str):
    needRetrain = False
    alreadyBuy = False
    account = 0

    koeff = 0.002
    iteration = 1
    sleep = 60
    koeffLoss = 0.002

    while True:
        candles = getCandles(client, market, resolution)

        cData = np.array(candles[-SEQLEN:])
        trd = client.public.get_trades(market=market)
        true = float(pd.DataFrame(trd.data['trades'])[0:2]['price'].astype({'price': 'float64'}).mean())

        cData = Variable(torch.from_numpy((cData - dydx_config.DATAMean) / dydx_config.DATAStd).float())
        cData = cData[None, :]
        pred = int(model(cData))

        upTriger = pred > true

        # Оцениваем необходимость размещения ордера - разница между реальным и предсказанным значением должна быть меншье 2*ожидание отклонения
        if abs(true - pred) < 2 * (true * koeff) and not alreadyBuy:
            sleep = 60
            # Оцениваем возможность разместить ордер
            if abs(true - pred) > (true * koeff):
                alreadyBuy = not alreadyBuy
                koeff = 0.002
                iteration = 1
                koeffLoss = 0.002

                if upTriger:
                    #TODO make real order
                    # Размещение ордера на покупку
                    print('NN buy ', true, '->', abs(true - pred))
                    order_cost = true
                    account -= true
                    print('NN acc ', account)
                else:
                    # TODO make real order
                    # Размещение ордера на продажу
                    print('NN sell ', true, '->', abs(true - pred))
                    order_cost = -true
                    account += true
                    print('NN acc ', account)

        # В случае большего ожидания отклонения
        elif not alreadyBuy:
            # При повторном слишком отличающемся предсказании - доучиваем модель
            if needRetrain:
                trainModel(getCandles(client, market, resolution),
                           INPUTDIM, HIDDENDIM, SEQLEN, COUNTPRED,
                           model=model,
                           epochs=40,
                           learningRate=0.0001)
            needRetrain = True
            sleep = 150

        # Если покупка уже совершена
        if alreadyBuy:
            sleep = 15

            # Понижаем коэффицент ожидания прибыли каждые 5 минут
            if not (iteration % 20):
                koeff *= 0.7

            if order_cost > 0 and (true - order_cost) > (order_cost * koeff):
                alreadyBuy = not alreadyBuy
                # TODO make real order
                print('NN sell ', true, iteration, koeff, (true - order_cost))
                account += true
                print('NN acc ', account)
                sleep = 60
                needRetrain = False
            elif order_cost < 0 and (abs(order_cost) - true) > (abs(order_cost) * koeff):
                alreadyBuy = not alreadyBuy
                # TODO make real order
                print('NN buy ', true, iteration, koeff, (abs(order_cost) - true))
                account -= true
                print('NN acc ', account)
                sleep = 60
                needRetrain = False

            if not (iteration % 80):
                koeffLoss *= 20

            elif (iteration > 60 and abs((abs(order_cost) - true)) < (abs(order_cost) * koeffLoss)):
                if order_cost > 0:
                    alreadyBuy = not alreadyBuy
                    # TODO make real order
                    print('NN sell ', true, iteration, koeff, (true - order_cost))
                    account += true
                    print('NN acc ', account)
                    sleep = 60
                    koeffLoss /= 20
                    if needRetrain:
                        trainModel(getCandles(client, market, resolution),
                                   INPUTDIM, HIDDENDIM, SEQLEN, COUNTPRED,
                                   model=model,
                                   epochs=40,
                                   learningRate=0.0001)
                    needRetrain = True
                else:
                    alreadyBuy = not alreadyBuy
                    # TODO make real order
                    print('NN buy ', true, iteration, koeff, (abs(order_cost) - true))
                    account -= true
                    print('NN acc ', account)
                    sleep = 60
                    koeffLoss /= 20
                    if needRetrain:
                        trainModel(getCandles(client, market, resolution),
                                   INPUTDIM, HIDDENDIM, SEQLEN, COUNTPRED,
                                   model=model,
                                   epochs=40,
                                   learningRate=0.0001)
                    needRetrain = True

            iteration += 1

        print(abs(true - pred), '|', true, '|', pred)
        time.sleep(sleep)