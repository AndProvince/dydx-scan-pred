HOST = 'https://api.dydx.exchange'
MODEL_FILE = 'model/model_LSTM.pt'
INPUTDIM = 6
HIDDENDIM = 21
SEQLEN = 50
COUNTPRED = 1
BATCHSIZE = 8

DATAMean = []
DATAStd = []
DATAMEAN_FILE = 'model/datamean.csv'
DATASTD_FILE = 'model/datastd.csv'