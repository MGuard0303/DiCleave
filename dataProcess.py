import torch
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np


# This function generates one_hot_encoding from input sequences
# After transforming the sequences, the function reshapes them into a proper shape
def one_hot_encoding(dataframe, column, token):
    sigma = token
    temp = []

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(sigma)

    for item in dataframe[column]:
        integer_seq = label_encoder.transform(list(item))
        temp.append(integer_seq)

    temp = np.array(temp)  # Convert python list to ndarray to speed up
    integer_tensor = torch.tensor(temp)
    one_hot_tensor = F.one_hot(integer_tensor)
    one_hot_tensor = torch.transpose(one_hot_tensor, 1, 2)
    one_hot_tensor = one_hot_tensor.type(torch.float)    # Sigmoid output layer requires Float type

    return one_hot_tensor
