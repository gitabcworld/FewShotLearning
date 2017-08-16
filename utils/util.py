import torch
import numpy as np

def createDictLabels(labels):

    """
    Creates dictionaries that fits data with non-sequential labels into a 
    sequential order label from [0...nClasses].
    :param labels: all the non-sequential labels
    :return: dict that converts from non-sequential to sequential, 
             dict that converts from sequential to non-sequential
    """

    # Re-arange the Target vectors between [0..nClasses_train]
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    dictLabels = {val: i for i, val in enumerate(unique_labels)}
    dictLabelsInverse = {i: val for i, val in enumerate(unique_labels)}
    return dictLabels,dictLabelsInverse


def fitLabelsToRange(dictLabels,labels):

    """
    Converts Tensor values to the values contained in the dictionary 
    :param dictLabels: dictionary with the conversion values
    :param labels: Tensor to convert
    :return: Tensor with the converted labels.
    """
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    labels_temp = np.array(labels)
    for i in dictLabels.keys():
        labels_temp[labels == i] = dictLabels[i]
    labels = labels_temp
    return torch.from_numpy(labels)
