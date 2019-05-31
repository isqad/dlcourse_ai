from collections import Counter

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    s  = ground_truth.shape[0]
    
    for i in range(s):
        if prediction[i] == True and ground_truth[i] == True: tp += 1
        if prediction[i] == False and ground_truth[i] == False: tn += 1
        if prediction[i] == False and ground_truth[i] == True: fn += 1
        if prediction[i] == True and ground_truth[i] == False: fp += 1
            
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2*precision*recall + 1) / (precision + recall + 1)

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    s  = ground_truth.shape[0]
    tp = 0
    for i in range(s):
        if prediction[i] == ground_truth[i]: tp += 1
    return tp / s
