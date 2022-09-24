def err_metric(CM): 
    TN, FP, FN, TP = CM.ravel()
    precision =(TP)/(TP+FP)
    accuracy_model  =(TP+TN)/(TP+TN+FP+FN)
    recall_score  =(TP)/(TP+FN)
    specificity_value =(TN)/(TN + FP)
      
    False_positive_rate =(FP)/(FP+TN)
    False_negative_rate =(FN)/(FN+TP)
    f1_score =2*(( precision * recall_score)/( precision + recall_score))
    print("Precision value of the model: ",precision)
    print("Accuracy of the model: ",accuracy_model)
    print("Recall of the model: ",recall_score)
    print("Specificity of the model: ",specificity_value)
    print("F1 Score of the model: ",f1_score)
    return [accuracy_model,f1_score]
