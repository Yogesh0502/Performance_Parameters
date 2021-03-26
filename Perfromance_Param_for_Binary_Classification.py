#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
y_test: Actual Test Response

y_pred: Predicted response
This can be calulated by "model.predict(X_test)"

y_pred_prob: Predicted probabilities of minority class
This can be calculated by "model.predict_probs(X_test)"

'''


def get_best_threshold_using_ROC(y_test,y_pred_prob):
    from sklearn.metrics import roc_curve
    fpr,tpr,threshold=roc_curve(y_test,y_pred_prob)
    # getting the geometric mean for each threshold. Gmeans because TPR and FPR are rates
    gmeans=np.sqrt(tpr*(1-fpr))
    # locate the index of the largest g-mean. It will give us the location of best threshold.
    locate=np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (threshold[locate], gmeans[locate]))
    
    
    
    #plot the ROC curve
    import matplotlib.pyplot
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Fitted Model')
    plt.scatter(fpr[locate], tpr[locate], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()
    
def get_best_threshold_from_PRcurve(y_test,y_pred_prob):
    #Calculate the precison for No skill model. A no skill model will predict all the response as 0.
    ns=len(y_test[y_test==1])/len(y_test)
    
    #Calculate the Precision, recall for different thresholds
    precision,recall,thres=metrics.precision_recall_curve(y_test,y_pred_prob)
    
    #Calucalte the location of threshold which has best F1 score
    fscore=(2*precision*recall)/(precision+recall)
    locate=np.argmax(fscore)
    
    print('Best Threshold=%f, F1_Score=%.3f' % (thres[locate], fscore[locate]))
    
    # plot the roc curve for the model
    plt.plot([0,1], [ns,ns], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.scatter(recall[locate], precision[locate], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    # show the plot
    plt.show()
    
    
    
def get_different_metric_using_new_threshold(y_test,y_pred_prob,thres):
    '''
    Here we can give a threshold value and see how our predicated probabilty of minority class(1) responds on that threshold.
    Mainly we can get the best threshold from get_best_threshold_using_ROC or get_best_threshold_from_PRcurve 
    and then use in this function to the the prefromance metrics.
    '''
    #get the y_pred using new threshold
    y_pred_opt=y_pred_prob>=thres
    from sklearn import metrics
    acc=accuracy_score(y_test,y_pred_opt)
    f1=metrics.f1_score(y_test,y_pred_opt)
    precision=metrics.precision_score(y_test,y_pred_opt)
    recall=metrics.recall_score(y_test,y_pred_opt)
    cm=confusion_matrix(y_test,y_pred_opt)
    print('Accuracy=%f, \nf1 score=%f,\nPrecision Score=%f,\nRecall=%f' %(acc,f1,precision,recall))
    print("Confusion Matrix:\n",cm)
    


# In[ ]:




