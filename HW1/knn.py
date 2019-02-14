import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    
   # Calculate the L2 distance
    def getL2Distance(pair1, pair2):
        return np.sqrt(np.sum(np.power(np.subtract(pair1,pair2, dtype=int),2)))



    # Get the most common element in a given list
    def getMostCommon(label_list):
        # Initialize a dictionary from a given list.
        # Keys: Set of the list, Values: 0.
        occurance=dict.fromkeys(set(label_list), 0)
        # Count the occurance of each unique element
        for i in label_list:
            occurance[i]+=1
        print("Occurance Dictionary", occurance)
        return max(occurance, key=lambda x: occurance.get(x))
    
    # Predict the label of the input
    def predict(test_data):
        # print("Input", input)
        
        # Calculate the distances between all training data
        distances=[]
        for pair in dataSet:
            distance=getL2Distance(test_data, pair)
            distances.append(distance)

        # Get the indices of the k nearest neighbors in the original list
        indices=np.array(distances).argsort()[:k]
        
        filtered_distances=np.take(distances,indices)
        print("Distances", filtered_distances)
        
        # Get the labels of the k nearest neighbors
        kLabels=np.take(labels,indices)
        print("KLabels", kLabels)
        
        # Get the most common label among the k nearest neighbors
        return getMostCommon(kLabels)
    
    for i in range(len(newInput)):
        prediction = predict(newInput[i])
        print("Prediction", prediction, "Actual", y_test[i])
        result.append(prediction)
        print("")
    
    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))