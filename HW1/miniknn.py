import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################
     # Calculate the L2 distance
    def getL2Distance(pair1, pair2):
        return np.sqrt(np.sum(np.power(np.subtract(pair1,pair2),2)))

    # Get the most common element in a given list
    def getMostCommon(labels):
        # Initialize a dictionary from a given list.
        # Keys: Set of the list, Values: 0.
        occurance=dict.fromkeys(set(labels),0)
        # Count the occurance of each unique element
        for i in labels:
            occurance[i]+=1
        print("Occurance Dictionary", occurance)
        return max(occurance, key=lambda x: occurance.get(x))
    
    # Predict the label of the input
    def predict(input):
        print("Input", input)
        
        # Calculate the distances between all training data
        distances=[]
        for pair in dataSet:
            distance=getL2Distance(input, pair)
            distances.append(distance)
        print("Distance", distances)

        # Get the indices of the k nearest neighbors in the original list
        indices=np.array(distances).argsort()[:k]
        print("Indices", indices)
        
        # Get the labels of the k nearest neighbors
        kLabels=labels[indices]
        print("KLabels", kLabels)
        
        # Get the most common label among the k nearest neighbors
        return getMostCommon(kLabels)
    
    for input in newInput:
        prediction = predict(input)
        print("Prediction", prediction)
        result.append(prediction)
        print("")
    
    
    ####################
    # End of your code #
    ####################
    return result

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,4)

print ('random test points are:', mini_test)
print ('knn classfied labels for test:', outputlabels)

# plot train data and classfied test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

#save diagram as png file
plt.savefig("miniknn.png")