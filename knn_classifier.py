import numpy as np
from collections import Counter

############# Step 1: Encoding of strings ###############
data = [
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
    ]

label_mapping = {'Apple': 0, 'Banana': 1, 'Orange': 2}
y_encoded = np.array([label_mapping[item[3]] for item in data])

X = np.array([item[:-1] for item in data])  #Features

############ Step 2: Euclidean Distance Function ###########
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

############ Step 3: Implement the KNN Classifier ##############
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])

    def predict_one(self, x):
        # Calculating distances to all training samples
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Finding the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Then finding the labels of the k nearest neighbors using the indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

############### Step 4: Test Your Classifier ################
test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
])

# Create an instance of the classifier with k=3
knn = KNN(k = 3)
# Fit it on the training data
knn.fit(X, y_encoded)
# Get the predicted labels
predictions = knn.predict(test_data)

# Print the predictions
label_mapping_inverse = {v: k for k, v in label_mapping.items()}
predicted_labels = [label_mapping_inverse[pred] for pred in predictions]
print("Predictions:", predicted_labels)

############## Step 5: Evaluation ####################
print("\nEvaluation: ")
expected_labels = ['Banana', 'Apple', 'Orange']
for expected, predicted in zip(expected_labels, predicted_labels):
    print(f"Expected: {expected}, Predicted: {predicted}")
