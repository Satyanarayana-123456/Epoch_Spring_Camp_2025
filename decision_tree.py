import numpy as np

############## Step 1: Encode the Dataset ###############
data = [
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
    ]

# Convert to numpy arrays
data = np.array(data)
X = data[:, :-1].astype(float)  # Features

# Encode labels into integers
label_mapping = {'Beer': 0, 'Wine': 1, 'Whiskey': 2}
y_encoded = np.array([label_mapping[item[3]] for item in data])

############# Step 2: Implement Gini Impurity ####################
def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    class_counts = np.bincount(labels)
    probabilities = class_counts/len(labels)
    return 1 - np.sum(probabilities ** 2)

############# Step 3: Implement the Best Split Finder ##################
def best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    n_samples, n_features = X.shape

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            left_indices = X[:, feature_index] < threshold
            right_indices = X[:, feature_index] >= threshold
            
            if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                continue
            
            left_labels = y[left_indices]
            right_labels = y[right_indices]
            
            gini_left = gini_impurity(left_labels)
            gini_right = gini_impurity(right_labels)
            weighted_gini = (len(left_labels)*gini_left + len(right_labels)*gini_right)/n_samples
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

############# Step 4: Implement Recursive Tree Building ###############
class Node:
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, value = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth = 0, max_depth = None, min_samples_split = 2):
    n_samples, n_features = X.shape
    unique_labels = np.unique(y)

    # Stopping conditions
    if len(unique_labels) == 1:
        return Node(value=unique_labels[0])
    if max_depth is not None and depth >= max_depth:
        return Node(value = np.bincount(y).argmax())
    if n_samples < min_samples_split:
        return Node(value = np.bincount(y).argmax())

    feature_index, threshold = best_split(X, y)
    if feature_index is None:
        return Node(value = np.bincount(y).argmax())

    left_indices = X[:, feature_index] < threshold
    right_indices = X[:, feature_index] >= threshold

    left_node = build_tree(X[left_indices], y[left_indices], depth + 1, max_depth, min_samples_split)
    right_node = build_tree(X[right_indices], y[right_indices], depth + 1, max_depth, min_samples_split)

    return Node(feature_index, threshold, left_node, right_node)

############### Step 5: Implement Prediction #############
def predict(sample, node):
    if node.value is not None:
        return node.value
    if sample[node.feature_index] < node.threshold:
        return predict(sample, node.left)
    else:
        return predict(sample, node.right)

############## Step 6: Evaluation #################
tree = build_tree(X, y_encoded)

test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
])

# Reverse mapping for prediction
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Predict
predicted_indices = [predict(sample, tree) for sample in test_data]
predicted_labels = [reverse_label_mapping[i] for i in predicted_indices]

# Output results
for i, label in enumerate(predicted_labels):
    print(f"Sample {i+1}: Predicted Class -> {label}")
