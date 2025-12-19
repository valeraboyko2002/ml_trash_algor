import numpy as np 

class DecisionTreeClassifier:

    def __init__(self, max_depth = None):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self,X,y):
        self.tree = self.build_tree(X,y)
        return self
    
    def build_tree(self,X,y,depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)


        if len(unique_classes)==1 or (self.max_depth is not None and depth >= self.max_depth):
            most_common_class = self.majority_class(y)
            return most_common_class
        best_feature, best_threshold = self.best_split(X,y,num_features)
        if best_feature is None:
            most_common_class = self.majority_class(y)
            return most_common_class
        left_indices = X[:,best_feature] < best_threshold
        right_indices = X[:,best_feature] >= best_threshold
        left_subtree = self.build_tree(X[left_indices],y[left_indices],depth+1)
        right_subtree = self.build_tree(X[right_indices],y[right_indices],depth+1)
        return (best_feature,best_threshold,left_subtree,right_subtree)
    
    def best_split(self,X,y,num_features):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(num_features):
            thresholds = np.unique(X[:,feature])
            for threshold in thresholds:
                gini = self.gini_index(X,y,feature,threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def gini_index(self,X,y,feature,threshold):
        left_indices = X[:,feature] < threshold
        right_indices = X[:,feature] >= threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return float('inf')
        gini_left = 1.0 - sum((np.sum(y[left_indices] == c) / len(y[left_indices]))**2 for c in np.unique(y))
        gini_right = 1.0 - sum((np.sum(y[right_indices] == c) / len(y[right_indices]))**2 for c in np.unique(y))
        weighted_gini = (len(y[left_indices]) * gini_left + len(y[right_indices]) * gini_right) / len(y)
        return weighted_gini
    
    def majority_class(self,y):
        values, counts = np.unique(y, return_counts=True)
        majority_class = values[np.argmax(counts)]
        return majority_class
    
    def predict(self,X):
        predictions = [self.predict_sample(sample,self.tree) for sample in X]
        return np.array(predictions)
    
    def predict_sample(self,sample,tree):
        if not isinstance(tree, tuple):
            return tree
        feature, threshold, left_subtree, right_subtree = tree
        if sample[feature] < threshold:
            return self.predict_sample(sample,left_subtree)
        else:
            return self.predict_sample(sample,right_subtree)
        