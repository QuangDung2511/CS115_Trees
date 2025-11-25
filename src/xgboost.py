import numpy as np
import pandas as pd
from math import e

class Node:
    '''
    A node object that is recursivly called within itslef to construct a regression tree. The only thing not 
    implemented in this version is sparsity aware fitting or the ability to handle NA values with a default direction.
    Inputs:
        x: pandas datframe of the training data
        gradient: negative gradient of the loss function
        hessian: second order derivative of the loss function
        idxs: used to keep track of samples (rows) within the tree structure
        subsample_cols: is an implementation of layerwise column subsample randomizing the structure of the trees
        (complexity parameter)
        min_leaf: minimum number of samples required in each child node (complexity parameter)
        min_child_weight: sum of the heassian inside required in each child node (complexity parameter)
        depth: limits the number of layers in the tree
        lambda: L2 regularization term on weights, preventing it from becoming too large, basically a denominator stabilizer.
        gamma: This parameter also prevents over fitting and is present in the the calculation of the gain, 
        penalizing unnecessary branches (structure score). 
        eps: This parameter is used in the quantile weighted skecth or 'approx' tree method roughly translates to 
        (1 / sketch_eps) number of bins
    Outputs: A single tree object that will be used for gradient boosintg.
    '''

    def __init__(self, x, gradient, hessian, idxs, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
      
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs 
        self.depth = depth
        self.min_leaf = min_leaf
        self.lambda_ = lambda_
        self.gamma  = gamma
        self.min_child_weight = min_child_weight
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.column_subsample = np.random.permutation(self.col_count)[:round(self.subsample_cols*self.col_count)]
        
        self.val = self.compute_gamma(self.gradient[self.idxs], self.hessian[self.idxs])
          
        self.score = float('-inf')
        self.find_varsplit()
        
        
    def compute_gamma(self, gradient, hessian):
        '''
        Calculates the optimal weight, aka output value for the leaf.
        '''
        return(-np.sum(gradient)/(np.sum(hessian) + self.lambda_))
        
    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Check condition for any further splitting.
        '''
        for c in self.column_subsample: self.weighted_qauntile_sketch(c) #switch to weighted_qauntile_sketch for approx method
        if self.is_leaf: return
        x = self.split_col
        # np.nonzero returns a tuple, first element is the array we want
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[lhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        self.rhs = Node(x = self.x, gradient = self.gradient, hessian = self.hessian, idxs = self.idxs[rhs], min_leaf = self.min_leaf, depth = self.depth-1, lambda_ = self.lambda_ , gamma = self.gamma, min_child_weight = self.min_child_weight, eps = self.eps, subsample_cols = self.subsample_cols)
        
    def find_greedy_split(self, var_idx):
        '''
         For a given feature greedily calculates the gain at each split.
         Globally updates the best score and split point if a better split point is found
        '''
        x = self.x.values[self.idxs, var_idx]
        
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            
            lhs_indices = np.nonzero(x <= x[r])[0]
            rhs_indices = np.nonzero(x > x[r])[0]
            if(rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or self.hessian[lhs_indices].sum() < self.min_child_weight
               or self.hessian[rhs_indices].sum() < self.min_child_weight): continue

            curr_score = self.gain(lhs, rhs)
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]
                
    def weighted_qauntile_sketch(self, var_idx):
        '''
        XGBOOST Mini-Version
        Uses the Weighted Quantile Sketch to find candidate split points based on 
        Hessian distribution (weights), rather than checking every single row.
        '''
        # Get the feature values and hessians for the current node only
        x = self.x.values[self.idxs, var_idx]
        hessian_ = self.hessian[self.idxs]
        
        # Create a temporary DataFrame for sorting
        df = pd.DataFrame({'feature': x, 'hess': hessian_})
        df.sort_values(by=['feature'], ascending=True, inplace=True)
        
        # We want to place a candidate split every time we accumulate 'eps' amount of weight
        total_hess = df['hess'].sum()
        hess_threshold = self.eps * total_hess # e.g. 0.1 * Total_Weight
        
        current_weight_sum = 0
        candidates = []
        
        # Iterate through sorted data to find bucket boundaries
        for index, row in df.iterrows():
            current_weight_sum += row['hess']
            
            # If bucket is full, mark this feature value as a candidate
            if current_weight_sum >= hess_threshold:
                candidates.append(row['feature'])
                current_weight_sum = 0 # Reset bucket
                
        # Instead of looping through all rows, we only loop through the candidates
        for split_value in candidates:
            
            # Create Boolean Masks using the local 'x'
            lhs = x <= split_value
            rhs = x > split_value
            
            # hessian_[lhs] is faster than accessing self.hessian[self.idxs][lhs]
            if (rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf 
               or hessian_[lhs].sum() < self.min_child_weight
               or hessian_[rhs].sum() < self.min_child_weight): 
                continue

            curr_score = self.gain(lhs, rhs)
            
            if curr_score > self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value
                
    def gain(self, lhs, rhs):
        '''
        Calculates the gain at a particular split point
        '''
        gradient = self.gradient[self.idxs]
        hessian  = self.hessian[self.idxs]
        
        lhs_gradient = gradient[lhs].sum()
        lhs_hessian  = hessian[lhs].sum()
        
        rhs_gradient = gradient[rhs].sum()
        rhs_hessian  = hessian[rhs].sum()
        
        gain = 0.5 *( (lhs_gradient**2/(lhs_hessian + self.lambda_)) + (rhs_gradient**2/(rhs_hessian + self.lambda_)) - ((lhs_gradient + rhs_gradient)**2/(lhs_hessian + rhs_hessian + self.lambda_))) - self.gamma
        return(gain)
                
    @property
    def split_col(self):
        '''
        splits a column 
        '''
        return self.x.values[self.idxs , self.var_idx]
                
    @property
    def is_leaf(self):
        '''
        checks if node is a leaf
        '''
        return self.score == float('-inf') or self.depth <= 0                 

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])
    
    def predict_row(self, xi):
        if self.is_leaf:
            return(self.val)

        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

    
class XGBoostTree:
    '''
    Wrapper class that provides a scikit learn interface to the recursive regression tree above
    '''
    def fit(self, x, gradient, hessian, subsample_cols = 0.8 , min_leaf = 5, min_child_weight = 1 ,depth = 10, lambda_ = 1, gamma = 1, eps = 0.1):
        self.dtree = Node(x, gradient, hessian, np.array(np.arange(len(x))), subsample_cols, min_leaf, min_child_weight, depth, lambda_, gamma, eps)
        return self
    
    def predict(self, X):
        return self.dtree.predict(X.values)
    
class XGBoostRegressor:
    '''
    Full application of the XGBoost algorithm as described in "XGBoost: A Scalable Tree Boosting System" for 
    regression.
    '''
    def __init__(self):
        self.estimators = []
    
    # first order gradient mean squared error
    @staticmethod
    def grad(preds, labels):
        return(2*(preds-labels))
    
    # second order gradient logLoss
    @staticmethod
    def hess(preds, labels):
        '''
        hessian of mean squared error is a constant value of two 
        returns an array of twos
        '''
        return(np.full((preds.shape[0], 1), 2).flatten().astype('float64'))
    
    
    def fit(self, X, y, subsample_cols = 0.8 , min_child_weight = 1, depth = 5, min_leaf = 5, learning_rate = 0.4, boosting_rounds = 5, lambda_ = 1.5, gamma = 1, eps = 0.1):
        self.X, self.y = X, y.values
        self.base_score = np.mean(y)
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight 
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds 
        self.lambda_ = lambda_
        self.gamma  = gamma
    
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten().astype('float64')
    
        for booster in range(self.boosting_rounds):
            Grad = self.grad(self.base_pred, self.y)
            Hess = self.hess(self.base_pred, self.y)
            boosting_tree = XGBoostTree().fit(self.X, Grad, Hess, depth = self.depth, min_leaf = self.min_leaf, lambda_ = self.lambda_, gamma = self.gamma, eps = self.eps, min_child_weight = self.min_child_weight, subsample_cols = self.subsample_cols)
            self.base_pred += self.learning_rate * boosting_tree.predict(self.X)
            self.estimators.append(boosting_tree)
          
    def predict(self, X):
        pred = np.zeros(X.shape[0])
        
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X) 
          
        return np.full((X.shape[0], 1), self.base_score).flatten().astype('float64') + pred