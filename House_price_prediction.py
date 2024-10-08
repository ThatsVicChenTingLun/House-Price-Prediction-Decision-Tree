
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# -------------------data preprocessing-----------------------------------------------
train_df=pd.read_csv("./data/train.csv")
test_df=pd.read_csv("/data/test.csv")
train_df=train_df.drop('Id',axis=1)
test_df=test_df.drop('Id',axis=1)
train_df.dropna(inplace=True,axis=1)
test_df.dropna(inplace=True,axis=1)

from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for column in train_df.columns:
    if(train_df[column].dtype==object):
        train_df[column] = label_encoder.fit_transform(train_df[column])
for column in test_df.columns:
    if(test_df[column].dtype==object):
        test_df[column] = label_encoder.fit_transform(test_df[column])


        train_Y=train_df['SalePrice']
train_X=train_df.drop('SalePrice', axis=1)
# Get the common columns between train_X and test_df
common_columns = train_X.columns.intersection(test_df.columns)

# Use the common columns to filter both train_X and test_df
train_X = train_X[common_columns]
test_df = test_df[common_columns]

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets with 20% for validation
X_train, X_validation, Y_train, Y_validation = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)


#------------------------manifest data feature-----------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns
n_cols = 5  # 每行顯示5個子圖
n_rows = (len(train_X.columns)) // n_cols + 1  # 計算需要多少行

plt.figure(figsize=(20,2 * n_rows)) 
for i, feature in enumerate(train_X.columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(x=train_X[feature], y=train_Y)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.tight_layout() #使圖保持一定的距離

plt.show()

#------------------------------build decision tree from scratch-------------------------------------


from collections import Counter

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import RMSprop


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


    def is_leaf_node(self):
        return self.value is not None

class DecisionNetwork_regression:
    def __init__(self,n_trees=None, min_samples_split=2, max_depth=20, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.mse_dict={}
        self.mse=[]
        self.feature_dict={}
    def fit(self, X, y):
        self.depth=0
        self.mse_dict={}
        self.feature_dict={}
        #self.n_feats=X.shape[1]
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
        sorted_dict = dict(sorted(self.mse_dict.items()))
        self.mse= [sum(values) / len(values) for values in sorted_dict.values()]
        return self.mse,self.depth+1,self.feature_dict

    def predict(self, X):
        y_pred=np.array([self._traverse_tree(x, self.root) for x in X])
        #print(y_pred.shape)
        return y_pred

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        #n_labels = len(np.unique(y))
        self.depth = max(self.depth,depth)
        #stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
        ):
            if(n_samples==0):
                leaf_value=0
            else:
                
                leaf_value=np.mean(self.label_network(X,y))
                #if(leaf_value.shape[0]>1):
                  #  leaf_value=np.mean(np.squeeze(leaf_value))
               # print(leaf_value)
               # print(leaf_value.shape)
            #self.entro.append(entropy(y))
           # if (self.depth!=0):
            #  self.depth+=1
           # self.depth = max(self.depth,depth)
                #print(leaf_value)

            return Node(value=leaf_value)


        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)


        # greedily select the best split according to information gain
        best_feat, best_thresh,min_mse = self._best_criteria(X, y, feat_idxs)

        #self.entro.append(entropy(y))
        # grow the children that result from the split
        #------------------mse_dict---------------------------
        if (depth not in self.mse_dict):
            self.mse_dict[depth]=[]
            self.mse_dict[depth].append(min_mse)
        else:
            self.mse_dict[depth].append(min_mse)
        #-----------------------------------------------------
        ##------------------feature importance------------
        if (best_feat not in self.feature_dict):
            self.feature_dict[best_feat]=0
        else:
            self.feature_dict[best_feat]=self.feature_dict[best_feat]+1

        ##-------------------------------------------------
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        min_mse=float('inf')
        min_diff_idx=float('inf')
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            if len(np.unique(X_column))>13:
                thersholds=self.calculate_quartiles(X_column)
            else:
                thersholds=np.unique(X_column)
            for threshold in thersholds:
                mse,diff_idx = self.calculate_mse(y, X_column, threshold)

                if diff_idx<min_diff_idx and mse<min_mse:
                    min_diff_idx=diff_idx
                    min_mse=mse
                    split_idx = feat_idx
                    split_thresh = threshold
        #print(min_diff_idx)
        # if (depth not in self.mse_dict):
        #   self.mse_dict[depth]=[]
        #   self.mse_dict[depth].append(min_mse)
        # else:
        #   self.mse_dict[depth].append(min_mse)
        return split_idx, split_thresh,min_mse

    def calculate_mse(self, y, X_column, split_thresh):

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        diff_idx=abs((len(left_idxs)-len(right_idxs))//4)
        
        #use mean value determine label and calculate mse:
        if len(left_idxs)!=0:
            mean_y_left=self.mean_label(y[left_idxs])
            #y_left=self.label_network(X_column[left_idxs],y[left_idxs])
            left_mse=(1/y[left_idxs].shape[0])*np.sum(np.abs(y[left_idxs]-mean_y_left))
        else:
            left_mse=0
        if len(right_idxs)!=0:
            mean_y_right=self.mean_label(y[right_idxs])
            #y_right=self.label_network(X_column[right_idxs],y[right_idxs])
            right_mse=(1/y[right_idxs].shape[0])*np.sum(np.abs(y[right_idxs]-mean_y_right))
        else:
            right_mse=0
        mse=left_mse+right_mse
        return mse,diff_idx

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        #print(str(left_idxs.shape)+' '+str(right_idxs.shape))
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
           # print(np.squeeze(node.value).shape)
            return np.squeeze(node.value)
        
        elif (x[node.feature] <= node.threshold ):
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def mean_label(self,y):
        label=np.mean(y)
      
        return label
    def median_label(self, y):
        return np.median(y)
    def calculate_quartiles(self,X):
        
        b=np.percentile(X,(25,50,75),method='midpoint')
        quartiles=b.tolist()
        return quartiles
    def label_network(self,X,y):
        #print(X.shape)
        model = Sequential()
        # Add the input layer and a hidden layer with 5 neurons
        model.add(Dense(5, input_dim=X.shape[1], activation='linear'))

        # Add another two hidden layers with 5 neurons each
        model.add(Dense(10, activation='linear'))
        #model.add(Dense(50, activation='linear'))
        model.add(Dense(10, activation='linear'))

        # Add the output layer with 1 neuron for binary classification
        model.add(Dense(1, activation='linear'))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer= RMSprop(learning_rate=0.01), metrics=['mean_squared_error'])

        # Train the model
        if(X.shape[0]//10==0):
            batch=1
        else:
            batch=int(X.shape[0]//3)
        model.fit(X, y, epochs=20, batch_size=batch,verbose=0)
        y_pred=model.predict(X,verbose=0)
        
        
       # del model
        return y_pred
#-------------------------training and testing-------------------------------------------------

train_x=X_train.to_numpy()
train_y=Y_train.to_numpy()
validation_x=X_validation.to_numpy()
validation_y=Y_validation.to_numpy()
clf = DecisionNetwork_regression(max_depth=50,min_samples_split=3)
mse,depth,feature_dict=clf.fit(train_x, train_y)

#---------------------how mse training result----------------------------------

plt.figure()
plt.plot(mse)
plt.title('mse')
plt.xlabel('training epoch')

y_pred_validation=clf.predict(validation_x)
#y_pre_validation = np.array([item[0] for item in y_pred_validation]).ravel()
plt.figure(figsize=(10, 6))
plt.scatter(validation_y, y_pred_validation, alpha=0.5)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.plot([validation_y.min(), validation_y.max()], [validation_y.min(), validation_y.max()], 'k--', lw=2)  # 绘制对角线
plt.show()

#------------------feature importance-------------------------------------

sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)

# 解包列表，得到排序后的特征和它们对应的值
features_idx, values = zip(*sorted_features)
column_names=train_X.columns.tolist()
selected_feature_names = [column_names[i] for i in features_idx]
# 绘制长条图
plt.figure(figsize=(10, 6))
plt.barh(selected_feature_names, values, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.gca().invert_yaxis()  # 将特征由重要到不重要从上往下排列
plt.show()

#-----------------test y predict result--------------------------------------------

test_x=test_df.to_numpy()
y_pred=clf.predict(test_x)
plt.plot(y_pred)
