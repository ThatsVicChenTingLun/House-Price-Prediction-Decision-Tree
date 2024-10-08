## House prices prediction in kaggle

- the main goal of this competition is to predict the house prices of feature ,according to current collect house feature


- Training data  and testing data including various feature of House ex. MSSubClass: The building class
MSZoning: The general zoning classification. Utilities: Type of utilities available


- We want to prediction y: sale prices ,it's a regression problem



- The competition evaluation using Root-Mean-Squared-Error (RMSE):

![image](https://github.com/user-attachments/assets/93fbefea-eccb-4911-8c3c-89ecdb00ed38)

    
## Module build -decision tree from scratch



### Decision tree theory
* Main Structure
    ![image](https://hackmd.io/_uploads/BkwIcfukC.png)

* What parameters do I need to define during the training process?
    * Splitting criteria for each node: As shown in the figure, training data needs to define how to split the data at each node using specific criteria to divide the data into two categories.
    * Final node (leaf): How to define the class label of its data.

* Defining Module Parameters Principles:
    * The definition of splitting criteria should ensure that the class labels are divided as cleanly as possible.
    &rarr; It is necessary to first define how to quantify a node and what standard constitutes cleanliness, which involves using the calculation method of entropy:
  ![image](https://github.com/user-attachments/assets/f1380918-3233-4421-b080-b38e9e1ade09)
    &rarr; The optimal splitting standard is the one that can maximize the reduction of entropy. The evaluation criterion will use information gain. If the data is split from one node (parent) into two nodes (children), the calculation method for information gain is:
    ![image](https://github.com/user-attachments/assets/556924db-59f2-44a3-bd9d-341fc9ff7fea)
    &rarr; Select data features and standard values that can maximize information gain.

* Common Python Modules: Random Forest, XGBoost, AdaBoost





### How to implement in python


1. **data preprocessing**:

- load data and store in pandas dataframe
- processing nan data drap

- label encoding : let object type data transform to digit

- split train to trainx, trainy 

- split a part of train data to validation



2. **manifest data each feature and target distribution using matplotlib scatter**

![image](https://github.com/user-attachments/assets/bc22605c-3162-4d69-a9e7-18ecc05e143f)


3. **Build a decision tree class from scratch**:

- Define the node of decision tree:

&rarr; each node has feature, critira ,and label (optional)

- inititalize feature of training: 
```python!
        self.min_samples_split = min_samples_split ## represent training min sample of nodes
        self.max_depth = max_depth ## training maximum depth
        self.n_feats = n_feats ## num of features comparing during training
```
- initialize tree nodes: roots, mse dictionary, feature importance ditionary


- Build a fit fucntion: using for training
    - use grow tree fucntion to find all node spliting feature, critiria
    - using best critira function to calculate information gain to find best spliting feature and critiria

    - using recursive method to find left and right node

- Build a prediction function:
    - using traverse tree function to predict each data 


4. **find feature importance**:

- using the feature split times in decision tree to define the feature importance

## Prediction result: 

![image](https://github.com/user-attachments/assets/73754e34-976c-4700-8349-a6fb1ae3b1e3)
![image](https://github.com/user-attachments/assets/bb461207-7b4a-49fe-8a73-1785a404630a)
![image](https://github.com/user-attachments/assets/81f674c0-a096-4941-89a3-de40d9c5dd52)
