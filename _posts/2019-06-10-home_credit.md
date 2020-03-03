---
title: "Home Credit Analysis "
date: 2019-06-10
tags: [data science]
---

- Group Members: Shiying Wang, Qing Gao, Ruochen Zhong, Zhixin Zheng, Tianrun Zhu

## **ABSTRACT**
An approach to identify what factors contribute to clients’ default rate and evaluation on lending risk to a specific client by using past data is described in this paper. We use feature engineering to extract and select useful features in order to determine the ability of clients to repay the loan on time. Through the investigation of some trending classification algorithms and the help of the ensemble learning, we hope to improve the performance of our final model. The result shows that our best model is LightGBM with combined data and upsampling method. To go beyond the current model performance, we need more information about clients and more estimators in the LightGBM model to identify clients’ default risk.

## I.	INTRODUCTION
Home Credit is an international non-bank, consumer finance group. It provides loans to people who need financial support and to those who have insufficient or non-existent credit histories due to lack of bank accounts. In order to provide loan services to a wider range of population such those are disqualified due to less or no bank histories, there will exist a higher risk for providing loan to this kind of clients which they may have a weaker ability to handle financial difficulties and default their loans due to multiple conditions or struggles. Our purpose is to evaluate each client by different models is trained with Home credit provided data, and try to determine what types of clients most likely default.

## II.	EXPLORATORY DATA ANALYSIS
We have in total of seven datasets downloaded from Kaggle, which the main dataset  “application_{train|test}.csv” (122 features) includes information about loans and loan applicants at application time.[1] In order to have a more comprehensive profile of clients, we decide to combine features all seven datasets. Therefore, the final dataset we are using contains 339 features. 

**Data Overview.** To better understand the recorded clients in the Home Credit datasets, we visualize the distributions of their family status, educational background, income type, and house type. From Figure 1, despite the client's ability to repay the loan, there is a majority group of clients in each feature signed contracts. According to these distribution graphs, our first instinct to conclude the majority group of people who need financial support are married working people who also have a high school diploma and own a house/apartment.

**Data Preprocessing.** Throughout data exploration and a trail with some algorithms, we notice there are several issues that we need to deal with before formal modeling: 1) The data is highly unbalanced: approximately 8.07% of Target is in group 1 (unable to repay the loan) while around 91.93% of Target is in group 0 (other cases), and we will use Upsampling and Downsampling to resolve the problem. 2) In the other six datasets, each client has several records, so we decide to deal with the numeric features by merging each of them by their mean, minimum, maximum, and sum and the categorical features by building several new columns to present each classes’ value count in one categorical feature or by choosing the most frequent class of each client. 3) A high proportion of the missing value for some features. We dropped features with more than 40% of missing data. After that, we used a heatmap to observe missing value correlation and determine to drop features which are highly uncorrelated with the TARGET feature. For those correlated features, we fill missing values with the median for numeric features and use ‘XNA’ (not available) to replace the missing value. 

**Feature Selection.** We first manually select some features based on our domain knowledge, and we combine some categories in certain features based on their distributions in target 0 and target 1. We also use  Random Forest Feature Importance to select the feature where we remove the features with 0 importance. The higher frequency of using some features to split the decision tree means these features will cause a smaller entropy or impurity. It indicates that each decision tree may have a better performance on classifying new observation with proper hyperparameter tuning. We build 1000 estimators as trees in the forest to see the feature importance. We also try to use hierarchical clustering to split the features into 4, 6 and 12 groups, and we remove those that were not in the same group with our Target; however, the results do not improve at all, so we decide to keep those features.

Table 1 is the top-ten result of importance based ranking, which implying these features are more frequently chosen when splitting dataset and produce smaller entropies. It indicates these features contain more useful information and are more capable to train a higher performance model than other lower importance score features.
![image](https://drive.google.com/uc?export=view&id=1sB4_3TVa4poORyOf1sMFBlRrRSxf1S_y)
                                                                                                                  
In Figure 2, we visualize some important features chosen from the feature selection of random forest. The distributions of these features are slightly different comparing Target 1 and Target 0 classes. For the ‘Age’ feature, Target 1 group is relatively younger than Target 0, which indicates that young people are more likely to be evaluated as an unqualified client by Home Credit. For the ‘Last Phone Changed’ feature, it suggests that clients who have consistent phone are more likely to repay the loan on time. For the ‘Days Credited’ feature and the ‘Days Employed’ feature, they imply for those who have longer credit time and days of employed record are more likely to repay the loan on time. The different distribution of Target 1 and Target 0 tell us those young people who don’t have enough income, working experience or reachable contact information are more likely to be classified as Target 1. 
![image](https://drive.google.com/uc?export=view&id=1dgNqLUJWt-jnKoZ6zoBP-c7HDpwVINp9)

After selecting features and dealing with missing values, we have in total of 53 feature for base data and 263 features for the full data.

## III. METHOD
### **A. Modeling**
In order to achieve higher accuracy or AUC (Area under Curve) for the final model, here are some strong candidates for classification problem in traditional and modern machine learning algorithms:

+ **Support Vector Machine (SVM).** SVM is one of the most powerful traditional supervised learning algorithms for classification. It tries to find the maximum distance margin between two or multiple classes to create a more robust classification rule than perceptron or logistic regression.  Since we are facing the classical binary classification problem, we attempt to use SVM with linear kernel and rbf Gaussian kernel to classify whether the target belongs to class 0 or 1. 

+ **Artificial Neural Network (ANN).** ANN is one of the most versatile and powerful deep learning algorithms when we need to deal with traditional regression and classification problems. Each node in ANN solves a simple regression or classification problem, and these nodes function together and structure as a neural network. We will have a huge hypothesis set  given its complex structure. For instance, when we deal with a cat image classification problem, given enough high-quality dataset after a decent number of back propagation, each node in the ANN will find its different role such that some nodes will identify the color pattern and some nodes will identify the eye, the whisker, or etc. Thus, ANN can approximate some very complex target functions (solve some complex problems). 

+ **K-Nearest Neighbor (KNN).** KNN is a non-parametric method which can be used for both classification and regression. In the classification case, KNN can classify classes by non-linear and disjoint boundaries without the limitation of linearity. Since other features are not linearly correlated with our Target, we tried KNN to classify the Target. The classification boundary based on a majority vote of its k nearest neighbor, where k is the hyperparameter which is the number of the nearest neighbor. KNN uses the distance function to measure the nearest neighbors, and here we use Minkowski as the distance function.

+ **Random Forest.** Random Forest is a popular machine learning algorithm for classification by bagging method with multiple weak decision trees. In this ensemble, Random Forest constructs multiple decision trees with downsampling of the train data, make conclusions about our target value by the majority of the weak classifier. By the design of decision tree, random forest can automatically select a useful feature for reaching the lowest entropy when it splits dataset. 

+ **LightGBM.** LightGBM is a trending machine learning algorithm due to its high efficiency and state of arts performance. It is a modified and more efficient version of Gradient boosting decision tree, where GBDT is a boosted decision tree by adding predictors to fit the residual errors from the previous predictor, compared to the original implementation of  Gradient boosting tree which using all data points and all features. In order to speed up the training time without losing model accuracy, LightGBM modifies GBDT with downsampling method (Gradient-based One-Side Sampling) and feature merging (Exclusive Feature Bundling). 

Gradient-based One-Side Sampling selects some datasets which have large gradients. These data points have large gradients can contribute more information for model training. Thus, lightGBM can get almost the same amount of information from the downsampling dataset which this smaller size of the dataset will speed up the training process. 

Exclusive Feature Bundling combines or bundles some features by assuming most features in a large number of feature set are almost independent or mutually exclusive. Thus,  these exclusive features can be combined together without losing a significant amount of information in an ideal situation. 

By using GOSS and EBF, lightGBM can tremendously increasing training efficiency with the similar original GBDT’s performance. These traits make lightGBM one of the most powerful boosting methods in modern algorithms.

### **B. Balancing Data**

**Upsampling.** To solve the unbalancing problem, one of the methods is to use upsampling. The traditional way of upsampling is simply resampling the same percentage of each class with replacement until both classes are roughly the same number of observations. This may help the model to solve the highly unbalanced issue, but the new observation is duplicated from the original data. The model won’t be able to gain more information to further understand or classify small percentage labels. It will only use the same data point to train the model. This may also cause overfitting in some cases.[3] Thus, We use an advanced up-sampling method called SMOTE (Synthetic Minority Over-sampling Technique).  SMOTE doesn’t generate duplicates from the minority class data. It produces ‘synthetic’ new data by random generate a data point within or between the chosen neighbors of each minority class data point.[3] This method will generate new information which may help some models’ generalization ability.

**Downsampling with Ensemble (Balanced Bagging).** In order to solve the problem of unbalanced data, we also tried the method of downsampling to balance data. We hope to train a bunch of weak models with balanced downsampling dataset. According to the ensemble method and the law of large numbers, the ensemble with a large number of weak but independent estimators will have a much higher accuracy in the hard voting classifier setting. 

We split the data between Target 0 and 1 and selected 30% from both datasets as test data and the rest 70% as train data. We randomly select data from train data to ensure the proportion of target 1 is 50%, trained the balanced train data and predicted n times of randomly selected train data on the same test data.   

## IV. RESULT & DISCUSSION
Through investigation and exploration of some algorithms with the main dataset which have 53 features, we get the following result. By considering their performance without remedy and with remedy (resampling methods), we will pick some better performance models for further investigation given full dataset with 263 features. We hope to reach a higher performance in the AUC standard by the help of resampling methods and voting classifier. 




