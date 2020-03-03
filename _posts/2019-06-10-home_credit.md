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
In Figure 2, we visualize some important features chosen from the feature selection of random forest. The distributions of these features are slightly different comparing Target 1 and Target 0 classes. For the ‘Age’ feature, Target 1 group is relatively younger than Target 0, which indicates that young people are more likely to be evaluated as an unqualified client by Home Credit. For the ‘Last Phone Changed’ feature, it suggests that clients who have consistent phone are more likely to repay the loan on time. For the ‘Days Credited’ feature and the ‘Days Employed’ feature, they imply for those who have longer credit time and days of employed record are more likely to repay the loan on time. The different distribution of Target 1 and Target 0 tell us those young people who don’t have enough income, working experience or reachable contact information are more likely to be classified as Target 1. 
After selecting features and dealing with missing values, we have in total of 53 feature for base data and 263 features for the full data.
![image](https://drive.google.com/file/d/1sB4_3TVa4poORyOf1sMFBlRrRSxf1S_y/view?usp=sharing)
![image](https://drive.google.com/open?id=1dgNqLUJWt-jnKoZ6zoBP-c7HDpwVINp9)
