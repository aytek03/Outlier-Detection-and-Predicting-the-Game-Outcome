# Outlier-Detection-and-Predicting-the-Game-Outcome
In this project, we analyze the NBA (National Basketball Association) statistics data for predicting the game outcome and detecting outliers. Firstly, some feature selection algorithms are implemented and their MSE values are calculated by ANN. NCA-C shows a better performance. Then, some classification algorithms are applied by using NCA-C. SVM shows a better performance whereas Naïve Bayes shows a worse performance. After that, we classify the data using two hidden layer (2x2 and 64x64 neurons) pattern recognition network. tansig shows a better performance whereas hardlim shows a worse performance. Secondly, when it comes to outlier detection, our aim is finding the players which break apart from ordinary or standard ones. Our outlier dataset variables are greater than one (1) so we focus on multivariate outliers. In order to find the most contributing features, we used Principle Component Analysis (PCA). It helps us extracting decreased dimensional set of features from the features approximating between 17 to 21. Afterward we practices PyOD toolkit which is specifically designed for outlier detection. We use 5 algorithms, Isolation Forest, Feature Bagging Detector, k-NN, Cluster Based Local Outlier and Average k-NN. More or less results of first 4 of them are similar but Average k-NN performs worse than others.


MATLAB Code => PREDICTING_THE_OUTCOME



Python Code => OUTLIER_DETECTION


We use databasebasketball2.0.zip
