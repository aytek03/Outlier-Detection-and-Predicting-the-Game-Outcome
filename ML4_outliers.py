# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:45:41 2019

@author: ali
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:29:50 2019

@author: ali
"""
# Import models
import pandas as pd
import numpy as np
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
import matplotlib.pyplot as plt
import matplotlib.font_manager


from scipy import stats

df = pd.read_excel("player_playoff_career_avg.xlsx")

df.columns

df.plot.scatter('ppts', 'pasts')


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df[['ppts','pasts', 'preb']] = scaler.fit_transform(df[['ppts','pasts','preb']])
df[['ppts','pasts','preb']].head()


X1 = df['ppts'].values.reshape(-1,1)
X2 = df['pasts'].values.reshape(-1,1)
X3 = df['preb'].values.reshape(-1,1)
X = np.concatenate((X1,X2),axis=1)

random_state = np.random.RandomState(42)
outliers_fraction = 0.010

classifiers = {
        
       'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
       # 'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        #'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False,        random_state=random_state),
       # 'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
       # 'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
         # 'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}


for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1
        
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    plt.figure(figsize=(10, 10))
    threshold = stats.scoreatpercentile(scores_pred,100 *       outliers_fraction)
    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()
    dfy=df[['ilkid','firstname','lastname', 'ppts','pasts','preb','outlier']]
    
    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 =  np.array(dfx['ppts'][dfx['outlier'] == 0]).reshape(-1,1)
    IX2 =  np.array(dfx['pasts'][dfx['outlier'] == 0]).reshape(-1,1)
    IX3 =  np.array(dfx['preb'][dfx['outlier'] == 0]).reshape(-1,1)
    
    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 =  dfx['ppts'][dfx['outlier'] == 1].values.reshape(-1,1)
    OX2 =  dfx['pasts'][dfx['outlier'] == 1].values.reshape(-1,1)
    OX2 =  dfx['preb'][dfx['outlier'] == 1].values.reshape(-1,1)
         
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers, clf_name)
        
    # threshold value to consider a datapoint inlier or outlier
  
    
    # create a meshgrid 
    xx , yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])*-1
    Z = Z.reshape(xx.shape)
          
    # fill blue map colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        
    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        
    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        
    b = plt.scatter(IX1,IX2, c='white',s=20, edgecolor='k')
    
    c = plt.scatter(OX1,OX2, c='black',s=20, edgecolor='k')
       
    plt.axis('tight')  
    
    # loc=2 is used for the top left corner 
    plt.legend(
        [a.collections[0], b,c],
        ['learned decision function', 'inliers','outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)
      
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()