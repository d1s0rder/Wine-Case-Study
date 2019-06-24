import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram

winedata = pd.read_csv('wine.csv',index_col = 0,header = 0, encoding = 'utf-8')

# Extract year from title column
winedata['year'] = winedata['title'].str.findall('(\d{4})').str[-1]

# Drop related columns
winedata_clear = winedata.drop(['province', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle'], axis = 1)

# Drop the row if its price is a missing value; For categorical, replace missing value to string "others"
winedata_clear.dropna(subset = ['price', 'year'], inplace = True)
winedata_clear.fillna('Others',inplace=True)

winedata_clear['year'] = winedata_clear['year'].astype('int64')
winedata_clear = winedata_clear.loc[(winedata_clear['year'] >= 1950) & (winedata_clear['year'] <= 2019)]
winedata_clear.reset_index(inplace = True, drop = True)

wine_category = winedata_clear.drop(['description'], axis = 1)

wine_category['country'][~wine_category['country'].isin(['US', 'France', 'Italy'])] = 'Other country'
wine_category['designation'][~wine_category['designation'].isin(['Reserve', 'Estate', 'Riserva'])] = 'Other designation'
wine_category['variety'][~wine_category['variety'].isin(['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon'])] = 'Other variety'
wine_category['winery'][~wine_category['winery'].isin(['Testarossa', 'Williams Selyem', 'DFJ Vinhos'])] = 'Other winery'

enc = OneHotEncoder()
wine_category_transform = enc.fit_transform(wine_category.drop(['title','year' ,'points', 'price'], axis = 1))

categorical_features = []
for category in enc.categories_:
    categorical_features.extend(category)
    
wine_no_text = pd.DataFrame(wine_category_transform.todense(), columns = categorical_features)
wine_std = pd.DataFrame(scale(winedata_clear['year']), columns = ['year'])

wine_no_text_combine = pd.merge(wine_no_text, wine_std, left_index = True, right_index = True)

hc = AgglomerativeClustering(n_clusters = 5)
hc.fit(wine_no_text_combine)

children = hc.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

fig = plt.figure(figsize = (12,8))
dendrogram(linkage_matrix, p = 100, truncate_mode = 'lastp')
fig.savefig('normal_hierarchical_clustering.png')

hc_label = pd.Series(hc.labels_, name = 'cluster_label')
hc_label_text = wine_no_text_combine.join(hc_label)

def top_four(category):
    
    top_four = []
    top_index = np.argsort(-category)[:4]
    for i in top_index:
        top_four.append(wine_no_text_combine.columns[i])
    return top_four

# Build logistic regression model for different clusters

def logreg_top_four(int):
    hc_label_text['convert_label'] = np.where(hc_label_text['cluster_label'] == int, 1, 0) # Convert the label to 1 and other label to 0
    X_train,X_test,Y_train,Y_test=train_test_split(hc_label_text.drop(['cluster_label', 'convert_label'], axis = 1),
                                                   hc_label_text['convert_label'],test_size=0.25)
    
    max_score = 0
    max_c = 0
    for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        log_reg = LogisticRegression(C = c, solver = 'lbfgs')
        log_reg.fit(X_train, Y_train)
        log_reg_score = log_reg.score(X_test, Y_test)
        
        if log_reg_score > max_score:
            max_score = log_reg_score
            max_c = c
            
    log_reg_max = LogisticRegression(C = max_c, solver = 'lbfgs')
    log_reg_max.fit(X_train, Y_train)
    top_word = top_four(log_reg_max.coef_.reshape(17,))

    return top_word

print(logreg_top_four(0))
print(logreg_top_four(1))
print(logreg_top_four(2))
print(logreg_top_four(3))
print(logreg_top_four(4))

hc_normal_result = winedata_clear[['title', 'description', 'price', 'points']].join(hc_label)
hc_normal_result.to_csv('hc_normal_result.csv')
