import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


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

clusters = list(range(2, 21))
silhouette = []

for cluster in clusters:
    kmeans_text = KMeans(n_clusters = cluster)
    kmeans_text.fit(wine_no_text)
    score = silhouette_score(wine_no_text, kmeans_text.labels_)
    silhouette.append(round(score,5))

fig = plt.figure()    
plt.bar(clusters, silhouette)
plt.plot(clusters, silhouette)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('Wine Text Data Silhouette Score Versus Clusters')
fig.savefig('Silhouette_Kmeans1_Ten_Cluster.png')

kmeans_10 = KMeans(n_clusters = 10)
kmeans_10.fit(wine_no_text)
silhouette_score(wine_no_text, kmeans_10.labels_)

label = pd.Series(kmeans_10.labels_, name = 'cluster_label')
label_no_text = wine_no_text.join(label)

def top_four(category):
    
    top_four = []
    top_index = np.argsort(-category)[:4]
    for i in top_index:
        top_four.append(wine_no_text.columns[i])
    return top_four
    
def logreg_top_four(int):
    label_no_text['convert_label'] = np.where(label_no_text['cluster_label'] == int, 1, 0) # Convert the label to 1 and other label to 0
    X_train,X_test,Y_train,Y_test=train_test_split(label_no_text.drop(['cluster_label', 'convert_label'], axis = 1),
                                                   label_no_text['convert_label'],test_size=0.25)
    
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
    top_word = top_four(log_reg_max.coef_.reshape(16,))
    
    return top_word
    
print(logreg_top_four(0))
print(logreg_top_four(1))
print(logreg_top_four(2))
print(logreg_top_four(3))
print(logreg_top_four(4))
print(logreg_top_four(5))
print(logreg_top_four(6))
print(logreg_top_four(7))
print(logreg_top_four(8))
print(logreg_top_four(9))

normal_result = winedata_clear[['title', 'price', 'points']].join(label)
normal_result.to_csv('kmeans1_ten_cluster.csv')


label_no_text = wine_no_text.join(label)
def kmeans_top_bottom_ten(cluster_label):
    text_df = label_no_text[label_no_text['cluster_label'] == cluster_label]
    label_df = kmeans_10.cluster_centers_[cluster_label].reshape(1,-1)
    text_df['cosine_similarity'] = cosine_similarity(text_df.drop('cluster_label', axis = 1), label_df)

    top_ten_percent = text_df.nlargest(int(text_df.shape[0] * 0.1), 'cosine_similarity')
    bottom_ten_percent = text_df.nsmallest(int(text_df.shape[0] * 0.1), 'cosine_similarity')

    top_ten_final = winedata_clear[['title', 'description', 'price', 'points']].join(
        top_ten_percent[['cluster_label', 'cosine_similarity']], how = 'right')

    bottom_ten_final = winedata_clear[['title', 'description', 'price', 'points']].join(
        bottom_ten_percent[['cluster_label', 'cosine_similarity']], how = 'right')

    return top_ten_final, bottom_ten_final


for i in range(10):
    top_center, bottom_center = kmeans_top_bottom_ten(i)
    top_center.to_csv('kmeans1_top_center_{}.csv'.format(i))
    bottom_center.to_csv('kmeans1_bottom_center_{}.csv'.format(i))

    top_rating =top_center.nlargest(20, 'points')
    top_rating.to_csv('kmeans1_top_rating_{}.csv'.format(i))

    bins = [0, 10, 20, 40, np.inf]
    labels = ['0-10', '10-20', '20-40', 'above 40']
    top_rating['price_bin'] = pd.cut(top_rating['price'], bins=bins, labels = labels)
    price_bin = top_rating.sort_values('price')
    price_bin.to_csv('kmeans1_price_bin_{}.csv'.format(i))

