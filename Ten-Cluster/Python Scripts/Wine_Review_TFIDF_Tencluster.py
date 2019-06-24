import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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

def no_number(tokens):
    r = re.sub('(\d)+', 'NUM', tokens.lower())
    return r
    
# Use tf-idf method to tokenize the column "description"

common_stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 'but', 'by', "can't",
                    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
                    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he",
                    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
                    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
                    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
                    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
                    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
                    "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
                    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
                    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
                    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", 'aren', 'can', 'couldn',
                    "didn", "doesn", "don", "hadn", "hasn", "haven", "isn", "let", "ll", "mustn", "re", "shan", "shouldn", "ve", 
                    "wasn", "weren", "won", "wouldn"]

customized_stopwords = ["NUM", "wine", "flavors", "aroma", "aromas", "palate", "finish", "drink", "notes", "nose", "now", 
                        "offers", "well", "fruits", "texture", "shows", "like","years", "character", "made", "just", "mouth",
                        "vineyard", "also", "bit", "note", "hint", "one", "give", "will", "flavor", "alongside", "along",
                       "ready", "yet", "mouthfeel"]
stopwords = common_stopwords + customized_stopwords

tfidf_vectorizer = TfidfVectorizer(stop_words = stopwords, preprocessor=no_number)
text=tfidf_vectorizer.fit_transform(winedata_clear['description'])

idf_rank_index = tfidf_vectorizer.idf_.argsort()[0 : 1000]
idf_rank_word = []
for i in idf_rank_index:
    for word, index in tfidf_vectorizer.vocabulary_.items():
        if i == index:
            idf_rank_word.append(word)
            
print(idf_rank_word)

new_text = pd.DataFrame(text[:, idf_rank_index].todense(), columns = idf_rank_word)

# Fit a Kmeans clustering model with different clusters

clusters_text = list(range(2, 21))
silhouette = []

for cluster in clusters_text:
    kmeans_text = KMeans(n_clusters = cluster, max_iter = 1000)
    kmeans_text.fit(new_text)
    score = silhouette_score(new_text, kmeans_text.labels_,metric = 'cosine')
    silhouette.append(round(score,5))
    
# Compute Silouette score to find the best cluster

fig = plt.figure()
plt.bar(clusters_text, silhouette)
plt.plot(clusters_text, silhouette)
plt.xlabel('Clusters')
plt.ylabel('Silhouette Score')
plt.title('Wine Text Data Silhouette Score Versus Clusters')
fig.savefig('Silouette_tfidf_ten_cluster.png')

kmeans_10_text = KMeans(n_clusters = 10, max_iter = 1000)
kmeans_10_text.fit(new_text)

label = pd.Series(kmeans_10_text.labels_, name = 'cluster_label')
label_text = new_text.join(label)

def top_ten(lda_list):
    
    top_ten = []
    top_index = np.argsort(-lda_list)[:10]
    for i in top_index:
        top_ten.append(new_text.columns[i])
    return top_ten
    
# Build logistic regression model for different clusters

def logreg_top_ten(int):
    label_text['convert_label'] = np.where(label_text['cluster_label'] == int, 1, 0) # Convert the label to 1 and other label to 0
    X_train,X_test,Y_train,Y_test=train_test_split(label_text.drop(['cluster_label', 'convert_label'], axis = 1),
                                                   label_text['convert_label'],test_size=0.25)
    
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
    top_word = top_ten(log_reg_max.coef_.reshape(1000,))
    
    return top_word

for i in range(10):
    print(str(i)+':')
    print(logreg_top_ten(i))

result = winedata_clear[['title', 'price', 'points']].join(label)
result.to_csv('tfidf_ten_cluster.csv')


label_text = new_text.join(label)

def kmeans_top_bottom_ten(cluster_label):
    text_df = label_text[label_text['cluster_label'] == cluster_label]
    label_df = kmeans_10_text.cluster_centers_[cluster_label].reshape(1,-1)
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
    top_center.to_csv('tfidf_top_center_{}.csv'.format(i))
    bottom_center.to_csv('tfidf_bottom_center_{}.csv'.format(i))

    top_rating =top_center.nlargest(20, 'points')
    top_rating.to_csv('tfidf_top_rating_{}.csv'.format(i))

    bins = [0, 10, 20, 40, np.inf]
    labels = ['0-10', '10-20', '20-40', 'above 40']
    top_rating['price_bin'] = pd.cut(top_rating['price'], bins=bins, labels = labels)
    price_bin = top_rating.sort_values('price')
    price_bin.to_csv('tfidf_price_bin_{}.csv'.format(i))

