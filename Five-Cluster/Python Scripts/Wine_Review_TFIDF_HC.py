import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
            
#print(idf_rank_word)

new_text = pd.DataFrame(text[:, idf_rank_index].todense(), columns = idf_rank_word)

hc = AgglomerativeClustering(n_clusters = 5)
hc.fit(new_text)

children = hc.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

fig = plt.figure(figsize = (12,8))
dendrogram(linkage_matrix, p = 100, truncate_mode = 'lastp')
fig.savefig('tfidf_hierarchical_clustering.png')

hc_label = pd.Series(hc.labels_, name = 'cluster_label')
hc_label_text = new_text.join(hc_label)

def top_ten(word_list):
    
    top_ten = []
    top_index = np.argsort(-word_list)[:10]
    for i in top_index:
        top_ten.append(new_text.columns[i])
    return top_ten

def logreg_top_ten(int):
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
    top_word = top_ten(log_reg_max.coef_.reshape(1000,))
    
    return top_word

print(logreg_top_ten(0))
print(logreg_top_ten(1))
print(logreg_top_ten(2))
print(logreg_top_ten(3))
print(logreg_top_ten(4))

hc_result = winedata_clear[['title', 'description', 'price', 'points']].join(hc_label)
hc_result.to_csv('hc_result_tfidf.csv')
