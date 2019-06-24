import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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
                        "offers", "well", "texture", "shows", "like","years", "character", "made", "just", "mouth",
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

cv = CountVectorizer(stop_words = stopwords, preprocessor=no_number)
cv_text = cv.fit_transform(winedata_clear['description'])

cv_text_index = []
for word in idf_rank_word:
    cv_text_index.append(cv.vocabulary_[word])
cv_text_new = pd.DataFrame(cv_text[:, cv_text_index].todense(), columns = idf_rank_word)

clusters_lda = list(range(2, 21))
score_lda = []

for cluster in clusters_lda:
    lda = LatentDirichletAllocation(n_components = cluster)
    lda_text = lda.fit_transform(cv_text_new)
    lda_score = lda.score(cv_text_new)
    score_lda.append(round(lda_score, 5))
    
fig = plt.figure()
plt.bar(clusters_lda, score_lda)
plt.plot(clusters_lda, score_lda)
plt.xlabel('Clusters')
plt.ylabel('Score')
plt.title('Wine Text Data LDA Score Versus Clusters')
fig.savefig('Score_LDA.png')

lda = LatentDirichletAllocation(n_components = 5)
lda_text = lda.fit_transform(cv_text_new)

text_table_lda = pd.DataFrame(lda_text, columns = [0,1,2,3,4])                                                 
def top_ten(lda_list):
    
    top_ten = []
    top_index = np.argsort(-lda_list)[:10]
    for i in top_index:
        top_ten.append(cv_text_new.columns[i])
    return top_ten
    

for i in range(5):
    print(str(i)+':')
    print(top_ten(lda.components_[i]))
    
lda_label = pd.Series(text_table_lda.idxmax(axis = 1), name = 'lda_label')

lda_result = winedata_clear[['title', 'price', 'points']].join(lda_label)
lda_result.to_csv('lda_fivecluster.csv')

lda_label_text = text_table_lda.join(lda_label)

def lda_top_bottom_ten(cluster_label):
    topic_df = lda_label_text[lda_label_text['lda_label'] == cluster_label]
    top_ten_percent = topic_df.nlargest(int(topic_df.shape[0] * 0.1), cluster_label)
    bottom_ten_percent = topic_df.nsmallest(int(topic_df.shape[0] * 0.1), cluster_label)
    
    top_ten_final = winedata_clear[['title', 'description', 'price', 'points']].join(
        top_ten_percent[['lda_label', cluster_label]], how = 'right')

    bottom_ten_final = winedata_clear[['title', 'description', 'price', 'points']].join(
        bottom_ten_percent[['lda_label', cluster_label]], how = 'right')
    
    return top_ten_final, bottom_ten_final

for i in range(5):
    top_center, bottom_center = lda_top_bottom_ten(i)
    top_center.to_csv('lda_top_center_{}.csv'.format(i))
    bottom_center.to_csv('lda_bottom_center_{}.csv'.format(i))
    
    top_rating =top_center.nlargest(20, 'points')
    top_rating.to_csv('lda_top_rating_{}.csv'.format(i))
    
    bins = [0, 10, 20, 40, np.inf]
    labels = ['0-10', '10-20', '20-40', 'above 40']
    top_rating['price_bin'] = pd.cut(top_rating['price'], bins=bins, labels = labels)
    price_bin = top_rating.sort_values('price')
    price_bin.to_csv('lda_price_bin_{}.csv'.format(i))
