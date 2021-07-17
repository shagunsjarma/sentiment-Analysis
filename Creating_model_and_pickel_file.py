#Importing necessary libaries
import pandas as pd
import numpy as np
#import nltk
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

#Reading the .CSV file using pandas
df = pd.read_csv("balanced_reviews.csv")
df.shape
df.columns.tolist()
print('Reading the File')

#Removing the NaN values
df.isnull().any(axis = 0)
df.dropna(inplace = True)
df.shape
print('Droping Nan Value')

# Droping the neutral values
df['overall'] != 3
df = df[df['overall'] != 3]
df.shape
df['overall'].value_counts()
print('Droped neutral reviews')

#Creating a column for positivity or as label
df['Positivity'] =  np.where(df['overall'] > 3, 1, 0)
df['Positivity'].value_counts()
print('Created Labels')

# =============================================================================
# corpus = []
# nltk.download('stopwords')
# #Removal of unwanted words
# print('---- Starting to remove stopwords ----')
# for i in range (0, df.shape[0]):
# 
#     review  =re.sub('[^a-zA-Z]' , " " , df.iloc[i, 1])
#     review = review.lower()
#     review  = review.split()
#     review = [word for word in review if not word in stopwords.words('english')]
#     ps = PorterStemmer()
#     review  = [ps.stem(word) for word in review]
#     review = " ".join(review)
#     corpus.append(review)
# print('Removal Done')
# =============================================================================

print('Features and labels declared')
#Declaring the features and Labels
features = df['reviewText']
labels = df['Positivity']

#spliting for train test split
print('Spliting of test and train data')
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2)

#Vectorization
vect = TfidfVectorizer(min_df = 5).fit(features_train)
len(vect.get_feature_names())
features_train_vectorized = vect.transform(features_train)
print('Vectorization Complete ')

# features_matrix_expanded = features_train_vectorized.toarray()
# vect.get_feature_names()[15000:15010]

print('Creating Model')
#Creating Model
model  = LogisticRegression(max_iter=1000)
model.fit(features_train_vectorized, labels_train)
# 0 - negative review
# 1 - positive review
print('Fitting Complete ----- Displaying Result Below')
predictions = model.predict(vect.transform(features_test))

print('Getting you the Accuracy of the Model')
#Creating_Confusion_matrix_for_accuracy
print(confusion_matrix(labels_test, predictions))
#Getting_to_know_accuracy_of_matrix
print(accuracy_score(labels_test, predictions))

print('Pickle File is been made')
#______________________________________
#Saving as pickel file for use as model
file = open("pickle_model.pkl", "wb")
pickle.dump(model, file)
#Vectorizer_Model_File
pickle.dump(vect.vocabulary_, open('feature.pkl','wb'))
print('Done')





