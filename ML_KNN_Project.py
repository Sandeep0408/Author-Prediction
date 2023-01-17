import json
import pandas as pd
import numpy as np
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


warnings.filterwarnings("ignore")

#Training Dataset

#Loading Train Dataset
data = open('train.json')

info = json.load(data)

#Initializing empty list
ls_authorId = []
ls_final = []
ls_title = []
ls_AuthorName = []

#Concatenating features, Performing pre-processing on text by 
#transforming them into lower text,remove stopwords, non alphabetical words and finally tokenizing them.  
#Eventually Appending them on ls_final[] which would be transformed into a Pandas Dataframe.
for item in info:
    ls_authorId.append(item['authorId'])
    ls_title.append(" ".join(
      set(  
       [t for t in [w for w in word_tokenize(item['title'].lower()) if w.isalpha()] if t not in stopwords.words('english') and len(t) >= 2 ] +
         [t for t in [w for w in word_tokenize(item['abstract'].lower()) if w.isalpha()] if t not in stopwords.words('english')  and len(t) >= 2] +  
          [t for t in [w for w in word_tokenize(item['venue'].lower()) if w.isalpha()] if t not in stopwords.words('english')] + [str(item['year'])]       
        )
      )
    )           
    ls_AuthorName.append(item['authorName'])

ls_final = [(i,j,k) for (i,j,k) in zip(ls_authorId,ls_AuthorName,ls_title)]

author_df_train = pd.DataFrame(ls_final,columns=['authorId','AuthorName','Text'])

author_df_train.head(5)


#TestSet

data = open('test.json')

#Loading Test Dataset
info = json.load(data)

#Initializing empty list
ls_final = []
ls_title = []
ls_paperId =  []

#Concatenating features, Performing pre-processing on text by 
#transforming them into lower text,remove stopwords non alphabetical words and finally tokenizing them.  
#Eventually Appending them on ls_final[] which would be transformed into a Pandas Dataframe.
for item in info:
    ls_title.append(" ".join(
      set(  
       [t for t in [w for w in word_tokenize(item['title'].lower()) if w.isalpha()] if t not in stopwords.words('english') and len(t) >= 2] +
         [t for t in [w for w in word_tokenize(item['abstract'].lower()) if w.isalpha()] if t not in stopwords.words('english') and len(t) >= 2] +  
          [t for t in [w for w in word_tokenize(item['venue'].lower()) if w.isalpha()] if t not in stopwords.words('english')] + [str(item['year'])]
        )         
      )
    )   
    ls_paperId.append(item['paperId'])         

ls_final = [(i,j) for (i,j) in zip(ls_title,ls_paperId)]

author_df_test = pd.DataFrame(ls_final,columns=['Text','paperId'])

author_df_test.head(5)


###Transforming text to numbers using TfIDF vector & performing K-fold on dataset to perform our prediction on Training dataset

#Initializing TfIDF Vectorizer
tfidf_vectorizer=TfidfVectorizer()

#Assigning X & Y values for Featured Text and Author ID
X = np.array(author_df_train['Text'].values)
y = np.array(author_df_train['authorId'])

#Initializing StratifiedKFold  on n_splits = 13 based on a simulation run to get the best accuracy score
skf = StratifiedKFold(n_splits=13, shuffle=True, random_state=21)

for train_index, test_index in skf.split(X,y):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]

x_train = tfidf_vectorizer.fit_transform(X_train)
x_val = tfidf_vectorizer.transform(X_val)

x_test = tfidf_vectorizer.transform(author_df_test['Text'].values)


###Hyperparameter tuning

#Setting parameter arguments for Hyperparameter tuning
knn = KNeighborsClassifier()
k_range = list(range(1, 50))
hyperparam_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, hyperparam_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(x_train, y_train)

print(grid_search.best_params_) # --> {'n_neighbors': 1}

###Model Prediction using KNN

#Based on one shot learning and grid_search.best_params, setting K= 1
knn = KNeighborsClassifier( n_neighbors=1)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_val) 

knn_acc = accuracy_score(y_val, y_pred_knn)
print("Accuracy score in Validation set is {:.3f} ".format(knn_acc))

author_test = knn.predict(x_test)


######Creating Prediction.json file

author_df_test['authorId'] = author_test

#Dropping Text column as not required for final submission
author_df_test = author_df_test.drop('Text', axis=1) 

#Creating predicted.json
author_df_test.to_json('predicted.json', orient='records')