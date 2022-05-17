#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pickle


# In[4]:


## Read the dataset, convert it into dataframe
dataframe = pd.read_csv(r'C:\Users\sid\Desktop\ass1\Fake_News_Detection-master\news.csv')
dataframe.head()


# In[5]:


## Split data into X & Y
x = dataframe['text']
y = dataframe['label']


# In[6]:


#data pre processing by checking the null values
dataframe.isnull().any()


# In[7]:


print(x)


# In[8]:


print(y)


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


## Now we need to fit the TFIDF Vectorizer.
# max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
# max_df = 25 means "ignore terms that appear in more than 25 documents".

tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


# In[11]:


## Now let's fit the Machine Learning Model
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)


# In[12]:


# Now let's check model accuracy. Let's fit model on the test data.

y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[13]:


pipeline = Pipeline ([('tfidf', TfidfVectorizer (stop_words='english' )),
                     ('nbmodel', MultinomialNB())])


# In[14]:


pipeline.fit(x_train, y_train)
score=pipeline.score(x_test,y_test)
print('accuracy',score)


# In[15]:


pred = pipeline.predict(x_test)
print(classification_report (y_test, pred))


# In[16]:


cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)


# In[17]:


with open(r'C:\Users\sid\Desktop\ass1\Fake_News_Detection-master\model.pkl', 'wb') as handle:
          pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)          


# In[ ]:





# In[18]:


## Let's create function for test the model on the real-time data.

def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


# In[ ]:





# In[19]:


fake_news_det("Mumbai indians lost all matches")


# In[1]:


cd


# In[ ]:




