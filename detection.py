import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#data collection and pre processing
raw_mail_data=pd.read_csv('mail_data.csv')
# print(raw_mail_data)

mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')

# print(mail_data.head())

#print(mail_data.shape) # this checks the size of the dataset

#label encoding spam is 0 and ham is 1
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1

X=mail_data['Message']
Y=mail_data['Category']
# print(X)
# print(Y)

#splitting the data into training adn test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
# print(X.shape)
# print(X_train.shape)
# print(X_test.shape)

#feature extraction
#transform the text data to feature vectors
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

#convert Y_train and Y_test values as integers

Y_train= Y_train.astype('int')
Y_test=Y_test.astype('int')
# print(X_train_features)
# print(X_test_features)\
    
#training the model with the traiging data
model = LogisticRegression()
model.fit(X_train_features,Y_train)

#evaluating the trained model
prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training__data=accuracy_score(Y_train,prediction_on_training_data)
# print('Accuracy of training data = ',accuracy_on_training__data)

#evaluating the trained model on test data
prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)
# print('Accuracy of test data = ',accuracy_on_test_data)


#building a predictive system
input_mail1 = ["U don't know how stubborn I am. I didn't even want to go to the hospital. I kept telling Mark I'm not a weak sucker. Hospitals are for weak suckers"]
input_mail2=["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a Â£1500 Bonus Prize, call 09066364589"]

#convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail2)

#making prediction

pred=model.predict(input_data_features)
if pred==1:
    print("this is not spam")
else:
    print("this is spam")
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Class Distribution
category_counts = mail_data['Category'].value_counts()
labels = ['Ham', 'Spam']
plt.figure(figsize=(6, 6))
plt.pie(category_counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'], startangle=90)
plt.title('Class Distribution of Mail Data')
plt.show()

# 2. Top 10 Important Features (Words) based on TF-IDF Scores
tfidf_features = feature_extraction.get_feature_names_out()
tfidf_scores = X_train_features.toarray().sum(axis=0)
top_indices = tfidf_scores.argsort()[-10:][::-1]
top_features = [tfidf_features[i] for i in top_indices]
top_scores = [tfidf_scores[i] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_scores, color='lightgreen')
plt.xlabel('TF-IDF Score')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()

# 3. Training vs. Test Data Accuracy
accuracy_values = [accuracy_on_training__data, accuracy_on_test_data]
labels = ['Training Accuracy', 'Test Accuracy']

plt.figure(figsize=(8, 5))
bars = sns.barplot(x=labels, y=accuracy_values, palette='viridis')
plt.ylim(0, 1)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')

# Adding the values on top of the bars
for bar in bars.patches:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position
        height + 0.02,  # y position (a bit above the bar)
        f'{height:.5f}',  # format the value
        ha='center', va='bottom', fontsize=12, color='black'
    )

plt.show()

# 4. Confusion Matrix
conf_matrix = confusion_matrix(Y_test, prediction_on_test_data)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
