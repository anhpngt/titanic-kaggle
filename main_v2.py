'''
    Based on https://www.kaggle.com/sinakhorami/titanic-best-working-classifier
'''

from os.path import join
from pprint import pprint

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

data_dir = 'datasets'

def getTitle(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def doFeatureEngineering(train_df, test_df):
    full_data = [train_df, test_df]
    # Pclass
    print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean(), '\n')
    
    # Sex
    print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean(), '\n')
    
    # SibSp and Parch
    # Create a family-size feature
    for df in full_data:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean(), '\n')
    # Create if with family feature
    for df in full_data:
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean(), '\n')
    
    # Embark
    train_df['Embarked'] = train_df['Embarked'].fillna('S') # Fill missing values
    
    # Fare
    test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median()) # Fill missing values
    for df in full_data:
        df['CategoricalFare'] = pd.qcut(df['Fare'], 4) 
    print(train_df[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
    
    # Age
    for df in full_data:
        age_avg = df['Age'].mean()
        age_std = df['Age'].std()
        age_null_count = df['Age'].isnull().sum()
        
        age_null_rand_list = np.random.randint(age_avg - age_std,
                                               age_avg + age_std,
                                               size=age_null_count)
        df['Age'][np.isnan(df['Age'])] = age_null_rand_list
        df['CategoricalAge'] = pd.cut(df['Age'], 5)
    print(train_df[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
    
    # Name
    # Obtain titles from names and categorize them
    for df in full_data:
        df['Title'] = df['Name'].apply(getTitle)
        df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr',
                                           'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle'], 'Miss')
        df['Title'] = df['Title'].replace(['Ms'], 'Miss')
        df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    print(pd.crosstab(train_df['Title'], train_df['Sex']))
    print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
            
    return train_df, test_df

def doDataCleaning(train_df, test_df):
    full_data = [train_df, test_df]
    for df in full_data:
        # Mapping Sex
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        
        # Mapping Title
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)
        
        # Mapping Embarked
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
        
        # Mapping Fare
        df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
        df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
        df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31.0), 'Fare'] = 2
        df.loc[(df['Fare'] > 31.0), 'Fare'] = 3
        df['Fare'] = df['Fare'].astype(int)
        
        # Mapping Age
        df.loc[df['Age'] <= 16, 'Age'] = 0
        df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
        df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
        df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
        df.loc[(df['Age'] > 64), 'Age'] = 4
        df['Age'] = df['Age'].astype(int)
    
    # Feature selection
    drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'FamilySize', 
                     'CategoricalFare', 'CategoricalAge', 'Cabin']
    train_df = train_df.drop(drop_elements, axis=1)
    test_df = test_df.drop(drop_elements, axis=1)
    
    return train_df, test_df

def doClassifying(train_df, test_df):
    classifiers = [KNeighborsClassifier(3),
                   SVC(probability=True),
                   DecisionTreeClassifier(),
                   RandomForestClassifier(),
                   AdaBoostClassifier(),
                   GradientBoostingClassifier(),
                   GaussianNB(),
                   LinearDiscriminantAnalysis(),
                   QuadraticDiscriminantAnalysis(),
                   LogisticRegression()]
    log_cols = ['Classifiers', 'Accuracy']
    log = pd.DataFrame(columns=log_cols)
    
    X = np.array(train_df.iloc[:, 1:])
    y = np.array(train_df.iloc[:, 0])
    
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        
    acc_dict = {}
    
    print("Training.....")
    for train_index, test_index in sss.split(X, y):
        trX, teX = X[train_index], X[test_index]
        trY, teY = y[train_index], y[test_index]
        
        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(trX, trY)
            predictions = clf.predict(teX)
            acc = accuracy_score(teY, predictions)
            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc
    
    for item in classifiers:
        clf = item.__class__.__name__
        acc_dict[clf] = acc_dict[clf]/10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)
    pprint(acc_dict)
    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy')
    
    sns.set_color_codes('muted')
    sns.barplot(x='Accuracy', y='Classifiers', data=log, color='r')
    
    # Using SVC since it gives the best result
    best_classifier = RandomForestClassifier()
    best_classifier.fit(X, y)
    result = best_classifier.predict(test_df)
    
    return result, best_classifier.__class__.__name__

if __name__=='__main__':
    # Import data
    train_df = pd.read_csv(join(data_dir, 'train.csv'), header=0, dtype={'Age': np.float64})
    test_df = pd.read_csv(join(data_dir, 'test.csv'), header=0, dtype={'Age': np.float64})
    
    print(train_df.info())
    print(test_df.info())
    
    train_df, test_df = doFeatureEngineering(train_df, test_df)
    train_df, test_df = doDataCleaning(train_df, test_df)
    print("Data info after feature engineered and cleaned: ")
    print(train_df.info())
    print(test_df.info())
    
    result, clf_name = doClassifying(train_df, test_df)
    submission_df = pd.read_csv(join(data_dir, 'submission.csv'))
    submission_df['Survived'] = result
    submission_df.to_csv(join(data_dir, 'submission.csv'), index=False)
    print("Submission file has been updated. Classifier used:", clf_name)
    plt.show()
     