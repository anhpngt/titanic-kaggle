import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# directories
data_dir = 'datasets'

def loadData(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    return train_df, test_df

def dropData(df, columns_dropped):
    return df.drop(columns_dropped, axis=1)

def procAge(train_df, test_df):
    # get average, std, and number of NaN values
    train_age_avg = train_df['Age'].mean()
    train_age_std = train_df['Age'].std()
    train_age_nan_count = train_df['Age'].isnull().sum()
    
    test_age_avg = test_df['Age'].mean()
    test_age_std = test_df['Age'].std()
    test_age_nan_count = test_df['Age'].isnull().sum()
    
    # generate random numbers between (mean - std) & (mean + std)
    train_rand = np.random.randint(train_age_avg - train_age_std, 
                                   train_age_avg + train_age_avg,
                                   size=train_age_nan_count)
    test_rand = np.random.randint(test_age_avg - test_age_std,
                                  test_age_avg + test_age_std,
                                  size=test_age_nan_count)
    
    # fill NaN values in Age column with random values generated
    train_df['Age'][np.isnan(train_df['Age'])] = train_rand
    test_df['Age'][np.isnan(test_df['Age'])] = test_rand
    
    return train_df, test_df

def procEmbarked(train_df, test_df):
    # fill the 2 missing values with the most occured value, 'S'
    train_df['Embarked'] = train_df['Embarked'].fillna('S')
#     sns.countplot(x='Embarked', data=train_df)
#     sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0])
    # actually, drop embarked since logically it is relevant in prediction
#     train_df = train_df.drop(['Embarked'], axis=1)
#     test_df = test_df.drop(['Embarked'], axis=1)

    # choose to include Embarked: make dummies
    embarked_dummies_train = pd.get_dummies(train_df['Embarked'])
    embarked_dummies_train.columns = ['C', 'Q', 'S']
    train_df = train_df.join(embarked_dummies_train)
    embarked_dummies_test = pd.get_dummies(test_df['Embarked'])
    embarked_dummies_test.columns = ['C', 'Q', 'S']
    test_df = test_df.join(embarked_dummies_test)
    
    train_df = train_df.drop(['Embarked'], axis=1)
    test_df = test_df.drop(['Embarked'], axis=1)
    
    return train_df, test_df

def procFare(train_df, test_df):
    # fill the missing fare value in test.csv using mean value
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
    # convert from float to intpd
#     train_df = train_df['Fare'].astype(int)
#     test_df = test_df['Fare'].astype(int)
    return train_df, test_df

def procSex(train_df, test_df):
    # create dummy variables
    person_dummies_train = pd.get_dummies(train_df['Sex'])
    person_dummies_train.columns = ['Female','Male']
    person_dummies_test = pd.get_dummies(test_df['Sex'])
    person_dummies_test.columns = ['Female','Male']
    train_df = train_df.join(person_dummies_train)
    test_df = test_df.join(person_dummies_test)
    
    # and drop the Sex column
    train_df = train_df.drop(['Sex'], axis=1)
    test_df = test_df.drop(['Sex'], axis=1)
    
    return train_df, test_df

def procPclass(train_df, test_df):
    # create dummy variables
    pclass_dummy_train = pd.get_dummies(train_df['Pclass'])
    pclass_dummy_train.columns = ['Class_1', 'Class_2', 'Class_3']
    pclass_dummy_test = pd.get_dummies(test_df['Pclass'])
    pclass_dummy_test.columns = ['Class_1', 'Class_2', 'Class_3']
    train_df = train_df.join(pclass_dummy_train)
    test_df = test_df.join(pclass_dummy_test)
    
    # and drop the PClass column
    train_df = train_df.drop(['Pclass'], axis=1)
    test_df = test_df.drop(['Pclass'], axis=1)
    
    return train_df, test_df

if __name__=='__main__':
    # import data
    train_df, test_df = loadData(data_dir)
    print(train_df.info())
    print(test_df.info())
    print('-------------------------')
    
    # drop unnecessary columns that are not useful in prediction
    train_df = dropData(train_df, ['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_df = dropData(test_df, ['Name', 'Ticket', 'Cabin'])

    # Processing each remaining feature
    train_df, test_df = procAge(train_df, test_df)
    train_df, test_df = procEmbarked(train_df, test_df)
    train_df, test_df = procFare(train_df, test_df)
    train_df, test_df = procSex(train_df, test_df)
    train_df, test_df = procPclass(train_df, test_df)
    
    # show plot and data
    print(train_df.info())
    print(test_df.info())
    plt.show()
    
    # do machine learning
    X_train = train_df.drop("Survived",axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId",axis=1).copy()
     
    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    print("Logistic Regression:", logreg.score(X_train, Y_train))
    Y_pred = logreg.predict(X_test)
#     # Random Forests
#     random_forest = RandomForestClassifier(n_estimators=100)
#     random_forest.fit(X_train, Y_train)
#     Y_pred = random_forest.predict(X_test)
#     print("Random Forest:", random_forest.score(X_train, Y_train))
     
    # output to csv file
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                               'Survived': Y_pred})
    submission.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)