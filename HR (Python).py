import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# to load the dataset into pandas dataframe
df = pd.read_csv('HR_dataset1.csv')

# to print statistical summary of dataset
print df.head()
print '\n', df.info()
print '\n', df.describe()

df=df.drop(['EmployeeCount','EmployeeNumber','StandardHours'], axis=1)

# to print the number of null values in each column
print '\n', df.isnull().sum()

df = df.fillna(value=df.mean())
print '\n', df.isnull().sum()

count=0
for col,i in df.iteritems():
    if i.dtype != 'object':
        count += 1
        plt.subplot(4,6,count)
        sns.violinplot(y=col,data=df,color=np.random.rand(4))
        plt.show()

plt.figure(2)
correlation = df.corr()
sns.heatmap(correlation, cmap='inferno')
plt.figure(2).show()

df=df.drop(['TotalWorkingYears','PercentSalaryHike','JobLevel','YearsInCurrentRole','YearsWithCurrManager'], axis=1)

target_map = {'Yes':1, 'No':0}
target = df['Attrition'].apply(lambda x: target_map[x])
df_target = pd.DataFrame(target)
df = df.drop(['Attrition'], axis=1)

plt.figure(3)
correlation1 = df.corr()
sns.heatmap(correlation1, cmap='inferno')
plt.figure(3).show()

df = pd.get_dummies(df)

print df.head()
print df_target.head()


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df, df_target, test_size=0.3)

from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
X_train, Y_train = oversampler.fit_sample(X_train,Y_train)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

lr.fit(X_train, Y_train)

print lr.intercept_

predictions = lr.predict(X_test)

from sklearn import metrics

print 'Accuracy (Logistic Regression)=', metrics.accuracy_score(Y_test, predictions)

y = lr.coef_[0]
x = df.columns.values

plt.figure(4)
sns.stripplot(x,y,size=8,hue=y,palette='viridis').legend_.remove()
plt.xticks(rotation='vertical')
plt.figure(4).show()


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)

rf.fit(X_train, Y_train)

predictions1 = rf.predict(X_test)

print 'Accuracy (Random Forest)=', metrics.accuracy_score(Y_test, predictions1)

y1 = rf.feature_importances_
x1 = df.columns.values
print x1

plt.figure(5)
sns.stripplot(x1,y1,size=8,hue=y1,palette='viridis').legend_.remove()
plt.xticks(rotation='vertical')
plt.figure(5).show()