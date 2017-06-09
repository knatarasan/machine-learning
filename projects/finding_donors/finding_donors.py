import numpy as np
import pandas as pd


# Load the Census dataset
data = pd.read_csv("census.csv")

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

#def income_2_digit(x):                 
#    y=10    
#    if x=='<=50K':
#        y=0
#    elif x=='>50K':
#        y=1
#        return y

def income_2_digit(x):
     y=10
     if x=='<=50K':
         y=0
     elif x=='>50K':
         y=1
     return y



# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(income_2_digit)


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

