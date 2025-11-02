import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import pickle

def model_1_train_and_dump():
    # retrieve the data
    df_coffee = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv')
    df_coffee.drop_duplicates(subset='desc_1', inplace=True)



    # split the data
    df_train, df_test = train_test_split(df_coffee, test_size=0.2, random_state=42)


    # define features and target
    feature_col = ['100g_USD']
    target_col = 'rating'


    # define what features/target constitutde X and y for training
    X = df_train[feature_col]
    y = df_train[feature_col]

    # instantiate the model
    model = LinearRegression()

    # fit the model
    model.fit(X, y)

    # write the model to pickle
    pickle.dump(model, open('model_1.pickle', 'wb'))


def write_map(dataframe, mapped_field, pickle_filename):
    mapping = {name: idx for idx, name in enumerate(dataframe[mapped_field].unique())}
    with open(pickle_filename, 'wb') as f:
        pickle.dump(mapping, f)

def read_mapping(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:
        mapping = pickle.load(f)
    return mapping

def roast_category():
    return read_mapping('roast_map.pickle')



def model_2_train_and_dump():
    # retrieve the data
    df_coffee = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/coffee_analysis.csv')
    df_coffee.drop_duplicates(subset='desc_1', inplace=True)

    # split the data
    df_train, df_test = train_test_split(df_coffee, test_size=0.2, random_state=42)



    # define features and target
    feature_col = ['100g_USD', 'roast']
    target_col = 'rating'

    # define what features/target constitutde X and y for training

    # first fully deal with x and the mapping
    # create the mapping
    write_map(df_train, 'roast', 'roast_map.pickle')

    # read the mapping pickle
    roast_map = read_mapping('roast_map.pickle')


    # create an X which is a copy of df_train but with the roast mapped
    X = df_train[feature_col].copy()
    # apply the mapping
    X['roast'] = X['roast'].map(roast_map)
    # X now has the two columns we need, but roast is mapped to integers

    # rename the field to suit what the autograder expects
    X = X.rename(columns={'roast': 'roast_cat'})

    # now deal with y
    y = df_train[target_col]

    # instantiate the model
    model = DecisionTreeRegressor(max_depth=2)

    # fit the model
    model.fit(X, y)

    # write the model to pickle
    pickle.dump(model, open('model_2.pickle', 'wb'))


model_1_train_and_dump()
model_2_train_and_dump()

# test code
dtr = pickle.load(open('model_2.pickle', 'rb'))


df_X = pd.DataFrame([
    [10.00, 1],
    [15.00, 3],
    [8.50, np.nan]], 
    columns=["100g_USD", "roast_cat"])

y_pred = dtr.predict(df_X.values)   # `dtr` is a DecisionTreeRegressor  
print(y_pred)

