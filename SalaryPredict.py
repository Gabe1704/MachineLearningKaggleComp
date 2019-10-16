"""
Income Prediction

Importing Data
Data imported from Kaggle competition https://www.kaggle.com/c/tcdml1920-income-ind
Pandas will be used to read csv files

Frameworks used
scikit learn
pandas
CSV
numpy

Basic Run Through of Solution
Import Data
Combine datasets for solution
Do not combine for testing

Impute Data to get rid of NaNs and Garbage Data
Drop non useful columns
Transform categorical data into nominal data
Scale data for training

Split dataset for training vs testing
Train Model
Predict Info needed with Algorithm below
Run any checks

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import csv
from sklearn.ensemble import RandomForestRegressor


class DataFrameImputer(TransformerMixin):
#Impute NaNs using scikit-learn SimpleImpute Class and getting rid of garbage data

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def ChangeStringToDummyNumbers(dataset):
    dataset = pd.get_dummies(dataset, dummy_na=True)
    return dataset

def PlotGraph(data):
    sns.set()
    sns.set_palette('Spectral')
    sns.set_style('whitegrid')

    sns.jointplot(x='Income in EUR', y='Body Height [cm]', data=dataset)


def main():
    #dataset = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
    #datasetTest = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

    reader = csv.reader(open("tcd ml 2019-20 income prediction training (with labels).csv"))
    reader1 = csv.reader(open("tcd ml 2019-20 income prediction test (without labels).csv"))
    f = open("combined.csv", "w")
    writer = csv.writer(f)

    for row in reader:
        writer.writerow(row)
    for row in reader1:
        writer.writerow(row)
    f.close()

    # had to manually combine the above CSVs as the aboce code was creating a Memory Error
    dataset = pd.read_csv("hope.csv")

    #dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    #dataset['Income'] = dataset['Income'].fillna(0)

    #Impute data with mean for numerical and most frequent for categorical
    dataset = pd.DataFrame(dataset)
    dataset = DataFrameImputer().fit_transform(dataset)

    #Drop instance column as it is not important in the training of the model
    dataset = dataset.drop(['Instance'], axis=1)


    #Making dummies for the categorical data
    datasetDummy = ChangeStringToDummyNumbers(dataset)
    datasetCat = datasetDummy.iloc[:, 6:]
    datasetNom = datasetDummy.iloc[:, :5]
    datasetY = datasetDummy['Income']


    #scaling the nom data to be used in the algorithm
    min_max_scaler = preprocessing.MinMaxScaler()
    datasetNomScaled = min_max_scaler.fit_transform(datasetNom[['Year of Record', 'Age', 'Size of City', 'Body Height [cm]']])
    datasetNomScaled = pd.DataFrame(datasetNomScaled)
    datasetX = pd.concat([datasetNomScaled, datasetCat], axis=1)

    #datasetX = datasetX.fillna(datasetX.mean())

    #Decided to split data manually since of the slightly awkward split needed
    #XTrain, XTest, YTrain, YTest = train_test_split(datasetX, datasetY, test_size=(73230/185223), random_state=1)
    XTrain = datasetX.head(111993)
    YTrain = datasetY.head(111993)
    XTest = datasetX.tail(73230)
    YTest = datasetY.tail(73230)


    #TRIED TO USE MLP REGRESSOR BUT KEPT RUNNING OUT OF ITERATIONS BEFORE ALL CALCULATIONS WERE FINISHED
    #mlp_regressor = MLPRegressor(XTrain, YTrain)
    #mlp_regressor = MLPRegressor(hidden_layer_sizes=(50,2),
    #                     activation='relu',
    #                     solver='adam',
    #                     learning_rate='adaptive',
    #                     max_iter=1000,
    #                     learning_rate_init=0.01,
    #                     alpha=0.01)
    # MLPRegressor -> hiddent_layer_sizes=(30,) solver='lbfgs'
    #mlp_regressor.fit(XTrain, YTrain)

    #PredictY = mlp_regressor.predict(XTest)

    #Changed to Random Forrest since it is less heavy on computational resources
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(XTrain, YTrain)
    PredictY = regressor.predict(XTest)

    #Root mean squared check - Meant only for when testing with data that has all the data IE not having Income missing
    rms = sqrt(mean_squared_error(YTest, PredictY))
    print(rms)
    print(PredictY)
    PredictY = pd.DataFrame(PredictY)
    PredictY.to_csv("output5.csv")

    #checking for leftover NaNs
    #print(datasetDummy.count())

if __name__ == '__main__':
  main()
