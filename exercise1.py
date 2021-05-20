import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv('irisdata.csv')
df = df.drop(list(range(0,50)))

#1a
def plot_2nd_3rd_classes(dataframe):
    dataframe = dataframe.drop(columns=['sepal_length', 'sepal_width'])
    grouped = dataframe.groupby(dataframe.species)
    df2 = grouped.get_group('versicolor')
    df3 = grouped.get_group('virginica')
    ax = df2.plot.scatter(x='petal_length',y='petal_width', c='g', label='versicolor')
    df3.plot.scatter(ax=ax, x='petal_length',y='petal_width', c='b', label='virginica')
    plt.grid()
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.show()


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(x)))

#1b
def nn_output(X, weight1, weight2, bias):
    weights = np.array([weight1, weight2]).T
    layer = sigmoid(np.dot(X, weights) + bias)
    output = []
    for i in range(0,len(layer)):
        if layer[i] < 0.5:
            output.append(1)
        else:
            output.append(0)
    return output

#1c
def plot_decision_boundary(dataframe):
    X = dataframe.iloc[:, [2,3]].values
    results = nn_output(X, -17, 68, -30)
    dataframe['classification'] = results
    grouped = dataframe.groupby(dataframe.classification)
    df2 = grouped.get_group(0)
    df3 = grouped.get_group(1)
    ax = df2.plot.scatter(x='petal_length',y='petal_width', c='g', label='0')
    df3.plot.scatter(ax=ax, x='petal_length',y='petal_width', c='b', label='1')
    plt.grid()
    plt.title("1c: Decision Boundary for Non-Linearity on Iris Data")
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    ax.legend(loc="upper left")
    plt.show()

#1d
def plot_surface(weight1, weight2, bias):
    x_axis = np.arange(2,8,0.0125)
    y_axis = np.arange(1,4,0.00625)
    X,Y = np.meshgrid(x_axis,y_axis)
    
    d = {'x':np.ravel(X),'y':np.ravel(Y)}
    dataframe = pd.DataFrame(data=d)
    weights = np.array([weight1, weight2]).T
    layer = sigmoid(np.dot(dataframe, weights) + bias)
    z_axis = layer.reshape(dataframe.x.shape)
    dataframe['classification'] = z_axis
    print(dataframe)
    Z = layer.reshape(X.shape)
    ax = Axes3D(plt.figure())
    ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0.2)
    plt.show()

#1e
def simple_classifier_examples():
    X = np.array([[3.9,1.2],[4.5,1.3],[5.0,1.7],[5.1,1.8],[5.8,2.2],[6.1,2.5]])
    dataframe = pd.DataFrame(X)
    dataframe['species'] = np.array([['versicolor'],['versicolor'],['versicolor'],['virginica'],['virginica'],['virginica']])
    results = nn_output(X, -17, 68, -30)
    dataframe['classification'] = results
    for i, row in dataframe.iterrows():
        print("Petal length: {}, Petal width: {}, Actual species: {}, Classification from neural net: {}".format(row[0], row[1], row['species'], row['classification']))

simple_classifier_examples()