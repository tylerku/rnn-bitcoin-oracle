import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class GraphResults():

    def __init__(self):
        pass

    def hist(self, data, title):
        pass

    def plot(self, x, y, title, xlabel, ylabel):
        x = np.asarray(x)
        y = np.asarray(y)
        ss = np.std(y)
        plt.plot(x,y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim((-1,len(x)+1))
        plt.ylim((min(y) - ss, max(y) + ss))
        plt.show()

    def scatter(self, x, y, title, xlabel, ylabel):
        x = np.asarray(x)
        y = np.asarray(y)
        ss = np.std(y)
        plt.scatter(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        print("SS: ", ss)
        plt.ylim((min(y) - ss, max(y) + ss))
        plt.show()