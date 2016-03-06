from MachineLearning.Reinforcement import ReinforcementLearner as RL
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target

    X1, X2, y1, y2 = train_test_split(X, y, test_size=0.80, random_state=0)

    #dt = DecisionTreeClassifier(max_depth=2, random_state=0)
    #clf = RL.ReinforcementLearner(clf=dt)
    #clf.fit(X1, y1, crossval=10)

    #clf = RL.ReinforcementLearner(load=True)
    #clf.fit(X2, y2)

    #clf = RL.ReinforcementLearner(load=True)
    #clf.fit(X, y)
