import sklearn.metrics as metrics
import sklearn.cross_validation as cv
from sklearn.externals import joblib
import MachineLearning.Reinforcement.InternalSQLManager as sqlManager

class ReinforcementLearner:

    def __init__(self, clf=None, load=False):
        """
        Pass either a classifier, or set load = True to load from a model.pkl file
        """
        if load:
            self.clf = joblib.load("model.pkl")
            self.reTrain = True
        else:
            self.clf = clf
            self.reTrain = False

    def fit(self, X, y, scoring="accuracy", crossval=5):
        if not self.reTrain:
            score = cv.cross_val_score(self.clf, X, y, scoring, cv=crossval)

            sqlManager.insertValue(self.clf.__name__, 0.0, score.mean(), 0, len(y), 1)
            joblib.dump(self.clf, "model.pkl")
            return True
        else:
            previousData = sqlManager.selectNewestRecord()
            oldSize = previousData[5]
            newSize = len(y)

            yPreds = self.clf.predict(X)
            accScore = metrics.accuracy_score(y, yPreds)

            score = cv.cross_val_score(self.clf, X, y, scoring, cv=crossval)
            newAccScore = score.mean()

            if accScore <= newAccScore:
                print("Reinforcement Learning : Newer model is superior. Saving Model.")
                sqlManager.insertValue(self.clf.__name__, accScore, newAccScore, oldSize, newSize, 1)
                joblib.dump(self.clf, "model.pkl")
                return True
            else:
                print("Reinforcement Learning : Newer model is inferior. Not saving model.")
                return False

    def predict(self, X):
        return self.clf.predict(X)

if __name__ == "__main__":
    pass

