import sklearn.metrics as metrics
import sklearn.cross_validation as cv
from sklearn.externals import joblib
import MachineLearning.Reinforcement.InternalSQLManager as sqlManager

class ReinforcementLearner:

    def __init__(self, clf=None, load=False, clfName=None):
        """
        Initialise the Classifier, either from the provided model or from the stored classifier

        :param clf: The current classifier, not yet fitted to the data
        :param load: Set to True in order to load a previously saved model
        """

        if load:
            self.clf = joblib.load("model.pkl")
            self.reTrain = True
        else:
            self.clf = clf
            self.reTrain = False

        if clfName == None:
            self.name = self.clf.__class__.__name__
        else:
            self.name = clfName

    def fit(self, X, y, scoring="accuracy", crossval=5):
        """
        Fit the Reinforcement classifier with data, either adding to previous previous data or learning for first time.

        :param X: Input Features
        :param y: Class Labels
        :param scoring: Scoring used for cross validation
        :param crossval: Cross Validation number of folds
        :return: True if a new model is fit to the data, or a previous model is updated
                 False if old model when fit to new data performs poorly in comparison to
                 earlier data
        """
        if not self.reTrain: # Train first time
            score = cv.cross_val_score(self.clf, X, y, scoring, cv=crossval)

            sqlManager.insertValue(self.name, 0.0, score.mean(), 0, len(y), 1) # Store the first result of clf
            self.clf.fit(X, y)

            joblib.dump(self.clf, "model.pkl") # Store the CLF
            print("Data Fit")
            return True
        else:
            previousData = sqlManager.selectNewestRecord(self.name) # Check the last entry of CLF
            if len(previousData) > 0:
                oldSize = previousData[5]
                newSize = len(y)

                accScore = previousData[3]

                score = cv.cross_val_score(self.clf, X, y, scoring, cv=crossval)
                newAccScore = score.mean()
                print("Old Accuracy Score : ", accScore)
                print("New Accuracy Score : ", newAccScore)

                if accScore <= newAccScore: # If new data is benefitial, increases accuracy
                    print("Reinforcement Learning : Newer model is superior. Saving Model.")
                    self.clf.fit(X, y)

                    sqlManager.insertValue(self.name, accScore, newAccScore, oldSize, newSize, 1)
                    joblib.dump(self.clf, "model.pkl")
                    return True
                else:
                    print("Reinforcement Learning : Newer model is inferior. Not saving model.")
                    return False

    def predict(self, X):
        return self.clf.predict(X)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sqlManager.close()

if __name__ == "__main__":
    pass

