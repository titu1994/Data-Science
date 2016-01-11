import sklearn.svm as svm

if __name__ == "__main__":
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

    y = [0, 1, 1, 0]

    print("X : ", X, "\nY : ", y, "\n")

    print("Beginning XOR gate test with Linear SVM")
    svc = svm.SVC(kernel="linear")
    svc.fit(X, y)

    print("Predictions : ", svc.predict(X), "\nFails due to problem being non linearly seperable.")

    print("\nBeginning XOR gate test with RBF Kernel SVM")

    svc = svm.SVC(kernel="rbf")
    svc.fit(X, y)

    print("Predictions : ", svc.predict(X), "\nSucceeds because Radial Basis Function Kernel trick is applied to linear SVM")