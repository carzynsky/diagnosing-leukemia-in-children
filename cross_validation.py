import select_features
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def run_crossvalid(X, y, k_best_features, random_state):
    scores = []
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values[0], random_state=random_state, test_size=0.2)
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(256),activation="relu",random_state=random_state).fit(X_trainscaled, y_train)
    y_pred = clf.predict(X_testscaled)
    print(clf.score(X_testscaled, y_test))



    # knn
    # split_algorithm = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # for train_samples_indexes, test_samples_indexes in split_algorithm.split(X_features, Y_diagnosis):
    #     X_train = X_features.iloc[train_samples_indexes]
    #     X_test = X_features.iloc[test_samples_indexes]
    #     Y_train = Y_diagnosis.iloc[train_samples_indexes]
    #     Y_test = Y_diagnosis.iloc[test_samples_indexes]

    #     X_train_best, X_test_best = select_features.select_k_best_features_train_and_test(X_train, Y_train, X_test, k_best_features)

    #     knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    #     knn.fit(X_train_best, Y_train.values.ravel())

    #     scores.append(knn.score(X_test_best, Y_test))

    return scores