import select_features
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_crossvalid(X, y, k_best_features, neurons_in_hidden_layer, momentum_param):

    best_features_columns = select_features.create_feature_ranking_from_given_file(k_best_features)
    X = X[best_features_columns]
    scores = []
    X = X.values
    y = y.values[0]

    clf = MLPClassifier(hidden_layer_sizes=(neurons_in_hidden_layer),random_state=1234, momentum=momentum_param, solver='sgd', max_iter=4000)
    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1234)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(clf.score(X_test, y_test))
    
    return scores