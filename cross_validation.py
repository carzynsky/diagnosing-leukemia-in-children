import select_features
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def run_crossvalid(X, y, k_best_features, neurons_in_hidden_layer):
    
    best_features_columns = select_features.create_feature_ranking_from_given_file(k_best_features)
    X = X[best_features_columns]
    scores = []
    X = X.values
    y = y.values[0]

    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2021)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sc_X = StandardScaler()
        X_trainscaled = sc_X.fit_transform(X_train)
        X_testscaled = sc_X.transform(X_test)

        clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(neurons_in_hidden_layer),random_state=2021, solver='lbfgs', max_iter=200).fit(X_trainscaled, y_train)
        y_pred = clf.predict(X_testscaled)
        scores.append(clf.score(X_testscaled, y_test))
    
    return scores