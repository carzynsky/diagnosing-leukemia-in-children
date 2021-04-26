import select_features
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def run_crossvalid(X, y, k_best_features, neurons_in_hidden_layer, random_state):

    best_features_columns = select_features.create_feature_ranking_from_given_file(k_best_features)
    X = X[best_features_columns]

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values[0], random_state=random_state, test_size=0.05)
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    clf = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(neurons_in_hidden_layer),random_state=random_state, solver='lbfgs', max_iter=200).fit(X_trainscaled, y_train)
    y_pred = clf.predict(X_testscaled)
    score = clf.score(X_testscaled, y_test)
    return score