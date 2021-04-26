from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def select_k_best_features(X_features, Y_diagnosis, k_best_features):
    # create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_features, Y_diagnosis)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_best_features = X_features.iloc[:, new_features]

    return X_best_features

def select_k_best_features_train_and_test(X_train, Y_train, X_test, k_best_features):
    # create and fit selector
    select_k_best_classifier = SelectKBest(score_func=chi2, k=k_best_features)
    select_k_best_classifier.fit(X_train, Y_train)

    # get columns to keep and create new DataFrame with those only
    new_features = select_k_best_classifier.get_support(indices=True)
    X_train_best_features = X_train.iloc[:, new_features]
    # create second DataFrame from test set which contains only selected features
    X_test_best_features = X_test.iloc[:, new_features]

    return (X_train_best_features, X_test_best_features)

def create_feature_ranking(X, y):
    X_values = X.values
    y_values = y.values[0]
    (chi, pval) = chi2(X_values, y_values)
    result = pd.DataFrame(X.columns, columns=['Feature name'])
    result["chi"] = chi
    result["pval"] = pval
    result.sort_values(by=['chi'], ascending=False, inplace=True)

    return result

def create_feature_ranking_from_given_file(k):
    f = pd.read_excel('Data/bestFeatures.xls')
    f = f.sort_values(by=['chi'], ascending=False)
    best_features_arr = f['cecha'].values
    return best_features_arr[:k]