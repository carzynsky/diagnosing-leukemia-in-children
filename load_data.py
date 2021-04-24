import pandas as pd
import numpy as np
import math

# load features from txt file
def load_features(features_path):
    features = pd.read_csv(features_path, header=None, error_bad_lines=False)
    features = features[0].tolist()
    return features

def load_data_from_files(data_path, features):
    X_features = pd.DataFrame(columns=features)
    Y_diagnosis = pd.DataFrame(columns=['Diagnoza'])

    X = pd.read_excel(data_path)

    y=[]
    X_val = X.fillna(0).values
    for i in range(len(X_val)):
            y.append(X_val[i][0])

    X = X.iloc[:,2:22]
    X.columns = features
    list = np.arange(len(y))
    Y = pd.DataFrame([y], columns=list)
    return (X,Y)
    # for i in range(len(Y)):
    #     Y[i] = y[i]
    # print(Y)
    # Y = pd.DataFrame(y) 