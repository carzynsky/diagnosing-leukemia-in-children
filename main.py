import load_data
import select_features
import cross_validation
import statistical_analysis

data_path = 'Data/białaczka.XLS'
features_path = 'Data/features.txt'

#load features
features = load_data.load_features(features_path)

# load data 
(X, y) = load_data.load_data_from_files(data_path, features)

#feature ranking
ranking = select_features.create_feature_ranking(X, y)

#cross_validation
k_best_features = 15
neurons_in_hidden_layer = 256
scores = cross_validation.run_crossvalid(X, y, k_best_features, neurons_in_hidden_layer, 0.9)
print('Scores: ', scores)

average=0
if(len(scores)!=0):
    average = sum(scores) / len(scores)

print('Average: ', average)

# run statistical analysis
statistical_analysis.run_statistical_analysis(X, y)

