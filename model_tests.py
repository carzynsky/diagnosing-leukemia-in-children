import cross_validation
import load_data
import pandas as pd

data_path = 'Data/białaczka.XLS'
features_path = 'Data/features.txt'

#load features
features = load_data.load_features(features_path)

# load data 
(X, y) = load_data.load_data_from_files(data_path, features)

# params
k_best_features = [10, 15, 20]
neurons_in_hidden_layer = [32, 64, 256]
momentum = [0, 0.9]

# data frame
df = pd.DataFrame(columns=['Best features', 'Neurons in hidden layer', 'Momentum', 'Average'])
for i in range(len(k_best_features)):
    for j in range(len(neurons_in_hidden_layer)):
        for k in range(len(momentum)):
            scores = cross_validation.run_crossvalid(X, y, k_best_features[i], neurons_in_hidden_layer[j], momentum[k])
            average=0
            if(len(scores)!=0):
                average = sum(scores) / len(scores)
            df = df.append({'Best features': k_best_features[i], 'Neurons in hidden layer': neurons_in_hidden_layer[j],'Momentum': momentum[k], 'Average': average}, ignore_index=True)

with pd.ExcelWriter('Data/model_tests.xls') as writer:            
    df.to_excel(writer, 'model_tests')
    writer.save()