from basic_info import *
import pandas as pd

df = pd.read_csv('Dataset_BicycleUse.csv')
df = categorise_dataset(df)

#df = pd.read_csv('test.csv')

#get_dataset_summary(df)

conditional_probabilities = conditional_prob(df)

for col, data in conditional_probabilities.items():
    print(f'{col}')
    for given, experiences in data.items():
        print(f'\t{given}:')
        for experience, result in experiences.items():
            print(f'\t\t{experience}: {result:.4f}')
        print('')
    print('')

#print(df['Is_Holiday'].dtype)