from basic_info import *
import pandas as pd

def write_disperion(sorted_df):
    with open("dispersion.txt", "w") as f:
        sorted_df = sort_types(df)
        disper = dispersion(sorted_df)
        for col, data in disper.items():
            f.write(f'{col}\n----------------\n')
            for k, v in data.items():
                f.write(f'{k}:\t\t\t{v}\n')
            f.write('\n')

def write_conditional_probs(df):
    with open('conditional_probabilities.txt', 'w') as f:
        conditional_probabilities = conditional_prob(df)
        for col, data in conditional_probabilities.items():
            f.write(f'{col}\n')
            for given, experiences in data.items():
                f.write(f'\t{given}:\n')
                for experience, result in experiences.items():
                    f.write(f'\t\t{experience}: {result:.4f}\n')
                f.write('\n')
            f.write('\n')

def write_central_tendancy(df):
    with open('central_tendancies.txt', 'w') as f:
        sorted_df = sort_types(df)
        f.write(central_tendancy(sorted_df))

df = pd.read_csv('Dataset_BicycleUse.csv')
sorted_df = sort_types(df)

write_central_tendancy(df)
write_conditional_probs(df)
write_disperion(df)
