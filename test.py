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
    cat_df = categorise_dataset(df)
    conditional_probabilities = conditional_prob(cat_df)
    with open('conditional_probabilities.txt', 'w') as f:
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

def trimmed_means(og_df):
    df = sort_types(og_df)
    with open('trimmed_means.txt', 'w') as f:
        trimmed = trimmed_mean(df)
        for col, mean in trimmed.items():
            f.write(f'{col}\t\t{mean}\n')

df = pd.read_csv('Dataset_BicycleUse.csv')
df = fix_weekend(df)
sorted_df = sort_types(df)

write_central_tendancy(df)
write_conditional_probs(df)
write_disperion(df)
trimmed_means(df)
