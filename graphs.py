from basic_info import *
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset_BicycleUse.csv')
df = sort_types(df)
columns = list(df.columns)
numerical_cols = df.select_dtypes(include=['number']).columns

for col in numerical_cols:
    plt.figure()
    plt.boxplot(df[col])
    plt.savefig(f'boxplots/{col}.png')
    plt.close()
