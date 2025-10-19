import pandas as pd
import numpy as np

def analyze_column_values(df, column_name):
    if df[column_name].dtype == 'object' or df[column_name].dtype == 'category':
        # Categorical column
        unique_values = df[column_name].unique().tolist()
        return {
            'type': 'categorical',
            'values': sorted(unique_values) if all(isinstance(x, (int, float)) or x is None for x in unique_values) else unique_values
        }
    else:
        # Numerical column
        return {
            'type': 'numerical',
            'range': {
                'min': df[column_name].min(),
                'max': df[column_name].max()
            }
        }

def get_dataset_summary(df):
    print('Summary\n--------------------------------------')
    for name in df.columns:
        column = analyze_column_values(df, name)
        print(f'\n{name}\nType : {column['type']}')
        if 'values' in column.keys():
            for x in column['values']:
                print(x, end=', ')
            print('')
        else:
            for x, y in column['range'].items():
                print(f'{x} : {y}')

def sort_types(df):
    sorted_df = df.copy()
    
    # Convert to Numerical
    numerical_columns = ['Age', 'Distance_Travelled(km)', 'Temperature(C)', 'Humidity(%)']
    for col in numerical_columns:
        sorted_df[col] = pd.to_numeric(sorted_df[col].astype(str), errors='coerce')

    # Convert to categorical
    categorical_columns = ['Bike_Type', 'Gender', 'Occupation', 'Weather', 'Day_of_Week', 
                        'Time_of_Day', 'Purpose_of_Ride', 'Road_Condition', 'User_Experience']
    for col in categorical_columns:
        sorted_df[col] = sorted_df[col].astype('category')

    # Convert to boolean
    boolean_columns = ['Is_Holiday', 'Is_Weekend', 'Helmet_Used']
    for col in boolean_columns:
        # First ensure they're strings, then map to boolean
        sorted_df[col] = sorted_df[col].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})

    # Handle ordinal/numerical columns
    ordinal_columns = ['Satisfaction_Level(of 5)', 'Traffic_Intensity(of 10)']
    for col in ordinal_columns:
        sorted_df[col] = pd.to_numeric(sorted_df[col].astype(str), errors='coerce')
    
    return sorted_df

def central_tendancy(df):
    output = ''
    means = df.select_dtypes(include=['number']).mean()
    modes = df.select_dtypes(include=['number']).mode()
    medians = df.select_dtypes(include=['number']).median()
    columns = list(df.columns)

    #Means
    output += f'Means\n-----------\n'
    output += f"{means}, \n"

    #Medians
    output += f'\nMedians\n-----------\n'
    output += f'{medians}, \n'

    #Modes
    output += f'\nModes\n----------\n'
    for col in columns:
        output += f'{col}:\t{df[col].mode().iloc[0]}\n'
    
    return output

def dispersion(df):
    output = {}
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        output[col] = {'Q1' : float(df[col].quantile(0.25)),
                       'Q2' : float(df[col].quantile(0.50)),
                       'Q3' : float(df[col].quantile(0.75)),
                       'IQR' : round( float(df[col].quantile(0.75) - df[col].quantile(0.25)), 3),
                       'Max' : float(df[col].max()),
                       'Min' : float(df[col].min()),
                       'Var' : round(float(df[col].var()), 3),
                       'Std' : round(float(df[col].std()), 3)}
        
    return output
        


def categorise_dataset(og_df):                                
    df = og_df.copy()
    df['Satisfaction_Level(of 5)'] = pd.Categorical(df['Satisfaction_Level(of 5)'], 
                                               categories=[1, 2, 3, 4, 5],
                                               ordered=True)

    df['Traffic_Intensity(of 10)'] = pd.Categorical(df['Traffic_Intensity(of 10)'], 
                                                categories=range(1, 11),
                                                ordered=True)
    
    df['Temperature(C)'] = pd.cut(df['Temperature(C)'], 
                                bins=[10, 15, 20, 25, 30, 35],
                                labels=['10-15', '15-20', '20-25', '25-30', '30-35'],
                                include_lowest=True)

    df['Age'] = pd.cut(df['Age'], 
                        bins=[18, 31, 44, 57, 70],
                        labels=['18-31', '31-44', '44-57', '57-70'],
                        include_lowest=True)

    df['Distance_Travelled(km)'] = pd.cut(df['Distance_Travelled(km)'], 
                                    bins=[0, 1, 2, 3, 4, 5],
                                    labels=['0-1', '1-2', '2-3', '3-4', '4-5'],
                                    include_lowest=True)

    df['Humidity(%)'] = pd.cut(df['Humidity(%)'], 
                                    bins=[30, 45, 60, 75, 90],
                                    labels=['30-45', '45-60', '60-75', '75-90'],
                                    include_lowest=True)
    
    df['Is_Holiday'] = pd.Categorical(df['Is_Holiday'],
                                categories=[True, False],
                                ordered=True)
    
    df['Is_Weekend'] = pd.Categorical(df['Is_Weekend'], 
                                categories=[True, False],
                                ordered=True)

    df['Helmet_Used'] = pd.Categorical(df['Helmet_Used'], 
                                categories=[True, False],
                                ordered=True)
    
    return df

def get_experiences(df):
    positive = df['User_Experience'].value_counts()['Positive']
    neutral = df['User_Experience'].value_counts()['Neutral']
    negative = df['User_Experience'].value_counts()['Negative']

    experiences = {'Positive' : positive * 1000, 
                   'Neutral' : neutral * 1000,
                   'Negative' : negative * 1000}
    
    return experiences

def conditional_prob(df):
    columns = list(df.columns)
    columns.remove('User_Experience')
    probabilities = {}

    for col in columns:
        probabilities[col] = {}

        #If categorical column
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            unique_values = df[col].unique().tolist()
            group = df.groupby(col, observed=False)['User_Experience'].value_counts(normalize=True).unstack(fill_value=0).stack()
            for category in unique_values:
                probabilities[col][f'Given {category}'] = {'Positive' : float(group[category]['Positive']),
                                                 'Neutral' : float(group[category]['Neutral']),
                                                 'Negative' : float(group[category]['Negative'])}
                
        else:
            print(f'{col} is {df[col].dtype}')

        
        #if categorical column treated as numerical
        '''elif col == 'Satisfaction_Level(of 5)' or col == 'Traffic_Intensity(of 10)':
            group = df.groupby(col)['User_Experience'].value_counts(normalize=True)
            max = len(group.keys()) / 3
            for x in range(1, max + 1):
                probabilities[col][f'Given {x}'] = {'Positive' : float(group.loc[(x, 'Positive')]),
                                                 'Neutral' : float(group.loc[(x, 'Neutral')]),
                                                 'Negative' : float(group.loc[(x, 'Negative')])}'''

        #ranges
        #else:
            #group = df.groupby(f'{col}')['User_Experience'].value_counts(normalize=True)

    return probabilities

        

if __name__ == "__main__":
    df = pd.read_csv('Dataset_BicycleUse.csv')
    
    # Get all column names
    columns = list(df.columns)
    print("All columns:")
    for x in columns:
        print(x)
    print(f"Num columns: {len(columns)}\n")
    
    get_dataset_summary(df)
