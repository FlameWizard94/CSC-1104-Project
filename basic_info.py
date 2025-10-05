import pandas as pd

def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def get_column_info(df):
    """
    Get information about all columns in the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        list: List of all column names
    """
    return list(df.columns)

def analyze_column_values(df, column_name):
    """
    Analyze the values in a specific column.
    For categorical columns: returns unique values
    For numerical columns: returns min and max range
    
    Args:
        df (pandas.DataFrame): The dataset
        column_name (str): Name of the column to analyze
        
    Returns:
        dict: Analysis results with 'type' and 'values' or 'range'
    """
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
    """
    Get a complete summary of all columns in the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset
        
    Returns:
        dict: Summary of all columns with their possible values or ranges
    """
    '''summary = {}
    for column in df.columns:
        summary[column] = analyze_column_values(df, column)
    return summary'''

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

def categorise_dataset(df):
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

    experiences = {'Positive' : positive, 
                   'Neutral' : neutral,
                   'Negative' : negative}
    
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
    df = load_dataset('Dataset_BicycleUse.csv')
    
    # Get all column names
    columns = get_column_info(df)
    print("All columns:")
    for x in columns:
        print(x)
    print(f"Num columns: {len(columns)}\n")
    
    get_dataset_summary(df)
