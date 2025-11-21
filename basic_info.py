import pandas as pd
from scipy.stats import trim_mean

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

def fix_weekend(og_df):
    df = og_df.copy()
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Friday', 'Saturday', 'Sunday'])
    return df

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

def trimmed_mean(df, trim_percent=2):
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    trimmed_means = {}
    proportion = trim_percent / 100
    for col in numerical_cols:        
        trimmed_means[col] = trim_mean(df[col], proportion)
    
    return trimmed_means


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
    
    '''df['Is_Holiday'] = pd.Categorical(df['Is_Holiday'],
                                categories=[True, False],
                                ordered=True)
    
    df['Is_Weekend'] = pd.Categorical(df['Is_Weekend'], 
                                categories=[True, False],
                                ordered=True)

    df['Helmet_Used'] = pd.Categorical(df['Helmet_Used'], 
                                categories=[True, False],
                                ordered=True)'''
    
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

    return probabilities
        
def attribute_impact(df, feature_name):
    user_experiences = ['Positive', 'Neutral', 'Negative']
    
    # Create cross-tabulation
    #cross_tab = pd.crosstab(df[feature_name], df['User_Experience'], margins=True)
    cross_tab_pct = pd.crosstab(df[feature_name], df['User_Experience'], 
                               normalize='index') * 100
    
    '''print(f"\nCounts:")
    print(cross_tab)
    print(f"\nPercentages (by row):")
    print(cross_tab_pct.round(1))'''
    
    # Add favors column - determines which experience has the highest percentage
    cross_tab_pct['Favors'] = cross_tab_pct[user_experiences].idxmax(axis=1)
    
    # For the 'All' row, we don't want to say it "favors" anything
    if 'All' in cross_tab_pct.index:
        cross_tab_pct.loc['All', 'Favors'] = 'Overall'
    
    #print(f"\nWith Favors Column:")
    #print(cross_tab_pct.round(1))
    
    return cross_tab_pct

def multi_attribute_impact(df, feature1, feature2, favor=True):
    user_experiences = ['Positive', 'Neutral', 'Negative']

    # Create a multi-index cross-tab
    cross_tab = pd.crosstab([df[feature1], df[feature2]], 
                        df['User_Experience'], 
                        normalize='index') * 100

    # Add favors column
    if favor:
        cross_tab['Favors'] = cross_tab[user_experiences].idxmax(axis=1)

    #print(f"{feature1} × {feature2} on User_Experience:")
    #print("=" * 60)

    return cross_tab

def attribute_impact_binned(df, feature_name):
    user_experiences = ['Positive', 'Neutral', 'Negative']
    
    # Check if the feature is numerical and should be binned
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        binning_strategies = {
            'Age': {'bins': [18, 31, 44, 57, 70], 'labels': ['18-31', '31-44', '44-57', '57-70']},
            'Temperature(C)': {'bins': [10, 15, 20, 25, 30, 35], 'labels': ['10-15', '15-20', '20-25', '25-30', '30-35']},
            'Distance_Travelled(km)': {'bins': [0, 1, 2, 3, 4, 5], 'labels': ['0-1', '1-2', '2-3', '3-4', '4-5']},
            'Humidity(%)': {'bins': [30, 45, 60, 75, 90], 'labels': ['30-45', '45-60', '60-75', '75-90']},
            'Satisfaction_Level(of 5)': {'bins': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 'labels': ['1', '2', '3', '4', '5']},
            'Traffic_Intensity(of 10)': {'bins': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
                                       'labels': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}
        }
        
        # Apply binning if strategy exists, otherwise create generic bins
        if feature_name in binning_strategies:
            strategy = binning_strategies[feature_name]
            binned_feature = pd.cut(df[feature_name], 
                                  bins=strategy['bins'], 
                                  labels=strategy['labels'],
                                  include_lowest=True)
        else:
            # Create generic bins for other numerical features
            unique_vals = df[feature_name].nunique()
            if unique_vals > 10:
                # For features with many unique values, create 5 bins
                binned_feature = pd.cut(df[feature_name], bins=5)
            else:
                # For features with few unique values, keep as is but convert to string
                binned_feature = df[feature_name].astype(str)
        
        # Use the binned feature for analysis
        working_feature = binned_feature
    else:
        # For categorical features, use as-is
        working_feature = df[feature_name]
    
    # Create cross-tabulation with percentages
    cross_tab_pct = pd.crosstab(working_feature, df['User_Experience'], 
                               normalize='index') * 100
    
    # Add favors column - determines which experience has the highest percentage
    cross_tab_pct['Favors'] = cross_tab_pct[user_experiences].idxmax(axis=1)
    
    # Sort the index for better readability (especially for binned numerical data)
    if hasattr(cross_tab_pct.index, 'categories'):
        # For categorical indexes, they're already ordered
        pass
    else:
        try:
            cross_tab_pct = cross_tab_pct.sort_index()
        except:
            # If sorting fails, keep as-is
            pass
    
    return cross_tab_pct


if __name__ == "__main__":
    df = pd.read_csv('Dataset_BicycleUse.csv')
    df = fix_weekend(df)
    df = sort_types(df)
    
    # Get all column names
    with open('Attribute_impact_binned.txt', 'w') as f:
        attributes = list(df.columns)
        attributes.remove('User_Experience')
        for feature in attributes:
            if feature in df.columns:
                table = attribute_impact_binned(df, feature)
                f.write(f'{feature}\n' + ('=' * 50) + f'\n{table}\n\n')

    time_and_weekend = multi_attribute_impact(df, 'Is_Weekend', 'Time_of_Day')
    print(time_and_weekend)
    #days = pd.crosstab(df['Is_Weekend'], df['Day_of_Week'], normalize='index')
    #print(days)