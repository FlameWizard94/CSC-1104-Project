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
    if df[column_name].dtype == 'object':
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
    summary = {}
    for column in df.columns:
        summary[column] = analyze_column_values(df, column)
    return summary

# Example usage:
if __name__ == "__main__":
    # Load the dataset
    df = load_dataset('Dataset_BicycleUse.csv')
    
    # Get all column names
    columns = get_column_info(df)
    print("All columns:")
    for x in columns:
        print(x)
    print(f"Num columns: {len(columns)}")
    
    # Analyze a specific column (categorical example)
    weather_info = analyze_column_values(df, 'Weather')
    print("\nWeather column analysis:", weather_info)
    
    # Analyze a specific column (numerical example)
    distance_info = analyze_column_values(df, 'Distance_Travelled(km)')
    print("\nDistance column analysis:", distance_info)
    
    # Get complete dataset summary
    full_summary = get_dataset_summary(df)
    print("\nFull dataset summary:")
    for col, info in full_summary.items():
        print(f"{col}: {info}")