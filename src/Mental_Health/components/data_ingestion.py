
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Drop missing values and duplicates
    data = data.dropna()
    data = data.drop_duplicates()
    
    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Apply Label Encoding to categorical columns
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    return data  # Returns the complete dataset after encoding
