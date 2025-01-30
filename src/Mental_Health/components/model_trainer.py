from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score

def train_models(data, target_column):
    """
    Train multiple machine learning models and store their records.
    
    Parameters:
    data (pd.DataFrame): The data to train the models on.
    target_column (str): The name of the target column.
    
    Returns:
    dict: Dictionary with model names as keys and (model, accuracy, f1, precision) as values.
    """
    X = data.drop([target_column], axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=500)  # Avoid convergence issues
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")  # Use 'weighted' for multi-class
        precision = precision_score(y_test, y_pred, average="weighted")  # Use 'weighted' for multi-class

        results[name] = (model, accuracy, f1, precision)

    return results