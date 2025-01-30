import mlflow
import mlflow.sklearn

def log_and_register_experiment(model, model_name, accuracy, f1, precision, params):
    """
    Log the experiment with MLflow and register the model in one step.
    
    Parameters:
    model: Trained model.
    model_name (str): Name of the model to register.
    accuracy (float): Accuracy score.
    f1 (float): F1 score.
    precision (float): Precision score.
    params (dict): Parameters used for training.
    """
    # Set the MLflow experiment
    mlflow.set_experiment("Mental Health Project2")
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Get the run_id
        run_id = run.info.run_id
        
        # Generate the model_uri
        model_uri = f"runs:/{run_id}/model"
        
        # Register the model
        mlflow.register_model(model_uri, model_name)
        print(f"Model '{model_name}' registered with URI: {model_uri}")