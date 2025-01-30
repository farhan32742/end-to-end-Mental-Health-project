import mlflow
from src.Mental_Health.components.model_trainer import train_models
from src.Mental_Health.components.data_ingestion import load_data,preprocess_data
from src.Mental_Health.mlflow_Integration import log_and_register_experiment
from src.Mental_Health.logging import logging
#from model_registry import model_register

def main():
    # Load and preprocess data
    data = load_data("D:\\DATA SCIENCE\\Mental health project\\Mental Health Professional Dataset.csv")
    logging.info("successfully load the model")
    data = preprocess_data(data)
    
    # Train model
    target_column = "Mental health affected"  # Replace with your target column name
    results = train_models(data, target_column)
    #model_name ="logestic regression"
    #run_id = "61a6115d70cc45c2a5264f7bbf87aeec"
    #model_uri = f"runs:/{run_id}/model"

    # Log experiments for each model
    for model_name, (model, accuracy, f1, precision) in results.items():
        params = {
            "model_type": model_name,
            "test_size": 0.2,
            "random_state": 42
        }
        log_and_register_experiment(model, model_name, accuracy, f1, precision, params)
        #model_register(model_name,model)

if __name__ == "__main__":
    main()