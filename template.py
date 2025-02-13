import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
project_name =  "Mental_Health"

list_of_files = [
".github/workflows/.gitkeep",
f"src/{project_name}/__init__.py",
f"src/{project_name}/artifacts",
f"src/{project_name}/components/__init__.py",
f"src/{project_name}/components/data_ingestion.py",
f"src/{project_name}/components/model_trainer.py",
f"src/{project_name}/experimen_file/__init__.py",
f"src/{project_name}/experimen_file/mental_health.ipynb",
f"src/{project_name}/logging.py",
f"src/{project_name}/exception.py",
f"src/{project_name}/mlflow_Integration.py",
f"templates/indix.html",
f"static",
"app.py",
"Dockerfile",
"requirements.txt",
"setup.py",
"main.py",
"app.py",
]


for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory:{file_dir} for the file {file_name}")


    if (not os.path.exists(file_path)) or (os.path.getsize(file_path==0)):
        with open(file_path,"w") as f:
            pass
            logging.info(f"creating empty files")

    else:
        logging.info(f"{file_name} is already exists")
