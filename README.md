# SCNSN_project
Performance Comparison of SGD, XGBoost, and Deep Neural Networks on the dataset of the Higgs Boson Machine Learning Challenge

# Content
This project aims at comparing the performances of three different architectural approaches: the linear SGD classifier, the decision-tree-based XGBoost classifier and a Keras/TensorFlow Deep Neural Network evaluated on the training dataset of the Higgs Boson Machine Learning Challenge.

# Dependencies
The git hub repositories contains the requirements.txt files, in which all the dependencies on which the projects relies were listed. However, in order to ensure the reproducibility fo the project, a container in which the script Higgs.py can be executed safely was created, and its image can be downloaded with the following command:

```js
docker pull feenaldi/scnsn_project:latest
```
# Execution
In order to execute the program, the content of the git repository must be downloaded and mounted during the execution of the container. To run the container, the following command can be used:

```js
docker run --user $(id -u):$(id -g) -v $PWD:/home/docker <image_id>
```

where the option `--user $(id -u):$(id -g)` ensures that the output files are owned by the current user; the option `-v $PWD:/home/docker` mounts the volume of the current working directory into the container, so that python script Higgs.py and the dataset training.csv can be accessed; finally, `<image_id>` must be replaced with the image identifier.

As soon as the container is run, the `Higgs.py` script is executed thanks to the following instruction: 
```js 
CMD ["python", "-W", "ignore", "Higgs.py"]
```
in the Dockerfile.

# Content of the Git Repository
The Git repository contains the following files:
* the `Higgs.py` script, which contains the source code and is executed by the container;
* the `Higgs.ipynb` notebook, which provides a visual representation of the script;
* the dataset `training.csv`, which is accessed by the `Higgs.py` script when executed;
* the `Dockerfile`, which contains the instructions to build the container;
* the `requirements.txt` file, which contains the dependencies of the project and is used to build the container;
* the `.gitignore` file, which was used to ignore the virtual environment in which the libraries were installed.

## Note on Dataset and Terms of Use
To streamline the evaluation process and ensure immediate reproducibility of the code—without requiring the examiners to register for the Kaggle competition—the dataset files have been directly included in this repository.

Please note that this [dataset](https://www.kaggle.com/competitions/higgs-boson/data) originate from the [**"Higgs Boson Machine Learning Challenge"**](https://www.kaggle.com/competitions/higgs-boson/overview) and are subject to its official terms ([Official Rules](https://www.kaggle.com/competitions/higgs-boson/rules)). Since the dataset cannot be freely redistributed, this repository is and will remain strictly **private**, intended solely for academic review by the professors.