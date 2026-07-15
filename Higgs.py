#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary libraries

#-------------- Python Libraries ------------------
import numpy as np # For linear algebra and numerical operations
import pandas as pd # For data processing and reading CSV files
import matplotlib.pyplot as plt # For basic visualizations
import seaborn as sns # For more appealing visualizations
import os
import scipy.stats as stats
import time

#-------------- Scikit Learn ----------------------
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

#-------------- XGBoost ----------------------------
import xgboost as xgb

#-------------- Keras ------------------------------
from tensorflow.keras import models
from tensorflow.keras.layers import Dense


# set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# do not use graphic interface
plt.switch_backend('Agg')

print("Libraries imported successfully!")


# In[ ]:


# find the directory of the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# data path
DATA_PATH = os.path.join(SCRIPT_DIR, "training.csv")

print(f"I am looking for the dataset in: {DATA_PATH}\n")
dataset_df = pd.read_csv(DATA_PATH)


# 1. Data inspection: after inspecting its format and shape, the dataset is going to be split into features (X_data) and labels (y_data), and the distribution of lables is going to be analysed. After that, the format and shape of X_data and y_data is also going to be inspected.

# In[ ]:


# display the first 5 rows of the dataset
print("Dataset Head:")
print(dataset_df.head())

print("Data loaded successfully\n")


# In[4]:


# split the dataset in features and labels, remove weights 
X_data = dataset_df.drop(columns=["EventId", "Weight", "Label"])
y_data = dataset_df["Label"]


# In[ ]:


print(f"Features shape: {X_data.shape}\n")
print(f"Features info: {X_data.info()}\n")


# In[ ]:


print(f"Labels shape: {y_data.shape}\n")
print(f"Labels info: {y_data.info()}\n")


# In[ ]:


print("Dataset Features Description:\n")
print(f"{X_data.describe()}\n")


# In[ ]:


# analyse label distribution
print("Target Label Distribution:\n")
print(f"{y_data.value_counts()}\n")


# 2. Explorative Data Analysis: In this part, the distribution of three sample features ("DER_mass_MMC", "PRI_tau_eta", "DER_mass_jet_jet") in the training set is goign to be plotted. At first, this is done taking into consideration all the values for that feature; secondly, singal events are separated from background events and, for each sample feature, the distribution is plotted separately. The information on the ditribution of the features is going to be used to decide if nan values should be filled with mean or median of a given feature.

# In[9]:


# split the training set into signal events and background events
signal = X_data[y_data == "s"]
background = X_data[y_data == "b"]


# In[ ]:


os.makedirs(f"{SCRIPT_DIR}/output/EDA/DER_mass_MMC/", exist_ok=True)

# In[ ]:


X_data["DER_mass_MMC"].hist(bins=50)
plt.title("Distribuzione DER_mass_MMC")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/DER_mass_MMC/full.pdf", bbox_inches='tight')


# In[ ]:


plt.hist(signal["DER_mass_MMC"], bins=50, alpha=0.5, label="Signal")
plt.hist(background["DER_mass_MMC"], bins=50, alpha=0.5, label="Background")

plt.legend()
plt.title("Distribuzione DER_mass_MMC")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/DER_mass_MMC/split.pdf", bbox_inches='tight')
plt.close()

# In[ ]:


os.makedirs(f"{SCRIPT_DIR}/output/EDA/PRI_tau_eta/", exist_ok=True)


# In[ ]:


X_data["PRI_tau_eta"].hist(bins=50)
plt.title("Distribuzione PRI_tau_eta")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/PRI_tau_eta/full.pdf", bbox_inches='tight')


# In[ ]:


plt.hist(signal["PRI_tau_eta"], bins=50, alpha=0.5, label="Signal")
plt.hist(background["PRI_tau_eta"], bins=50, alpha=0.5, label="Background")

plt.legend()
plt.title("Signal vs Background - PRI_tau_eta")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/PRI_tau_eta/split.pdf", bbox_inches='tight')
plt.close()

# In[ ]:


os.makedirs(f"{SCRIPT_DIR}/output/EDA/DER_pt_tot/", exist_ok=True)


# In[ ]:


X_data["DER_pt_tot"].hist(bins=50)
plt.title("Distribuzione DER_pt_tot")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/DER_pt_tot/full.pdf", bbox_inches='tight')


# In[ ]:


plt.hist(signal["DER_pt_tot"], bins=50, alpha=0.5, label="Signal")
plt.hist(background["DER_pt_tot"], bins=50, alpha=0.5, label="Background")

plt.legend()
plt.title("Signal vs Background - DER_pt_tot")
plt.savefig(f"{SCRIPT_DIR}/output/EDA/DER_pt_tot/split.pdf", bbox_inches='tight')
plt.close()

# 3. Data Preprocessing: the dataset format is going to be preprocessed so that it can be prepared for the training part according to the project purposes. In particular, labels "b" and "s" are going to be converted into integers "0" and "1", respectively. To add to this, -999.0 values are going to be handled by replacing them with "nan", so that they do not affect the statistics for that column. After splitting the dataset into a training set and a test set, for each feature, "nan" values are going to be filled with the corresponding median. 

# In[ ]:


# convert string-type lables into integers 
y_data.replace("b", 0, inplace=True)
y_data.replace("s", 1, inplace=True)
print(f"replacement of lables:\n")
print(f"{y_data}\n")


# In[ ]:


# replace -999.0 with nan
X_data.replace(-999.0, np.nan, inplace=True)

# check the amount of missing values
missing_percent = X_data.isna().mean() * 100

# sort them descendingly
missing_percent = missing_percent.sort_values(ascending=False)

print("percentage of missing values:\n")
print(f"{missing_percent}\n")


# In[18]:


# split the dataset into traing and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    stratify=y_data,    
    test_size=0.2,
    random_state=42
)


# In[19]:


# fill the missing values with median
X_tr = X_train.fillna(X_train.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))


# Note that the missing values for training set and test set were filled separately in order to avoid contaminating the training set with information coming from the test set.

# 4. Definition of Machine Learning Models: a linear SGD classifier, a decision-tree-based XGBoost classifier and a simple Deep Neural Network are going to be trained on the training set, which needs to be scaled to imporve the performance of the SGD Classifier and of the DNN.

# In[20]:


# scale the dataset first; so that features lies between 0 and 1
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_tr)
X_test_scaled = scaler1.fit_transform(X_test)


# In[ ]:


#Model 1: linear SGDC classifier
sgd_clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42) #log_loss -> predicts probability

start_time_1 = time.time()
sgd_clf.fit(X_train_scaled, y_train.astype('int'))

end_time_1 = time.time()
training_time_1 = end_time_1 - start_time_1

print(f"SGD training time: {training_time_1}\n")


# In[ ]:


xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    device="cpu", #force GPU usage if possible
    tree_method="hist",
    n_estimators=300,
    n_jobs=-1
)

start_time_2 = time.time()

xgb_model.fit(
    X_tr,
    y_train
)

end_time_2 = time.time()
training_time_2 = end_time_2 - start_time_2

print(f"XGBoost training time: {training_time_2}\n")


# In[23]:


# Model 3: DNN
dnn = models.Sequential([
    Dense(64, activation='relu', input_shape=(X_tr.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[24]:


dnn.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy', 'AUC'])


# In[ ]:


start_time_3 = time.time()

history_callback = dnn.fit(X_train_scaled, y_train.astype('int'), epochs=10, batch_size=128, validation_split=0.2)

end_time_3 = time.time()
training_time_3 = end_time_3 - start_time_3

print(f"DNN training time: {training_time_3}\n")


# In[ ]:


fig = pd.DataFrame(history_callback.history).plot().get_figure()
fig.savefig(f"{SCRIPT_DIR}/output/overfit.pdf", bbox_inches='tight')


# 5. Model evaluation and comparison: the performance of the three models is going to be evaluated according to the following metrics: accuracy, precision, recall, F1-score, and ROC-AUC.

# In[27]:


# evaluation function

def evaluate_model(model, X, y, name="Model"):

    # predictions for SGDC and XGBoost 
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        # SGDC implemented with log_loss and XGBoost
        y_score = model.predict_proba(X)[:, 1]
    else:
        # DNN case (Keras)
        y_score = model.predict(X).ravel() #falltened arrai
        y_pred = (y_score > 0.5).astype(int)

    # metrics
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_score)
    }

    print(f"\n--{name}--")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    current_path = os.getcwd()
    os.makedirs(f"{SCRIPT_DIR}/output/results/{name}", exist_ok=True)


    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(f"{SCRIPT_DIR}/output/results/{name}/AUC_{name}.pdf", bbox_inches='tight')
    plt.close()


    # --- Confusion matrix ---
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{SCRIPT_DIR}/output/results/{name}/CM_{name}.pdf", bbox_inches='tight')
    plt.close()

    return metrics


# In[28]:


# SGDC Classifier
sgd_metrics = evaluate_model(
    sgd_clf,
    X_test_scaled,
    y_test.astype('int'),
    name="SGD"
)


# In[29]:


# Xgboost mdel
xgb_metrics = evaluate_model(
    xgb_model,
    X_test,
    y_test.astype('int'),
    name="XGBoost"
)


# In[30]:


# DNN
dnn_metrics = evaluate_model(
    dnn,
    X_test_scaled,
    y_test.astype('int'),
    name="DNN"
)


# 6. Possible improvements for XGBoost Classifier: three XGBoost Classifiers are going to be built, introducing variatios with respect to the first implementation of the first classifier. In particular, xgb_model_1 is going to be trained on a dataset that was not imputed (in which np.nan values are not replaced)m xgb_model_2 is going to be trained on a dataset that contains only the top 10 features for mutual information, while xgb_model_3 is going to undergo hyperparameter tuning before being trained on te original dataset.

# In[ ]:


# Model 1 -- do not replace np.nan values

xgb_model_1 = xgb.XGBClassifier(
    tree_method='hist',  
    device='cpu',       
    n_estimators=100
)

start_time_4 = time.time()
xgb_model_1.fit(X_train, y_train)

end_time_4 = time.time()
training_time_4 = end_time_4 - start_time_4

print(f"Model 1 training time: {training_time_4}\n")


# In[ ]:


xgb_metrics_1 = evaluate_model(
    xgb_model_1,
    X_test,
    y_test.astype("int"),
    name="XGBoost_1"
)


# In[ ]:


X_sample = X_tr.sample(n=20000, random_state=42)
y_sample = y_train.loc[X_sample.index]

# mutual information
print("Mutual Information score\n")
mi_scores = mutual_info_classif(X_sample, y_sample.astype("int"), random_state=42)

# show results
df_mi = pd.DataFrame({
    'Feature': X_tr.columns,
    'MI_Score': mi_scores
}).sort_values(by='MI_Score', ascending=False)

print("in order:\n")
print(df_mi.head(10))

#top k feature selection
non_linear_selector = SelectKBest(score_func=mutual_info_classif, k=10)

# fit
non_linear_selector.fit(X_sample, y_sample.astype("int"))

# keep only selected feature
X_tr_2 = non_linear_selector.transform(X_tr)
X_test_2 = non_linear_selector.transform(X_test)

# show selected feature
saved_feature = X_tr.columns[non_linear_selector.get_support()]
print(f"Selected feautres: ({len(saved_feature)}): {list(saved_feature)}\n")


# In[ ]:


# Model 2 -- select top ten features
xgb_model_2 = xgb.XGBClassifier(
    tree_method='hist',  
    device='cpu',
    n_estimators=100
)

start_time_5 = time.time()
xgb_model_2.fit(X_tr_2, y_train)

end_time_5 = time.time()
training_time_5 = end_time_5 - start_time_5

print(f"Model 2 training time: {training_time_5}\n")


# In[35]:


xgb_metrics_2 = evaluate_model(
    xgb_model_2,
    X_test_2,
    y_test.astype("int"),
    name="XGBoost_2"
)


# In[ ]:


# Model 3 -- Hyperparamter tuning

# define the hyperparameter distribution
param_dist = {
    "max_depth": stats.randint(3, 10),
    "learning_rate": stats.uniform(0.01, 0.1),
    "subsample": stats.uniform(0.5, 0.5),
    "n_estimators":stats.randint(50, 200)
}

xgb_model_3 = xgb.XGBClassifier()

start_time_6 = time.time()
random_search = RandomizedSearchCV(xgb_model_3, param_distributions=param_dist, n_iter=10, cv=5, scoring="roc_auc")
random_search.fit(X_tr, y_train)

end_time_6 = time.time()
training_time_6 = end_time_6 - start_time_6

print(f"Random Search time: {training_time_6}\n")

best_par = random_search.best_params_
print("Best set of hyperparameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)


# In[37]:


xgb_tuned = xgb.XGBClassifier(
    **best_par,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    device="cpu",
    n_jobs=-1
)


# In[38]:

start_time_7 = time.time()
xgb_tuned.fit(
    X_tr,
    y_train
)
end_time_7 = time.time()
training_time_7 = end_time_7 - start_time_7

print(f"Model 3: {training_time_7}\n")


# In[39]:


xgb_tuned_metric = evaluate_model(
    xgb_tuned,
    X_test,
    y_test.astype("int"),
    name="XGBoost_tuned"
)


# 8. Results Summary: Create two .csv files, one for the comparison of the three different models; one for the cmomparison of the different implementation of the XGBoost classifier. Write them to a file.

# In[ ]:


# comparison between SGD, XGBoost and DNN
results_1 = [sgd_metrics, xgb_metrics, dnn_metrics]
df_1 = pd.DataFrame(results_1, index=["SGDC", "XGBoost", "DNN"]).T

df_1.to_csv(f"{SCRIPT_DIR}/output/results/benchmark_ML.csv")
print("File 'benchmark_ML.csv' successfully saved!")


# In[ ]:


# comparison between different implementation of XGBOOST
results_2 = [xgb_metrics, xgb_metrics_1, xgb_metrics_2, xgb_tuned_metric]
df_2 = pd.DataFrame(results_2, index=["Model 0", "Model 1", "Model 2", "Model 3"]).T

df_2.to_csv(f"{SCRIPT_DIR}/output/results/benchmark_XGB.csv")
print("File 'benchmark_XGB.csv' successfully saved!")

