import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingClassifier

import pickle

# Load the data
df_original = pd.read_csv("../data/raw/train.csv")

# Copy the data so as not to modify the original information and visualize it
df = df_original.copy()

# Delete the personal information columns in order to work with anonimous data, as well as parental consent and medical institution name and location which are not relevant
df1 = df.drop(["Patient Id", "Patient First Name", "Family Name", "Father's name", "Parental consent", "Institute Name", "Location of Institute"], axis=1)

# Test and Symptom columns don't have any specific information, so were deleted
df1 = df1.drop(["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Symptom 1", "Symptom 2", "Symptom 3", "Symptom 4", "Symptom 5"], axis=1)

# Rename columns
df1.rename(columns={"Patient Age": "Patient_Age", "Genes in mother's side": "Mother_inherit", "Inherited from father": "Father_inherit",
                    "Maternal gene": "Maternal_gene", "Paternal gene": "Paternal_gene", "Blood cell count (mcL)": "Blood_cell_count",
                    "Mother's age": "Mother_age", "Father's age": "Father_age", "Respiratory Rate (breaths/min)": "Respiratory_rate",
                    "Heart Rate (rates/min": "Heart_rate", "Follow-up": "Follow_up", "Birth asphyxia": "Birth_asphyxia",
                    "Autopsy shows birth defect (if applicable)": "Autopsy_birth_defect", "Place of birth": "Place_birth",
                    "Folic acid details (peri-conceptional)": "Folic_acid", "H/O serious maternal illness": "Maternal_illness",
                    "H/O radiation exposure (x-ray)": "Radiation_exposure", "H/O substance abuse": "Substance_abuse",
                    "Assisted conception IVF/ART": "Assisted_conception", "History of anomalies in previous pregnancies": "History_previous_pregnancies",
                    "No. of previous abortion": "Number_abortions", "Birth defects": "Birth_defects", "White Blood cell count (thousand per microliter)": "WBC_count",
                    "Blood test result": "Blood_test", "Genetic Disorder": "Genetic_disorder", "Disorder Subclass": "Disorder_subclass"}, inplace=True)

# Replacing missing information categories to NaN
df1["Birth_asphyxia"] = df1["Birth_asphyxia"].replace("No record",np.NaN)
df1["Birth_asphyxia"] = df1["Birth_asphyxia"].replace("Not available",np.NaN)

df1["Autopsy_birth_defect"] = df1["Autopsy_birth_defect"].replace("None",np.NaN)
df1["Autopsy_birth_defect"] = df1["Autopsy_birth_defect"].replace("Not applicable",np.NaN)

df1["Radiation_exposure"] = df1["Radiation_exposure"].replace("-",np.NaN)
df1["Radiation_exposure"] = df1["Radiation_exposure"].replace("Not applicable",np.NaN)

df1["Substance_abuse"] = df1["Substance_abuse"].replace("-",np.NaN)
df1["Substance_abuse"] = df1["Substance_abuse"].replace("Not applicable",np.NaN)

df1.dropna(subset=["Genetic_disorder", "Disorder_subclass"], axis=0, inplace=True)

# Categorical columns
df1["Autopsy_birth_defect"].fillna(df1["Autopsy_birth_defect"].mode()[0], inplace=True)
df1["Birth_asphyxia"].fillna(df1["Birth_asphyxia"].mode()[0], inplace=True)
df1["Radiation_exposure"].fillna(df1["Radiation_exposure"].mode()[0], inplace=True)
df1["Substance_abuse"].fillna(df1["Substance_abuse"].mode()[0], inplace=True)
df1["Maternal_gene"].fillna(df1["Maternal_gene"].mode()[0], inplace=True)
df1["History_previous_pregnancies"].fillna(df1["History_previous_pregnancies"].mode()[0], inplace=True)
df1["Place_birth"].fillna(df1["Place_birth"].mode()[0], inplace=True)
df1["Assisted_conception"].fillna(df1["Assisted_conception"].mode()[0], inplace=True)
df1["Follow_up"].fillna(df1["Follow_up"].mode()[0], inplace=True)
df1["Gender"].fillna(df1["Gender"].mode()[0], inplace=True)
df1["Respiratory_rate"].fillna(df1["Respiratory_rate"].mode()[0], inplace=True)
df1["Birth_defects"].fillna(df1["Birth_defects"].mode()[0], inplace=True)
df1["Folic_acid"].fillna(df1["Folic_acid"].mode()[0], inplace=True)
df1["Blood_test"].fillna(df1["Blood_test"].mode()[0], inplace=True)
df1["Maternal_illness"].fillna(df1["Maternal_illness"].mode()[0], inplace=True)
df1["Heart_rate"].fillna(df1["Heart_rate"].mode()[0], inplace=True)
df1["Father_inherit"].fillna(df1["Father_inherit"].mode()[0], inplace=True)


# Numerical columns
df1["Mother_age"].fillna(df1.groupby(["Disorder_subclass"])["Mother_age"].transform("mean"),inplace=True)
df1["Father_age"].fillna(df1.groupby(["Disorder_subclass"])["Father_age"].transform("mean"),inplace=True)
df1["WBC_count"].fillna(df1.groupby(["Disorder_subclass"])["WBC_count"].transform("mean"),inplace=True)
df1["Patient_Age"].fillna(df1.groupby(["Disorder_subclass"])["Patient_Age"].transform("mean"),inplace=True)
df1["Number_abortions"].fillna(df1.groupby(["Disorder_subclass"])["Number_abortions"].transform("mean"),inplace=True)

numeric_cols = df1.select_dtypes(include=[np.number]).columns
categoric_cols = df1.select_dtypes(exclude=[np.number]).columns

# Encode categorical columns
le = LabelEncoder()
df2 = df1.copy()
df2[categoric_cols] = df2[categoric_cols].apply(le.fit_transform)
df2 = df2.astype("float32")

df3 = df2.drop(["Autopsy_birth_defect"], axis=1)
df3 = df_eda

# Create Genetic disorder dataframe
Genetic_disorder = df_eda.drop(["Disorder_subclass"], axis=1)

# Split target column from the rest of the columns of the Genetic_disorder dataframe
Genetic_X = Genetic_disorder.drop(["Genetic_disorder"], axis=1)
Genetic_Y = Genetic_disorder["Genetic_disorder"]

# Standardize the data
Genetic_X_scaled = StandardScaler().fit_transform(Genetic_X)

# Divide dataframe into train and test sets
X_Gen_train, X_Gen_test, y_Gen_train, y_Gen_test = train_test_split(Genetic_X_scaled, Genetic_Y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier for Genetic Disorder Dataframe with best hyperparameters
gb = GradientBoostingClassifier(learning_rate=0.7, max_depth=2, max_features='auto', n_estimators=30, random_state=42)
gb_Gen_model = gb.fit(X_Gen_train, y_Gen_train)

# Create Disorder Subclass dataframe
Disorder_subclass = df_eda.drop(["Genetic_disorder"], axis=1)

# Split target column from the rest of the columns of the Disorder_subclass dataframe
Subclass_X = Disorder_subclass.drop(["Disorder_subclass"], axis=1)
Subclass_Y = Disorder_subclass["Disorder_subclass"]

# Standardize the data
Subclass_X_scaled = StandardScaler().fit_transform(Subclass_X)

# Divide dataframe into train and test sets
X_Sub_train, X_Sub_test, y_Sub_train, y_Sub_test = train_test_split(Subclass_X_scaled, Subclass_Y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier for Disorder Subclass Dataframe with best hyperparameters
gb = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features='auto', n_estimators=30, random_state=42)
gb_Sub_model = gb.fit(X_Sub_train, y_Sub_train)

Gen_model_def = gb_Gen_model
Sub_model_def = gb_Sub_model
genetic_model_def = "Gen_model_def.pkl"
subclass_model_def = "Sub_model_def.pkl"

pickle.dump(genetic_model_def, open('../model/genetic_model_def', 'wb'))
pickle.dump(subclass_model_def, open('../model/subclass_model_def', 'wb'))
