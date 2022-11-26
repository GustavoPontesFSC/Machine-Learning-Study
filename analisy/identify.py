# Scope
import os, sys
p = os.path.abspath('.')
sys.path.insert(1,p)

# -----------------------------
# Imports
from math_func.group import dirtoread, y_ml
from spectra.classes import spectra_SNV
from pathlib import Path
import numpy as np
# -----------------------------
# Machine Learning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# -----------------------------


# Definition values training and predict
ROOT_training = Path(__file__).parent.parent.joinpath('database/medidas_training')
ROOT_predict = Path(__file__).parent.parent.joinpath('database/medidas_predict')

DIR_NOC = ROOT_training.joinpath('Amostra A sem colágeno/')
DIR_WIC = ROOT_training.joinpath('Amostra B com colágeno/')
# -----------------------------
all_sample = sorted(os.listdir(DIR_NOC))

# data training
matrix = []
dirtoread(DIR=DIR_NOC, all_sample=all_sample, matrix=matrix, spectra=spectra_SNV)
shape1 = len(matrix)
all_sample = sorted(os.listdir(DIR_WIC))
dirtoread(DIR=DIR_WIC, all_sample=all_sample, matrix=matrix, spectra=spectra_SNV)
shape2 = len(matrix)-shape1

shape = (shape1,shape2)
# -----------------------------

# predict
X = np.array(matrix)
y = np.array(y_ml(shape))

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
# -----------------------------

# Data predict
DIR_1 = ROOT_predict.joinpath('Amostra A sem colágeno/')
DIR_2 = ROOT_predict.joinpath('Amostra B com colágeno/')
predict = [] 
all_sample = sorted(os.listdir(DIR_1))
dirtoread(DIR=DIR_1, all_sample=all_sample, matrix=predict, spectra=spectra_SNV)
all_sample = sorted(os.listdir(DIR_2))
dirtoread(DIR=DIR_2, all_sample=all_sample, matrix=predict, spectra=spectra_SNV)
# -----------------------------

predict = np.array(predict)
final_predict = np.zeros(len(predict))
for i, spec in enumerate(predict):
    final_predict[i] = int(clf.predict([spec]))

print(final_predict)



