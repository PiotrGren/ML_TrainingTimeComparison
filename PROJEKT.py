import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from time import process_time
import plotly.graph_objects as go

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem

df = pd.read_csv("Data/CPU1.csv")
df.info()

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.boxplot(data = df, y = 'Wykorzystanie procesora', color = 'darkgreen', linewidth = 2)
plt.title("Wykorzystanie procesora")
plt.ylabel("Procent wykorzystania")
plt.show()

df2 = pd.read_csv("Data/CPU2.csv")
df2.info()

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.boxplot(data = df2, y = 'Wykorzystanie procesora', color = 'darkred', linewidth = 2)
plt.title("Wykorzystanie procesora")
plt.ylabel("Procent wykorzystania")
plt.show()


df3 = pd.read_csv("Data/CPU3.csv")

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.boxplot(data = df3, y = 'Wykorzystanie procesora', color = 'darksalmon', linewidth = 2)
plt.title("Wykorzystanie procesora")
plt.ylabel("Procent wykorzystania")
plt.show()

df.drop('CZAS', axis = 1, inplace = True)
df2.drop('CZAS', axis = 1, inplace = True)
df3.drop('CZAS', axis = 1, inplace = True)


an = []
for i in df["Wykorzystanie procesora"]:
    if i >= 85:
        an.append(1)
    else:
        an.append(0)
df["Anomalia"] = an
df.head()








'''
LOGISTIC REGRESSION
'''

#1 Podstawowy Model
model = LogisticRegression()

X = df[["Wykorzystanie procesora"]].values
y = df["Anomalia"].values

t = process_time()
   
model.fit(X , y)

XY_LR_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
t = process_time()
y_pred = model.predict(Z)
XY_LR_pred = process_time() - t


XY_LR_Time = XY_LR_fit + XY_LR_pred
XY_LR_fit
XY_LR_pred

df_logistic = df2
df_logistic["anomaly"] = y_pred
df_logistic['anomalia'] = df_logistic['anomaly'].apply(lambda x: 
                'outlier' if x==1  else 'inlier')
fig = px.scatter(df_logistic, y = "Wykorzystanie procesora", color='anomalia')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie LogisticRegression",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()

#Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_ss = LogisticRegression(class_weight = 'balanced')

t = process_time()
model_ss.fit(X_scaled, y)
LR_Scaler_fit = process_time() - t

t = process_time()
model_ss.predict(Z)
LR_Scaler_pred = process_time() - t

LR_Scaler_fit
LR_Scaler_time = LR_Scaler_fit + LR_Scaler_pred





#2 Trenowanie z wykorzsytaniem funkcji train_test_split
model1 = LogisticRegression(calss_weight ='blanced')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

t = process_time()

model1.fit(X_train , y_train)

TRAIN_fit = process_time() - t
TRAIN_fit

#3 Trenowanie ze zmianą parametru solver w modelu na szybszy i obsługujący większe zbiory danych oraz paramateru n_jobs tak aby model mógł używać wszystkich dostępnych rdzeni CPU
model2 = LogisticRegression(n_jobs = -1, class_weight = "balanced")

t = process_time()

model2.fit(X, y)

Solver_time = process_time() - t

t = process_time()
model2.predict(Z)
Solver_pred = process_time() - t

Solver_LR_time = Solver_time + Solver_pred
Solver_time

#4 Trenowanie poprzez samplowanie zbioru danych
X_train1, y_train1 = resample(X, y, n_samples = int(len(df)*0.7), random_state = 42)

model3 = LogisticRegression(class_weight='balanced')

t = process_time()
model3.fit(X_train1, y_train1)
Resample_LR_fit = process_time() - t

t = process_time()
model3.predict(Z)
Resample_LR_pred = process_time() - t


Resample_LR_time = Resample_LR_fit + Resample_LR_pred



#5 Trenowanie przez wrzucanie zbioru po koleii
l = [int(len(df)*0.2), int(len(df)*0.4), int(len(df)*0.6), int(len(df)*0.8), int(len(df))]
a = 0
model4 = LogisticRegression(n_jobs = -1, class_weight = "balanced")
t = process_time()
for i in l:
    X_train2 = df[["Wykorzystanie procesora"]][a:i].values
    y_train2 = df["Anomalia"][a:i].values
    model4.fit(X_train2, y_train2)
    a = i
Sample_LR_fit = process_time() - t

t = process_time()
model4.predict(Z)
Sample_LR_pred = process_time() - t

Sample_LR_fit


#6 Trenowanie jak poprzednio z dodatkowym zmniejszeniem liczby iteracji
l = [int(len(df)*0.2), int(len(df)*0.4), int(len(df)*0.6), int(len(df)*0.8), int(len(df))]
a = 0
model6 = LogisticRegression(n_jobs = -1, class_weight = "balanced", max_iter = 60)
t = process_time()
for i in l:
    X_train2 = df[["Wykorzystanie procesora"]][a:i].values
    y_train2 = df["Anomalia"][a:i].values
    model6.fit(X_train2, y_train2)
    a = i
Sample_iter_fit = process_time() - t

t = process_time()
model6.predict(Z)
Sample_iter_pred = process_time() - t







#próba przykładowego wstępnego wykresu (na tej podstawie został potem stworzony ostateczny)
LR_time = [XY_LR_fit, LR_Scaler_fit, Solver_time, TRAIN_fit, Resample_LR_fit, Sample_LR_fit]
LR_names = ["Podstawowy model", "Standard Scaler", "n_jobs = -1", "Train Test Split", "Resample", "Próbkowanie danych"]

l = {"Name":LR_names, "Time":LR_time}
LR_Data = pd.DataFrame(l)
LR_Data = LR_Data.sort_values(by = "Time", ascending = False)


sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = LR_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette='viridis')
plt.title("Porównanie czasów trenowania modeli Logistic Regression")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.figure(figsize=(10,5))
plt.show()



#Analiza przyspieszenia przewidywania
#Metoda stochastycznego gradientu
sgd_clf = SGDClassifier(loss='log', max_iter=1000, tol=1e-3, warm_start=True)


sgd_clf.coef_ = model3.coef_
sgd_clf.intercept_ = model3.intercept_

t = process_time()
y_pred = sgd_clf.predict(X)
SGD_LR_pred = process_time() - t


#biblioteka joblib
joblib.dump(model, 'model.joblib')

loaded_model = joblib.load('model.joblib')
t = process_time()
predictions = loaded_model.predict(X)
JOBLIB_LR_pred = process_time() - t


#Użycie do przewidywań biblioteki numpy
t = process_time()
predictions = np.dot(X, model3.coef_.T) + model3.intercept_
NP_LR_pred = process_time() - t

#użycie biblioteki modin
modin_df = pd.DataFrame(data=df)

t = process_time()
predictions = modin_df.apply(loaded_model.predict)
Modin_LR_pred = process_time() - t




'''
KLASTERYZACJA DANYCH
'''

X = df3[["Wykorzystanie procesora"]].values

# tworzymy model k-means
kmeans = KMeans(n_clusters=1)

t = process_time()
kmeans.fit(X)
# obliczamy centroidy dla każdej z grup
centroids = kmeans.cluster_centers_
centroids
Standard_KM_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans.fit_transform(Z)
# oznaczamy punkty, dla których odległość od centroidu jest większa niż 53, jako outliery
outliers = Z[distances > 53]
Standard_KM_predict = process_time() - t
outliers


df_kmeans = df2
df_kmeans["anomaly"] = 0
for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)



Standard_KM_time = Standard_KM_predict + Standard_KM_fit
Standard_KM_fit


df_kmeans['anomalia'] = df_kmeans['anomaly'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
fig = px.scatter(df_kmeans, y = "Wykorzystanie procesora", color='anomalia')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie K-Means (klasteryzacja danych)",
                  yaxis_title="Wykorzystanie CPU [%]")
fig.update_layout(width=1050, height=500)
fig.show()



#metoda losowej inicjalizacji i początkowej liczby inicjalizacji na 20
init = "random"
kmeans2 = KMeans(n_clusters=1, init = init, n_init = 2)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans2.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans2.cluster_centers_
centroids
InRandom_fit = process_time() - t
InRandom_fit

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans2.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InRandom_predict = process_time() - t


df_kmeans = df2
df_kmeans["anomaly"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)

InRandom_time = InRandom_fit + InRandom_predict
InRandom_time

'''
NIE DZIAŁA / DOESN'T WORK

#metoda losowej inicjalizacji i początkowej liczby inicjalizacji na 20
init = "random"
kmeans2 = KMeans(n_clusters=1, init = init, n_init = 10)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans2.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans2.cluster_centers_
centroids
InRandom_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans2.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InRandom_predict = process_time() - t

df2.drop('Anomalia', axis = 1, inplace = True)

df_kmeans = df2
df_kmeans["anomaly"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)

InRandom_time = InRandom_fit + InRandom_predict
InRandom_time





#metoda losowej inicjalizacji i początkowej liczby inicjalizacji na 20
init = "random"
kmeans2 = KMeans(n_clusters=1, init = init, n_init = 5)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans2.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans2.cluster_centers_
centroids
InRandom_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans2.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InRandom_predict = process_time() - t

df2.drop('Anomalia', axis = 1, inplace = True)

df_kmeans = df2
df_kmeans["anomaly"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)

InRandom_time = InRandom_fit + InRandom_predict
InRandom_time





#metoda losowej inicjalizacji i początkowej liczby inicjalizacji na 20
init = "random"
kmeans2 = KMeans(n_clusters=1, init = init, n_init = 1)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans2.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans2.cluster_centers_
centroids
InRandom_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans2.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InRandom_predict = process_time() - t

df2.drop('Anomalia', axis = 1, inplace = True)

df_kmeans = df2
df_kmeans["anomaly"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)

InRandom_time = InRandom_fit + InRandom_predict
InRandom_time




#metoda losowej inicjalizacji i początkowej liczby inicjalizacji na 20
init = "random"
kmeans2 = KMeans(n_clusters=1, init = init, n_init = 12)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans2.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans2.cluster_centers_
centroids
InRandom_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans2.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InRandom_predict = process_time() - t

df2.drop('Anomalia', axis = 1, inplace = True)

df_kmeans = df2
df_kmeans["anomaly"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly"] = -1
print(df_kmeans)

InRandom_time = InRandom_fit + InRandom_predict
InRandom_time
'''


init = "k-means++"
kmeans31 = KMeans(n_clusters=1, init = init, n_init = 1)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans31.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans31.cluster_centers_
centroids
InKplus_ninit1_fit = process_time() - t
InKplus_ninit1_fit



#metoda inicjalizacji k_means++ i początkowej liczby inicjalizacji na 20
init = "k-means++"
kmeans3 = KMeans(n_clusters=1, init = init, n_init = 15)

# używamy modelu k-means do przydzielenia danych do grup
t = process_time()
kmeans3.fit(X)

# obliczamy centroidy dla każdej z grup
centroids = kmeans3.cluster_centers_
centroids
InKplus_fit = process_time() - t
InKplus_fit

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans3.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
InKplus_predict = process_time() - t

InKplus_time = InKplus_fit + InKplus_predict
InKplus_time



#sprawdzenie parametru n_init
n_init_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for n_init in n_init_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    kmeans_new = KMeans(n_clusters = 1, init = "k-means++", n_init = n_init)
    
    t = process_time()
    kmeans_new.fit(X)
    centroids = kmeans_new.cluster_centers_
    end_time = process_time() - t
    print(f"n_init = {n_init}, czas trenowania: {end_time}")


for n_init in n_init_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    kmeans_new = KMeans(n_clusters = 1, init = "random", n_init = n_init)
    
    t = process_time()
    kmeans_new.fit(X)
    centroids = kmeans_new.cluster_centers_
    end_time = process_time() - t
    print(f"n_init = {n_init}, czas trenowania: {end_time}")




#wstępne przetwarzanie danych
scaler = StandardScaler()

# Dopasowanie StandardScaler do danych i zastosowanie go do danych wejściowych
X_scaled = scaler.fit_transform(X)

#Użycie przetworzonych danych X_scaled do trenowania modelu zamiast pelnych danych
kmeans5 = KMeans(n_clusters=1, n_init = 1, init = 'k-means++')

t = process_time()
kmeans5.fit(X_scaled)

centroids = kmeans5.cluster_centers_
Scaled_fit = process_time() - t
Scaled_fit

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans5.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
Scaled_predict = process_time() - t

Scaled_time = Scaled_fit + Scaled_predict
Scaled_time



'''
NIE DZIAŁA / DOESN'T WORK

#to samo co wyżej tylko z argumentem pozwalającym użyć wszystkie dostępne rdzenie procesora
kmeans6 = KMeans(n_clusters=1, n_init = 1, n_jobs = -1)

t = process_time()
kmeans6.fit(X)

centroids = kmeans6.cluster_centers_
Scaled_iter_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values
# obliczamy odległość każdego punktu od jego centroidu
t = process_time()
distances = kmeans6.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
Scaled_iter_predict = process_time() - t

Scaled_iter_time = Scaled_iter_fit + Scaled_iter_predict
Scaled_iter_time
'''

#przetworzone dane podobnie jak w metodzie LogisticRegression, spróbujemy wrzucić dane podzielone na 5 próbek

l = [int(len(X_scaled)*0.2), int(len(X_scaled)*0.4), int(len(X_scaled)*0.6), int(len(X_scaled)*0.8), int(len(X_scaled))]
a = 0
kmeans7 = KMeans(n_clusters = 1, n_init = 1, init = "k-means++")
t = process_time()
for i in l:
    X_train = X[a:i]
    kmeans7.fit(X_train)
    a = i
Prob20_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values

t = process_time()
distances = kmeans7.fit_transform(Z)

# oznaczamy punkty, dla których odległość od centroidu jest większa niż 85, jako outliery
outliers = Z[distances > 53]
Prob20_predict = process_time() - t

Prob20_time = Prob20_fit + Prob20_predict
Prob20_fit


#train test split
X = df3[["Wykorzystanie procesora"]].values
X_train, X_test= train_test_split(X, random_state = 42)
kmeans8 = KMeans(n_clusters = 1, n_init = 12)

t = process_time()
kmeans8.fit(X_train)
centroids = kmeans8.cluster_centers_
TTS_KMeans_fit = process_time() - t

Z = df2[["Wykorzystanie procesora"]].values

t = process_time()
distances = kmeans8.fit_transform(Z)
outliers = Z[distances > 53]
TTS_KMeans_pred = process_time() - t


TTS_KMeans_time = TTS_KMeans_fit + TTS_KMeans_pred
TTS_KMeans_fit

df_kmeans = df2
df_kmeans.drop('anomaly_label', axis = 1, inplace = True)
df_kmeans["anomaly_label"] = 0


for i in range(len(Z)):
    if Z[i] in outliers:
        df_kmeans.loc[i, "anomaly_label"] = -1

df_kmeans.drop('anomalia', axis = 1, inplace = True)

df_kmeans['anomaly'] = df_kmeans['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')

fig = px.scatter(df_kmeans, y = "Wykorzystanie procesora", color='anomaly')
fig.show()




KM_time = [Standard_KM_fit, InRandom_fit, InKplus_fit, InKplus_ninit1_fit, Scaled_fit, Prob20_fit, TTS_KMeans_time]
KM_names = ["Podstawowy model", "Losowa inicjalizacja", "Inicjalizacja k_means++", "Podstawowa inicjalizacja + n_init = 1", "StandardScaler", "Próbkowanie danych", "Train Test Split"]

l = {"Name":KM_names, "Time":KM_time}
KM_Data = pd.DataFrame(l)
KM_Data = KM_Data.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = KM_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette='cubehelix')
plt.title("Porównanie czasów trenowania modeli Logistic Regression")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.show()




#próba przyspieszenia detekcji
#metoda mini-batch
kmeans_an = KMeans(n_clusters = 1, init = "k-means++", n_init = 1)

kmeans_an.fit(X)
centroids = kmeans_an.cluster_centers_

Z_batch1, Z_batch2, Z_batch3, Z_batch4, Z_batch5 = np.array_split(Z, 5)

t = process_time()
distances = kmeans_an.fit_transform(Z_batch1)
outliers_batch1 = Z_batch1[distances > 53]
distances = kmeans_an.fit_transform(Z_batch2)
outliers_batch2 = Z_batch2[distances > 53]
distances = kmeans_an.fit_transform(Z_batch3)
outliers_batch3 = Z_batch3[distances > 53]
distances = kmeans_an.fit_transform(Z_batch4)
outliers_batch4 = Z_batch4[distances > 53]
distances = kmeans_an.fit_transform(Z_batch5)
outliers_batch5 = Z_batch5[distances > 53]
mini_batch5_pred = process_time() - t
mini_batch5_pred

MINIBATCH5_time = Scaled_fit + mini_batch5_pred


Z_batch1, Z_batch2, Z_batch3 = np.array_split(Z, 3)

t = process_time()
distances = kmeans_an.fit_transform(Z_batch1)
outliers_batch1 = Z_batch1[distances > 53]
distances = kmeans_an.fit_transform(Z_batch2)
outliers_batch2 = Z_batch2[distances > 53]
distances = kmeans_an.fit_transform(Z_batch3)
outliers_batch3 = Z_batch3[distances > 53]
mini_batch3_pred = process_time() - t
mini_batch3_pred

MINIBATCH3_time = Scaled_fit + mini_batch3_pred





#Nyostream
transformer = Nystroem(n_components = 100)

Z_transformed = transformer.fit_transform(Z)

t = process_time()
distances = kmeans8.fit_transform(Z_transformed)
outliers = Z[distances > 53]
Nystroem_pred = process_time() - t
Nystroem_pred

Nystroemtime = Scaled_fit + Nystroem_pred

X = df3[["Wykorzystanie procesora"]].values
Z = df2[["Wykorzystanie procesora"]].values
n_components_values = [2, 10, 20, 25, 30, 50, 60, 75, 80, 100]

for n_component in n_components_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    transformer = Nystroem(n_components = n_component)
    Z_transformed = transformer.fit_transform(Z)

    t = process_time()
    distances = kmeans_an.fit_transform(Z_transformed)
    outliers = Z[distances > 53]
    end_time = process_time() - t
    print(f"n_components = {n_component}, czas przewidywania: {end_time}")

transformer1 = Nystroem(n_components = 1)
Z_transformed = transformer1.fit_transform(Z)

t = process_time()
distances = kmeans8.fit_transform(Z_transformed)
outliers = Z[distances > 53]
Nystroem1_pred = process_time() - t
Nystroem1_pred

transformer2 = Nystroem(n_components = 1)
Z_transformed = transformer2.fit_transform(Z)

t = process_time()
distances = kmeans8.fit_transform(Z_transformed)
outliers = Z[distances > 53]
Nystroem2_pred = process_time() - t
Nystroem2_pred

nystroeam2_time = Scaled_fit + Nystroem2_pred



#próba przykładowego wstępnego wykresu (na tej podstawie został potem stworzony ostateczny)
KM_pred_time = [Standard_KM_predict, mini_batch3_pred, mini_batch5_pred, Nystroem_pred, Nystroem2_pred]
KM_pred_names = ["Przed optymalizacją", "Mini-batch[3]", "Mini-batch[5]", "Nystroem", "Nystroem n_c = 1"]


l = {"Name":KM_pred_names, "Time":KM_pred_time}
KM_Data = pd.DataFrame(l)
KM_Data = KM_Data.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = KM_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette='ch:s=-.2,r=.6')
plt.title("Porównanie czasów przewidywania modeli KMeans")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.show()


#próba przykładowego wykresu końcowego
KM_time_end = [Standard_KM_time, Scaled_time, MINIBATCH3_time, MINIBATCH5_time, Nystroemtime, nystroeam2_time]
KM_namesend = ["Podstawowy model", "Przed optymalizacją", "Mini-batch[3]", "Mini-batch[5]", "Nystroem", "Nystroem n_c = 1"]


l = {"Name":KM_namesend, "Time":KM_time_end}
KM_Data = pd.DataFrame(l)
KM_Data = KM_Data.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = KM_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette="ch:start=.2,rot=-.3")
plt.title("Porównanie czasów całkowitego działania modeli KMeans")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.show()










'''
Local Outlier Factor
'''
#Początkowy model
df4 = df2
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]]


t = process_time()
lof1 = LocalOutlierFactor(contamination = 0.062, novelty = True)
lof1.fit(X)
Standard_fit_LOF = process_time() - t


t = process_time()
y_pred = lof1.predict(X1)
Standard_pred_LOF = process_time() - t



df4['anomaly_label'] = y_pred
df4['anomaly'] = df4['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
df4.drop("anomalia", axis = 1, inplace = True)
df4
fig = px.scatter(df4, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()





#Standardowy model
X = df3[["Wykorzystanie procesora"]].values
X1 = df2[["Wykorzystanie procesora"]].values


t = process_time()
lof = LocalOutlierFactor(contamination = 0.062, n_neighbors = 2000)
lof.fit(X)
Standard_fit_LOF = process_time() - t

Standard_fit_LOF

t = process_time()
y_pred = lof.fit_predict(X1)
Standard_pred_LOF = process_time() - t
Standard_pred_LOF


df_lof = df2
df_lof['anomaly_label'] = y_pred
df_lof['anomaly'] = df_lof['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
fig = px.scatter(df_lof, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()

Standard_time_LOF = Standard_fit_LOF + Standard_pred_LOF


#próba zmniejszenia liczby sąsiadów
df4 = pd.DataFrame(df2)
df4
X = df4[["Wykorzystanie procesora"]]


t = process_time()
lof2 = LocalOutlierFactor(contamination = 0.06, n_neighbors = 1500)
y_pred = lof2.fit_predict(X)
Less_Neighbours = process_time() - t

df4['anomaly_label'] = y_pred
df4['anomaly'] = df4['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
df4[df4['anomaly'] == 'outlier']


fig = px.scatter(df4, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie (n_neighbors = 1500)",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()

#1650
df4 = pd.DataFrame(df2)
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]]


t = process_time()
lof2 = LocalOutlierFactor(contamination = 0.062, n_neighbors = 1600)
lof2.fit(X)
Less_Neighbors_fit = process_time() - t
Less_Neighbors_fit

t = process_time()
y_pred = lof2.fit_predict(X1)
Less_Neighbors_pred = process_time() - t


Less_Neighbors_time = Less_Neighbors_fit + Less_Neighbors_pred


df4['anomaly_label'] = y_pred
df4['anomaly'] = df4['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
df4[df4['anomaly'] == 'outlier']


fig = px.scatter(df4, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie (n_neighbors = 1600)",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()




#Prókowanie 20% ponieważ algorytm działa lepiej jak ma mniej punktów do przetworzenia
df4 = pd.DataFrame(df2)
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]].values

lof3 = LocalOutlierFactor(contamination = 0.062, n_neighbors = 1400, novelty = True)


l = [int(len(X)*0.5), int(len(X))]
a = 0
t = process_time()
for i in l:
    X_train = X[a:i]
    lof3.fit(X_train)
    a = i
fit_50 = process_time() - t
fit_50
t = process_time()
y_pred = lof3.predict(X1)
pred_50 = process_time() - t


Time_50 = fit_50 + pred_50

df4['anomaly_label'] = y_pred
df4['anomaly'] = df4['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
df4[df4['anomaly'] == 'outlier']


fig = px.scatter(df4, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie (próbkowanie danych)",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()


#próbkowanie + standard scaler
scaler = StandardScaler()
df4 = pd.DataFrame(df2)
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]].values
X_scaled = scaler.fit_transform(X)


lof4 = LocalOutlierFactor(contamination = 0.062, n_neighbors = 1400, novelty = True)


l = [int(len(X)*0.5), int(len(X))]
a = 0
t = process_time()
for i in l:
    X_train = X_scaled[a:i]
    lof4.fit(X_train)
    a = i
fit_50_scaler = process_time() - t
fit_50_scaler

t = process_time()
y_pred = lof4.predict(X1)
pred_50_scaler = process_time() - t
pred_50_scaler





#Scaler ale z każdym wolnym CPU
scaler = StandardScaler()
df4 = pd.DataFrame(df2)
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]].values
X_scaled = scaler.fit_transform(X)


lof5 = LocalOutlierFactor(contamination = 0.062, n_neighbors = 1400, novelty = True, n_jobs = -1)


l = [int(len(X)*0.5), int(len(X))]
a = 0
t = process_time()
for i in l:
    X_train = X_scaled[a:i]
    lof5.fit(X_train)
    a = i
fit_50_scaler_CPU = process_time() - t
fit_50_scaler_CPU

t = process_time()
y_pred = lof5.predict(X1)
pred_50_scaler_CPU = process_time() - t




#algorytm KD_Tree
scaler = StandardScaler()
df4 = pd.DataFrame(df2)
X = df[["Wykorzystanie procesora"]]
X1 = df4[["Wykorzystanie procesora"]].values
X_scaled = scaler.fit_transform(X)


lof6 = LocalOutlierFactor(contamination = 0.062, n_neighbors = 1400, novelty = True, algorithm = "kd_tree")


l = [int(len(X)*0.5), int(len(X))]
a = 0
t = process_time()
for i in l:
    X_train = X_scaled[a:i]
    lof6.fit(X_train)
    a = i
fit_50_scaler_kdtree = process_time() - t
fit_50_scaler_kdtree

t = process_time()
y_pred = lof6.predict(X1)
pred_50_scaler_kdtree = process_time() - t
pred_50_scaler_kdtree



#próba wykresu
LOF_time = [Standard_fit_LOF, Less_Neighbors_fit, fit_50, fit_50_scaler, fit_50_scaler_CPU, fit_50_scaler_kdtree]
LOF_names = ["Podstawowy model", "Zmniejszona liczba sąsiadów", "Podział danych na pół", "Podział danych z przetworzeniem StandardScaler", "Dodanie n_jobs = -1", "Dodanie KD-Tree"]

l = {"Name":LOF_names, "Time":LOF_time}
LOF_Data = pd.DataFrame(l)
LOF_Data = LOF_Data.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = LOF_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette="flare")
plt.title("Porównanie czasów trenowania modeli Local Outlier Factor")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.show()










#próba optymalizacji algorytmu poprzez zastosowanie grafu k-najbliższych
t = process_time()
#określenie stopnia odosobnienia dla każdego punktu danych
scores_pred = lof4.negative_outlier_factor_
treshold = -3.94
y_pred = (scores_pred < treshold)
K_pred_time = process_time() - t

K_pred_time



#przewidywanie za pomocą funkcji decyzyjnej
t = process_time()
scores = lof4.decision_function(X1)

anomalies_index = np.where(scores < -167)
decfun_pred_time = process_time() - t
decfun_pred_time


df_lof = df2
df_lof['anomaly_label'] = 0
for i in df_lof.index:
    if i in anomalies_index[0]:
        df_lof.loc[i, "anomaly_label"] = -1


df_lof['anomaly'] = df_lof['anomaly_label'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
df_lof.iloc[136, :]
fig = px.scatter(df_lof, y = "Wykorzystanie procesora", color='anomaly')
fig.update_layout(template="plotly_dark")
fig.update_layout(title="Wykryte anomalie (funkcja decyzyjna)",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()
























'''
LAS IZOLACJI
'''


def IsolationForesst(df, df2):

    X = df3[["Wykorzystanie procesora"]].values

    iforest = IsolationForest(n_estimators = 200, contamination = 0.062, random_state = 200, bootstrap = True)
    
    t = process_time()
    iforest.fit(X)
    IForest_fit = process_time() - t

    Z = df2[["Wykorzystanie procesora"]].values
    y_pred = iforest.predict(Z)
 
    scores = iforest.decision_function(Z)

    return y_pred, scores, IForest_fit

list1 = IsolationForesst(df, df2)
list1[2]
if_y_pred = list1[0]

df_forest = df2
df_forest["anomaly"] = if_y_pred
df_forest['anomalia'] = df_forest['anomaly'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
fig = px.scatter(df_forest, y = "Wykorzystanie procesora", color='anomalia')
fig.update_layout(template="plotly_dark")
fig.update_traces(marker=dict(color='red', size=8),
                  selector=dict(anomalia='outlier'))
fig.update_traces(marker=dict(color='lightblue', size=8),
                  selector=dict(anomalia='inlier'))
fig.update_layout(title="Wykryte anomalie",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()


#Poprawa algorytmu tak aby wykrywał poprawne anomalie
X = df3[["Wykorzystanie procesora"]].values
Z = df2[["Wykorzystanie procesora"]].values

class MyIsolationForest(IsolationForest):
    def predict(self, X):
        y_pred = super().predict(X)
        threshold = 85
        y_pred[X[:, 0] > threshold] = 1
        y_pred[X[:, 0] <= threshold] = -1
        return y_pred

my_iforest = MyIsolationForest(contamination=0.062)
t = process_time()
my_iforest.fit(X)
Standard_IsForest_fit = process_time() - t

t = process_time()
y_pred = my_iforest.predict(Z)
Standard_IsForest_predict = process_time() - t

Standard_IsForest_fit
Standard_IsForest_predict

df_forest = df2
df_forest["anomaly"] = y_pred
df_forest['anomalia'] = df_forest['anomaly'].apply(lambda x: 
                'outlier' if x==-1  else 'inlier')
fig = px.scatter(df_forest, y = "Wykorzystanie procesora", color='anomalia')
fig.update_layout(template="plotly_dark")
fig.update_traces(marker=dict(color='red', size=8),
                  selector=dict(anomalia='outlier'))
fig.update_traces(marker=dict(color='lightblue', size=8),
                  selector=dict(anomalia='inlier'))
fig.update_layout(title="Wykryte anomalie",
                  yaxis_title="Procent wykorzystania")
fig.update_layout(width=1050, height=500)
fig.show()


Standard_IsForest_time = Standard_IsForest_fit + Standard_IsForest_predict
Standard_IsForest_time

#testowanie różnych wartości parametru n_estimators
n_estimators_values = [100, 200, 300, 400, 500]
train_time = []
for n_estimator in n_estimators_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    my_iforest = MyIsolationForest(n_estimators = n_estimator, contamination = 0.062)
    
    
    start_time = process_time()
    y_pred = my_iforest.predict(Z)
    end_time = process_time()
    training_time = end_time - start_time
    print(f"n_estimators = {n_estimator}, czas tranowania: {training_time}")
    train_time.append(training_time)


#n_estimators = 100
my_iforest1 = MyIsolationForest(contamination=0.062, n_estimators = 100, randomstate = 200)
t = process_time()
my_iforest.fit(X)
Nest_IsForest_fit = process_time() - t

t = process_time()
y_pred = my_iforest.predict(Z)
Nest_IsForest_predict = process_time() - t

Nest_IsForest_fit









#train_test_split
X = df[["Wykorzystanie procesora"]].values
X_train, X_test = train_test_split(X, random_state = 42)
Z = df2[["Wykorzystanie procesora"]].values

my_iforest2 = MyIsolationForest(n_estimators=100, contamination=0.062, random_state = 200)

t = process_time()
my_iforest2.fit(X_train)
TTS_IsForest_fit = process_time() - t
TTS_IsForest_fit

t = process_time()
y_pred = my_iforest2.predict(Z)
TTS_IsForest_pred = process_time() - t

TTS_IsForest_fit

TTS_IsForest_time = TTS_IsForest_fit + TTS_IsForest_pred
TTS_IsForest_time


#Standard Scaler
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

my_iforest3 = MyIsolationForest(n_estimators=100, contamination=0.062, random_state = 200)

t = process_time()
my_iforest3.fit(X_scaled)
Scaler_IsForest_fit = process_time() - t

t = process_time()
y_pred = my_iforest3.predict(Z)
Scaler_IsForest_pred = process_time() - t

Scaler_IsForest_time = Scaler_IsForest_pred + Scaler_IsForest_fit
Scaler_IsForest_fit





#Połączenie lasu izolacji z algorytmem PCA
# Tworzenie obiektu PCA z 1 składową główną
pca = PCA(n_components=1)
# Redukcja wymiarów danych za pomocą PCA
X_reduced = pca.fit_transform(X)
# Tworzenie obiektu lasu izolacji
my_iforest5 = MyIsolationForest(n_estimators=100, contamination=0.062, random_state=200)
# Trenowanie lasu izolacji na danych zredukowanych wymiarowo
t = process_time()
my_iforest5.fit(X_reduced)
PCA_IsForest_fit = process_time() - t
t = process_time()
y_pred = my_iforest5.predict(Z)
PCA_IsForest_pred = process_time() - t

PCA_IsForest_fit

PCA_IsForest_time = PCA_IsForest_fit + PCA_IsForest_pred



#Próbkowanie danych trenujących
X = df[["Wykorzystanie procesora"]].values
X_train, X_test = train_test_split(X, random_state = 42)
Z = df2[["Wykorzystanie procesora"]].values

my_iforest6 = MyIsolationForest(n_estimators=100, contamination=0.062, random_state=200, bootstrap=True)


l = [int(len(X_train)*0.2), int(len(X_train)*0.4), int(len(X_train)*0.6), int(len(X_train)*0.8), int(len(X_train))]
a = 0
t = process_time()
for i in l:
    X_tr = X_train[a:i]
    my_iforest6.fit(X_tr)
    a = i
TTS20_IsForest_fit = process_time() - t



t = process_time()
y_pred = my_iforest6.predict(Z)
TTS20_IsForest_pred = process_time() - t

TTS20_IsForest_time = TTS20_IsForest_fit + TTS20_IsForest_pred

TTS20_IsForest_fit

PCA_IsForest_fit = 0.28125
Standard_IsForest_fit

IsForest_fit = [Standard_IsForest_fit, Nest_IsForest_fit, TTS_IsForest_fit, Scaler_IsForest_fit, PCA_IsForest_fit, TTS20_IsForest_fit]
IsForest_names = ["Podstawowy model","N_estimators = 100", "Train Test Split", "Standard Scaler", "Algorytm PCA", "Train Test Split + Próbkowanie danych"]

l = {"Name":IsForest_names, "Time":IsForest_fit}
IsForest_Data = pd.DataFrame(l)
IsForest_Data = IsForest_Data.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
custom_palette = sns.color_palette(["#1f77b4"])
sns.barplot(data = IsForest_Data, x="Name", y="Time", hue="Name", saturation = 0.90, palette='rocket')
plt.title("Porównanie czasów działania modeli Isolation Forest")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.figure(figsize=(10,5))
plt.show()




#Optyalizacja przewidywania
#MIERZENIE NIEZALEŻNOŚCI NA LICZBĘ PRÓBEK

import time
# Lista wartości max_samples do przetestowania
max_samples_values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1800, 1900, 2000]

for max_samples in max_samples_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    my_iforest3.max_samples_ = max_samples
    
    
    start_time = time.time()
    y_pred = my_iforest3.predict(Z)
    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"max_samples = {max_samples}, czas przewidywania: {prediction_time}")



max_samples_values = [1375, 1390, 1400, 1420, 1425]
pred_time = []
for max_samples in max_samples_values:
    # Tworzenie nowego obiektu modelu z nową wartością max_samples
    my_iforest3.max_samples_ = max_samples
    
    
    start_time = time.time()
    y_pred = my_iforest3.predict(Z)
    end_time = time.time()
    prediction_time = end_time - start_time
    print(f"max_samples = {max_samples}, czas przewidywania: {prediction_time}")
    pred_time.append(prediction_time)

pred_time
pred_time.sort()
print(pred_time)


TTS_IsForest_fit
pred_time[0]

TTS_New_IsForest_time = TTS_IsForest_fit + pred_time[0]
Standard_IsForest_time

IsForest_time = [TTS_IsForest_time, TTS_New_IsForest_time]
IsForest_time_names = ["Przed optymalizacją przewidywania", "Po optymalizacji przewidywania"]
list10 = {"Name":IsForest_time_names, "Time":IsForest_time}

IsForest_Data_END = pd.DataFrame(list10)
IsForest_Data_END
IsForest_Data_END = IsForest_Data_END.sort_values(by = "Time", ascending = False)

sns.set_style("darkgrid")
sns.set_context("notebook", rc={"grid.linewidth": 1, "lines.linewidth": 2.5})
sns.barplot(data = IsForest_Data_END, x="Name", y="Time", hue="Name", saturation = 0.90, palette='rocket')
plt.title("Porównanie czasów działania modeli Isolation Forest - optymalizacja przewidywania")
plt.xlabel("Model")
plt.ylabel("Czas działania [s]")
plt.legend(title=None)
plt.show()




'''
PRZYKŁAD EKSPORTU WYSZKOLONEGO MODELU / EXAMPLE OF EXPORT OF TRAINED MODEL
'''

import joblib

joblib.dump(my_iforest5, 'TrainedModels/isolation_forest.joblib')