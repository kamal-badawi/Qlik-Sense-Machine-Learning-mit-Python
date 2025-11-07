#importiere die benötigten Bibs und Packages
#Operatives System
import os
#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

#Pandas & Numpy
import pandas as pd
import numpy as np

#Visualisierung
import matplotlib.pyplot as plt
import seaborn as  sns
#Nachricht
print('Bitte haben Sie etwas Geduld 😊')
print('Python Code wird gerade ausgeführt...')

#heutiges Datum berechnen
heute = pd.to_datetime('today').date()


#*************************************************************
#*************************************************************
#training Daten
#*************************************************************
#*************************************************************
#lade die training-Daten
training_daten = pd.read_excel(
                        r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Machine Learning Training Data.xlsx',
                        dtype={
                        'Product (Training)': 'category',
                        'City (Training)': 'category'},
                        sheet_name="Data",
                        usecols=['City (Training)','Product (Training)','Price/pc. (Training)','Personell (Training)',
                                 'Number of Machines (Training)','R&D (Training)','Customer Support (Training)',
                                 'Marketing & Advertising (Training)','Sales Quantity (Training)'])

#Wähle  100 Datensätze
training_daten_visuals = training_daten.sample(n=100, random_state=123)

#Berechne die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) (training)
desc_training = training_daten.describe()
for i in range(1,5):
    print('')
print('Statistische Kennzahlen (training):')
print(desc_training)

#Speichere die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) lokal (training)
desc_training.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\desc_training.csv',
    decimal=',',
    sep=',')

#Berechne die Korrelationen zwischen den nummerischen Kennzahlen (training)
corr_training = training_daten.corr(numeric_only=True)
for i in range(1,5):
    print('')
print('Korrelationen zwischen den nummerischen Kennzahlen (Training):')
print(corr_training)

#Speichere die Korrelationen zwischen den nummerischen Kennzahlen (training) lokal (training)
corr_training.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\corr_training.csv',
    decimal=',',
    sep=',')

#Datentypen der Spalten ausgeben  (training)
dt_training = training_daten.dtypes
for i in range(1,5):
    print('')
print('Datentypen')
print(dt_training)

#Daten encodieren =>String Daten in 0 & 1 konvertieren (training)
training_daten = pd.get_dummies(training_daten)
for i in range(1,5):
    print('')
print('Encodierte Daten (Training)')
print(training_daten)

#Spaltennamen
cols_training = training_daten.columns.values
for i in range(1,5):
    print('')
print('Spaltennamen (Training)')
print(cols_training)


#Features Spalten festlegen  (Training)
features_training = np.setdiff1d(cols_training, ['Sales Quantity (Training)'])
for i in range(1,5):
    print('')
print('Features (Training)')
print(features_training)

#Target Spalte festlegen  (training)
target_training = ['Sales Quantity (Training)']


#X & Y festlegen  (training)
X_training = training_daten[features_training]
y_training = training_daten[target_training]


#X-Y-Test & X-Y-Train festlegen  (training)
X_train_training,X_test_training,y_train_training,y_test_training  = train_test_split(X_training,y_training,test_size=0.2,random_state=123)

#Das Modell mit Daten futtern  (training)
lin_training = LinearRegression()
lin_training.fit(X_train_training,y_train_training)

#Prediction der Test-Daten für Profit (training)
y_predict_training = lin_training.predict(X_test_training)

#Koeffizienten und Intersect berechnen und lokal speichern
coef = lin_training.coef_.tolist()
for i in range(1,5):
    print('')
function_lr =pd.DataFrame(coef,columns=features_training)
print('Koeffizienten Dataframe (Model)')
print(coef)

intersect = lin_training.intercept_.tolist()
function_lr['Intersect (Training)'] = intersect
function_lr.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Function\y = ax + b .csv',
    sep=';',
    decimal=',')



#Genauigkeit des Modells überprüfen  (training)
accur_training= r2_score(y_test_training,y_predict_training)
for i in range(1,5):
    print('')
print('R²-Wert (training)')
print(accur_training)

#Speiche die Genauigkeit des Modells lokal
data = {'R²-Wert': [accur_training]}
accur_actual = pd.DataFrame.from_dict(data)
accur_actual.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Accuracy\R²-Wert.csv',
    sep=',',
    decimal=',',
    index=False)




#Visualisieren (training)
#Ordner für die Visualisierungen erstellen (training)
path_training = rf"C:\Users\kamal\OneDrive\Dokumente\Qlik\Sense\Content\Default\Machine Learning\Python Images\Training\{heute}"
isExist = os.path.exists(path_training)
if not isExist:
   os.makedirs(path_training)

#Scatter-Plot Price vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='Price/pc. (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['Price/pc. (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\price_sq_training.png')

#Scatter-Plot Personell vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='Personell (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['Personell (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\personell_sq_training.png')

#Scatter-Plot Number of Machines vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='Number of Machines (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['Number of Machines (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\Number of Machines_sq_training.png')

#Scatter-Plot R&D vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='R&D (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['R&D (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\R&D_sq_training.png')

#Scatter-Plot Customer Support vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='Customer Support (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['Customer Support (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\Customer Support_sq_training.png')

#Scatter-Plot Marketing & Advertising vs. Sales Quantity (training)
sns.jointplot(data=training_daten_visuals,x='Marketing & Advertising (Training)',y='Sales Quantity (Training)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(training_daten_visuals['Marketing & Advertising (Training)'],
                                                    training_daten_visuals['Sales Quantity (Training)'])[0,1],2)))
plt.savefig(path_training+r'\Marketing & Advertising_sq_training.png')



#*************************************************************
#*************************************************************
#Plan Daten
#*************************************************************
#*************************************************************
#lade die Plan-Daten
plan_daten =  pd.read_csv(
                        r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Machine Learning Data (Plan_Read).csv',
                        sep=';',
                        decimal=',',
                        dtype={
                            'Product (Plan)': 'category',
                            'City (Plan)': 'category',
                            'Price/pc. (Plan)':'float64'},
                        usecols=['City (Plan)','Product (Plan)','Price/pc. (Plan)','Personell (Plan)',
                                 'Number of Machines (Plan)','R&D (Plan)','Customer Support (Plan)',
                                 'Marketing & Advertising (Plan)','Sales Quantity (Plan)'])



#Datentypen der Spalten ausgeben  (plan)
dt_plan = plan_daten.dtypes
for i in range(1,5):
    print('')
print('Datentypen (Plan)')
print(dt_plan)

#Daten encodieren =>String Daten in 0 & 1 konvertieren (plan)
plan_daten_encodiert = pd.get_dummies(plan_daten)
for i in range(1,5):
    print('')
print('Encodierte Daten (Plan)')
print(plan_daten_encodiert)

#Spaltennamen
cols_plan = plan_daten_encodiert.columns.values
for i in range(1,5):
    print('')
print('Spaltennamen (plan)')
print(cols_plan)




#Features Spalten festlegen  (plan)
features_plan = np.setdiff1d(cols_plan, ['Sales Quantity (Plan)'])


#X festlegen  (Plan)
X_plan = plan_daten_encodiert[features_plan]

#Spalten umbennen in Training, damit das Modell sie akzeptiert
print("Spaltennamen vor dem Umbennen (Plan)")
print(X_plan.columns)

X_plan.columns = np.setdiff1d(cols_training, ['Sales Quantity (Training)'])

print("Spaltennamen nach dem Umbennen (Plan)")
print(X_plan.columns)

print(X_plan)
#Prediction der Plan-Daten für Profit (Plan)
y_predict_plan = lin_training.predict(X_plan)




#Füge die Spalte zu den Daten
plan_daten['Sales Quantity (Plan)']= y_predict_plan
plan_daten['Sales Quantity (Plan)'] = plan_daten['Sales Quantity (Plan)'].astype('int')
plan_daten['Price/pc. (Plan)'] = plan_daten['Price/pc. (Plan)'].astype('float')




#Speichere die Daten lokal
plan_daten.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Machine Learning Data (Plan_Read).csv',
    sep=';',
    decimal=',',
    index=False)




#Berechne die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) (plan)
desc_plan = plan_daten.describe()
for i in range(1,5):
    print('')
print('Statistische Kennzahlen (plan):')
print(desc_plan)

#Speichere die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) lokal (plan)
desc_plan.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\desc_plan.csv',
    decimal=',',
    sep=',')

#Berechne die Korrelationen zwischen den nummerischen Kennzahlen (plan)
corr_plan = plan_daten.corr(numeric_only=True)
for i in range(1,5):
    print('')
print('Korrelationen zwischen den nummerischen Kennzahlen (plan):')
print(corr_training)

#Speichere die Korrelationen zwischen den nummerischen Kennzahlen (training) lokal (plan)
corr_plan.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\corr_plan.csv',
    decimal=',',
    sep=',')


#Visualisieren (Plan)
#Ordner für die Visualisierungen erstellen (Plan)
path_plan = rf"C:\Users\kamal\OneDrive\Dokumente\Qlik\Sense\Content\Default\Machine Learning\Python Images\Plan\{heute}"
isExist = os.path.exists(path_plan)
if not isExist:
   os.makedirs(path_plan)

#Scatter-Plot Price vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='Price/pc. (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['Price/pc. (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\price_sq_plan.png')

#Scatter-Plot Personell vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='Personell (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['Personell (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\personell_sq_plan.png')

#Scatter-Plot Number of Machines vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='Number of Machines (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['Number of Machines (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\Number of Machines_sq_plan.png')

#Scatter-Plot R&D vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='R&D (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['R&D (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\R&D_sq_plan.png')

#Scatter-Plot Customer Support vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='Customer Support (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['Customer Support (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\Customer Support_sq_plan.png')

#Scatter-Plot Marketing & Advertising vs. Sales Quantity (Plan)
sns.jointplot(data=plan_daten,x='Marketing & Advertising (Plan)',y='Sales Quantity (Plan)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(plan_daten['Marketing & Advertising (Plan)'],
                                                    plan_daten['Sales Quantity (Plan)'])[0,1],2)))
plt.savefig(path_plan+r'\Marketing & Advertising_sq_plan.png')


#*************************************************************
#*************************************************************
#Ist Daten
#*************************************************************
#*************************************************************
#lade die Ist-Daten
actual_daten = pd.read_excel(
                        r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Machine Learning Data (Actual_Read).xlsx',
                        dtype={
                        'City (Actual)': 'category',
                        'Product (Actual)': 'category'},
                        sheet_name="Data",
                        usecols=['City (Actual)','Product (Actual)', 'Price/pc. (Actual)',  'Personell (Actual)',
                                'Number of Machines (Actual)', 'R&D (Actual)', 'Customer Support (Actual)',
                                'Marketing & Advertising (Actual)', 'Sales Quantity (Actual)'])





#Berechne die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) (actual)
desc_actual = actual_daten.describe()
for i in range(1,5):
    print('')
print('Statistische Kennzahlen (actual):')
print(desc_actual)

#Speichere die statistischen Kennzahlen (Min, Max, Standard Abweichung, Mittelwert, etc.) lokal (actual)
desc_actual.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\desc_actual.csv',
    decimal=',',
    sep=',')

#Berechne die Korrelationen zwischen den nummerischen Kennzahlen (actual)
corr_actual = actual_daten.corr(numeric_only=True)
for i in range(1,5):
    print('')
print('Korrelationen zwischen den nummerischen Kennzahlen (actual):')
print(corr_actual)

#Speichere die Korrelationen zwischen den nummerischen Kennzahlen (actual) lokal
corr_actual.to_csv(
    r'C:\Users\kamal\OneDrive\Dokumente\Qlik\Machine Learning\Statistics and Correlations\corr_actual.csv',
    decimal=',',
    sep=',')

#Genauigkeit-Wert
print(accur_actual)




#Visualisieren (Actual)
#Ordner für die Visualisierungen erstellen (Actual)
path_actual = rf"C:\Users\kamal\OneDrive\Dokumente\Qlik\Sense\Content\Default\Machine Learning\Python Images\Actual\{heute}"
isExist = os.path.exists(path_actual)
if not isExist:
   os.makedirs(path_actual)

#Scatter-Plot Price vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='Price/pc. (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['Price/pc. (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\price_sq_actual.png')

#Scatter-Plot Personell vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='Personell (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['Personell (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\personell_sq_actual.png')

#Scatter-Plot Number of Machines vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='Number of Machines (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['Number of Machines (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\Number of Machines_sq_actual.png')

#Scatter-Plot R&D vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='R&D (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['R&D (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\R&D_sq_actual.png')

#Scatter-Plot Customer Support vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='Customer Support (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['Customer Support (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\Customer Support_sq_actual.png')

#Scatter-Plot Marketing & Advertising vs. Sales Quantity (Actual)
sns.jointplot(data=actual_daten,x='Marketing & Advertising (Actual)',y='Sales Quantity (Actual)',kind='reg')
plt.suptitle("Korrelation: "+ str( np.round(np.corrcoef(actual_daten['Marketing & Advertising (Actual)'],
                                                    actual_daten['Sales Quantity (Actual)'])[0,1],2)))
plt.savefig(path_actual+r'\Marketing & Advertising_sq_actual.png')




print('Python Code wurde erfolgreich ausgeführt :)')