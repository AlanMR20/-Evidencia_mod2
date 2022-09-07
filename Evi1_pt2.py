import pandas as pd #SOLO para importar datos y visualizar datos
import numpy as np #Para cambiar el formato de los arreglos
import matplotlib.pyplot as plt #Para poder graficar los resultados obtenidos
from sklearn.model_selection import train_test_split #Solo para separar datos de entrenamiento y prueba de manera mas sencilla

#Importamos los datos de entrada
#cols = ['Species', 'Weight',  'Length1',  'Length2',  'Length3',  'Height', 'Width']
data = pd.read_csv('Fish.csv')

# Separamos nuestras caracteristicas utiles y nuestra variable de interes
x =  data[['Length1', 'Length2',  'Length3','Width']]
y = data[['Weight']]
cor_data = data[['Weight',  'Length1',  'Length2',  'Length3',  'Height', 'Width']]
cor_d = cor_data.corr()
cor_d.style.background_gradient (cmap = 'coolwarm')
# Separamos un conjunto de datos para entrenar y otro para probar
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Arreglos para las caracteristicas para datos de entrenamiento
lens_train = X_train['Length1'] * X_train['Length2'] * X_train['Length3']
x_train = lens_train**(1/3)
x2_train = X_train['Width']
y_train = y_train['Weight']
# Cambiamos el tamaños de los arrays para manejarlos más fácil
x_train = (np.asarray(x_train)).flatten()
x2_train = (np.asarray(x2_train)).flatten()
y_train = (np.asarray(y_train)).flatten()

# Arreglos para las caracteristicas para datos de prueba
lens_test = X_test['Length1'] * X_test['Length2'] * X_test['Length3']
x_validate = lens_test**(1/3)
x2_validate = X_test['Width']
y_validate = y_test['Weight']
# Cambiamos el tamaños de los arrays para manejarlos más fácil
x_validate = (np.asarray(x_validate)).flatten()
x2_validate = (np.asarray(x2_validate)).flatten()
y_validate = (np.asarray(y_validate)).flatten()
# Modelo de regresión lineal
h = lambda x,theta,x2: theta[0]+theta[1]*x+theta[2]*x2

# Condiciones iniciales
theta = [1.0,1.0,1.0]
its = 100000 #100000
alpha = 0.0001 #0.000000001

n_train = len(y_train)
n_validate = len(y_validate)
# Ajuste de nuestro modelo
for idx in range(its):
  acumDelta = []
  acumDeltaX = []
  acumDeltaX2 = []
  for x_i, y_i, x2_i in zip(x_train, y_train,x2_train): # Agregar la nueva dimensión
    #print(y_i)
    acumDelta.append(h(x_i,theta,x2_i)-y_i)
    acumDeltaX.append((h(x_i,theta,x2_i)-y_i)*x_i)
    acumDeltaX2.append((h(x_i,theta,x2_i)-y_i)*x2_i) # Acumular para el nuevo theta

  sJt0 = sum(acumDelta)
  sJt1 = sum(acumDeltaX)
  sJt2 = sum(acumDeltaX2)
  theta[0] = theta[0]-alpha/n_train*sJt0
  theta[1] = theta[1]-alpha/n_train*sJt1
  theta[2] = theta[2]-alpha/n_train*sJt2 # ACtualizar el nuevo theta

print(theta)

j2_i = lambda x,y,theta,x2: (h(x,theta,x2)-y)**2

# Validación
acumDelta = []
for x_i, y_i,x2_i in zip(x_validate,y_validate,x2_validate):
  acumDelta.append(j2_i(x_i,y_i,theta,x2_i))

sDelta = sum(acumDelta)
J_validate = 1/(2*n_validate)*sDelta

# Entrenamiento
acumDelta = []
for x_i, y_i,x2_i in zip(x_train,y_train,x2_train):
  acumDelta.append(j2_i(x_i,y_i,theta,x2_i))
sDelta = sum(acumDelta)
J_train = 1/(2*n_train)*sDelta

print(J_validate)
print(J_train)
print(theta)

ls =  (data['Length1'] * data['Length2'] * data['Length3'])**(1/3)
Y = data['Weight']
X = data['Width']*ls

X = (np.asarray(X)).flatten()
Y = (np.asarray(Y)).flatten()

plt.scatter(X,Y)
plt.scatter(x_validate*x2_validate,y_validate, color='r')
plt.legend(["Real data" , "Some predictions"])
plt.title('Fish Weigth prediction')
plt.xlabel('Volume*Width')
plt.ylabel('Weight')
plt.show()
