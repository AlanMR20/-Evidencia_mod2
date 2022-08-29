import pandas as pd #SOLO para importar datos y visualizar datos
from sklearn.model_selection import train_test_split #Solo para separar datos de entrenamiento y prueba de manera mas sencilla

#Importamos los datos de entrada
cols = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Class']
data = pd.read_csv('iris.data',names = cols)

x = data[['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']]
y = data[['Class']]


# Separamos un conjunto de datos para entrenar y otro para probar
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

# Arreglos para las caracteristicas para datos de entrenamiento
x_train = X_train['Sepal_Length']/X_train['Sepal_Length']
x2_train = X_train['Petal_Length']/X_train['Petal_Width']
y_train = y_train[['Class']]
# Arreglos para las caracteristicas para datos de prueba
x_validate = X_test['Sepal_Length']/X_test['Sepal_Length']
x2_validate = X_test['Petal_Length']/X_test['Petal_Width']
y_validate = y_test[['Class']]

# Modelo de regresión lineal
h = lambda x,theta,x2: theta[0]+theta[1]*x+theta[2]*x2

# Condiciones iniciales
theta = [1,0.5,0.5]
its = 10000 #100000
alpha = 0.0001 #0.000000001

# Ajuste de nuestro modelo

for idx in range(its):
  acumDelta = []
  acumDeltaX = []
  acumDeltaX2 = []
  for x_i, y_i, x2_i in zip(x_train, y_train,x2_train): # Agregar la nueva dimensión
    print(y_i)
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

n_train = len(y_train)
n_validate = len(y_validate)

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
