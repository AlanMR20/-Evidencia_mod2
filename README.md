# Evidencia 1, Módulo 2: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución

__NOTA:__ Los script utilizados para esta evidencia se encuetran en la carpeta de __Codes__.

## Fish Market Dataset
Para esta evidencia se utilizó un dataset de Kaggle el cual nos proporciona información acerca de 7 especies de peces que se comercializan. El fin de este dataset es poder realizar un modelo de machine learning que pueda predecir con base a sus características un cierto peso y con este posteriormente poder asignarle un cierto peso arbitrario para comercializarlos.

Entre las características de los pescados podemos encontrar lo siguiente:
* __Species:__ Clase del pez
* __Weight:__ Peso del pez en gramos (g)
* __Length1:__ Longitud vertical (cm)
* __Length2:__ Longitud diagonal (cm)
* __Length3:__ Longitud cruzada (cm)
* __Height:__ Altura del pez (cm)
* __Width:__ Anchura diagonal del pez (cm)

## Análisis previo de los datos
Antes de escoger un modelo de regresión para hacer nuestras predicciones, primero debemos de conocer cuál es la relación de los datos. Para ello podemos ver la matriz de correlaciones de nuestros datos para ver si existe linealidad entre dos o más variables, teniendo lo siguiente:

![image](https://user-images.githubusercontent.com/101605777/188798206-84b3f85b-fcba-480b-b4c3-cf1dfa1d974b.png)

Gracias a esta matriz de correlaciones podemos darnos cuenta de que todas las longitudes están fuertemente relacionadas. Para el caso de la altura vemos que realmente no está tan fuertemente relacionada ni con el peso ni con las longitudes por lo que puede que no sea una característica que nos sea útil para implementar nuestro modelo. Por otro lado, vemos que la anchura diagonal se relaciona correlaciona más con los demás datos que la altura, por lo que podemos usar esta característica para poder realizar nuestro modelo.

## Modelo de regresión utilizado
Para nuestro conjunto de datos tenemos que las longitudes 1,2 y 3 están fuertemente relacionadas con el peso pero sobre todo entre sí, por lo que haremos una nueva característica donde se incluyan las tres longitudes de tal manera que siga siendo útil para nuestro modelo y sobre todo reducir el número de dimensiones. Lo que haremos será multiplicar las 3 longitudes y sacarle la raíz cúbica a dicha operación para no tener un valor tan grande para los coeficientes del modelo. Esta nueva característica será la primera de nuestro modelo. Mientras que la anchura diagonal por sí sola será nuestra segunda característica y con buscaremos un modelo de regresión lineal de segundo orden que se pueda ajustar a nuestras características para encontrar un modelo para predecir el peso de los peces. Por lo que buscaremos que nuestro modelo se ajuste a estas dos características.

### RStudio
En modulo de estadística utilizamos la herramienta de RStudio, la cual tiene una sintaxis muy simple al igual que python con la diferencia que R es más especializada para trabajar con herramientas estadísticas. Por lo que para analizar cómo sería nuestro modelo de regresión lineal tomando en cuenta la anchura y nuestra característica creada de las longitudes como variables independientes una de la otra es decir $\theta_1$ y $\theta_2$, así como la interacción de ambas características, lo cual nos ayuda para ver que tan significante es tomar en cuenta la interacción entres estas. El script de R se encuentra dentro de la carpeta __Codes__ con el normbre de **Fish_test.Rmd**.
![image](https://user-images.githubusercontent.com/101605777/189255881-66731729-5bba-47f3-a34d-662dcadb069e.png)

Lo que podemos ver en la imagen anterior es un resumen del modelo generado en R. La parte de coeficientes es la que más nos interesa analizar ya que en la columna de __Estimate__ tenemos los valores de $\theta$, $\theta_1$, $\theta_2$ y $\theta_3$ ; donde $\theta_3$ corresponde a la interacción de $x_1$ y $x_2$. Sabemos que esta interacción es significativa ya que tenemos un valor t mucho mayor de 0 que nos indica que la interación es significativa para hallar nuestra $H_0$ por lo que nuestro modelo quedaría de la siguiente manera: $h_\theta=\theta+\theta_1x_1+\theta_2x_2+\theta_3x_1x_2$.

Sustituyendo los coeficientes y nuestras variables tenemos lo siguiente: $F_{weight}=71.1874-53.4969F_{width}-6.7464L+5.3301F_{width}L$

### Implementación del modelo con Python
Gracias a esta nueva característica que creamos a partir de una relación entre las longitudes, podemos utilizar esta información para implementarla en python con sklearn y probar con diferentes modelos de regresión con diferentes características para saber cuáles de esas características son más relevantes para poder hacer nuestra estimación del peso de los peces. Por lo que vamos a probar 4 modelos de distintas características y comparar sus desempeños y para ello necesitamos usar las siguientes librerías.

**Librerías Utilizadas**
* __Pandas:__ Para importar y visualizar el dataset
* __Matplotlib:__ Para poder graficar los resultados obtenidos
* __Sklearn:__ Librería para el uso de herramientas de Machine Learning, pero solo usaremos las siguientes:
    * __Train_test_split:__ Nos permite separar datos de entrenamiento y prueba de manera más sencilla
    * __Linear_model:__ Para usar el modelo ya implementado  de regresión lineal
    * __Metrics:__  Para usar métricas estadísticas que nos ayuden a evaluar el desempeño de los modelos

**Añadimos las nuevas caracteristicas al modelo**
 * __L123:__ Contiene la raíz cúbica del producto de las longitudes 1,2 y 3
 * __WidthXL123:__ La interacción entre Anchura la nuestra nueva característica

![image](https://user-images.githubusercontent.com/101605777/189273261-04f96749-26a9-430b-8ef2-2125b3c023bd.png)

Al agregar estas nuevas columnas quiere decir que vamos a tener más interacciones por lo que hay que checar la matriz de correlación con estos cambios y ver qué tan relevantes pueden ser para nuestro modelo lineal.

![image](https://user-images.githubusercontent.com/101605777/189273345-f475a467-29e6-48af-b7bf-5685bbafcd87.png)

### Implementación de diferentes modelos
Para poder comparar que se use el modelo con las características óptimas para poder predecir nuestra y que es el peso del pez, se probarán __6 modelos__ donde cada uno posee distintas características. Además se creó un arreglo donde contiene todos las __X__ y este arreglo se mete en una función donde se evalúa cada modelo para ver el rendimiento y poder decir cual es es la mejor opción para predecir el peso del pez.

![image](https://user-images.githubusercontent.com/101605777/189424803-d12a7fd5-4ab4-48ea-8149-64013a90cd20.png)

Se creó una función donde se crea el modelo de regresión lineal múltiple y se evalúa su rendimiento dependiendo de los conjuntos de entrenamiento y prueba y de las características de $X_n$.

![image](https://user-images.githubusercontent.com/101605777/189431358-93899cfa-b6b2-400e-943e-be0a1f929c14.png)

### Métricas de desempeño
Para evaluar el desempeño de cada modelo usamos dos métricas estadísticas para evaluar que tan bien se ajustó el conjunto de entrenamiento para hacer las predicciones con el conjunto de prueba.
* __Coeficiente de correlación ( $r^2$ ):__ Nos indica que que tan bien se ajustan los datos con nuestro modelo mediante los residuos entre los valores reales y los estimados
* __Error cuadrático medio (ECM):___ Es la la suma de la varianza y el cuadrado sesgo de las predicciones

## Predicciones de prueba
* **Entradas**: Conjuntos de entrenamiento  y prueba para nuestros modelos 
* **Valor esperado:** Valores de los pesos de los peces reales de nuestro dataset
* **Valor obtenido:**  Valores de los pesos de los peces predecidos por los modelos

Se usaron de manera aleatoria __5__ números que son un hiperparámetro del modelo de regresión lineal de Sklearn para que se escogieron de manera aleatoria distintas muestras para los conjuntos de prueba y entrenamiento y estas muestras aleatorias se evaluaron en los 6 diferentes modelos creados, donde los resultados fueron los siguientes:

![image](https://user-images.githubusercontent.com/101605777/189426763-2dee53b5-ca21-409c-9b22-dafd30f261ff.png)


El script de Python se encuentra de igual manera en la carperta de __Codes__ con el nombre de **Evi1_pt2.py**.

## Conclusión
Se mamó xd
