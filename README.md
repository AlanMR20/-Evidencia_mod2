# Evidencia 1, Módulo 2: Análisis y Reporte sobre el desempeño del modelo

El script utilizado para esta evidencia se llama __Evi1_pt2.py__, el cual es el único archivo que se encuentra en el repositorio.

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

**Librerias Utiilizadas**
* __Pandas:__ Para importar y visualizar el dataset
* __Numpy:__ Para el manejo de arreglos y medidas de estadística
* __Matplotlib:__ Para poder graficar los resultados obtenidos
* __Train_test_split de Sklearn:__ Esto nos permite separar datos de entrenamiento y prueba de manera más sencilla

## Análisis previo de los datos
Antes de escoger un modelo de regresión para hacer nuestras predicciones, primero debemos de conocer cuál es la relación de los datos. Para ello podemos ver la matriz de correlaciones de nuestros datos para ver si existe linealidad entre dos o más variables, teniendo lo siguiente:

![image](https://user-images.githubusercontent.com/101605777/188798206-84b3f85b-fcba-480b-b4c3-cf1dfa1d974b.png)

Gracias a esta matriz de correlaciones podemos darnos cuenta de que todas las longitudes están fuertemente relacionadas. Para el caso de la altura vemos que realmente no está tan fuertemente relacionada ni con el peso ni con las longitudes por lo que puede que no sea una característica que nos sea útil para implementar nuestro modelo. Por otro lado, vemos que la anchura diagonal se relaciona correlaciona más con los demás datos que la altura, por lo que podemos usar esta característica para poder realizar nuestro modelo.

## Modelo de regresión lineal múltiple
Para nuestro conjunto de datos tenemos que las longitudes 1,2 y 3 están fuertemente relacionadas con el peso pero sobre todo entre sí, por lo que haremos una nueva característica donde se incluyan las tres longitudes de tal manera que siga siendo útil para nuestro modelo y sobre todo reducir el número de dimensiones. Lo que haremos será multiplicar las 3 longitudes y sacarle la raíz cúbica a dicha operación para no tener un valor tan grande para los coeficientes del modelo. Esta nueva característica será la primera de nuestro modelo. Mientras que la anchura diagonal por sí sola será nuestra segunda característica y con buscaremos un modelo de regresión lineal de segundo orden que se pueda ajustar a nuestras características para encontrar un modelo para predecir el peso de los peces. Por lo que nuestro mdelo nos quedaría de la siguiene manera:

$h_\theta=\theta+\theta_1x_1+\theta_2x_2$

## Métricas de desempeño (valor logrado sobre el subset de prueba)

## Predicciones de prueba (entradas, valor esperado, valor obtenido)

![image](https://user-images.githubusercontent.com/101605777/188798597-7575f2a8-3617-476d-8eb7-7fb6541493bf.png)


## Conclusión
