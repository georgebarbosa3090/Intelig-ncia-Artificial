# 📊 KNN (K-Nearest Neighbors) - Python

Este repositório apresenta uma implementação simples do algoritmo **KNN (K-Nearest Neighbors)** utilizando Python, com visualização gráfica e explicação detalhada linha por linha.

---

## 🚀 Objetivo

Demonstrar de forma didática:

* Como funciona o algoritmo KNN
* Como treinar um modelo com `scikit-learn`
* Como visualizar os dados com `matplotlib`
* Como classificar um novo ponto

---

## 📦 Tecnologias utilizadas

* Python 3
* scikit-learn
* matplotlib

---

## 🧠 Conceito do KNN

O KNN é um algoritmo de aprendizado supervisionado que classifica novos dados com base nos **K vizinhos mais próximos**.

📌 Ideia central:

> “Diga-me quem são seus vizinhos e eu direi quem você é.”

---

## 💻 Código completo

```python
import matplotlib.pyplot as plt

# Dados (features)
num_1 = [21, 22, 27, 21, 20, 28, 31, 23, 27, 29]
num_2 = [38, 36, 41, 34, 33, 42, 41, 39, 38, 38]

# Classes (rótulos)
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]

# Plot inicial
plt.scatter(num_1, num_2, c=classes)
plt.show()

from sklearn.neighbors import KNeighborsClassifier

# Junta os dados (x,y)
data = list(zip(num_1, num_2))
print(data)

# Modelo KNN (k=1)
knn = KNeighborsClassifier(n_neighbors=1)

# Treinamento
knn.fit(data, classes)

# Novo ponto
new_x = 25
new_y = 38
new_point = [(new_x, new_y)]

# Predição
prediction = knn.predict(new_point)

# Plot com novo ponto
plt.scatter(num_1 + [new_x], num_2 + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
```

---

## 🔍 Explicação linha por linha

### 📌 Importação da biblioteca de gráficos

```python
import matplotlib.pyplot as plt
```

Importa a biblioteca responsável por criar gráficos.

---

### 📌 Definição dos dados (features)

```python
num_1 = [...]
num_2 = [...]
```

Representam duas variáveis de entrada (eixos X e Y).

---

### 📌 Classes (rótulos)

```python
classes = [...]
```

Define a categoria de cada ponto:

* `0` → Classe A
* `1` → Classe B

---

### 📌 Visualização inicial

```python
plt.scatter(num_1, num_2, c=classes)
```

Cria um gráfico de dispersão:

* Eixo X → `num_1`
* Eixo Y → `num_2`
* Cor → classe

```python
plt.show()
```

Exibe o gráfico.

---

### 📌 Importação do KNN

```python
from sklearn.neighbors import KNeighborsClassifier
```

Importa o algoritmo KNN da biblioteca `scikit-learn`.

---

### 📌 Organização dos dados

```python
data = list(zip(num_1, num_2))
```

Combina os dados em pares `(x, y)`:

```python
[(21,38), (22,36), ...]
```

```python
print(data)
```

Exibe os dados no console.

---

### 📌 Criação do modelo

```python
knn = KNeighborsClassifier(n_neighbors=1)
```

Define o modelo com:

* `k = 1` → considera apenas o vizinho mais próximo

---

### 📌 Treinamento

```python
knn.fit(data, classes)
```

O modelo aprende a relação entre os dados e suas classes.

---

### 📌 Novo ponto para classificação

```python
new_x = 25
new_y = 38
```

Define um ponto desconhecido.

```python
new_point = [(new_x, new_y)]
```

Formata no padrão exigido pelo modelo.

---

### 📌 Predição

```python
prediction = knn.predict(new_point)
```

O modelo:

1. Calcula distâncias
2. Encontra o vizinho mais próximo
3. Define a classe

---

### 📌 Visualização final

```python
plt.scatter(num_1 + [new_x], num_2 + [new_y], c=classes + [prediction[0]])
```

Adiciona o novo ponto ao gráfico.

```python
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
```

Exibe a classe do novo ponto no gráfico.

```python
plt.show()
```

Mostra o resultado final.

---

## 📊 Resultado esperado

* Pontos coloridos por classe
* Novo ponto classificado automaticamente
* Visualização clara do funcionamento do KNN

---

## 📌 Autor

George Barbosa


---
