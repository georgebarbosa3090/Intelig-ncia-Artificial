# 📊 Análise Exploratória de Dados (EDA) com Python

## 🎯 Introdução

A **Análise Exploratória de Dados (EDA – Exploratory Data Analysis)** é uma etapa fundamental em qualquer projeto de Ciência de Dados. Seu principal objetivo é **compreender os dados antes da modelagem**, identificando padrões, inconsistências, tendências e relações entre variáveis.

A EDA envolve o uso de técnicas estatísticas e visuais para extrair insights iniciais, sendo considerada uma fase essencial para garantir qualidade nos resultados de modelos de Machine Learning. ([ia-labs.com.br][1])

---

## 🧠 Objetivos da EDA

* Entender a estrutura dos dados
* Identificar valores nulos ou inconsistentes
* Detectar outliers
* Analisar distribuição das variáveis
* Descobrir relações entre variáveis

📌 Em termos práticos:

> A EDA é o processo de “fazer perguntas aos dados”.

---

## 🧰 Bibliotecas utilizadas

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### 🔍 Explicação linha por linha

```python
import pandas as pd
```

* Importa a biblioteca **Pandas**
* Usada para manipulação de dados em formato de tabela (DataFrame)

```python
import matplotlib.pyplot as plt
```

* Biblioteca para criação de gráficos
* `plt` é um apelido padrão

```python
import seaborn as sns
```

* Biblioteca de visualização baseada no matplotlib
* Facilita criação de gráficos estatísticos

---

## 📥 Leitura do dataset

```python
df = pd.read_csv("dados.csv")
```

### 🔍 Explicação

* `read_csv()` → lê arquivos CSV
* `df` → DataFrame (estrutura tabular com linhas e colunas)

---

## 🔎 Visualização inicial dos dados

```python
df.head()
```

* Mostra as **5 primeiras linhas**
* Útil para entender a estrutura inicial

```python
df.tail()
```

* Mostra as **últimas linhas**

---

## 📊 Estrutura dos dados

```python
df.info()
```

### 🔍 Explicação

* Mostra:

  * Tipos de dados (int, float, object)
  * Valores não nulos
  * Quantidade de colunas

---

```python
df.describe()
```

### 🔍 Explicação

Gera estatísticas básicas:

* média
* desvio padrão
* mínimo e máximo
* quartis

👉 Essencial para análise quantitativa

---

## 🧹 Limpeza de dados

```python
df.isnull().sum()
```

### 🔍 Explicação

* Verifica valores nulos
* `.sum()` conta quantos existem

---

```python
df.dropna()
```

* Remove linhas com valores nulos

---

```python
df.fillna(0)
```

* Substitui valores nulos por 0

---

## 🔄 Manipulação de dados

```python
df['coluna'].unique()
```

* Retorna valores únicos

---

```python
df['coluna'].nunique()
```

* Conta valores únicos

---

```python
df.sort_values(by='coluna')
```

* Ordena os dados

---

## 📈 Visualização de dados

### 📊 Histograma

```python
df['idade'].hist()
```

* Mostra distribuição dos dados

---

### 📦 Boxplot

```python
sns.boxplot(x=df['idade'])
```

* Identifica outliers

---

### 🔵 Scatter Plot

```python
plt.scatter(df['x'], df['y'])
```

* Mostra relação entre duas variáveis

---

### 🔥 Heatmap (correlação)

```python
sns.heatmap(df.corr(), annot=True)
```

### 🔍 Explicação

* `corr()` → calcula correlação
* `heatmap()` → mostra visualmente

👉 Muito usado para Machine Learning

---

## 🔗 Relação entre variáveis

```python
sns.pairplot(df)
```

* Mostra relações entre todas as variáveis
* Muito usado em EDA inicial

---

## 📊 Interpretação (parte mais importante)

A EDA não é apenas código — é análise.

Durante essa etapa, o analista deve responder:

* Existe padrão nos dados?
* Há correlação entre variáveis?
* Existem dados inconsistentes?
* Os dados estão balanceados?

---

## 🎓 Conexão com os vídeos

[Análise Exploratória de Dados - Aula prática](https://www.youtube.com/watch?v=woObL4Mx9ns&utm_source=chatgpt.com)

Os vídeos demonstram exatamente esse fluxo:

1. Carregamento dos dados
2. Limpeza e organização
3. Análise estatística
4. Visualização gráfica
5. Interpretação

---

## ☁️ Uso no Google Colab

O Google Colab permite executar esse processo diretamente no navegador:

### 📌 Vantagens:

* Não precisa instalar nada
* Execução por células
* Ideal para EDA

---

## 🚀 Pipeline completo de EDA

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dados.csv")

print(df.head())
print(df.info())
print(df.describe())

df.isnull().sum()

sns.heatmap(df.corr(), annot=True)
plt.show()
```

---

## 🧠 Conclusão

A Análise Exploratória de Dados é uma etapa **crítica e indispensável** na ciência de dados, pois:

* Evita erros na modelagem
* Garante qualidade dos dados
* Permite descobrir insights ocultos
* Orienta decisões

📌 Sem EDA, qualquer modelo de Machine Learning está comprometido.

---

## 📌 Autor

Material estruturado para ensino e uso acadêmico (IFPA / Ciência de Dados)

---

[1]: https://ia-labs.com.br/cursos/analise-exploratoria-de-dados-python/?utm_source=chatgpt.com "Análise Exploratória de Dados (python) – IA-Labs"
