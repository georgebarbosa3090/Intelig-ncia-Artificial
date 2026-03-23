# 📊 Análise Exploratória de Dados (EDA) com Python

## 🎯 Introdução

A **Análise Exploratória de Dados (EDA – Exploratory Data Analysis)** é uma etapa fundamental em qualquer projeto de Ciência de Dados. Seu principal objetivo é **compreender os dados antes da modelagem**, identificando padrões, inconsistências, tendências e relações entre variáveis. É o processo de investigar e analisar conjuntos de dados para compreender suas características principais, frequentemente utilizando métodos de visualização e estatística descritiva. O objetivo central é **construir uma compreensão intuitiva dos dados** sem necessariamente ter vivido a experiência de coleta, permitindo identificar padrões, detectar anomalias (outliers) e testar hipóteses antes de aplicar modelos complexos de aprendizado de máquina.

Abaixo estão os principais componentes e etapas de uma EDA detalhada conforme as fontes:

### 1. Etapas Iniciais e Preparação
Uma análise bem estruturada geralmente segue um fluxo que começa pelo **entendimento do negócio** e dos dados, seguido pelo tratamento e limpeza.
*   **Inspeção Visual:** Utiliza-se funções como `head()` para visualizar as primeiras linhas e `info()` para verificar a estrutura do dataset e identificar dados ausentes.
*   **Limpeza de Dados:** Envolve a remoção de colunas irrelevantes para o problema e o tratamento de valores nulos (usando técnicas como `dropna()`), garantindo que a base de dados seja consistente para a análise.

### 2. Tipos de Dados e Estatística Descritiva
Compreender a natureza das variáveis é fundamental para escolher as técnicas de análise corretas. Os dados podem ser divididos em:
*   **Numéricos:** Podem ser **discretos** (contáveis) ou **contínuos** (infinitas possibilidades dentro de um intervalo).
*   **Categóricos:** Envolvem qualidades subjetivas, podendo ser **ordinais** (com hierarquia lógica, como tamanho P, M, G) ou **nominais** (sem ordem, como cores).

Para resumir as informações, utilizam-se medidas de:
*   **Centralidade:** Média, mediana (valor posicional que evita distorções por outliers) e moda.
*   **Dispersão:** Desvio padrão, variância, valores máximos e mínimos, e quartis.

### 3. Visualização de Dados
A visualização permite "bater o olho" e entender distribuições e relações complexas que apenas números não revelariam.
*   **Histograma:** Mostra a frequência e a distribuição dos dados (como a distribuição normal em formato de sino). É útil para detectar a densidade de ocorrências em determinados intervalos (bins).
*   **Box Plot:** Fornece uma visão clara dos quartis, da mediana e, principalmente, de **outliers** (pontos fora da curva).
*   **Scatter Plot (Gráfico de Dispersão):** Utilizado para identificar a **correlação** entre duas variáveis numéricas, mostrando se uma variável tende a aumentar ou diminuir conforme a outra muda.

### 4. Correlação e Causalidade
A EDA busca entender como as variáveis se relacionam através de coeficientes como o de **Pearson** (sensível a magnitudes) e o de **Spearman** (focado na ordem de crescimento). É importante notar que uma alta correlação (ex: entre preço e qualidade de um vinho) indica uma relação estatística, mas não necessariamente implica que uma variável causa a outra diretamente.

### Importância para o Machine Learning
A análise exploratória é crucial para o sucesso de algoritmos como o **KNN**. Durante a EDA, identifica-se a necessidade de **normalização ou padronização dos dados**, pois o KNN baseia-se em cálculos de distância; se as escalas forem muito diferentes, a análise será distorcida. Além disso, a detecção de outliers e a redução de dimensionalidade (como o PCA) são decisões frequentemente tomadas com base nos insights gerados pela análise exploratória.

**Outliers**, também conhecidos como **pontos fora da curva ou anomalias**, são observações que se afastam significativamente do padrão geral de um conjunto de dados. Eles representam valores discrepantes que podem surgir por erros de medição ou por variações naturais extremas, como o exemplo de um bilionário residindo em um bairro de classe média, o que distorceria a média de renda local.

O **Box Plot** (ou diagrama de caixa) é uma das ferramentas mais eficazes da análise exploratória para identificar esses pontos visualmente. Ele funciona da seguinte maneira:

*   **Estrutura do Gráfico:** O Box Plot apresenta uma caixa central que contém 50% dos dados, delimitada pelo primeiro e terceiro **quartis**, com uma linha interna representando a **mediana** (o segundo quartil).
*   **Bigodes (Whiskers):** A partir dessa caixa, estendem-se linhas conhecidas como "bigodinhos", que indicam os limites dos valores esperados (mínimo e máximo dentro de uma faixa estatística comum).
*   **Identificação Visual:** Qualquer ponto de dado que esteja localizado **além das extremidades desses bigodes** é plotado individualmente no gráfico e classificado como um **outlier**. 

Essa identificação é crucial para algoritmos como o **KNN**, que é altamente sensível a outliers quando o valor de **K** é pequeno (como K=1). Nesses casos, um único outlier pode levar o modelo ao erro, classificando incorretamente um novo dado apenas porque ele está fisicamente próximo de uma anomalia. Por isso, utilizar o Box Plot para detectar e tratar essas inconsistências é uma etapa fundamental antes de treinar o modelo.

As bibliotecas **Pandas** e **Seaborn** são ferramentas fundamentais na Análise Exploratória de Dados (EDA), permitindo que o analista compreenda a estrutura, qualidade e padrões de um conjunto de dados antes da aplicação de modelos de machine learning.

### O papel da biblioteca Pandas
O Pandas é utilizado principalmente para a **manipulação, limpeza e resumo estatístico** dos dados. Suas principais contribuições incluem:

*   **Carregamento e Visualização Inicial:** Permite importar dados (como arquivos CSV) e visualizar rapidamente as primeiras linhas com a função `head()` para entender a composição das colunas.
*   **Inspeção da Estrutura:** Através da função `info()`, é possível verificar o tipo de dado em cada coluna e identificar a presença de **valores ausentes**, o que é crucial para decidir se colunas ou linhas devem ser descartadas.
*   **Estatística Descritiva:** A função `describe()` gera um resumo estatístico das variáveis numéricas, apresentando média, valores mínimos, máximos e quartis, ajudando a identificar a escala dos dados e a dispersão.
*   **Limpeza e Transformação:** Facilita a remoção de colunas irrelevantes com `drop()`, a eliminação de dados nulos com `dropna()` e a criação de filtros complexos para segmentar o dataset (como agrupar dados por país ou faixa de preço).
*   **Análise de Frequência:** Com o `value_counts()`, é possível identificar quais categorias são mais frequentes, como os países que mais produzem determinados produtos ou os tipos de uva mais avaliados.

### O papel da biblioteca Seaborn
O Seaborn, construído sobre o Matplotlib, foca na **visualização estatística**, tornando mais fácil a detecção visual de padrões que números sozinhos podem omitir. Suas principais funções na EDA são:

*   **Distribuição de Dados (Histogramas):** Permite visualizar a frequência dos dados e verificar se eles seguem uma **distribuição normal** (curva em formato de sino), o que ajuda a entender a densidade de ocorrências em certos intervalos.
*   **Detecção de Outliers (Box Plots):** O `boxplot` é uma das ferramentas mais eficazes para identificar **anomalias ou pontos fora da curva**, além de mostrar visualmente a média e os quartis de diferentes categorias.
*   **Análise de Correlação (Scatter Plots):** Através de gráficos de dispersão (como o `scatterplot` ou `regplot`), o analista pode verificar se existe uma **relação positiva ou negativa** entre duas variáveis, como o preço de um produto e sua qualidade.
*   **Comparação entre Categorias:** Facilita a criação de gráficos de barras (`barplot`) para comparar métricas entre diferentes grupos, como a média de pontuação por país.

### Importância Conjunta para o Machine Learning
O uso combinado dessas bibliotecas permite realizar intervenções críticas para o sucesso de algoritmos como o **KNN**. Por meio da EDA com Pandas e Seaborn, o desenvolvedor identifica a necessidade de **normalizar ou padronizar** os dados (essencial para cálculos de distância) e decide sobre a redução de dimensionalidade para evitar a lentidão do modelo em grandes conjuntos de dados.

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
