import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Lendo o arquivo CSV em um dataframe
df = pd.read_csv("ecommerce_estatistica.csv")
print(df.info())
print(df.describe())

# Gráfico de Histograma
plt.hist(df['Preço'], bins=30, color='blue', edgecolor='black')
plt.title('Distribuição de Preços')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()

# Gráfico de Dispersão - Exemplo: Relação entre preço e quantidade vendida
plt.scatter(df['Preço'], df['Qtd_Vendidos'], alpha=0.5)
plt.title('Relação entre Preço e Quantidade Vendida')
plt.xlabel('Preço')
plt.ylabel('Quantidade Vendida')
plt.show()

# Mapa de calor - Exemplo: Correlação entre variáveis
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Correlações')
plt.show()

# Gráfico de Barra - Exemplo: Quantidade vendida por marca
df_grouped = df.groupby('Marca')['Qtd_Vendidos'].sum().sort_values(ascending=False)
df_grouped.plot(kind='bar', color='green')
plt.title('Quantidade Vendida por Marca')
plt.xlabel('Marca')
plt.ylabel('Quantidade Vendida')
plt.show()

# Gráfico de Pizza - Exemplo: Proporção de vendas por temporada
season_counts = df['Temporada'].value_counts()
season_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Proporção de Vendas por Temporada')
plt.ylabel('')  # Remove o rótulo do eixo Y
plt.show()

# Gráfico de Densidade - Exemplo: Distribuição de descontos
sns.kdeplot(df['Desconto'], shade=True, color='purple')
plt.title('Distribuição de Descontos')
plt.xlabel('Desconto')
plt.ylabel('Densidade')
plt.show()

# Gráfico de Regressão - Exemplo: Relação entre preço e quantidade vendida
sns.regplot(x='Preço', y='Qtd_Vendidos', data=df, scatter_kws={'alpha':0.5})
plt.title('Regressão Linear entre Preço e Quantidade Vendida')
plt.xlabel('Preço')
plt.ylabel('Quantidade Vendida')
plt.show()
