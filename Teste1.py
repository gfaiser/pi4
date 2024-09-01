import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
from jinja2 import Template
from tkinter import Tk, filedialog

def load_data():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Selecione a planilha Excel", filetypes=[("Excel files", "*.xlsx *.xls")])
    if file_path:
        data = pd.read_excel(file_path)
        return data, file_path
    else:
        return None, None

def perform_pca(data):
    numeric_data = data.select_dtypes(include=[np.number])
    pca = PCA(n_components=2)
    components = pca.fit_transform(numeric_data)
    return components, pca.explained_variance_ratio_

def train_model(data):
    numeric_data = data.select_dtypes(include=[np.number])
    
    X = numeric_data.iloc[:, :-1]
    y = numeric_data.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, y_test, y_pred

def generate_plots(data, pca_results, pca_variance, y_test, y_pred, output_dir):
    # 1. PCA Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=data.iloc[:, -1], palette="Set1")
    plt.title("PCA - Componentes Principais")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    pca_plot_path = os.path.join(output_dir, 'pca_plot.png')
    plt.savefig(pca_plot_path, bbox_inches='tight', facecolor='white')
    plt.close()

    # 2. Histogram of the first numeric column
    plt.figure(figsize=(10, 6))
    sns.histplot(data.select_dtypes(include=[np.number]).iloc[:, 0], kde=True, color='blue')
    plt.title("Histograma da Primeira Coluna Numérica")
    plt.xlabel(data.select_dtypes(include=[np.number]).columns[0])
    hist_plot_path = os.path.join(output_dir, 'hist_plot.png')
    plt.savefig(hist_plot_path, bbox_inches='tight', facecolor='white')
    plt.close()

    # 3. Regression Results Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title("Regressão Linear - Resultados")
    plt.xlabel("Valores Reais")
    plt.ylabel("Valores Previstos")
    reg_plot_path = os.path.join(output_dir, 'reg_plot.png')
    plt.savefig(reg_plot_path, bbox_inches='tight', facecolor='white')
    plt.close()

    return pca_plot_path, hist_plot_path, reg_plot_path

def generate_html_report(data, pca_plot, hist_plot, reg_plot, pca_variance, mse, r2, output_dir, report_name):
    html_template = """
    <html>
    <head>
        <title>Relatório de Análise de Dados</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 40px; background-color: #ffffff;}
            h1, h2 {text-align: center;}
            img {display: block; margin-left: auto; margin-right: auto;}
            table {width: 100%; border-collapse: collapse; margin-bottom: 40px;}
            table, th, td {border: 1px solid #ddd; padding: 8px;}
            th {background-color: #f2f2f2; text-align: left;}
            tr:hover {background-color: #f5f5f5;}
            p {text-align: justify;}
        </style>
    </head>
    <body>
        <h1>Relatório de Análise de Dados</h1>
        
        <h2>Análise de Componentes Principais (PCA)</h2>
        <img src="{{ pca_plot }}" alt="PCA Scatter Plot" width="600">
        <p>A análise de Componentes Principais (PCA) é uma técnica de redução de dimensionalidade que transforma variáveis correlacionadas em um conjunto de variáveis não correlacionadas, chamadas de componentes principais. No gráfico, o eixo X representa o primeiro componente principal, enquanto o eixo Y representa o segundo componente principal. Cada ponto no gráfico representa uma observação no conjunto de dados. O colorido dos pontos indica a classe da última coluna do dataset, ajudando a visualizar como as observações se agrupam em relação aos componentes principais.</p>
        <p>Variância explicada pelos componentes principais: {{ pca_variance }}</p>
        
        <h2>Histograma da Primeira Coluna Numérica</h2>
        <img src="{{ hist_plot }}" alt="Histogram" width="600">
        <p>O histograma mostra a distribuição dos valores da primeira coluna numérica do dataset. No eixo X estão os valores da coluna, enquanto no eixo Y está a frequência com que esses valores ocorrem. O histograma fornece uma visão geral da distribuição dos dados, indicando a densidade e a variabilidade dos valores dessa coluna.</p>
        
        <h2>Resultados da Regressão Linear</h2>
        <img src="{{ reg_plot }}" alt="Regression Results" width="600">
        <p>O gráfico de resultados da regressão linear mostra a relação entre os valores reais e os valores previstos pelo modelo. No eixo X estão os valores reais, enquanto no eixo Y estão os valores previstos. A linha preta (k--) representa a linha de identidade, onde os valores reais e previstos são iguais. Os pontos que se aproximam dessa linha indicam uma boa correspondência entre os valores reais e previstos, enquanto os pontos distantes indicam erros de previsão.</p>
        <p>Mean Squared Error: {{ mse }}</p>
        <p>R-squared: {{ r2 }}</p>
        
        <h2>Dados Originais (Primeiras 10 Linhas)</h2>
        <table>
            <thead>
                <tr>
                    {% for col in data.columns %}
                        <th>{{ col }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in data.head(10).itertuples() %}
                    <tr>
                        {% for value in row[1:] %}
                            <td>{{ value }}</td>
                        {% endfor %}
                    </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h2>Conclusão</h2>
        <p>Este relatório apresentou uma análise abrangente dos dados fornecidos. Utilizamos técnicas como PCA e regressão linear para explorar e fornecer insights sobre os dados inseridos. As análises realizadas ajudaram a entender a estrutura dos dados e a eficácia do modelo de regressão linear aplicado.</p>
    </body>
    </html>
    """

    template = Template(html_template)
    html_content = template.render(
        pca_plot=pca_plot,
        pca_variance=pca_variance,
        hist_plot=hist_plot,
        reg_plot=reg_plot,
        mse=mse,
        r2=r2,
        data=data
    )
    
    report_path = os.path.join(output_dir, f'{report_name}.html')
    with open(report_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    return report_path

def main():
    data, file_path = load_data()
    
    if data is not None:
        output_dir = os.path.dirname(file_path)
        report_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Realizar PCA
        pca_results, pca_variance = perform_pca(data)
        
        # Treinar o modelo de Machine Learning
        model, mse, r2, y_test, y_pred = train_model(data)
        
        # Gerar os gráficos
        pca_plot, hist_plot, reg_plot = generate_plots(data, pca_results, pca_variance, y_test, y_pred, output_dir)
        
        # Gerar o relatório HTML
        report_path = generate_html_report(data, pca_plot, hist_plot, reg_plot, pca_variance, mse, r2, output_dir, report_name)
        
        print(f"Relatório gerado: {report_path}")
    else:
        print("Nenhum arquivo selecionado.")

if __name__ == "__main__":
    main()
