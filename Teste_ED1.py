import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pdfkit
import io
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import base64

# Função para realizar PCA
def perform_pca(data):
    numeric_data = data.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    pca = PCA()
    components = pca.fit_transform(scaled_data)
    return components, pca

# Função para treinar o modelo
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

# Função para salvar gráficos como imagens PNG e retornar o base64
def save_plot_as_base64(fig):
    img_bytes = fig.to_image(format="png", width=700, height=500)  # Ajuste o tamanho aqui
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# Função para gerar o PDF com as imagens inline
def generate_pdf_report(content):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Salva o conteúdo HTML em um arquivo temporário
        html_path = os.path.join(temp_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Configura o caminho do wkhtmltopdf
        config = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')  # Ajuste o caminho conforme necessário
        
        # Gera o PDF a partir do arquivo HTML
        pdf_path = os.path.join(temp_dir, 'report.pdf')
        pdfkit.from_file(html_path, pdf_path, configuration=config)
        
        # Lê o PDF gerado
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        return pdf_bytes

# Função principal do aplicativo Streamlit
def main():
    st.set_page_config(page_title="Projeto Integrador IV - UNIVESP 2024", page_icon=":bar_chart:")
    
    st.title("Projeto Integrador IV - UNIVESP 2024")
    st.markdown("""
    **Objetivo:** Este relatório apresenta a análise de dados e técnicas de machine learning aplicadas aos dados fornecidos.
    """)

    # Upload de arquivo
    uploaded_file = st.file_uploader("Selecione a planilha Excel", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write("Dados carregados com sucesso!")
        st.dataframe(data.head(10))
        
        # Realizar PCA
        pca_results, pca = perform_pca(data)
        pca_variance = pca.explained_variance_ratio_
        num_components = st.sidebar.slider("Número de Componentes PCA", min_value=2, max_value=min(len(pca_variance), 5), value=2)
        
        # Gráficos Interativos com Plotly
        st.subheader("Análise de Componentes Principais (PCA)")
        pca_df = pd.DataFrame(data=pca_results, columns=[f'PC{i+1}' for i in range(len(pca_variance))])
        pca_df['Target'] = data.iloc[:, -1]
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="PCA - Componentes Principais", 
                         labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'},
                         color_continuous_scale=px.colors.sequential.Plasma)  # Use color_continuous_scale para garantir a cor
        plot_file_pca = save_plot_as_base64(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Análise de Componentes Principais (PCA):** A PCA é uma técnica que reduz a dimensionalidade dos dados, mantendo a maior variação possível dos dados originais. Os componentes principais (PCs) são combinações lineares das variáveis originais.
        
        - **Componente Principal 1 e 2:** Esses componentes explicam a maior parte da variância dos dados. O gráfico mostra a projeção dos dados nesses dois componentes principais. É útil para identificar padrões e agrupamentos nos dados.
        
        **Conclusão:** A análise PCA ajuda a visualizar como as variáveis influenciam a estrutura dos dados e a identificar possíveis padrões ou agrupamentos que podem não ser evidentes nas dimensões originais.
        """)

        # Treinar o modelo de regressão linear
        model, mse, r2, y_test, y_pred = train_model(data)
        
        # Gráficos de Regressão Linear
        st.subheader("Resultados da Regressão Linear")
        regression_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
        fig = px.scatter(regression_df, x='Real', y='Previsto', trendline='ols', title="Regressão Linear - Resultados", 
                         color_discrete_sequence=['#636EFA'])
        plot_file_regression = save_plot_as_base64(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Resultados da Regressão Linear:**
        
        - **Mean Squared Error (MSE):** {mse:.2f}
        - **R-squared (R²):** {r2:.2f}
        
        O gráfico mostra a relação entre os valores reais e previstos pela regressão linear. O modelo de regressão linear é utilizado para prever a variável dependente com base nas variáveis independentes. A linha de tendência (ols) ajuda a visualizar o ajuste do modelo.
        
        **Conclusão:** O MSE e o R² são métricas importantes para avaliar a qualidade do modelo. O MSE indica o erro médio quadrático, enquanto o R² mostra a proporção da variância explicada pelo modelo. Um R² próximo de 1 indica um bom ajuste do modelo.
        """)

        # Gráficos Adicionais
        st.subheader("Distribuição das Variáveis")
        
        # Histograma
        st.write("Histograma da Primeira Coluna Numérica")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            fig = px.histogram(data, x=numeric_columns[0], title=f"Histograma da Coluna: {numeric_columns[0]}", 
                               color_discrete_sequence=['#FF7F0E'])
            plot_file_histogram = save_plot_as_base64(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Boxplot
        st.write("Boxplot das Variáveis Numéricas")
        fig = px.box(data, y=numeric_columns[0], title="Boxplot das Variáveis Numéricas", 
                     color_discrete_sequence=['#2CA02C'])
        plot_file_boxplot = save_plot_as_base64(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de Barras da Média das Variáveis
        mean_values = data[numeric_columns].mean()
        st.write("Média das Variáveis Numéricas")
        fig = px.bar(x=mean_values.index, y=mean_values.values, title="Média das Variáveis Numéricas", 
                     labels={'x': 'Variável', 'y': 'Média'}, color_discrete_sequence=['#D62728'])
        plot_file_bars = save_plot_as_base64(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Distribuição das Variáveis:**

        - **Histograma:** Mostra a distribuição dos dados da primeira coluna numérica. Ajuda a entender a frequência das diferentes faixas de valores.
        - **Boxplot:** Mostra a distribuição e os outliers das variáveis numéricas. Ajuda a identificar a dispersão dos dados e detectar valores extremos.
        - **Média das Variáveis:** O gráfico de barras mostra a média das variáveis numéricas. Indica as tendências gerais nos dados.

        **Conclusão:** Esses gráficos ajudam a entender melhor a estrutura dos dados, identificando padrões e tendências importantes. A distribuição das variáveis é fundamental para a interpretação dos dados e para a aplicação de técnicas de análise multivariada e machine learning.
        """)

        # Gerar PDF
        content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Relatório de Análise de Dados</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #004d99; }}
                h2 {{ color: #007acc; }}
                p {{ font-size: 14px; line-height: 1.6; }}
                img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; }}
                header {{ text-align: center; margin-bottom: 40px; }}
                footer {{ text-align: center; margin-top: 40px; font-size: 12px; color: #888; }}
                .chart {{ max-width: 90%; margin: auto; }}
            </style>
        </head>
        <body>
            <header>
                <h1>Relatório de Análise de Dados</h1>
                <p>Projeto Integrador IV - UNIVESP 2024</p>
            </header>
            
            <h2>Objetivo</h2>
            <p>Este relatório apresenta a análise de dados e técnicas de machine learning aplicadas aos dados fornecidos. O objetivo é explorar as principais características dos dados e construir modelos preditivos usando as técnicas discutidas ao longo do curso.</p>

            <h2>Análise de Componentes Principais (PCA)</h2>
            <p>A Análise de Componentes Principais (PCA) foi realizada para reduzir a dimensionalidade dos dados e explorar as relações entre as variáveis. Abaixo está o gráfico dos componentes principais.</p>
            <img class="chart" src="{plot_file_pca}" alt="PCA">
            <p>**Análise de Componentes Principais (PCA):** A PCA é uma técnica que reduz a dimensionalidade dos dados, mantendo a maior variação possível dos dados originais. Os componentes principais (PCs) são combinações lineares das variáveis originais.</p>
            <p>- **Componente Principal 1 e 2:** Esses componentes explicam a maior parte da variância dos dados. O gráfico mostra a projeção dos dados nesses dois componentes principais. É útil para identificar padrões e agrupamentos nos dados.</p>
            <p>**Conclusão:** A análise PCA ajuda a visualizar como as variáveis influenciam a estrutura dos dados e a identificar possíveis padrões ou agrupamentos que podem não ser evidentes nas dimensões originais.</p>

            <h2>Resultados da Regressão Linear</h2>
            <p>Os resultados da regressão linear indicam a qualidade do ajuste do modelo preditivo. As métricas como MSE e R² estão incluídas abaixo.</p>
            <img class="chart" src="{plot_file_regression}" alt="Regressão Linear">
            <p>**Resultados da Regressão Linear:**</p>
            <p>- **Mean Squared Error (MSE):** {mse:.2f}</p>
            <p>- **R-squared (R²):** {r2:.2f}</p>
            <p>O gráfico mostra a relação entre os valores reais e previstos pela regressão linear. O modelo de regressão linear é utilizado para prever a variável dependente com base nas variáveis independentes. A linha de tendência (ols) ajuda a visualizar o ajuste do modelo.</p>
            <p>**Conclusão:** O MSE e o R² são métricas importantes para avaliar a qualidade do modelo. O MSE indica o erro médio quadrático, enquanto o R² mostra a proporção da variância explicada pelo modelo. Um R² próximo de 1 indica um bom ajuste do modelo.</p>

            <h2>Distribuição das Variáveis</h2>
            <p>A distribuição das variáveis foi explorada usando histogramas e boxplots. Estes gráficos ajudam a entender a dispersão e as características principais dos dados.</p>
            <img class="chart" src="{plot_file_histogram}" alt="Histograma">
            <p>**Distribuição das Variáveis:**</p>
            <p>- **Histograma:** Mostra a distribuição dos dados da primeira coluna numérica. Ajuda a entender a frequência das diferentes faixas de valores.</p>
            <p>- **Boxplot:** Mostra a distribuição e os outliers das variáveis numéricas. Ajuda a identificar a dispersão dos dados e detectar valores extremos.</p>
            <p>- **Média das Variáveis:** O gráfico de barras mostra a média das variáveis numéricas. Indica as tendências gerais nos dados.</p>
            <p>**Conclusão:** Esses gráficos ajudam a entender melhor a estrutura dos dados, identificando padrões e tendências importantes. A distribuição das variáveis é fundamental para a interpretação dos dados e para a aplicação de técnicas de análise multivariada e machine learning.</p>

            <h2>Média das Variáveis Numéricas</h2>
            <p>O gráfico de barras abaixo mostra a média das variáveis numéricas, o que ajuda a identificar tendências gerais nos dados.</p>
            <img class="chart" src="{plot_file_bars}" alt="Média das Variáveis">

            <footer>
                <p><b>Projeto Integrador IV</b></p> 
                <p>desenvolvido por alunos do curso de Engenharia da Computação e Ciência de Dados</p>
                <p>UNIVESP, 2º Semestre de 2024</p>
            </footer>
        </body>
        </html>
        """
        
        pdf_bytes = generate_pdf_report(content)
        
        st.download_button(
            label="Baixar Relatório em PDF",
            data=pdf_bytes,
            file_name="Relatorio_Analise_de_Dados.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
