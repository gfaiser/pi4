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
    img_bytes = fig.to_image(format="png")
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
                         color_discrete_sequence=px.colors.sequential.Plasma)
        plot_file_pca = save_plot_as_base64(fig)
        st.plotly_chart(fig)
        
        st.markdown("""
        A PCA (Análise de Componentes Principais) é uma técnica de redução de dimensionalidade que transforma os dados em um novo sistema de coordenadas. As novas coordenadas, ou componentes principais, são combinações lineares das variáveis originais.
        
        **Variáveis Consideradas para PCA:**
        - Todas as variáveis numéricas presentes no dataset foram usadas para a análise PCA. Isso inclui variáveis como idade, número de infecções, e número de vacinas.
        """)

        # Treinar o modelo de regressão linear
        model, mse, r2, y_test, y_pred = train_model(data)
        
        # Gráficos de Regressão Linear
        st.subheader("Resultados da Regressão Linear")
        regression_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
        fig = px.scatter(regression_df, x='Real', y='Previsto', trendline='ols', title="Regressão Linear - Resultados", 
                         color_discrete_sequence=['#636EFA'])
        plot_file_regression = save_plot_as_base64(fig)
        st.plotly_chart(fig)
        
        st.markdown(f"""
        **Métricas de Regressão Linear:**
        - **Mean Squared Error (MSE):** {mse:.2f}
        - **R-squared (R²):** {r2:.2f}
        
        A regressão linear foi utilizada para prever a variável dependente com base nas variáveis independentes. O gráfico acima mostra a relação entre os valores reais e previstos pela regressão linear.
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
            st.plotly_chart(fig)
        
        # Boxplot
        st.write("Boxplot das Variáveis Numéricas")
        fig = px.box(data, y=numeric_columns[0], title="Boxplot das Variáveis Numéricas", 
                     color_discrete_sequence=['#2CA02C'])
        plot_file_boxplot = save_plot_as_base64(fig)
        st.plotly_chart(fig)
        
        # Gráfico de Barras da Média das Variáveis
        mean_values = data[numeric_columns].mean()
        st.write("Média das Variáveis Numéricas")
        fig = px.bar(x=mean_values.index, y=mean_values.values, title="Média das Variáveis Numéricas", 
                     labels={'x': 'Variável', 'y': 'Média'}, color_discrete_sequence=['#D62728'])
        plot_file_bars = save_plot_as_base64(fig)
        st.plotly_chart(fig)
        
        st.markdown("""
        O histograma mostra a distribuição dos dados, ajudando a entender a frequência das diferentes faixas de valores.
        O boxplot ajuda a identificar outliers e a dispersão dos dados.
        O gráfico de barras da média das variáveis pode mostrar quais variáveis têm maiores médias, indicando tendências gerais.
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
                h1, h2 {{ color: #333; }}
                p {{ font-size: 14px; line-height: 1.6; }}
                ul {{ margin: 0; padding: 0; list-style: none; }}
                li {{ margin-bottom: 5px; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Análise de Dados</h1>
            <p>Este relatório apresenta a análise de dados e técnicas de machine learning aplicadas aos dados fornecidos.</p>
            <h2>Análise de Componentes Principais (PCA)</h2>
            <p>A PCA (Análise de Componentes Principais) é uma técnica de redução de dimensionalidade que transforma os dados em um novo sistema de coordenadas. As novas coordenadas, ou componentes principais, são combinações lineares das variáveis originais.</p>
            <p>Variáveis Consideradas para PCA:</p>
            <p>Todas as variáveis numéricas presentes no dataset foram usadas para a análise PCA. Isso inclui variáveis como idade, número de infecções, e número de vacinas.</p>
            <p>PCA:</p>
            <img src="{plot_file_pca}" alt="Gráfico PCA">
            <h2>Resultados da Regressão Linear</h2>
            <p>Métricas de Regressão Linear:</p>
            <ul>
                <li><strong>Mean Squared Error (MSE):</strong> {mse:.2f}</li>
                <li><strong>R-squared (R²):</strong> {r2:.2f}</li>
            </ul>
            <p>A regressão linear foi utilizada para prever a variável dependente com base nas variáveis independentes. O gráfico acima mostra a relação entre os valores reais e previstos pela regressão linear.</p>
            <img src="{plot_file_regression}" alt="Gráfico de Regressão Linear">
            <h2>Distribuição das Variáveis</h2>
            <p>O histograma mostra a distribuição dos dados, ajudando a entender a frequência das diferentes faixas de valores.</p>
            <p>Histograma:</p>
            <img src="{plot_file_histogram}" alt="Histograma">
            <p>O boxplot ajuda a identificar outliers e a dispersão dos dados.</p>
            <p>Boxplot:</p>
            <img src="{plot_file_boxplot}" alt="Boxplot">
            <p>O gráfico de barras da média das variáveis pode mostrar quais variáveis têm maiores médias, indicando tendências gerais.</p>
            <p>Gráfico de Barras:</p>
            <img src="{plot_file_bars}" alt="Gráfico de Barras">
            <h2>Conclusões</h2>
            <ul>
                <li>Os componentes principais revelam as direções de maior variância nos dados, ajudando a identificar padrões importantes.</li>
                <li>A regressão linear fornece uma visão de como as variáveis independentes afetam a variável dependente.</li>
                <li>O histograma mostra a distribuição dos dados, ajudando a entender a frequência das diferentes faixas de valores.</li>
                <li>O boxplot ajuda a identificar outliers e a dispersão dos dados.</li>
                <li>O gráfico de barras da média das variáveis pode mostrar quais variáveis têm maiores médias, indicando tendências gerais.</li>
            </ul>
        </body>
        </html>
        """

        if st.button('Gerar Relatório em PDF'):
            pdf_bytes = generate_pdf_report(content)
            st.download_button(label="Download do Relatório em PDF", data=pdf_bytes, file_name="relatorio.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
