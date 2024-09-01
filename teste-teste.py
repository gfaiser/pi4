import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configurações
num_rows = 1000
product_types = ['Frutas', 'Vegetais', 'Carnes', 'Laticínios', 'Grãos', 'Bebidas', 'Produtos de Limpeza', 'Padaria', 'Snacks']
min_price = 1.0
max_price = 100.0
min_quantity = 1
max_quantity = 20
min_frequency = 1
max_frequency = 10
min_age = 18
max_age = 80
sex_options = ['Masculino', 'Feminino']
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

# Gerar datas
dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_rows)]

# Gerar dados
data = {
    'Tipo de Produto': [random.choice(product_types) for _ in range(num_rows)],
    'Preço do Produto (R$)': [round(random.uniform(min_price, max_price), 2) for _ in range(num_rows)],
    'Quantidade': [random.randint(min_quantity, max_quantity) for _ in range(num_rows)],
    'Frequência (nº de compras por mês)': [random.randint(min_frequency, max_frequency) for _ in range(num_rows)],
    'Data da Aquisição': [date.strftime('%Y-%m-%d') for date in dates],
    'Sexo do Consumidor': [random.choice(sex_options) for _ in range(num_rows)],
    'Idade do Consumidor': [random.randint(min_age, max_age) for _ in range(num_rows)]
}

# Criar DataFrame
df = pd.DataFrame(data)

# Salvar em um arquivo Excel
output_file = 'consumo_supermercado_sp.xlsx'
df.to_excel(output_file, index=False, engine='openpyxl')

print(f"Arquivo Excel '{output_file}' gerado com sucesso.")
