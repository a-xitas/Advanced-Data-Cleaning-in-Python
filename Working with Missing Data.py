## 1. Introduction ##

import pandas as pd
mvc = pd.read_csv("nypd_mvc_2018.csv")

# contar o nmr de NaN's no N/DataSet usando o metodo isnull() encadeado com o sum():
null_counts = mvc.isnull().sum()

mvc.shape

mvc_null_pct = null_counts / mvc.shape[0]*100

# criar um DataFrame com o nmr total de NaN's para cada coluna, assim como as percetangens que cada uma destas colunas com NaN's representa no nmr total de datapoints:
null_sum_pct = pd.DataFrame({'null_sum':null_counts, 'null_pct':mvc_null_pct})
# passamos a tabela para a horizontal com o metodo transpose() ou diminutivo T(), isto para se ler melhor, e acrescentamos o tipo dos ficheiros para integer, mudando de float (astype(int)). Isto para arredondar os nmrs à unidade:
null_sum_pct_h = null_sum_pct.transpose().astype(int)

## 2. Verifying the Total Columns ##

killed_cols = [col for col in mvc.columns if 'killed' in col]
killed = mvc[killed_cols].copy()

killed_manual_sum = killed.iloc[:,0] + killed.iloc[:,1] + killed.iloc[:,2]

killed_mask = killed_manual_sum != killed['total_killed']

killed_non_eq = killed[killed_mask]


## 3. Filling and Verifying the Killed and Injured Data ##

import numpy as np

# fix the killed values
killed['total_killed'] = killed['total_killed'].mask(killed['total_killed'].isnull(), killed_manual_sum)
killed['total_killed'] = killed['total_killed'].mask(killed['total_killed'] != killed_manual_sum, np.nan)

# Create an injured dataframe and manually sum values
injured = mvc[[col for col in mvc.columns if 'injured' in col]].copy()
injured_manual_sum = injured.iloc[:,:3].sum(axis=1)

injured['total_injured'] = injured['total_injured'].mask(injured['total_injured'].isnull(), injured_manual_sum)

injured['total_injured'] = injured['total_injured'].mask(injured['total_injured'] != injured_manual_sum, np.nan)

## 4. Assigning the Corrected Data Back to the Main Dataframe ##

mvc['total_injured'] = injured['total_injured']

mvc['total_killed'] = killed['total_killed']

mvc_top5 = mvc.head()

## 5. Visualizing Missing Data with Plots ##

import matplotlib.pyplot as plt
import seaborn as sns

def plot_null_correlations(df):
    # create a correlation matrix only for columns with at least
    # one missing value
    cols_with_missing_vals = df.columns[df.isnull().sum() > 0]
    missing_corr = df[cols_with_missing_vals].isnull().corr()
    
    # create a triangular mask to avoid repeated values and make
    # the plot easier to read
    missing_corr = missing_corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(missing_corr), k=1)
    
    # plot a heatmap of the values
    plt.figure(figsize=(20,14))
    ax = sns.heatmap(missing_corr, vmin=-1, vmax=1, cbar=False,
                     cmap='RdBu', mask=mask, annot=True)
    
    # format the text in the plot to make it easier to read
    for text in ax.texts:
        t = float(text.get_text())
        if -0.05 < t < 0.01:
            text.set_text('')
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks(rotation=90, size='x-large')
    plt.yticks(rotation=0, size='x-large')

    plt.show()
    

# criar uma list comprehension para me retornar as colunas do DS mvc, para todas as colunas que façam parte deste DS e que tenham no titulo das colunas 'vehicle':
vehicle = mvc[[col for col in mvc.columns if 'vehicle' in col]].copy()

plot_null_correlations(vehicle)



## 6. Analyzing Correlations in Missing Data ##

col_labels = ['v_number', 'vehicle_missing', 'cause_missing']

# mvc_v = mvc[['vehicle_1', 'vehicle_2', 'vehicle_3', 'vehicle_4', 'vehicle_5']]

# mvc_c =  mvc[['cause_vehicle_

vc_null_data = []

for v in range(1,6):
    v_col = 'vehicle_{}'.format(v)
    c_col = 'cause_vehicle_{}'.format(v)
    v_null = (mvc[v_col].isnull() & mvc[c_col].notnull()).sum()
    c_null = (mvc[c_col].isnull() & mvc[v_col].notnull()).sum()
    # a seguir junto as 3 variáveis numa lista de uma lista, isto para criar um DataFrame, q é nada + nada - q um Dataframe:
    vc_null_data.append([v, v_null, c_null])
# finalizamos criando um Dataframe com a list of lists criada anteriormente e usando o método pd.DataFrame do pandas. Usando para cols names as labels que temos em cima na lista col_labels:
vc_null_df = pd.DataFrame(vc_null_data, columns=col_labels)

#mvc['vehicle_3'].isnull().sum()

# for v in range(1,6):
# #     v_1_null = (mvc['vehicle_{}'.format(v)].isnull().sum()) & (mvc['cause_vehicle_{}'.format(v)].notnull().sum())
    
#     v_1_null = (mvc[v_col].isnull() & mvc[c_col].notnull()).sum()

    
# print(v_1_null)



## 7. Finding the Most Common Values Across Multiple Columns ##

v_cols = [c for c in mvc.columns if c.startswith("vehicle")]

v_cols = mvc[v_cols]

top10_vehicles = v_cols.stack(dropna=False).value_counts().head(10)

#criar uma list comprehension para sacar/adicionar à lista da variável c_cols, as colunas em mvc.columns que tenham 'cause_' em todas elas:
c_cols = [c for c in mvc.columns if 'cause_' in c]

#usar um boolean mask para filtrar todas as rows no DataSet mvc que correspondam às colunas anteriormente filtradas/criadas em c_cols:
causa = mvc[c_cols]

#transformar o Dataframe 'causa' numa Series, usando o método stack() para que consigamos trabalhar com o método value_counts() q é usado apenas como um método de Series. Para percebermos qual o motivo de causa de acidente que surge com mais frequência para o imputarmos aos outros q têm NaN values:
causa_series = causa.stack(dropna=False)

causa_series.value_counts()


## 8. Filling Unknown Values with a Placeholder ##

def summarize_missing():
    v_missing_data = []

    for v in range(1,6):
        v_col = 'vehicle_{}'.format(v)
        c_col = 'cause_vehicle_{}'.format(v)

        v_missing = (mvc[v_col].isnull() & mvc[c_col].notnull()).sum()
        c_missing = (mvc[c_col].isnull() & mvc[v_col].notnull()).sum()

        v_missing_data.append([v, v_missing, c_missing])

    col_labels = columns=["vehicle_number", "vehicle_missing", "cause_missing"]
    return pd.DataFrame(v_missing_data, columns=col_labels)

summary_before = summarize_missing()

for v in range(1,6):
    v_col = 'vehicle_{}'.format(v)
    c_col = 'cause_vehicle_{}'.format(v)
    v_missing = (mvc[v_col].isnull() & mvc[c_col].notnull())
    c_missing = (mvc[c_col].isnull() & mvc[v_col].notnull())
    mvc[v_col] = mvc[v_col].mask(v_missing, 'Unspecified')
    mvc[c_col] = mvc[c_col].mask(c_missing, 'Unspecified')
    
summary_after = summarize_missing()
                     
                     
# v_missing_1 = (mvc['vehicle_1'].isnull() & mvc['cause_vehicle_1'].notnull()).value_counts()                 

## 10. Imputing Location Data ##

sup_data = pd.read_csv('supplemental_data.csv')

location_cols = ['location', 'on_street', 'off_street', 'borough']
null_before = mvc[location_cols].isnull().sum()

#c.mask(x, sup_data[c] for c in mvc[location_cols] if c.isnull() or not

for c in location_cols:
    boolean_mask = (mvc[c].isnull()) #| mvc[c].notnull())
    mvc[c] = mvc[c].mask(boolean_mask, sup_data[c])
    
null_after = mvc[location_cols].isnull().sum()
    
                
    
    