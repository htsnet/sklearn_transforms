from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
#Preenche o campo indicado (coluna) com o valor desejado    
class FillWith(BaseEstimator, TransformerMixin):
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #faz a inclusão do valor recebido quando o campo for NaN
        data[self.column] = data[self.column].fillna(self.value)  # insere valor recebido quando não tem a informação
        # Retornamos um novo dataframe
        return data     
    

    
#cria uma coluna com média geral das notas
class MediaGeral(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #def AvaliaCaso(faltoso, nota_de, nota_em, nota_mf, nota_go):
        def AvaliaCaso(nota_de, nota_em, nota_mf, nota_go):            
            conta = 0
            valores = 0
            if nota_de > 0 and not pd.isna(nota_de):
                conta += 1
                valores += nota_de
            if nota_em > 0 and not pd.isna(nota_em):
                conta += 1
                valores += nota_em
            if nota_mf > 0 and not pd.isna(nota_mf):
                conta += 1
                valores += nota_mf
            if nota_go > 0 and not pd.isna(nota_go):
                conta += 1
                valores += nota_go
            if conta > 0:    
                media = valores / conta
            else:
                media = 0
            return media
        
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #faz a criação da coluna "MEDIA_GERAL" sem considerar 0 para não faltosos
        #data['MEDIA_GERAL'] = data.apply(lambda x: AvaliaCaso(x.FALTOSO, x.NOTA_DE, x.NOTA_EM, x.NOTA_MF, x.NOTA_GO), axis=1)
        data['MEDIA_GERAL'] = data.apply(lambda x: AvaliaCaso(x.NOTA_DE, x.NOTA_EM, x.NOTA_MF, x.NOTA_GO), axis=1)
        # Retornamos um novo dataframe
        return data     
    
    
#Altera nota quando está NaN ou 0 e limita a nota em 10
class NotaUnica(BaseEstimator, TransformerMixin):
    def __init__(self, qual):
        self.qual = qual
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #def AvaliaNota(faltoso, nota_media, qual, nota):
        def AvaliaNota(nota_media, qual, nota):    
            #verifica se nota é 0 ou NaN
            if pd.isna(nota) or nota == 0:
                if nota_media > 0:
                    nota = 10
                else:
                    nota = 0
            elif nota > 10:    
              nota = 10
            return nota
        
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #faz verificação se a nota é 0 ou NaN e coloca novo valor
        #data[self.qual] = data.apply(lambda x: AvaliaNota(x.FALTOSO, x.MEDIA_GERAL, self.qual, x[self.qual]), axis=1)
        data[self.qual] = data.apply(lambda x: AvaliaNota(x.MEDIA_GERAL, self.qual, x[self.qual]), axis=1)
        # Retornamos um novo dataframe
        return data       
    
    
#cria uma coluna com TRUE para os TUDO ZERO e FALSE para os demais    
class Dificuldade(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        #faz a criação da coluna "DIFICULDADE" 
        data['DIFICULDADE'] = data.apply(lambda x: True if (x['MEDIA_GERAL'] == 0) else False, axis=1)
        # Retornamos um novo dataframe
        return data     
    
    
#transforma um campo em determinado tipo
class TrocaTipo(BaseEstimator, TransformerMixin):
    def __init__(self, coluna, tipo):
        self.coluna = coluna
        self.tipo = tipo
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # faz a transformação do tipo da coluna
        data[self.coluna] = data[self.coluna].astype(self.tipo) 
        # Retornamos um novo dataframe
        return data 
    
    
#executa o SMOTE
class ExecutaSmote(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    
    def transform(self, X, y):
        #separa features do target
        print(X.head())
        print(y.head())
        
        # Preparação dos argumentos para os métodos da biblioteca ``scikit-learn``
        X_apoio, y_apoio = SMOTE().fit_sample(X, y)
        
        print(X.apoio())
        print(y.apoio())
        
        #data = pd.merge(X_apoio, y_apoio, left_index=True, right_index=True)
        
        # Retornamos um novo dataframe
        return X.apoio, y.apoio
