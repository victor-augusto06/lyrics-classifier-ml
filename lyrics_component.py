import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

class LyricsClassifier:
    def __init__(self):
        self.pipeline = None
        try:
            self.stop_words = stopwords.words("portuguese")
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = stopwords.words("portuguese")

    def _limpar_texto(self, texto):
        if not isinstance(texto, str):
            return ""
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = texto.replace('\n', ' ')
        return texto.lower()

    def treinar(self, caminho_dataset, test_size=0.30, modelo='bayes'):
        print(f"Carregando dataset: {caminho_dataset}")
        df = pd.read_excel(caminho_dataset)
        df.dropna(inplace=True)

        print("Aplicando pré-processamento (limpeza)...")
        df['musica_tratada'] = df['musica'].apply(self._limpar_texto)

        X = df['musica_tratada']
        y = df['genero']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if modelo == 'svm':
            print("Usando modelo: Linear SVM")
            clf = LinearSVC(random_state=42, dual='auto')
        elif modelo == 'rf':
            print("Usando modelo: Random Forest")
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            print("Usando modelo: Naive Bayes (Baseline)")
            clf = MultinomialNB()

        self.pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words=self.stop_words, ngram_range=(1, 2))),
            ('clf', clf)
        ])

        print("Treinando o modelo...")
        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)
        
        print(f"\nRelatório de Classificação ({modelo}):")
        print(metrics.classification_report(y_test, preds))
        
        print(f"\nMatriz de Confusão ({modelo}):")
        print(metrics.confusion_matrix(y_test, preds))
        
        return metrics.accuracy_score(y_test, preds)

    def exportar_modelo(self, nome_arquivo):
        if self.pipeline:
            joblib.dump(self.pipeline, nome_arquivo)
            print(f"Modelo exportado com sucesso para: {nome_arquivo}")
        else:
            print("Erro: O modelo precisa ser treinado antes de exportar.")

    def carregar_modelo(self, nome_arquivo):
        self.pipeline = joblib.load(nome_arquivo)
        print(f"Modelo carregado de: {nome_arquivo}")

    def prever(self, letra_musica):
        if not self.pipeline:
            raise Exception("Modelo não carregado!")
        
        letra_limpa = self._limpar_texto(letra_musica)
        
        predicao = self.pipeline.predict([letra_limpa])[0]
        
        if hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            probs = self.pipeline.predict_proba([letra_limpa])[0]
            classes = self.pipeline.classes_
            probs_dict = dict(zip(classes, probs))
        else:
            probs_dict = {"info": "Probabilidade não disponível para este modelo"}
        
        return predicao, probs_dict