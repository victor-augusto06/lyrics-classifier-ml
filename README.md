# Classificador de Gênero Musical por Letras (Lyrics Classifier)

Este projeto implementa um pipeline de Machine Learning para classificar gêneros musicais baseados em letras de músicas. O componente foi desenvolvido para ser reutilizável, permitindo treinamento, exportação e inferência.

## Funcionalidades

- **Ingestão de Dados**: Leitura de dataset em formato Excel (.xlsx).
- **Pré-processamento**: Limpeza de texto (remoção de pontuação, stopwords, normalização).
- **Modelos Suportados**:
  - Naive Bayes (Multinomial) - Baseline
  - Linear SVM (Support Vector Machine) - Alternativa
  - Random Forest - Alternativa
- **Exportação**: O modelo treinado é serializado via joblib para consumo local ou via API.

## Tecnologias Utilizadas

- Python 3.8+
- Scikit-learn (Modelagem)
- Pandas (Manipulação de dados)
- NLTK (Processamento de Linguagem Natural)
- Joblib (Serialização de objetos)

## Como Executar

### 1. Pré-requisitos
Certifique-se de ter o Python instalado. Recomenda-se o uso de um ambiente virtual (venv).

### 2. Instalação das Dependências
Execute o comando abaixo para instalar as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```
### 3. Estrutura de Arquivos
Certifique-se de que o dataset "dataset_genero_musical.xlsx" esteja na raiz do diretório junto com os scripts "main.py" e "lyrics_component.py".

### 4. Executando o Pipeline
Para treinar o modelo, validar as métricas e testar uma predição, execute:
```bash
python main.py
```
Ao rodar, o script solicitará a configuração do dataset:
- **Opção Padrão**: Pressione **ENTER** (deixe vazio) para carregar automaticamente o arquivo `dataset_genero_musical.xlsx`.
- **Opção Personalizada**: Digite o nome do seu arquivo (ex: `meus_dados.xlsx`) e pressione Enter.

## Resultados e Métricas

O script main.py executa o treinamento comparativo. O modelo padrão exportado é o Naive Bayes.

Exemplo de saída esperada no terminal:
- Relatório de Classificação: Exibe Precision, Recall e F1-Score.
- Matriz de Confusão: Exibida no console após o treino.
- Arquivo Gerado: ai_component_cla_lyrics.joblib (Artefato reutilizável).

## Consumo do Componente (Exemplo)

Para utilizar o classificador em outro script python, basta importar a classe:
```python
    from lyrics_component import LyricsClassifier

    # Instancia e carrega o modelo treinado
    clf = LyricsClassifier()
    clf.carregar_modelo("ai_component_cla_lyrics.joblib")

    # Realiza a predição
    genero, probs = clf.prever("Eu sei que vou te amar...")
    print(f"Gênero: {genero}")
    print(f"Probabilidades: {probs}")
```
## Autores
- Victor Augusto Farias Ferreira
- Felipe Alexandre Pereira
- Lucas Barroso Silvestrini
- Matheus de Farias Garcia
