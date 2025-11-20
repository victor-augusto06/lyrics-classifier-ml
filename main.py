import os
from lyrics_component import LyricsClassifier

ARQUIVO_PADRAO = "dataset_genero_musical.xlsx"
ARQUIVO_MODELO = "ai_component_cla_lyrics.joblib"

def main():
    print("\n=== Configuração do Dataset ===")
    print(f"Pressione ENTER para usar o padrão ('{ARQUIVO_PADRAO}')")
    print("Ou digite o nome do arquivo .xlsx que está na raiz:")
    
    entrada_usuario = input(">> ").strip()
    
    arquivo_dados = entrada_usuario if entrada_usuario else ARQUIVO_PADRAO

    if not os.path.exists(arquivo_dados):
        print(f"\n[ERRO CRÍTICO] O arquivo '{arquivo_dados}' não foi encontrado!")
        print("Certifique-se de que o arquivo está na mesma pasta do script.")
        return

    print(f"\nDataset selecionado: {arquivo_dados}")
    
    classificador = LyricsClassifier()
    
    try:
        print("\n--- TREINAMENTO 1: BASELINE (NAIVE BAYES) ---")
        acc_bayes = classificador.treinar(arquivo_dados, modelo='bayes')
        
        classificador.exportar_modelo(ARQUIVO_MODELO)

        print("\n--- TREINAMENTO 2: ALTERNATIVA (SVM) ---")
        acc_svm = classificador.treinar(arquivo_dados, modelo='svm')

        print(f"\nComparativo: Bayes ({acc_bayes:.2f}) vs SVM ({acc_svm:.2f})")

    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")
        return

    print("\n--- Recarregando Modelo Exportado (Bayes) ---")
    classificador.carregar_modelo(ARQUIVO_MODELO)

    print("\n--- Teste de Predição ---")
    letra_teste = """
    Eu sei que vou te amar
    Por toda a minha vida eu vou te amar
    """
    
    genero, probabilidades = classificador.prever(letra_teste)
    print(f"Letra: {letra_teste.strip().splitlines()[0]}...")
    print(f"Gênero Previsto: {genero}")
    print(f"Probabilidades: {probabilidades}")

if __name__ == "__main__":
    main()