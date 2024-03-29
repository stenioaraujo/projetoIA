from models import criar_classificador
from util import carregar_dados, save_teste, carregar_teste
import matplotlib as mpl
mpl.use('Agg')


def rodar_modelo():
    #Comentarios e Labels dos arquivos test_with_solutions.csv e train.csv
    comentarios, categorias = carregar_dados()
    #Carregar comentarios para teste posterior ao treinamento
    comentarios_teste = carregar_teste("testes_para_saida.csv")

    #Carrega o classificador
    classificador = criar_classificador()
    classificador.fit(comentarios, categorias)
    resultado_fuzzy = classificador.predict_proba(comentarios_teste)
    save_teste(resultado_fuzzy[:, 1], "saida.csv",
               ds="testes_para_saida.csv")

    print("Treinamento Concluido")


if __name__ == "__main__":
    rodar_modelo()
