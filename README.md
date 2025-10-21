# HW1---Inteligência-Computacional-Aplicada

## 1. Objetivos e necessidades
Compreender os fatores que influenciam no sucesso acadêmico é uma etapa essencial na construção de medidas para melhorar a educação e o desempenho dos estudantes. Neste projeto, será analisado, de forma detalhada, um conjunto de dados sintéticos chamado ”Student Habits vs Academic Performance”, advindos da plataforma Kaggle.

- Você pode acessar o dataset aqui:
[Dataset Student Habits vs Academic Performance ](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance).

O dataset contém dados sobre os hábitos de vida, de estudo e de tecnologia de 1.000 estudantes. Nosso principal objetivo é identificar padrões e correlações entre variáveis como horas de estudo, uso de redes sociais, horas de sono e saúde mental, bem como avaliar o impacto desses fatores na pontuação final dos exames escolares.

## 2. Preparação do ambiente
Precisamos instalar algumas bibliotecas para o funcionamento do nosso projeto.

**Observação:** sempre use um ambiente virtual (venv) para instalar as dependências com o objetivo de não causar conflito de versões e bibliotecas do seu computador.

Abra o terminal e faça:

### no Windows:

```bash
# Verificar se Python esta instalado
python --version
# ou
py --version

# Criar ambiente virtual
py -m venv .venv
# ou
python -m venv .venv

# Ativar o ambiente virtual
.venv\Scripts\activate

# Verificar se esta funcionando
python --version
```

### no Linux:

```bash
# Verificar Python
python3 --version

# Criar venv
python3 -m venv .venv

# Ativar venv
source .venv/bin/activate

# 4. Verificar que esta usando a venv
which python
python --version
```

Tendo feito isso aparecerá um (venv) no seu terminal.

Para a instalação das bibliotecas em python necessárias você obrigatoriamente irá precisar do pip, o gerenciador de pacotes python.

Por fim, tanto no linux quanto no windows você pode instalar as bibliotecas dessa maneira:

```bash
pip install -r requirements.txt
```
Esse comando instala todas as bibliotecas que iremos precisar.

## 3. Arquivos
Nosso projeto tem duas partes principais, os Notebooks e os códigos python. Nos Notebooks fizemos uma discussão mais técnica e detalhada, que não foi possível no artigo. Comentamos gráficos e tabelas além de aprofundar mais nas interpretações desses. Os códigos python foram feitos para desenvolver as funções que são usadas no notebook, garantindo que os aqruivos .ipynb foquem mais em comentar resultados do que na implementação do código em si, os códigos também apresentam comentários sobre sua lógica em python e como essas funções foram pensadas. Os arquivos python estão numa pasta chamada _src_ que contêm as funções de pré-processamento, da análise univariada, da bivariada e da multivariada com PCA. Os notebooks estão na pasta de mesmo nome, dentro temos 4 notebooks, um pro pré processsamento, um pra análise univariada, outro pra análise bivariada e outro pra implementação do PCA. Também existe uma pasta de dados onde estão armazenados nosso dataset bruto e nosso dataset pré processado. Por fim a última pasta importante contêm nosssa documentação usada.

## 4. Execução
Após termos instalados nossa biblioteca e estando numa venv, podemos iniciar respectivamente o fluxo do nosso trabalho:

1. Rode o arquivo 01_pre_processamento.ipynb para obter a explicação detalhada do pré-processamento.
2. Rode o arquivo 02_analise_univariada.ipynb para obter a explicação detalhada da análise univariada.
3. Rode o arquivo 03_analise_bivariada.ipynb para obter a explicação detalhada da análise bivariada.
4. Rode o arquivo 04_implementacao_pca_multi.ipynb para obter a explicação detalhada do PCA.

Além dos notebooks, implementamos nossos resultados gerais no artigo presente na pasta _docs_ , caso queira entrar em detalhes sobre as funções usadas, existem comentários nos arquivos pyton.

## 5. Método de trabalho

Decidimos fazer uma divisão parcial e não total das atividades pedidas. Dessa forma cada pessoa focou em desenvolver um modelo "esqueleto" de cada uma das 4 partes principais, ficando essa divisão:

- Pré processsamento: José Lessa
- Análise univariada: Nataniel Marques
- Análise bivariada: Matheus Rocha
- Implementação do PCA: Victor Guedes

Ao término desse esqueleto nos reunimos e compartilhamos o que foi feito em cada parte, com cada um ficando ciente da parte um do outro e nesse encontro sugerimos implementações nas partes dos outros. Além disso houve muito contato entre pessoas de partes adjacentes como uma corrigindo e falando com uma pessoa de uma etapa anterior e resumindo isso pra uma pessoa de uma etapa superior, sempre repassando o conteúdo pras não adjacentes. Também foi realizadas reuniões onde debatamos todos sobre todas as partes enquanto implementavamos tudo (principalmente o artigo) em conjunto, cada um revisando e melhorando.
