# Tech Challenge Fase 5 — Datathon Passos Mágicos

Projeto desenvolvido para o Tech Challenge da Fase 5 da Pós Tech em Data Analytics.

A solução analisa os dados educacionais da Passos Mágicos de 2022 a 2024 e utiliza Machine Learning para estimar a probabilidade de um aluno entrar em defasagem no ano seguinte. A aplicação em Streamlit reúne contexto do projeto, dashboard analítico, respostas às dez perguntas de negócio, sistema preditivo e informações sobre o modelo.

## Estrutura do projeto

```text
datathon-passos-magicos-fase-5/
├── app.py
├── requirements.txt
├── README.md
├── style.css
├── test_app.py
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── base_pede.xlsx
│   ├── alunos_ano.csv
│   ├── desenvolvimento.csv
│   ├── teste_temporal.csv
│   ├── dicionario.pdf
│   ├── pontos_importantes.docx
│   └── links_adicionais.docx
├── models/
│   ├── modelo_risco.pkl
│   └── metadados_modelo.json
├── notebooks/
│   └── passos_magicos.ipynb
└── outputs/
    ├── respostas_negocio.md
    └── graficos_negocio.json
```

A pasta `data/` reúne a base de origem (`base_pede.xlsx`), os três documentos de referência fornecidos e os três CSVs gerados pelo notebook. `alunos_ano.csv` contém somente os 14 campos utilizados no dashboard; a planilha de origem permanece completa.

## Principais resultados da análise

- Nas fases 0–7, a proporção de alunos com defasagem passou de **69,9% em 2022 para 50,7% em 2024**.
- Entre os mesmos 441 alunos acompanhados nos três anos, a proporção passou de **66,0% para 37,0%**.
- O desempenho acadêmico médio caiu em 2024, indicando a necessidade de acompanhamento mesmo com a melhora na adequação de nível.

Essas mudanças são observadas nos dados; sem grupo de comparação, não é possível atribuí-las exclusivamente ao programa.

## Modelo final

O modelo selecionado foi a **Regressão Logística**, com padronização das variáveis **IDA, IEG, IPV e fase**. Foram comparadas também Árvore de Decisão e Random Forest; a Regressão Logística apresentou a maior Average Precision média na validação interna.

O desenvolvimento utilizou indicadores de **2022 e desfechos de 2023**, com 186 alunos. A avaliação temporal utilizou indicadores de **2023 e desfechos de 2024**, com 296 alunos.

Com ponto de corte de **27,8%**, os resultados na avaliação temporal foram:

- Acurácia: **78,0%**
- Precisão dos alertas: **63,8%**
- Recall — casos de defasagem identificados: **52,4%**
- F1-score: **57,5%**
- ROC AUC: **0,813**
- Average Precision: **0,652**

Foram emitidos **69 alertas: 44 casos confirmados e 25 falsos alertas**. Outros **40 casos não receberam alerta**. O corte prioriza a assertividade das revisões pedagógicas, com perda de cobertura.

A avaliação é **exploratória**, pois os resultados de 2024 já haviam sido examinados antes das revisões do modelo e da escolha final do corte. Uma nova avaliação independente continua necessária.

## Como executar localmente

Na pasta do projeto, com o ambiente Python ativado:

1. Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

2. Execute a aplicação:

```bash
python -m streamlit run app.py --server.headless false
```

Para reproduzir a preparação, as análises e a modelagem, execute `notebooks/passos_magicos.ipynb` em ordem. No Colab, envie o ZIP completo quando solicitado.

## Observação importante

O arquivo `modelo_risco.pkl` foi salvo usando **scikit-learn==1.6.1**. Essa versão está fixada no `requirements.txt` para preservar a compatibilidade ao carregar o modelo.

O projeto é compatível com **Python 3.9.6** no Mac. Para o Streamlit Community Cloud, a aplicação foi testada com **Python 3.12**, utilizando as mesmas dependências.

## Limites de uso e privacidade

- O modelo atende alunos **sem defasagem atual, nas fases 0–6**, com IDA, IEG e IPV conhecidos. Valores fora das faixas de treinamento recebem aviso de extrapolação.
- As probabilidades subestimaram o risco médio observado. O resultado apoia a revisão pedagógica e não substitui o acompanhamento dos alunos.
- Os dados digitados no sistema preditivo não são gravados pela aplicação em arquivo ou banco de dados.
- O pacote contém dados individuais e deve permanecer em repositório privado até a definição das condições de divulgação. Nomes codificados não garantem anonimização completa.
- Projeto acadêmico independente, sem caráter de sistema oficial da Associação Passos Mágicos.

Fonte: base PEDE 2022–2024 e materiais fornecidos para o desafio. [Associação Passos Mágicos](https://passosmagicos.org.br/).
