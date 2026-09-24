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

## Observação importante

O arquivo `modelo_risco.pkl` foi salvo usando **scikit-learn==1.6.1**. Essa versão está fixada no `requirements.txt` para preservar a compatibilidade ao carregar o modelo.

O projeto é compatível com **Python 3.9.6** no Mac. Para o Streamlit Community Cloud, a aplicação foi testada com **Python 3.12**, utilizando as mesmas dependências.

## Limites de uso e privacidade

- O modelo atende alunos **sem defasagem atual, nas fases 0–6**, com IDA, IEG e IPV conhecidos. Valores fora das faixas de treinamento recebem aviso de extrapolação.
- As probabilidades subestimaram o risco médio observado. O resultado apoia a revisão pedagógica e não substitui o acompanhamento dos alunos.
- Os dados digitados no sistema preditivo não são gravados pela aplicação em arquivo ou banco de dados.
- A base fornecida passou por anonimização e sua publicação foi autorizada.
- Projeto acadêmico independente, sem caráter de sistema oficial da Associação Passos Mágicos.

Fonte: base PEDE 2022–2024 e materiais fornecidos para o desafio. [Associação Passos Mágicos](https://passosmagicos.org.br/).
