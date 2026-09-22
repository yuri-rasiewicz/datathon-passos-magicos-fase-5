# Passos Mágicos

Análise dos dados PEDE de 2022–2024 e previsão de novos casos de defasagem no ano seguinte. A aplicação reúne contexto, dashboard interativo, perguntas de negócio, sistema preditivo e informações do modelo.

## Resultados

- Nas fases 0–7, a defasagem caiu de 69,9% para 50,7%; o IDA médio recuou em 2024.
- A Regressão Logística usa IDA, IEG, IPV e fase para alunos sem defasagem nas fases 0–6.
- Corte de **27,8%**: 44 casos identificados, 25 falsos alertas e 40 casos sem alerta. Precisão de **63,8%** e recall de **52,4%**.
- Avaliação exploratória em 296 alunos: o período posterior já havia sido examinado. As probabilidades subestimaram o risco médio. O alerta apoia a revisão pedagógica; não substitui o acompanhamento.

## Executar no Mac / VS Code

Compatível com **Python 3.9.6**; modelo gerado com **scikit-learn 1.6.1**. Extraia o ZIP em uma pasta nova e abra a pasta que contém `app.py`. No terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abra http://localhost:8501. Para encerrar, use `Control+C`. Nas próximas vezes, execute somente os dois comandos de ativação e abertura da aplicação.

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `data/` | Fonte, referências e três bases preparadas |
| `models/` | Pipeline `modelo_risco.pkl` e metadados |
| `notebooks/` | Um notebook: preparação, dez perguntas e modelagem |
| `outputs/` | Respostas e gráficos das dez perguntas |

`app.py` contém a aplicação; `style.css` e `.streamlit/config.toml`, a apresentação visual. Tecnologias: pandas, scikit-learn, Matplotlib, Plotly e Streamlit.

## Reproduzir a análise

Abra `notebooks/passos_magicos.ipynb` e execute em ordem. No Colab, envie o ZIP completo quando solicitado. Localmente, use as dependências acima e selecione esse ambiente no notebook. A execução gera os CSVs, o `.pkl`, os metadados e os gráficos para a aplicação. As respostas estão em `outputs/`. A aplicação já inclui esses arquivos.

Testes: `python -m unittest test_app -v`.

O pacote de trabalho contém dados individuais. `data/` está excluída do versionamento automático; a publicação dos dados ainda precisa ser definida. GitHub, deploy, apresentação e vídeo permanecem pendentes.

Fonte: PEDE 2022–2024 e documentos fornecidos para o projeto. [Passos Mágicos](https://passosmagicos.org.br/).
