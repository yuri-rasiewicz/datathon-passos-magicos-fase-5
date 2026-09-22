"""Passos Mágicos — análise educacional e triagem inicial.

Executar na pasta do projeto: python -m streamlit run app.py
O notebook é a fonte da preparação, das análises e da avaliação do modelo.
"""
from pathlib import Path
import hashlib
import base64
import html
import json

import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE, RED = '#1265AB', '#ED8524', '#D85245'
PALETTE = [BLUE, ORANGE, RED, '#6289BB', '#8269AC', '#449991', '#8192A5']
YEARS = [2022, 2023, 2024]
STONES = ['Quartzo', 'Ágata', 'Ametista', 'Topázio', 'Sem registro']
CLASSES = ['Em fase', 'Moderada', 'Severa']
LABELS = {'ian':'Adequação do nível', 'ida':'Desempenho acadêmico',
          'ieg':'Engajamento', 'iaa':'Autoavaliação', 'ips':'Psicossocial',
          'ipp':'Psicopedagógico', 'ipv':'Ponto de virada', 'inde':'Índice global'}
WEIGHTS = {'ian':.1, 'ida':.2, 'ieg':.2, 'iaa':.1, 'ips':.1, 'ipp':.1, 'ipv':.2}


@st.cache_data
def load_data():
    """Usar somente campos necessários ao painel; sem nomes ou dados de contato."""
    cols = ['ra', 'ano', 'fase', 'pedra', 'defasagem', 'classe_defasagem', *LABELS]
    data = pd.read_csv(ROOT / 'data/alunos_ano.csv', usecols=cols)
    if data.duplicated(['ra', 'ano']).any():
        raise ValueError('Há observações anuais duplicadas. Reexecute o notebook.')
    data['pedra'] = data.pedra.fillna('Sem registro')
    return data


@st.cache_resource
def load_model():
    """Carregar somente o artefato local confiável e verificar sua integridade."""
    meta = json.loads((ROOT / 'models/metadados_modelo.json').read_text())
    path = ROOT / 'models/modelo_risco.pkl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256_modelo']:
        raise ValueError('O modelo não corresponde aos metadados do projeto.')
    import sklearn
    if sklearn.__version__ != meta['versoes']['scikit_learn']:
        raise ValueError('Instale as versões do requirements.txt para carregar o modelo.')
    with path.open('rb') as arquivo:
        model = pickle.load(arquivo)
    if list(model.feature_names_in_) != meta['features']:
        raise ValueError('As entradas do modelo não correspondem à documentação.')
    return model, meta


def number(value, digits=0):
    if pd.isna(value):
        return '—'
    return f'{value:,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def kpi(label, value, note):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
                f'<div class="kpi-value">{html.escape(value)}</div>'
                f'<div class="kpi-note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{html.escape(text)}</div>', unsafe_allow_html=True)


def chart(fig, key=None, height=330):
    fig.update_layout(height=height, paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', color='#344F6B', size=12),
        margin=dict(l=25, r=20, t=35, b=30),
        legend=dict(orientation='h', y=-.18, title=None),
        colorway=PALETTE, hoverlabel=dict(bgcolor='white'),
        separators=',.')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#EEF2F7', zeroline=False)
    st.plotly_chart(fig, use_container_width=True, key=key, config={'displayModeBar':False})
    captions = {'adequacy': 'Cada barra conta os alunos em fase ou com defasagem moderada ou severa no ano selecionado.', 'phase_performance': 'Cada barra mostra a média de desempenho dos alunos daquela fase. Notas ausentes não entram na média.', 'indicator_trend': 'Cada ponto é a média do indicador naquele ano. A linha permite comparar 2022, 2023 e 2024 com os filtros escolhidos.', 'psycho_groups': 'A linha dentro da caixa é a mediana. A caixa reúne os 50% centrais das notas. Pontos além das hastes são valores atípicos pelo critério do gráfico, não necessariamente erros.', 'ips_drop': 'Cada barra mostra a porcentagem de alunos de uma faixa de IPS que teve queda maior que 0,1 ponto no ano seguinte.', 'turning_signals': 'À direita de zero, notas anteriores maiores acompanham maior mudança de IPV; à esquerda, menor mudança. Perto de zero há pouca relação.', 'inde_profiles': 'Cada barra mostra a média de INDE de um perfil. Os quatro indicadores comparados são IDA, IEG, IPS e IPP.'}
    if key in captions: st.caption(captions[key])
    elif key and (key.startswith('engagement_') or key.startswith('self_')):
        st.caption('Cada ponto representa um aluno. À direita, maior nota no eixo horizontal; mais acima, maior nota no eixo vertical.')


def school_panel(data, cohort, phases, stones):
    """Painel fixo usa fase/pedra de 2022 para preservar as mesmas pessoas."""
    school = data[data.fase.between(0, 7)].copy()
    if cohort == 'Mesmos alunos nos três anos':
        ids = school.groupby('ra').ano.nunique().loc[lambda s:s.eq(3)].index
        start = school[school.ra.isin(ids) & school.ano.eq(2022)]
        ids = start[start.fase.isin(phases) & start.pedra.isin(stones)].ra
        return school[school.ra.isin(ids)].copy()
    return school[school.fase.isin(phases) & school.pedra.isin(stones)].copy()


def correlation(data, x, y):
    pairs = data[[x, y]].dropna()
    if len(pairs) < 3 or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
        return np.nan, len(pairs)
    return pairs[x].corr(pairs[y], method='spearman'), len(pairs)


def evaluate_student(model, meta, status, phase, values, confirmed, allow_outside):
    """Validar elegibilidade antes de estimar; sem imputar entradas ausentes."""
    if status != 'Sem defasagem atual':
        return {'error':'A previsão é para novos casos. Confirme ausência de defasagem atual antes de usar o modelo.'}
    if phase is None or phase not in range(7):
        return {'error':'Este modelo foi desenvolvido apenas para alunos nas fases 0 a 6.'}
    if not confirmed:
        return {'error':'Confirme que os três indicadores são conhecidos e pertencem à mesma avaliação anual.'}
    if any(v is None or not np.isfinite(v) or v < 0 or v > 10 for v in values.values()):
        return {'error':'Preencha IDA, IEG e IPV com números de 0 a 10. Não substitua informação ausente por zero.'}
    outside = [name.upper() for name, value in values.items()
               if not meta['faixas_treino'][name]['min'] <= value <= meta['faixas_treino'][name]['max']]
    if outside and not allow_outside:
        return {'error':f"{', '.join(outside)} fora da faixa observada no treino. Confira os valores; para uma simulação, reconheça a limitação de extrapolação abaixo."}
    row = pd.DataFrame([{**values, 'fase':phase}])[meta['features']]
    positive_index = list(model.classes_).index(meta['classe_positiva'])
    probability = float(model.predict_proba(row)[0, positive_index])
    return {'probability':probability, 'alert':probability >= meta['limiar'],
            'outside':outside, 'values':{**values, 'fase':phase}}


def navigate(page):
    st.session_state['nav'] = page


def parse_scores(raw):
    """Aceitar vírgula ou ponto decimal, sem formatar durante a digitação."""
    import re
    values = {}
    for key in ['ida', 'ieg', 'ipv']:
        text = str(raw.get(key, '')).strip()
        if not text:
            raise ValueError(f'Preencha {key.upper()}. Informação ausente não deve virar zero.')
        if not re.fullmatch(r'(?:\d+(?:[.,]\d+)?|[.,]\d+)', text):
            raise ValueError(f'{key.upper()}: digite um número de 0 a 10, como 7,5 ou 7.5, sem separador de milhar.')
        value = float(text.replace(',', '.'))
        if not np.isfinite(value) or not 0 <= value <= 10:
            raise ValueError(f'{key.upper()}: o valor deve estar entre 0 e 10.')
        values[key] = value
    return values


def annual_summary(data):
    return data.groupby('ano').agg(
        alunos=('ra', 'size'), defasados=('defasagem', lambda s: s.lt(0).sum()),
        ida_n=('ida', 'count'), ida_media=('ida', 'mean')).reset_index().assign(
            percentual=lambda d: 100 * d.defasados / d.alunos)


def context_page(data):
    st.markdown('<div class="hero"><div class="eyebrow">Contexto do projeto · FIAP Pós-Tech</div>'
        '<h1>Educação que transforma.<br>Dados para acompanhar.</h1>'
        '<p>Uma análise das trajetórias dos alunos da Passos Mágicos, com uma ferramenta '
        'para apoiar a identificação antecipada de novos casos de defasagem.</p></div>', unsafe_allow_html=True)
    st.write('A Passos Mágicos atua na formação de crianças e jovens em vulnerabilidade social. '
             'Neste projeto, usamos os dados PEDE para entender a evolução educacional e apoiar o acompanhamento dos alunos.')
    school = data[data.fase.between(0, 7)]
    current = school[school.ano.eq(2024)]
    for col, args in zip(st.columns(3), [
        ('Período analisado', '2022–2024', 'Três avaliações anuais'),
        ('Alunos analisados em 2024', number(len(current)), 'Recorte escolar · fases 0–7'),
        ('Objetivo da previsão', 'Próximo ano', 'Antecipar novos casos de defasagem')]):
        with col: kpi(*args)
    left, right = st.columns([1.55, 1])
    rows = annual_summary(school)
    with left:
        st.subheader('A defasagem diminuiu no período observado')
        fig = px.bar(rows, x='ano', y='percentual', text=rows.percentual.map(lambda v: number(v, 1)+'%'),
            custom_data=['defasados', 'alunos'], color_discrete_sequence=[BLUE],
            labels={'ano': 'Ano', 'percentual': 'Alunos com defasagem (%)'})
        fig.update_traces(width=.5, textposition='outside',
            hovertemplate='%{x}<br>%{customdata[0]} de %{customdata[1]} alunos<br>%{y:.1f}%<extra></extra>')
        fig.update_xaxes(type='category'); fig.update_yaxes(range=[0, 85]); chart(fig, key='context_trend')
        st.caption('Fases 0–7 em cada ano. A composição do grupo pode variar.')
    with right:
        st.subheader('O que queremos descobrir?')
        st.write('Como evoluem a adequação de nível e o desempenho? Quais indicadores ajudam a compreender '
                 'as necessidades dos alunos? Como antecipar novos casos de defasagem?')
        insight(f'Em 2024, {int(current.defasagem.lt(0).sum())} dos {number(len(current))} alunos do recorte '
                'ainda apresentavam defasagem. A melhora observada convive com a necessidade de acompanhamento.')
        st.button('Abrir Dashboard Analítico', key='open_dashboard', width='stretch',
                  on_click=navigate, args=('Dashboard Analítico',))
    with st.expander('De onde vêm os números?'):
        st.write('Fonte: planilha PEDE fornecida pela FIAP. Em 2024, há 1.156 alunos na base: '
                 '1.054 nas fases 0–7 e 102 nas fases 8–9. O dashboard concentra a comparação escolar nas fases 0–7; '
                 'os outros registros permanecem na base e são contextualizados no notebook.')
        st.write('Defasagem = fase efetiva − fase ideal. Um resultado negativo indica defasagem. '
                 'O percentual divide os alunos com resultado negativo pelo total do recorte. '
                 'Fase é um nível do programa, não automaticamente um ano escolar.')
        st.dataframe(rows[['ano', 'alunos', 'defasados', 'percentual']].rename(columns={
            'ano': 'Ano', 'alunos': 'Alunos', 'defasados': 'Com defasagem', 'percentual': 'Percentual (%)'}).round(1), hide_index=True)
        st.caption('A evolução é descritiva: sem grupo de comparação, não isolamos o efeito causal do programa. '
                   'O dashboard também permite acompanhar os mesmos alunos.')
    st.caption('Projeto acadêmico independente. Não é um sistema oficial da Associação Passos Mágicos.')


def relationship_plot(data, x, y, title, key):
    """Uma pergunta por gráfico; a associação técnica fica em segundo plano."""
    valid = data[[x, y]].dropna()
    st.subheader(title)
    if len(valid) < 3:
        st.info('Há menos de três avaliações completas neste recorte. Amplie os filtros.'); return
    fig = px.scatter(valid, x=x, y=y, opacity=.45, color_discrete_sequence=[BLUE],
        labels={x: LABELS[x]+' · '+x.upper(), y: LABELS[y]+' · '+y.upper()})
    fig.update_traces(marker=dict(size=7)); chart(fig, key=key, height=310)
    r, n = correlation(valid, x, y)
    if pd.isna(r):
        meaning = 'Não há variação suficiente para resumir a relação.'
    elif abs(r) < .2:
        meaning = 'A relação entre os indicadores é fraca neste grupo.'
    elif r > 0:
        meaning = 'Notas maiores em um indicador tendem a acompanhar notas maiores no outro.'
    else:
        meaning = 'Notas maiores em um indicador tendem a acompanhar notas menores no outro.'
    st.write(meaning)
    with st.expander('Detalhes da associação'):
        st.write(f'Cada ponto representa um aluno com as duas avaliações disponíveis. São {n} pares. '
                 f'Correlação de Spearman: {number(r, 2)}. De −1 a +1; perto de zero, pouca associação. '
                 'Na interface, valores absolutos abaixo de 0,2 são descritos como fracos.')
        st.caption('Associação no mesmo ano não demonstra causa nem antecipação de risco.')


def progress_page(panel, now, year, cohort):
    st.subheader(f'Situação no ano selecionado · {year}')
    left, right = st.columns(2)
    with left:
        st.write('**Quantos alunos estão abaixo da fase esperada?**')
        counts = now.classe_defasagem.value_counts().reindex(CLASSES, fill_value=0)
        fig = go.Figure(go.Bar(x=CLASSES, y=counts.values, text=counts.values,
            textposition='outside', marker_color=[BLUE, ORANGE, RED]))
        fig.update_yaxes(title='Alunos', range=[0, max(1, counts.max()*1.2)])
        chart(fig, key='adequacy')
    with right:
        st.write('**Como está o desempenho em cada fase?**')
        by_phase = now.groupby('fase').ida.agg(media='mean', n='count').reset_index().dropna(subset=['media'])
        fig = px.bar(by_phase, x='fase', y='media', text=by_phase.media.round(2), custom_data=['n'],
            labels={'fase': 'Fase', 'media': 'IDA médio'}, color_discrete_sequence=[BLUE])
        fig.update_xaxes(type='category'); fig.update_yaxes(range=[0, 10.5])
        fig.update_traces(hovertemplate='Fase %{x}<br>IDA: %{y:.2f}<br>%{customdata[0]} avaliações<extra></extra>')
        chart(fig, key='phase_performance')
    st.divider()
    st.subheader('Evolução entre anos · 2022–2024')
    st.caption('Esta parte compara os três anos, independentemente do ano selecionado acima. Mantém os filtros de população, fase e pedra.')
    indicator = st.selectbox('Qual indicador acompanhar?', ['ida', 'ieg', 'ipv', 'inde'],
        format_func=lambda v: LABELS[v]+' · '+v.upper(), key='trend')
    trend = panel.groupby('ano')[indicator].agg(media='mean', n='count').reset_index()
    fig = px.line(trend, x='ano', y='media', markers=True, custom_data=['n'],
        labels={'ano': 'Ano', 'media': indicator.upper()+' médio'}, color_discrete_sequence=[BLUE])
    fig.update_xaxes(tickvals=YEARS); fig.update_yaxes(range=[0, 10])
    fig.update_traces(hovertemplate='%{x}<br>Média: %{y:.2f}<br>%{customdata[0]} avaliações<extra></extra>')
    chart(fig, key='indicator_trend', height=300)
    values = '; '.join(f'{int(r.ano)}: {number(r.media, 2)} (n={int(r.n)})' for r in trend.itertuples())
    st.write(f'**Médias observadas:** {values}. Use a evolução para localizar anos e fases que merecem atenção.')
    if cohort == 'Mesmos alunos nos três anos':
        complete = panel.pivot(index='ra', columns='ano', values=indicator).reindex(columns=YEARS).dropna()
        if len(complete):
            delta = (complete[2024]-complete[2022]).mean()
            st.write(f'Entre os mesmos {len(complete)} alunos com esse indicador nos três anos, '
                     f'a mudança média de 2022 a 2024 foi {number(delta, 2)} ponto(s).')
    with st.expander('Critérios e limites da comparação'):
        st.write('Em fase: diferença de fases ≥0; defasagem moderada: de −2 até abaixo de 0; severa: abaixo de −2. '
                 'Cada média utiliza apenas as avaliações preenchidas. Fases, avaliações e a composição dos grupos podem mudar. '
                 'Nem uma melhora no painel fixo, isoladamente, prova efeito causal do programa.')


def relations_page(now):
    question = st.selectbox('Qual pergunta explorar?', [
        'Engajamento acompanha desempenho e ponto de virada?',
        'Autoavaliação acompanha desempenho e engajamento?',
        'Avaliação psicopedagógica acompanha a defasagem?'], key='relation_question')
    if question.startswith('Engajamento'):
        for col, target, title in zip(st.columns(2), ['ida', 'ipv'], ['Engajamento e desempenho', 'Engajamento e ponto de virada']):
            with col: relationship_plot(now, 'ieg', target, title, 'engagement_'+target)
        insight('Acompanhe engajamento e desempenho em conjunto. A associação no mesmo ano não garante previsão de risco futuro.')
    elif question.startswith('Autoavaliação'):
        if now.iaa.eq(0).any():
            st.caption(f'Há {int(now.iaa.eq(0).sum())} registros de autoavaliação com zero, de interpretação incerta.')
            if st.checkbox('Comparar sem os zeros de autoavaliação', key='iaa_zero'):
                now = now[now.iaa.gt(0)]
        for col, target, title in zip(st.columns(2), ['ida', 'ieg'], ['Percepção e desempenho', 'Percepção e engajamento']):
            with col: relationship_plot(now, 'iaa', target, title, 'self_'+target)
        insight('Diferenças entre autoavaliação e outras notas podem orientar uma conversa com o aluno. As avaliações medem aspectos diferentes.')
    else:
        valid = now.dropna(subset=['ipp', 'ian', 'classe_defasagem'])
        if valid.empty:
            st.info('Não há avaliações psicopedagógicas disponíveis neste recorte. IPP não foi fornecido em 2022.'); return
        summary = valid.groupby('classe_defasagem').ipp.agg(mediana='median', n='count').reindex(CLASSES).dropna().reset_index()
        st.subheader('A avaliação psicopedagógica muda entre os grupos?')
        fig = go.Figure()
        for group, color in zip(CLASSES, [BLUE, ORANGE, RED]):
            scores = valid.loc[valid.classe_defasagem.eq(group), 'ipp']
            if scores.empty: continue
            fig.add_trace(go.Box(y=scores, name=f'{group} (n={len(scores)})',
                boxpoints='all' if len(scores) < 5 else 'outliers',
                quartilemethod='linear', marker_color=color, line_color=color,
                marker=dict(size=5, opacity=.7), jitter=.12, pointpos=0))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title='Avaliação psicopedagógica · IPP',
            range=[min(0, float(valid.ipp.min())-.3), max(10.5, float(valid.ipp.max())+.3)])
        chart(fig, key='psycho_groups', height=370)
        if summary.n.lt(5).any():
            st.caption('Nos grupos com menos de cinco alunos, mostramos todas as notas. Uma amostra tão pequena não permite caracterizar bem a distribuição.')
        insight('IPP e adequação se complementam. Uma avaliação psicopedagógica alta não descarta defasagem.')
        with st.expander('Detalhes desta comparação'):
            r, n = correlation(valid, 'ipp', 'ian')
            st.write(f'{n} avaliações completas; associação IPP–IAN de {number(r, 2)}. '
                     'As categorias de defasagem correspondem aos níveis de IAN. '
                     'As hastes usam o limite de 1,5 vez a distância entre os quartis; os pontos além delas são destacados, sem exclusão dos dados.')


def followup_pairs(data, panel, origin):
    start = panel[panel.ano.eq(origin)]
    future = data[data.ano.eq(origin+1) & data.fase.between(0, 7)]
    return start, start.merge(future, on='ra', suffixes=('_inicio', '_fim'), validate='one_to_one')


def signals_page(data, panel, origin):
    st.subheader(f'O que os indicadores anteriores mostram? · {origin} → {origin+1}')
    st.write('Comparamos a avaliação de um ano com a mudança do mesmo aluno no ano seguinte.')
    start, pairs = followup_pairs(data, panel, origin)
    st.caption(f'{len(start)} alunos no recorte inicial; {len(pairs)} com registro escolar no ano seguinte. '
               'Só entram alunos com acompanhamento disponível.')
    question = st.selectbox('Qual pergunta explorar?', ['IPS antecede queda no desempenho?',
        'Quais indicadores se relacionam com a mudança no ponto de virada?'], key='signals_question')
    if question.startswith('IPS'):
        outcome = st.radio('Acompanhar a mudança em', ['Desempenho · IDA', 'Engajamento · IEG'], horizontal=True, key='drop_target')
        target = 'ida' if outcome.startswith('Desempenho') else 'ieg'
        pairs['mudanca'] = pairs[target+'_fim']-pairs[target+'_inicio']
        valid = pairs[['ips_inicio', 'mudanca']].dropna().copy()
        if len(valid) < 4 or valid.ips_inicio.nunique() < 2:
            st.info('Não há avaliações suficientes ou variação de IPS para formar grupos. Amplie os filtros.'); return
        valid['grupo'] = pd.qcut(valid.ips_inicio, 4, duplicates='drop')
        valid['queda'] = valid.mudanca.lt(-.1-1e-9)
        summary = valid.groupby('grupo', observed=True).agg(n=('queda', 'size'), percentual=('queda', 'mean'),
            minimo=('ips_inicio', 'min'), maximo=('ips_inicio', 'max')).reset_index()
        if summary.empty:
            st.info('Os valores de IPS são muito concentrados para formar grupos. Amplie os filtros.'); return
        summary['percentual'] *= 100
        summary['faixa'] = [f'{number(r.minimo, 2)} a {number(r.maximo, 2)}' for r in summary.itertuples()]
        fig = px.bar(summary, x='faixa', y='percentual', text=summary.percentual.map(lambda v: number(v, 1)+'%'),
            custom_data=['n'], color_discrete_sequence=[BLUE], labels={'faixa': f'Faixa de IPS em {origin}', 'percentual': 'Alunos com queda (%)'})
        fig.update_yaxes(range=[0, 105]); fig.update_traces(hovertemplate='IPS %{x}<br>Queda: %{y:.1f}%<br>n=%{customdata[0]}<extra></extra>')
        chart(fig, key='ips_drop')
        r, n = correlation(valid, 'ips_inicio', 'mudanca')
        message = ('A relação entre IPS anterior e mudança posterior é fraca neste grupo.'
                   if pd.isna(r) or abs(r) < .2 else 'Há associação entre IPS anterior e mudança posterior neste grupo.')
        insight(f'{int(valid.queda.sum())} de {n} alunos tiveram queda superior a 0,1 ponto. {message} '
                'O IPS isolado não comprova risco futuro.')
        with st.expander('Como os grupos e a queda foram definidos'):
            st.write('As faixas dividem os valores de IPS em até quatro grupos. Empates podem reduzir esse número. '
                     'Queda significa redução maior que 0,1 ponto; o notebook também confere qualquer redução. As faixas não são categorias clínicas.')
            st.write(f'Associação de Spearman entre IPS anterior e mudança: {number(r, 2)}. '
                     'A equivalência da avaliação de IPS entre os anos não está estabelecida.')
    else:
        pairs['mudanca_ipv'] = pairs.ipv_fim-pairs.ipv_inicio
        rows = []
        for field in ['ida', 'ieg', 'ips', 'iaa', 'ipp']:
            r, n = correlation(pairs, field+'_inicio', 'mudanca_ipv')
            if pd.notna(r): rows.append({'Indicador anterior': field.upper(), 'Associação': r, 'Alunos': n})
        if not rows:
            st.info('Não há pares completos suficientes neste recorte. Amplie os filtros.'); return
        summary = pd.DataFrame(rows)
        fig = px.bar(summary, x='Associação', y='Indicador anterior', orientation='h', custom_data=['Alunos'],
            text=summary['Associação'].round(2), color_discrete_sequence=[BLUE])
        fig.update_xaxes(range=[-1, 1]); fig.add_vline(x=0, line_color='#8192A5')
        fig.update_traces(hovertemplate='%{y}<br>Associação: %{x:.2f}<br>n=%{customdata[0]}<extra></extra>')
        chart(fig, key='turning_signals')
        st.write('Essas associações descrevem o acompanhamento entre anos. Não são os pesos do modelo preditivo.')
        weak = summary['Associação'].abs().max() < .2
        insight(('As relações são fracas neste grupo. ' if weak else
                 'As relações variam entre os indicadores. ') +
                'Use os sinais para investigar com a equipe; eles não demonstram causas.')
        st.caption('Cada indicador usa seus pares completos. IPP não permite a transição 2022→2023 porque está ausente em 2022.')


def inde_page(now):
    st.subheader('O que é o INDE?')
    st.write('É a nota global que combina sete indicadores. **IDA, IEG e IPV têm peso de 20% cada**; '
             'IAN, IAA, IPS e IPP, de 10% cada. Uma nota global alta pode esconder dificuldade em uma dimensão.')
    with st.expander('Ver os componentes e seus pesos'):
        st.dataframe(pd.DataFrame([{'Indicador':k.upper()+' · '+LABELS[k], 'Peso (%)':int(v*100)}
            for k,v in WEIGHTS.items()]),hide_index=True)
        st.caption('Estes são os pesos da fórmula do INDE; não são os pesos do modelo preditivo.')
    complete = now.dropna(subset=list(WEIGHTS)+['inde'])
    if complete.empty:
        st.info('Não há registros completos para comparar os componentes neste recorte. '
                'IPP não está disponível em 2022; não reconstruímos avaliações ausentes.'); return
    fields = ['ida','ieg','ips','ipp']
    medians = now[fields].median()
    st.subheader('Como formamos os dois grupos?')
    st.write('Comparamos **desempenho (IDA), engajamento (IEG), aspectos psicossociais (IPS) e psicopedagógicos (IPP)**. '
             'Chamamos de “alto” o valor **igual ou superior à mediana** de cada indicador: a nota central ao ordenar as avaliações disponíveis. '
             'É uma comparação com o grupo selecionado, não uma nota mínima de aprovação.')
    st.dataframe(pd.DataFrame({'Indicador':[f'{k.upper()} · {LABELS[k]}' for k in fields],
        'Alto a partir de (mediana)':[number(medians[k],4) for k in fields]}),hide_index=True)
    st.caption('As referências mudam com ano, população, fase e pedra. Os cálculos usam os valores completos, sem arredondamento.')
    st.write('**Quatro indicadores altos:** o aluno alcança as quatro referências ao mesmo tempo. '
             '**Demais combinações:** fica abaixo da referência em pelo menos um indicador, mesmo que os outros três sejam altos.')
    high = complete[fields].ge(medians).all(axis=1)
    groups = complete.assign(perfil=np.where(high,'Quatro indicadores altos','Demais combinações'))
    summary = groups.groupby('perfil').inde.agg(media='mean',n='count').reset_index()
    st.subheader('Qual grupo apresenta maior INDE médio?')
    fig = px.bar(summary,x='perfil',y='media',text=summary.media.map(lambda v:number(v,2)),custom_data=['n'],
        labels={'perfil':'Grupo de alunos','media':'INDE médio'},color_discrete_sequence=[BLUE])
    fig.update_yaxes(range=[0,10.5])
    fig.update_traces(hovertemplate='%{x}<br>INDE médio: %{y:.2f}<br>%{customdata[0]} alunos<extra></extra>')
    chart(fig,key='inde_profiles')
    st.caption(f'A comparação usa {len(complete)} alunos com INDE e seus sete componentes preenchidos. '
               f'{len(now)-len(complete)} registros do recorte ficaram fora por falta de alguma dessas avaliações.')
    description = '; '.join(f'{r.perfil}: INDE {number(r.media,2)} em {int(r.n)} alunos' for r in summary.itertuples())
    insight(description+'. A altura compara as médias dos grupos, não a nota de cada aluno.')
    st.write('**O que concluir:** esses perfis ajudam a olhar o conjunto das avaliações. Parte da diferença é esperada, '
             'pois os quatro indicadores já entram na fórmula do INDE. IAN, IAA e IPV também variam entre os grupos; '
             'o gráfico não mede o efeito isolado de melhorar um indicador.')
    if summary.n.lt(20).any(): st.caption('Há grupo com menos de 20 alunos; sua média pode ser instável.')
    with st.expander('Conferência da fórmula'):
        calculated = sum(complete[k]*v for k,v in WEIGHTS.items())
        st.write(f'INDE médio informado: {number(complete.inde.mean(),2)}. '
                 f'Média calculada pelos componentes: {number(calculated.mean(),2)}.')


def dashboard(data):
    st.markdown('<div class="eyebrow">Dashboard analítico</div>', unsafe_allow_html=True)
    st.title('Uma pergunta de cada vez')
    topic = st.selectbox('O que você quer entender?', ['Defasagem e desempenho', 'Engajamento e percepções',
        'Sinais para acompanhamento', 'Entendendo o INDE'], key='topic')
    is_transition = topic == 'Sinais para acompanhamento'
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        if is_transition:
            year = c1.selectbox('Comparar a passagem entre', [2022, 2023], format_func=lambda y: f'{y} → {y+1}', key='origin')
        else:
            year = c1.selectbox('Ano selecionado', [2024, 2023, 2022], key='year')
        cohort = c2.selectbox('Quem acompanhar?', ['População escolar de cada ano', 'Mesmos alunos nos três anos'], key='cohort')
        anchor = ' em 2022' if cohort == 'Mesmos alunos nos três anos' else (' no ano inicial' if is_transition else ' em cada ano')
        c1, c2 = st.columns(2)
        phases = c1.multiselect('Fases'+anchor, list(range(8)), default=list(range(8)), key='phases')
        stones = c2.multiselect('Pedras'+anchor, STONES, default=STONES, key='stones')
    if not phases or not stones:
        st.info('Selecione pelo menos uma fase e uma pedra.'); return
    panel = school_panel(data, cohort, phases, stones)
    now = panel[panel.ano.eq(year)].copy()
    if now.empty:
        st.info('Não há alunos nesse recorte. Amplie os filtros.'); return
    if cohort == 'Mesmos alunos nos três anos':
        st.caption('Grupo fixo: 441 alunos escolares antes dos filtros. Fase e pedra selecionam a situação em 2022; '
                   'seguimos essas mesmas pessoas depois. Cada indicador utiliza suas avaliações disponíveis.')
    else:
        st.caption('Recorte escolar: fases 0–7. Na comparação anual, o grupo pode mudar. '
                   'Na passagem entre anos, os filtros selecionam a situação inicial e o acompanhamento aceita mudança de fase/pedra.')
    if len(now) < 20: st.caption('Menos de 20 alunos neste recorte: interprete os resultados com cautela.')
    if is_transition:
        signals_page(data, panel, year)
    else:
        for col, args in zip(st.columns(4), [
            ('Alunos no recorte', number(len(now)), f'Fases selecionadas · {year}'),
            ('Com defasagem', number(100*now.defasagem.lt(0).mean(), 1)+'%', f'{int(now.defasagem.lt(0).sum())} de {len(now)} alunos'),
            ('Desempenho médio · IDA', number(now.ida.mean(), 2), f'{now.ida.count()} avaliações preenchidas'),
            ('Engajamento médio · IEG', number(now.ieg.mean(), 2), f'{now.ieg.count()} avaliações preenchidas')]):
            with col: kpi(*args)
        if topic == 'Defasagem e desempenho': progress_page(panel, now, year, cohort)
        elif topic == 'Engajamento e percepções': relations_page(now)
        else: inde_page(now)
    with st.expander('Dados ausentes, fases e significado dos indicadores'):
        quality = pd.DataFrame({'Indicador': [k.upper()+' · '+v for k, v in LABELS.items()],
            'Preenchidos': [int(now[k].count()) for k in LABELS], 'Ausentes': [int(now[k].isna().sum()) for k in LABELS]})
        st.dataframe(quality, hide_index=True)
        st.caption(f'Contagens da situação em {year}. As análises entre anos exigem também a avaliação posterior.')
        st.write('Erros de planilha, como #DIV/0!, são tratados como ausentes, nunca como zero. '
                 'Só deixam o cálculo que exige aquele indicador. Nos registros sem IDA, também faltam as notas componentes; '
                 'não há dados suficientes para recalcular o IDA. Os valores e os zeros originais foram preservados.')
        st.write('Fases 8–9 ficam fora do recorte escolar principal e têm grande ausência de avaliações acadêmicas. '
                 'Os registros permanecem na base. Pedras são categorias de INDE, diferentes das fases; seus cortes variam entre anos.')


def screening(model, meta):
    st.markdown('<div class="eyebrow">Sistema preditivo</div>', unsafe_allow_html=True)
    st.title('Antecipar para acompanhar')
    st.write('Estime o risco de **entrar em defasagem no próximo ano**. Esta ferramenta é para alunos '
             '**sem defasagem atual, nas fases 0–6**, com IDA, IEG e IPV da mesma avaliação anual.')
    left, right = st.columns([1.35, 1])
    with left:
        with st.form('screening'):
            st.subheader('Dados do aluno')
            phase = st.selectbox('Fase atual', list(range(7)), index=None, placeholder='Selecione a fase',
                format_func=lambda x: '0 · Alfa' if x == 0 else str(x), key='phase')
            st.caption('Digite livremente: 10, 7,5 ou 7.5. Use vírgula ou ponto como separador decimal, sem separador de milhar.')
            raw = {key: st.text_input(key.upper()+' · '+LABELS[key], value='', max_chars=20, placeholder=example, key=key)
                   for key, example in [('ida', 'Ex.: 6,5'), ('ieg', 'Ex.: 7'), ('ipv', 'Ex.: 6,800')]}
            confirmed = st.checkbox('Confirmo que o aluno está sem defasagem e os indicadores são da mesma avaliação anual.', key='confirmed')
            with st.expander('Simulação fora das faixas de treinamento'):
                allow = st.checkbox('Reconheço a limitação de extrapolação e desejo simular caso algum valor esteja fora da faixa de treino.', key='outside')
                st.caption('Use somente valores conhecidos. Informação ausente não deve ser preenchida com zero.')
            submitted = st.form_submit_button('Estimar risco', type='primary', width='stretch')
        st.caption('Não solicitamos nome ou RA. As entradas não são gravadas em arquivo pela aplicação.')
    with right:
        st.subheader('Como funciona o alerta?')
        st.write(f'O modelo calcula uma probabilidade estimada. A partir de **{number(meta["limiar"]*100, 1)}%**, '
                 'a aplicação recomenda uma revisão inicial do histórico do aluno.')
        st.write('Por exemplo: uma estimativa de 30% gera alerta; 15% não. '
                 'O corte não é a taxa de acerto do modelo.')
        st.write('**Com alerta:** conferir os registros e conversar com a equipe responsável. '
                 '**Sem alerta:** manter o acompanhamento habitual.')
        with st.expander('Faixas observadas no treinamento'):
            st.dataframe(pd.DataFrame([{'Entrada': k.upper(), 'Mínimo': v['min'], 'Máximo': v['max']}
                for k, v in meta['faixas_treino'].items()]), hide_index=True)
        st.caption('Uso acadêmico de apoio. Não é diagnóstico nem encaminhamento automático.')
    if submitted:
        try: values = parse_scores(raw)
        except ValueError as exc:
            st.error(str(exc)); return
        result = evaluate_student(model, meta, 'Sem defasagem atual' if confirmed else 'Não confirmado',
            phase, values, confirmed, allow)
        if 'error' in result:
            st.error(result['error']); return
        label = 'Alerta para acompanhamento' if result['alert'] else 'Sem alerta pelo modelo'
        css = 'result alert' if result['alert'] else 'result'
        st.markdown(f'<div class="{css}"><strong>{label}</strong><div class="prob">{number(100*result["probability"], 1)}%</div>'
            '<p>Probabilidade estimada de entrada em defasagem na próxima avaliação anual.</p></div>', unsafe_allow_html=True)
        # Exibir exatamente os valores interpretados, sem reescrever o texto digitado.
        used = ' · '.join(k.upper()+': '+format(v, '.15g').replace('.', ',') for k, v in values.items())
        st.caption(f'Valores utilizados: fase {phase} · {used}. Após editar, clique novamente em Estimar risco.')
        if result['outside']: st.warning('Fora da faixa de treinamento: '+', '.join(result['outside'])+'. Confira os dados antes de interpretar a simulação.')
        st.info('As estimativas subestimaram o risco na avaliação realizada. O percentual não é uma chance individual exata comprovada. '
                'A equipe deve considerar o histórico do aluno, inclusive quando não houver alerta.')


def about(meta):
    st.markdown('<div class="eyebrow">Sobre o modelo</div>', unsafe_allow_html=True)
    st.title('Como chegamos à previsão')
    st.write('Usamos **IDA, IEG, IPV e fase** para estimar a entrada em defasagem no próximo ano, '
             'entre alunos sem defasagem atual nas fases 0–6.')
    st.dataframe(pd.DataFrame([
        {'Etapa':'Treinamento','Ano dos indicadores':2022,'Ano do resultado':2023,'Alunos':186,'Novos casos':60},
        {'Etapa':'Avaliação posterior','Ano dos indicadores':2023,'Ano do resultado':2024,'Alunos':296,'Novos casos':84}]),hide_index=True)
    st.caption('Entram na avaliação apenas alunos com situação conhecida no ano seguinte, em fases 0–7.')

    st.subheader('Por que Regressão Logística?')
    st.write('Ela estima uma probabilidade e teve a melhor AP média entre os três modelos comparados com as mesmas entradas. '
             'A comparação usou cinco divisões do desenvolvimento, sem avaliar as alternativas em 2024.')
    table=pd.DataFrame(meta['comparacao_quatro_features']['resultados'])
    shown=table[['modelo','ap_media']].copy()
    shown['ap_media']=shown.ap_media.map(lambda v:number(100*v,1)+'%')
    st.dataframe(shown.rename(columns={'modelo':'Modelo','ap_media':'Qualidade média da priorização (AP)'}),hide_index=True)
    st.write('**AP mede como o modelo prioriza os casos ao variar o corte de alerta.** '
             'AP média é a média das cinco divisões. A Logística chegou a **63,9%**, frente a **32,3%** '
             'da referência sem discriminação, próxima de uma seleção aleatória. É um resultado favorável, '
             'quase duas vezes a referência; não é acurácia nem garantia de desempenho futuro.')

    st.subheader('Corte de alerta: 27,8%')
    st.write('Priorizamos alertas mais assertivos para concentrar a revisão pedagógica. '
             'No período avaliado, **44 de 69 alertas se confirmaram: precisão de 63,8%**.')
    metrics=meta['metricas_politica_adotada_exploratorias']
    for col,args in zip(st.columns(3),[
        ('Casos identificados','44 de 84','52,4% dos casos observados'),
        ('Falsos alertas','25','Alunos sinalizados que não entraram em defasagem'),
        ('Casos sem alerta','40','O acompanhamento deve continuar para todos')]):
        with col:kpi(*args)
    st.dataframe(pd.DataFrame([
        {'Corte':'21,7% (anterior)','Alunos sinalizados':111,'Casos identificados':58,'Falsos alertas':53,'Casos sem alerta':26},
        {'Corte':'27,8% (adotado)','Alunos sinalizados':69,'Casos identificados':44,'Falsos alertas':25,'Casos sem alerta':40}]),hide_index=True)
    insight('São 42 revisões e 28 falsos alertas a menos, ao custo de deixar 14 casos adicionais sem alerta. '
            'A escolha prioriza assertividade; a capacidade real da equipe e o custo financeiro não foram medidos.')
    st.caption('O corte é 0,278 exato. Ele altera o alerta, não a probabilidade estimada.')
    st.warning('2024 já havia sido examinado quando revisamos o modelo e escolhemos a regra. '
               'A avaliação é exploratória e precisa ser confirmada com novos dados.')

    with st.expander('O que cada indicador representa na previsão?'):
        st.write('IPV e fase têm os maiores pesos padronizados, ambos negativos. IDA também tem peso negativo; '
                 'IEG tem peso positivo neste ajuste. Isso descreve o modelo mantendo as outras entradas iguais, '
                 'não significa que engajamento cause defasagem.')
        st.write('IPS saiu por comparabilidade incerta entre anos e idade pela forte relação com fase. '
                 'INDE já combina vários indicadores. Esses campos continuam nas análises de negócio.')
    with st.expander('Resultados e cuidados na interpretação'):
        st.write('Na avaliação de **296 alunos**, o modelo acertou a classificação de **231**. '
                 'Para decidir sobre os alertas, os números mais úteis são:')
        st.dataframe(pd.DataFrame([
            {'O que avaliamos':'Quantos alertas se confirmaram?','Resultado':f"{number(metrics['precisao']*100,1)}%",'Em alunos':'44 dos 69 alertas'},
            {'O que avaliamos':'Quantos casos foram identificados?','Resultado':f"{number(metrics['recall']*100,1)}%",'Em alunos':'44 dos 84 casos'},
            {'O que avaliamos':'Quantas classificações estavam certas?','Resultado':f"{number(metrics['acuracia']*100,1)}%",'Em alunos':'231 dos 296 alunos'}]),hide_index=True)
        st.write('**O principal cuidado:** 40 alunos entraram em defasagem sem receber alerta. '
                 'Uma previsão baixa não dispensa o acompanhamento.')
        st.write('**O percentual não é exato:** a média prevista foi 21,0%, mas ocorreram casos em 28,4% dos alunos. '
                 'O modelo subestimou o risco nesse período.')
        st.write('**Ainda precisamos confirmar o resultado em novos anos.** O desenvolvimento teve 186 alunos '
                 'e só avaliamos quem tinha acompanhamento disponível. A fase 6 tem poucos exemplos; '
                 'os três indicadores precisam estar disponíveis quando a previsão for usada.')
        st.caption('O notebook reúne as métricas complementares, os parâmetros e a comparação de treino e validação.')


@st.cache_data
def load_business_charts():
    return json.loads((ROOT/'outputs/graficos_negocio.json').read_text(encoding='utf-8'))


def business_answers():
    text = (ROOT/'outputs/respostas_negocio.md').read_text(encoding='utf-8')
    sections = text.split('## Pergunta ')[1:]
    topics = ['Adequação do nível','Desempenho acadêmico','Engajamento','Autoavaliação',
              'Aspectos psicossociais','Avaliação psicopedagógica','Ponto de virada',
              'Composição do INDE','Previsão de risco','Evolução do programa']
    selected = st.selectbox('Qual pergunta de negócio consultar?',list(range(1,11)),
        format_func=lambda i:f'{i}. {topics[i-1]}',key='business_question')
    section = sections[selected-1].split('\n',1)[1].strip()
    question_and_graph, answer = section.split('**Resposta:**',1)
    st.markdown(question_and_graph)
    st.caption('Síntese da análise completa de 2022–2024. Estes gráficos não usam os filtros do Dashboard Analítico.')
    for graph in load_business_charts()[str(selected)]:
        st.image(base64.b64decode(graph['png']),use_container_width=True)
        st.caption(graph['legenda'])
    st.markdown('**Resposta:**'+answer)


def main():
    st.set_page_config(page_title='Passos Mágicos | Observatório educacional', page_icon='📘', layout='wide')
    st.markdown('<style>'+(ROOT/'style.css').read_text()+'</style>', unsafe_allow_html=True)
    pages = [('Contexto do Projeto', ':material/home:', 'context'), ('Dashboard Analítico', ':material/bar_chart:', 'dashboard'),
             ('Perguntas de Negócio', ':material/quiz:', 'questions'),
             ('Sistema Preditivo', ':material/online_prediction:', 'predict'), ('Sobre o Modelo', ':material/info:', 'about')]
    if st.session_state.get('nav') not in [p[0] for p in pages]: st.session_state['nav'] = pages[0][0]
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-dots"><i></i><i></i><i></i></div>'
            '<div class="brand-name">passos<br><span>mágicos</span></div>'
            '<div class="brand-tag">Observatório educacional</div></div>', unsafe_allow_html=True)
        st.write('**Selecione a seção**')
        for label, icon, key in pages:
            st.button(label, icon=icon, key='nav_'+key, width='stretch',
                type='primary' if st.session_state['nav'] == label else 'secondary', on_click=navigate, args=(label,))
        st.divider(); st.caption('PEDE · 2022 / 2023 / 2024\n\nFIAP Pós-Tech · Data Analytics')
        st.markdown('[Conheça a Passos Mágicos ↗](https://passosmagicos.org.br/)')
    try:
        data = load_data(); model, meta = load_model()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        st.error('Não foi possível carregar os arquivos. Confira data/ e models/ conforme o README.')
        with st.expander('Detalhes para quem está executando'): st.code(str(exc))
        st.stop()
    page = st.session_state['nav']
    if page == 'Contexto do Projeto': context_page(data)
    elif page == 'Dashboard Analítico': dashboard(data)
    elif page == 'Perguntas de Negócio':
        st.title('As dez perguntas de negócio')
        business_answers()
    elif page == 'Sistema Preditivo': screening(model, meta)
    else: about(meta)
    st.markdown('<div class="footer">PASSOS MÁGICOS · PROJETO ACADÊMICO FIAP &nbsp; | &nbsp; Dados para apoiar o acompanhamento.</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()
