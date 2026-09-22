"""Verificações essenciais. Executar: python -m unittest test_app -v"""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from streamlit.testing.v1 import AppTest

import app


class ModelAndDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.meta = app.load_model()
        cls.data = app.load_data()

    def test_saved_policy_reproduces_temporal_results(self):
        test = pd.read_csv(app.ROOT / 'data/teste_temporal.csv')
        prob = self.model.predict_proba(test[self.meta['features']])[:,1]
        actual = confusion_matrix(test.alvo, prob >= self.meta['limiar'], labels=[0,1])
        np.testing.assert_array_equal(actual, [[187,25],[40,44]])
        self.assertEqual(self.model.named_steps['escala'].n_samples_seen_,186)
        self.assertFalse(self.meta['nova_validacao_independente'])

    def test_fixed_panel_preserves_same_students(self):
        panel = app.school_panel(self.data,'Mesmos alunos nos três anos',list(range(8)),app.STONES)
        self.assertEqual(panel.ra.nunique(),441)
        self.assertTrue(panel.groupby('ra').ano.nunique().eq(3).all())
        filtered = app.school_panel(self.data,'Mesmos alunos nos três anos',[0,1],app.STONES)
        self.assertTrue(filtered.groupby('ra').ano.nunique().eq(3).all())
        self.assertTrue(filtered[filtered.ano.eq(2022)].fase.isin([0,1]).all())

    def test_missing_and_ineligible_inputs_are_blocked(self):
        valid = dict(ida=6.5,ieg=7.0,ipv=6.8)
        base = [self.model,self.meta,'Sem defasagem atual',2,valid,True,False]
        for pos, value in [(2,'Já apresenta defasagem'),(2,'Não sei informar'),(3,7),(3,None),(5,False)]:
            args=base.copy();args[pos]=value
            self.assertIn('error',app.evaluate_student(*args))
        for value in [None,np.nan,-1,11]:
            args=base.copy();args[4]={**valid,'ida':value}
            self.assertIn('error',app.evaluate_student(*args))

    def test_prediction_and_extrapolation(self):
        values=dict(ida=6.5,ieg=7.0,ipv=6.8)
        result=app.evaluate_student(self.model,self.meta,'Sem defasagem atual',2,values,True,False)
        expected=self.model.predict_proba(pd.DataFrame([{**values,'fase':2}])[self.meta['features']])[0,1]
        self.assertAlmostEqual(result['probability'],expected)
        self.assertEqual(result['alert'],expected>=self.meta['limiar'])
        values['ida']=0.0
        self.assertIn('error',app.evaluate_student(self.model,self.meta,'Sem defasagem atual',2,values,True,False))
        result=app.evaluate_student(self.model,self.meta,'Sem defasagem atual',2,values,True,True)
        self.assertIn('IDA',result['outside'])


class InterfaceTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(app.ROOT/'app.py'),default_timeout=30).run()

    def test_navigation_and_filters(self):
        ui=self.make_app();self.assertFalse(ui.exception)
        ui.button(key='nav_dashboard').click().run()
        self.assertFalse(ui.exception)
        before=[m.value for m in ui.markdown if 'Alunos no recorte' in m.value][0]
        ui.multiselect(key='phases').set_value([0]).run()
        after=[m.value for m in ui.markdown if 'Alunos no recorte' in m.value][0]
        self.assertNotEqual(before,after)
        ui.multiselect(key='phases').set_value([]).run()
        self.assertTrue(ui.info);self.assertFalse(ui.exception)
        ui.multiselect(key='phases').set_value(list(range(8))).run()
        ui.selectbox(key='cohort').set_value('Mesmos alunos nos três anos').run()
        self.assertTrue(any('441' in m.value for m in ui.markdown if 'Alunos no recorte' in m.value))
        ui.selectbox(key='topic').set_value('Engajamento e percepções').run()
        for question in ['Engajamento acompanha desempenho e ponto de virada?',
                         'Autoavaliação acompanha desempenho e engajamento?',
                         'Avaliação psicopedagógica acompanha a defasagem?']:
            ui.selectbox(key='relation_question').set_value(question).run();self.assertFalse(ui.exception)
        ui.selectbox(key='year').set_value(2022).run()
        self.assertTrue(ui.info);self.assertFalse(ui.exception)
        ui.selectbox(key='topic').set_value('Sinais para acompanhamento').run();self.assertFalse(ui.exception)
        self.assertNotIn('year',[e.key for e in ui.selectbox])
        ui.radio(key='drop_target').set_value('Engajamento · IEG').run();self.assertFalse(ui.exception)
        ui.selectbox(key='origin').set_value(2023).run();self.assertFalse(ui.exception)
        ui.selectbox(key='signals_question').set_value('Quais indicadores se relacionam com a mudança no ponto de virada?').run()
        self.assertFalse(ui.exception)
        ui.selectbox(key='topic').set_value('Entendendo o INDE').run();self.assertFalse(ui.exception)
        ui.selectbox(key='year').set_value(2022).run();self.assertTrue(ui.info)
        ui.button(key='nav_about').click().run();self.assertFalse(ui.exception)
        self.assertNotIn('business_question',[element.key for element in ui.selectbox])
        self.assertEqual(len([button for button in ui.sidebar.button if button.key.startswith('nav_')]),5)
        ui.button(key='nav_questions').click().run();self.assertFalse(ui.exception)
        self.assertEqual(ui.title[0].value,'As dez perguntas de negócio')
        for question in range(1,11):
            ui.selectbox(key='business_question').set_value(question).run()
            self.assertFalse(ui.exception)
            self.assertTrue(ui.get('imgs'))
        ui.button(key='nav_context').click().run();self.assertFalse(ui.exception)

    def test_form_requires_complete_inputs_and_shows_result(self):
        ui=self.make_app();ui.button(key='nav_predict').click().run()
        submit=lambda:ui.button(key='FormSubmitter:screening-Estimar risco').click().run()
        self.assertEqual(ui.selectbox(key='phase').options,['0 · Alfa','1','2','3','4','5','6'])
        submit();self.assertTrue(ui.error)
        ui.selectbox(key='phase').set_value(2)
        for key,value in [('ida','6,5'),('ieg','10'),('ipv','6.800')]:ui.text_input(key=key).set_value(value)
        submit();self.assertTrue(ui.error)
        ui.checkbox(key='confirmed').set_value(True)
        submit()
        self.assertFalse(ui.exception);self.assertFalse(ui.error)
        self.assertEqual(ui.text_input(key='ieg').value,'10')
        self.assertTrue(any('IEG: 10' in c.value for c in ui.caption))
        self.assertTrue(any('Probabilidade estimada de entrada' in m.value for m in ui.markdown))
        ui.text_input(key='ida').set_value('');submit()
        self.assertTrue(ui.error)
        self.assertFalse(any('Probabilidade estimada de entrada' in m.value for m in ui.markdown))
        ui.text_input(key='ida').set_value('10');submit();self.assertTrue(ui.error)
        ui.checkbox(key='outside').set_value(True);submit()
        self.assertFalse(ui.error);self.assertTrue(ui.warning)
        self.assertEqual(ui.text_input(key='ida').value,'10')


class CompatibilityAndAnalysisTests(unittest.TestCase):
    def test_decimal_input_preserves_intended_values(self):
        for text,expected in [('10',10.0),('7,5',7.5),('7.5',7.5),(' 6,800 ',6.8),('0',0.0),(',5',.5)]:
            with self.subTest(text=text):
                self.assertEqual(app.parse_scores(dict(ida=text,ieg='7',ipv='6.8'))['ida'],expected)
        for text in ['', 'nan','inf','11','-1','1,2.3','abc','1 0','7,']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):app.parse_scores(dict(ida=text,ieg='7',ipv='6.8'))

    def test_annual_counts_and_absence_are_preserved(self):
        data=app.load_data();summary=app.annual_summary(data[data.fase.between(0,7)])
        self.assertEqual(summary.alunos.tolist(),[860,951,1054])
        self.assertEqual(summary.defasados.tolist(),[601,552,534])
        self.assertEqual(summary.ida_n.tolist(),[860,937,1054])

    def test_followup_does_not_filter_out_changed_phase(self):
        data=app.load_data()
        panel=app.school_panel(data,'População escolar de cada ano',[0],app.STONES)
        start,pairs=app.followup_pairs(data,panel,2022)
        expected=data[data.ano.eq(2023)&data.fase.between(0,7)&data.ra.isin(start.ra)]
        self.assertEqual(set(pairs.ra),set(expected.ra))
        self.assertTrue(pairs.fase_fim.gt(0).any())

    def test_limited_review_preserves_operational_model(self):
        model,meta=app.load_model();review=meta['comparacao_quatro_features']
        self.assertFalse(review['alternativas_avaliadas_em_2024'])
        self.assertFalse(review['artefato_substituido'])
        self.assertEqual(len(review['resultados']),3)
        self.assertEqual(len(review['folds']),15)
        self.assertFalse(any(r['atende_criterio'] for r in review['resultados']))
        self.assertEqual(meta['limiar'],0.278)


if __name__=='__main__':unittest.main()
