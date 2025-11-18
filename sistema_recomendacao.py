# -*- coding: utf-8 -*-
from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from difflib import SequenceMatcher
from collections import Counter
import warnings
import os

warnings.filterwarnings('ignore')
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Recomendação Médica</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
         font-family: 'Roboto';
         background: linear-gradient(135deg, #c8102e 0%, #8b0015 100%);
         min-height: 100vh;
         padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .mack-logo {
            height: 140px;
            width: auto;
            margin-bottom: 20px;
        }

        h1 {
            color: #c8102e;
            font-size: 28px;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #c8102e;
            font-size: 14px;
        }

        .search-box {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #ddd;
            border-radius: 12px;
            font-size: 16px;
            margin-bottom: 20px;
            transition: border 0.3s;
        }

        .search-box:focus {
            outline: none;
            border-color: #c8102e;
        }

        .counter {
            background: white;
            color: #c8102e;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 600;
            border: 2px solid #c8102e;
        }

        .counter-number {
            font-size: 36px;
            font-weight: 700;
            display: block;
            margin-bottom: 5px;
            color: #c8102e;
        }

        .symptoms-container {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 12px;
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 12px;
            margin-bottom: 20px;
        }

        .symptom-btn {
            padding: 14px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            text-align: center;
            transition: all 0.3s;
            user-select: none;
        }

        .symptom-btn:hover {
            border-color: #c8102e;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(200, 16, 46, 0.2);
        }

        .symptom-btn.selected {
            background: linear-gradient(135deg, #c8102e 0%, #8b0015 100%);
            color: white;
            border-color: #c8102e;
            font-weight: 600;
        }

        .btn-analyze {
            width: 100%;
            padding: 18px;
            background: white;
            color: #c8102e;
            border: 2px solid #c8102e;
            border-radius: 12px;
            font-family:'Roboto';
            font-size: 18px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-analyze:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(200, 16, 46, 0.3);
            background: #f8f8f8;
        }

        .btn-analyze:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .result {
            margin-top: 30px;
            padding: 25px;
            background: #fff5f5;
            border-left: 6px solid #c8102e;
            border-radius: 12px;
        }

        .result-title {
            font-size: 18px;
            font-weight: 600;
            color: #c8102e;
            margin-bottom: 12px;
        }

        .result-content {
            font-size: 16px;
            color: #333;
            line-height: 1.6;
        }

        .footer {
            margin-top: 50px;
            text-align: center;
            padding-top: 30px;
            border-top: 2px solid #f0f0f0;
        }

        .footer h3 {
            color: #c8102e;
            margin-bottom: 20px;
            font-size: 20px;
        }

        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .team-member {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #f0f0f0;
            transition: all 0.3s;
        }

        .team-member:hover {
            border-color: #c8102e;
            transform: translateY(-2px);
        }

        .team-member strong {
            display: block;
            color: #c8102e;
            font-size: 14px;
            margin-bottom: 5px;
        }

        .team-member small {
            color: #666;
            font-size: 12px;
        }

        .footer-note {
            margin-top: 20px;
            font-size: 13px;
            color: #666;
        }

        @media (max-width: 768px) {
            .symptoms-container {
                grid-template-columns: 1fr;
            }
            h1 { font-size: 22px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-container">
                <img src="{{ url_for('static', filename='mackenzie-logo.png') }}" 
                     alt="Logo Mackenzie" 
                     class="mack-logo">
            </div>
            <h1>Sistema de Recomendação Médica</h1>
        </div>
        
        <input 
            type="text" 
            id="searchBox" 
            class="search-box" 
            placeholder="Buscar sintomas..."
            onkeyup="filterSymptoms()"
        >

        <div class="counter">
            <span class="counter-number" id="counter">0</span>
            sintomas selecionados
        </div>

        <form method="POST" onsubmit="return prepareSubmit()">
            <div class="symptoms-container" id="symptomsContainer">
                {% for symptom in symptoms %}
                <div class="symptom-btn" onclick="toggleSymptom(this)">
                    {{ symptom }}
                </div>
                {% endfor %}
            </div>

            <input type="hidden" name="selected_symptoms" id="selectedSymptoms">

            <button type="submit" class="btn-analyze" id="analyzeBtn" disabled>
                Analisar Sintomas
            </button>
        </form>

        {% if result %}
        <div class="result">
            <div class="result-title">Resultado da Análise</div>
            <div class="result-content">{{ result }}</div>
        </div>
        {% endif %}

        <div class="footer">
            <h3>Equipe do Projeto</h3>
            <div class="team-grid">
                <div class="team-member">
                    <strong>ALINE A. FERREIRA</strong>
                    <small>RA: 10433718</small>
                </div>
                <div class="team-member">
                    <strong>KAREN SANTOS SOUZA</strong>
                    <small>RA: 10342208</small>
                </div>
                <div class="team-member">
                    <strong>NATALLIA RODRIGUES DE OLIVEIRA</strong>
                    <small>RA: 10444681</small>
                </div>
                <div class="team-member">
                    <strong>RAFAEL FERREIRA ELOI</strong>
                    <small>RA: 10442962</small>
                </div>
            </div>
            <p class="footer-note">Projeto Aplicado III - 2025</p>
        </div>
    </div>

    <script>
        function toggleSymptom(element) {
            element.classList.toggle('selected');
            updateCounter();
        }

        function updateCounter() {
            const count = document.querySelectorAll('.symptom-btn.selected').length;
            document.getElementById('counter').textContent = count;
            document.getElementById('analyzeBtn').disabled = count === 0;
        }

        function filterSymptoms() {
            const search = document.getElementById('searchBox').value.toLowerCase();
            const symptoms = document.querySelectorAll('.symptom-btn');
            
            symptoms.forEach(symptom => {
                const text = symptom.textContent.toLowerCase();
                symptom.style.display = text.includes(search) ? 'block' : 'none';
            });
        }

        function prepareSubmit() {
            const selected = Array.from(document.querySelectorAll('.symptom-btn.selected'))
                .map(btn => btn.textContent.trim())
                .join(', ');
            
            document.getElementById('selectedSymptoms').value = selected;
            return selected.length > 0;
        }
    </script>
</body>
</html>
"""

class SistemaRecomendacaoMedica:
    def __init__(self, k=13):
        self.k = k
        self.modelo = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.encoder = LabelEncoder()
        self.colunas_sintomas = []
        self.dataset = None
        self.mapa_especialidades = {}

    def _similaridade_strings(self, str1, str2):
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _buscar_match_parcial(self, doenca, mapa_csv, threshold=0.6):
        melhor_match = None
        melhor_score = threshold
        doenca_lower = doenca.lower()
        palavras_doenca = set(doenca_lower.split())

        for doenca_csv, especialista in mapa_csv.items():
            doenca_csv_lower = doenca_csv.lower()
            palavras_csv = set(doenca_csv_lower.split())

            if palavras_doenca & palavras_csv:
                score = len(palavras_doenca & palavras_csv) / max(len(palavras_doenca), len(palavras_csv))
                if score > melhor_score:
                    melhor_score = score
                    melhor_match = especialista

            if doenca_lower in doenca_csv_lower or doenca_csv_lower in doenca_lower:
                score = min(len(doenca_lower), len(doenca_csv_lower)) / max(len(doenca_lower), len(doenca_csv_lower))
                if score > melhor_score:
                    melhor_score = score
                    melhor_match = especialista

            sim = self._similaridade_strings(doenca, doenca_csv)
            if sim > melhor_score:
                melhor_score = sim
                melhor_match = especialista

        return melhor_match, melhor_score if melhor_match else (None, 0)

    def carregar_dados(self, arquivo_sintomas, arquivo_especialidades=None):
        self.dataset = pd.read_csv(arquivo_sintomas)
        self.colunas_sintomas = [col for col in self.dataset.columns if col != 'diseases']
        
        if arquivo_especialidades:
            self._carregar_especialidades(arquivo_especialidades)
        else:
            self._mapear_especialidades_automatico()

    def _carregar_especialidades(self, arquivo):
        df = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(arquivo, encoding=enc)
                break
            except:
                continue
        if df is None:
            return

        df.columns = df.columns.str.strip()
        if 'Disease' not in df.columns or 'Specialist' not in df.columns:
            return

        mapa_csv = {str(row['Disease']).strip(): str(row['Specialist']).strip()
                    for _, row in df.iterrows() if pd.notna(row['Disease']) and pd.notna(row['Specialist'])}

        for doenca in self.dataset['diseases'].unique():
            if doenca in mapa_csv:
                self.mapa_especialidades[doenca] = mapa_csv[doenca]
            else:
                match_case = False
                for d_csv, e_csv in mapa_csv.items():
                    if d_csv.lower() == doenca.lower():
                        self.mapa_especialidades[doenca] = e_csv
                        match_case = True
                        break
                if not match_case:
                    especialista, _ = self._buscar_match_parcial(doenca, mapa_csv)
                    self.mapa_especialidades[doenca] = especialista or 'Clínico Geral'

    def _mapear_especialidades_automatico(self):
        for doenca in self.dataset['diseases'].unique():
            self.mapa_especialidades[doenca] = 'Clínico Geral'

    def preparar_dados(self):
        X = self.dataset[self.colunas_sintomas].values
        y = self.dataset['diseases'].values
        
        contagem = Counter(y)
        classes_raras = [c for c, n in contagem.items() if n < 2]
        if classes_raras:
            mask = ~self.dataset['diseases'].isin(classes_raras)
            X = self.dataset[mask][self.colunas_sintomas].values
            y = self.dataset[mask]['diseases'].values
        
        X = self.imputer.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def treinar(self, X_train, y_train):
        self.modelo = KNeighborsClassifier(n_neighbors=self.k, weights='distance', n_jobs=-1)
        self.modelo.fit(X_train, y_train)

    def prever_por_nomes_sintomas(self, lista_sintomas):
        sintomas_array = np.zeros(len(self.colunas_sintomas))
        
        for sintoma in lista_sintomas:
            sintoma_lower = sintoma.lower().strip()
            for i, col in enumerate(self.colunas_sintomas):
                if sintoma_lower in col.lower() or col.lower() in sintoma_lower:
                    sintomas_array[i] = 1
                    break
        
        sintomas_proc = self.imputer.transform([sintomas_array])
        sintomas_proc = self.scaler.transform(sintomas_proc)
        
        doenca_idx = self.modelo.predict(sintomas_proc)[0]
        doenca = self.encoder.inverse_transform([doenca_idx])[0]
        
        proba = self.modelo.predict_proba(sintomas_proc)[0]
        confianca = proba[doenca_idx]
        
        especialista = self.mapa_especialidades.get(doenca, 'Clínico Geral')
        
        return {
            'doenca': doenca,
            'especialista': especialista,
            'confianca': float(confianca)
        }

sistema = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        selected = request.form.get('selected_symptoms', '')
        if selected:
            sintomas_lista = [s.strip() for s in selected.split(',')]
            try:
                pred = sistema.prever_por_nomes_sintomas(sintomas_lista)
                result = f"Especialista: {pred['especialista']} |Diagnóstico: {pred['doenca']} |Confiança: {pred['confianca']:.0%}"
            except Exception as e:
                result = f"Erro ao processar: {str(e)}"
    
    return render_template_string(HTML_TEMPLATE, symptoms=sistema.colunas_sintomas, result=result)

def inicializar():
    global sistema
    print("\n" + "="*60)
    print("SISTEMA DE RECOMENDACAO MEDICA")
    print("="*60 + "\n")
    
    arquivo_sintomas = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
    arquivo_especialidades = 'Sintomas - Especialidade.csv'
    
    if not os.path.exists(arquivo_sintomas):
        print(f"ERRO: {arquivo_sintomas} não encontrado!")
        return False
    
    sistema = SistemaRecomendacaoMedica(k=13)
    
    print("Carregando dados...")
    sistema.carregar_dados(arquivo_sintomas, arquivo_especialidades)
    
    print("Preparando e treinando...")
    X_train, X_test, y_train, y_test = sistema.preparar_dados()
    sistema.treinar(X_train, y_train)
    
    y_pred = sistema.modelo.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Acurácia: {acc:.2%}")
    print(f"Registros: {len(sistema.dataset):,}")
    print(f"Sintomas: {len(sistema.colunas_sintomas)}")
    
    print("\n" + "="*60)
    print("PRONTO!")
    print("="*60 + "\n")
    
    return True

if __name__ == '__main__':
    if inicializar():
        port = int(os.environ.get('PORT', 5000))
        print(f"Acesse: http://localhost:{port}\n")
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("ERRO ao inicializar!")