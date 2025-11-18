# Sistema de Recomendação Médica 🏥

Sistema de recomendação de especialistas médicos baseado em sintomas usando Machine Learning (k-NN).

## 📋 Arquivos Necessários

Certifique-se de ter estes arquivos no repositório:

- ✅ `sistema_recomendacao.py` - Código principal da aplicação
- ✅ `requirements.txt` - Dependências Python
- ✅ `render.yaml` - Configuração do Render
- ✅ `Final_Augmented_dataset_Diseases_and_Symptoms.csv` - Dataset de sintomas
- ✅ `Sintomas - Especialidade.csv` - Mapeamento de especialidades
- ✅ `static/mackenzie-logo.png` - Logo (se disponível)

## 🚀 Deploy no Render

### Passo 1: Adicionar Arquivos
1. No GitHub, clique em **"Add file"** → **"Upload files"**
2. Faça upload de todos os arquivos acima
3. Clique em **"Commit changes"**

### Passo 2: Deploy no Render
1. Acesse [render.com](https://render.com)
2. Clique em **"New"** → **"Web Service"**
3. Conecte este repositório do GitHub
4. O Render detectará automaticamente o `render.yaml`
5. Clique em **"Create Web Service"**

### Passo 3: Configurações (se necessário)

Se o `render.yaml` não for detectado:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
gunicorn sistema_recomendacao:app
```

**Environment:** Python 3

## 📦 Tecnologias Utilizadas

- Python 3.11
- Flask 3.0
- scikit-learn 1.3.2
- pandas 2.1.3
- numpy 1.26.2
- K-Nearest Neighbors (k-NN) para classificação

## 👥 Equipe

- **GUILHERME FERREIRA FARIA** - RA: 10433718
- **KAREN SANTOS SOUZA** - RA: 10342208
- **NATALLIA RODRIGUES DE OLIVEIRA** - RA: 10444681
- **RAFAEL FERREIRA ELOI** - RA: 10442962

**Projeto Aplicado III - 2025**  
Universidade Presbiteriana Mackenzie

## ⚠️ Importante

- O sistema pode levar até 5 minutos para inicializar no primeiro acesso
- Plano gratuito do Render: 750 horas/mês
- O app pode "dormir" após 15 minutos de inatividade
