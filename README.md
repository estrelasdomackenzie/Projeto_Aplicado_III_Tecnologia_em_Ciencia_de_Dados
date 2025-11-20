# Sistema de Recomendação de Médicos Especialistas

Projeto desenvolvido como parte do **Projeto Aplicado III** do curso de **Tecnologia em Ciência de Dados** da Universidade Presbiteriana Mackenzie.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-red.svg)](https://www.mackenzie.br/)

---

## Índice

- [Descrição Geral](#descrição-geral)
- [Objetivo](#objetivo)
- [Funcionalidades](#funcionalidades)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Requisitos do Sistema](#requisitos-do-sistema)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Pipeline de Execução](#pipeline-de-execução)
- [Resultados Obtidos](#resultados-obtidos)
- [Arquivos do Projeto](#arquivos-do-projeto)
- [Troubleshooting](#troubleshooting)
- [Equipe](#equipe)
- [Documentação](#documentação)

---

## Descrição Geral

O projeto implementa um **Sistema de Recomendação de Médicos Especialistas** utilizando técnicas de aprendizado supervisionado (K-Nearest Neighbors - k-NN). O sistema recebe sintomas informados pelo usuário e recomenda o diagnóstico provável e o especialista adequado para atendimento.

O sistema utiliza o algoritmo **K-Nearest Neighbors (KNN)** otimizado, alcançando acurácia superior a **99%**.

### Características Principais

- Interface web intuitiva e responsiva
- Análise em tempo real de sintomas
- Múltiplas visualizações de performance
- Comparação entre diferentes algoritmos de ML
- Sistema de busca inteligente de sintomas
- Recomendação de especialista com nível de confiança

---

## Objetivo

Desenvolver um modelo de Machine Learning que:
- Identifique doenças prováveis com base em sintomas relatados
- Recomende o especialista médico ideal para atendimento
- Forneça nível de confiança da predição
- Apresente interface acessível para usuários finais

---

## Funcionalidades

### Sistema Web (Interface Interativa)

- Seleção múltipla de sintomas através de interface visual
- Busca e filtro de sintomas em tempo real
- Contador dinâmico de sintomas selecionados
- Recomendação instantânea de especialista
- Diagnóstico com percentual de confiança
- Design responsivo (desktop e mobile)

### Análise e Avaliação do Modelo

- Otimização automática do hiperparâmetro K
- Validação cruzada (3-fold)
- 8 tipos diferentes de visualizações gráficas
- Comparação entre 5 algoritmos de ML
- Métricas detalhadas: Acurácia, Precisão, Recall, F1-Score
- Análise de performance por especialidade médica
- Curvas ROC e Precisão-Recall
- Matriz de confusão
- Distribuição de confiança das predições

---

## Estrutura do Repositório

```
estrelasdomackenzie/
│
├── README.md                                    # Documentação principal
├── requirements.txt                             # Dependências do projeto
├── PROJETO_APLICADO_III_Documento_Tecnico.docx # Documentação técnica completa
│
├── Sistema_Recomendação.py                     # Aplicação web Flask
├── Analise_Sistema_Recomendacao.py             # Análise e avaliação do modelo
│
├── Final_Augmented_dataset_Dis...              # Dataset principal
├── Sintomas - Especialidade.csv                # Mapeamento especialidades
│
└── static/                                     # Recursos estáticos
    └── mackenzie-logo.png                      # Logo institucional
```

---

## Tecnologias Utilizadas

### Backend e Machine Learning

- **Python**: 3.8+ (3.11+ recomendado)
- **Flask**: 3.0.0 - Framework web
- **scikit-learn**: 1.3.2 - Algoritmos de ML
- **Pandas**: 2.1.4 - Manipulação de dados
- **NumPy**: 1.26.2 - Computação numérica

### Visualização de Dados

- **Matplotlib**: 3.8.2 - Gráficos estáticos
- **Seaborn**: 0.13.0 - Visualizações estatísticas

### Frontend

- **HTML5**: Estrutura semântica
- **CSS3**: Estilização (identidade visual Mackenzie)
- **JavaScript**: Interatividade e validações

### Algoritmos de Machine Learning

- **K-Nearest Neighbors (KNN)**: Algoritmo principal (K=13 otimizado)
- **Random Forest**: Comparação e benchmark
- **Support Vector Machine (SVM)**: Comparação e benchmark
- **Decision Tree**: Comparação e benchmark
- **Naive Bayes**: Comparação e benchmark

### Bibliotecas Auxiliares

- **difflib**: Similaridade de strings
- **collections**: Manipulação de dados

---

## Requisitos do Sistema

### Hardware

- **Processador**: Multi-core recomendado
- **Memória RAM**: 4GB mínimo (8GB recomendado)
- **Espaço em disco**: 500MB livres

### Software

- **Sistema Operacional**: Windows, Linux ou macOS
- **Python**: Versão 3.8 ou superior (3.11+ recomendado)
- **Navegador Web**: Chrome, Firefox, Safari ou Edge (versões atualizadas)

### Ambiente de Desenvolvimento (Opcional)

- Google Colab
- Jupyter Notebook
- Visual Studio Code
- PyCharm

---

## Instalação

### Passo 1: Clone o Repositório

```bash
git clone https://github.com/estrelasdomackenzie/PROJETO_APLICADO_III_-_Documento_T-cnico.git
cd PROJETO_APLICADO_III_-_Documento_T-cnico
```

### Passo 2: Verifique a Versão do Python

```bash
python --version
```

ou

```bash
python3 --version
```

Certifique-se de ter Python 3.8 ou superior instalado.

### Passo 3: Crie um Ambiente Virtual (Recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Passo 4: Instale as Dependências

```bash
pip install -r requirements.txt
```

**Instalação Manual (caso necessário):**
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

### Passo 5: Verifique os Arquivos de Dados

Certifique-se de que os seguintes arquivos CSV estão presentes no diretório:

- `Final_Augmented_dataset_Diseases_and_Symptoms.csv`
- `Sintomas - Especialidade.csv`

---

## Como Executar

### Opção 1: Sistema Web (Interface Interativa)

Recomendado para usuários finais que desejam utilizar o sistema de recomendação.

**Executar:**

```bash
python Sistema_Recomendação.py
```

**Saída esperada no console:**

```
============================================================
SISTEMA DE RECOMENDACAO MEDICA
============================================================

Carregando dados...
Preparando e treinando...
Acurácia: 99.XX%
Registros: X,XXX
Sintomas: XXX

============================================================
PRONTO!
============================================================

Acesse: http://localhost:5000
```

**Acessar a aplicação:**

Abra seu navegador e acesse: `http://localhost:5000`

**Como utilizar:**

1. Use a caixa de busca para filtrar sintomas específicos
2. Clique nos sintomas que você está apresentando
3. Observe o contador de sintomas selecionados
4. Clique no botão "ANALISAR SINTOMAS"
5. Visualize o resultado: especialista recomendado, diagnóstico e confiança

**Parar o servidor:**

Pressione `Ctrl + C` no terminal

---

### Opção 2: Análise Completa do Modelo

Recomendado para desenvolvedores, pesquisadores e avaliação do modelo.

**Executar:**

```bash
python Analise_Sistema_Recomendação.py
```

**O que será executado:**

Este script executa um pipeline completo de análise:

#### Fase 1: Carregamento e Preparação de Dados
- Carregamento dos datasets
- Exibição de estatísticas
- Top 10 doenças mais frequentes
- Preparação e normalização dos dados

#### Fase 2: Otimização de Hiperparâmetros
- Teste de valores de K (3, 5, 7, 9, 11, 13, 15)
- Validação cruzada para cada valor
- Visualização gráfica da otimização
- Seleção automática do melhor K

#### Fase 3: Treinamento e Avaliação
- Treinamento do modelo com K otimizado
- Cálculo de métricas de performance
- Geração de 8 visualizações:
  1. Matriz de Confusão (top 10 doenças)
  2. Distribuição de Especialidades
  3. Curvas ROC (5 doenças mais comuns)
  4. Curvas Precisão-Recall
  5. Distribuição de Confiança
  6. Comparação de Algoritmos (KNN vs RF vs SVM vs DT vs NB)
  7. Validação Cruzada (3-fold)
  8. Performance por Especialidade Médica

#### Fase 4: Teste e Validação
- Execução de predição com sintomas de exemplo
- Exibição de diagnósticos similares
- Validação final do sistema

**Nota importante:** Feche cada janela de gráfico para avançar para o próximo.

---

## Pipeline de Execução

O código segue um pipeline completo e estruturado:

### 1. Carregamento de Dados
- Leitura dos arquivos CSV
- Validação da integridade dos dados
- Mapeamento de sintomas e especialidades

### 2. Preparação e Normalização
- Tratamento de valores ausentes
- Normalização de features
- Encoding de labels
- Divisão treino/teste (80/20)

### 3. Otimização do Hiperparâmetro K
- Validação cruzada para múltiplos valores de K
- Seleção do K com melhor performance
- Visualização dos resultados

### 4. Treinamento e Avaliação
- Treinamento do modelo KNN otimizado
- Cálculo de métricas de desempenho
- Geração de visualizações
- Comparação com outros algoritmos

### 5. Predição
- Recepção de sintomas do usuário
- Processamento e normalização
- Predição de doença e especialista
- Cálculo de confiança

---

## Resultados Obtidos

### Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **Acurácia** | Superior a 99% |
| **Precisão** | Superior a 99% |
| **Recall** | Superior a 99% |
| **F1-Score** | Superior a 99% |
| **K Otimizado** | 13 vizinhos |
| **Tempo de Resposta** | Inferior a 100ms |

### Comparação de Algoritmos

O sistema KNN com K=13 demonstrou:

- Melhor acurácia entre todos os algoritmos testados
- Tempo de treinamento competitivo
- Excelente generalização (validação cruzada)
- Performance consistente em todas as especialidades
- Robustez contra overfitting

### Especialidades Cobertas

O sistema mapeia doenças para mais de 20 especialidades médicas, incluindo:

- Cardiologia
- Dermatologia
- Endocrinologia
- Gastroenterologia
- Neurologia
- Pneumologia
- Psiquiatria
- Clínico Geral
- Entre outras

---

## Arquivos do Projeto

### Códigos Python

| Arquivo | Descrição | Uso |
|---------|-----------|-----|
| `Sistema_Recomendação.py` | Aplicação web Flask com interface | Usuários finais |
| `Analise_Sistema_Recomendação.py` | Análise completa com visualizações | Desenvolvedores/Pesquisadores |

### Datasets

| Arquivo | Descrição | Obrigatório |
|---------|-----------|-------------|
| `Final_Augmented_dataset_Diseases_and_Symptoms.csv` | Dataset de sintomas e doenças | Sim |
| `Sintomas - Especialidade.csv` | Mapeamento doença-especialista | Sim |

**Estrutura do dataset principal:**
- Colunas de sintomas (valores binários: 0 ou 1)
- Coluna 'diseases' (nome da doença)

**Estrutura do dataset de especialidades:**
- Coluna 'Disease': Nome da doença
- Coluna 'Specialist': Especialista correspondente

### Recursos Adicionais

| Arquivo | Descrição | Obrigatório |
|---------|-----------|-------------|
| `requirements.txt` | Lista de dependências Python | Sim |
| `PROJETO_APLICADO_III_Documento_Tecnico.docx` | Documentação técnica completa do projeto | Sim (recomendado) |
| `static/mackenzie-logo.png` | Logo institucional | Não (opcional) |

---

## Troubleshooting

### Erro: ModuleNotFoundError

**Causa:** Biblioteca Python não instalada

**Solução:**
```bash
pip install -r requirements.txt
```

ou instale a biblioteca específica:
```bash
pip install [nome-da-biblioteca]
```

---

### Erro: FileNotFoundError

**Causa:** Arquivos CSV não encontrados no diretório

**Solução:**

1. Verifique se está no diretório correto:
   ```bash
   pwd  # Linux/macOS
   cd   # Windows
   ```

2. Liste os arquivos CSV:
   ```bash
   ls -la *.csv  # Linux/macOS
   dir *.csv     # Windows
   ```

3. Certifique-se de que os arquivos necessários estão presentes

---

### Erro: Address already in use (Porta 5000)

**Causa:** Porta 5000 já está sendo utilizada por outro processo

**Solução 1:** Use outra porta via variável de ambiente

```bash
# Linux/macOS
export PORT=8080
python Sistema_Recomendação.py

# Windows
set PORT=8080
python Sistema_Recomendação.py
```

**Solução 2:** Modifique o código (linha 574 em Sistema_Recomendação.py)

```python
port = int(os.environ.get('PORT', 8080))  # Altere 5000 para 8080
```

---

### Problema: Logo não aparece na interface

**Causa:** Arquivo de imagem não encontrado

**Solução:**

1. Crie a pasta `static/` se não existir
2. Coloque o arquivo `mackenzie-logo.png` dentro dela
3. Alternativamente, comente as linhas 252-256 no código

---

### Problema: Gráficos não são exibidos

**Causa:** Backend gráfico do matplotlib não configurado

**Solução Ubuntu/Debian:**
```bash
sudo apt-get install python3-tk
```

**Solução macOS:**
```bash
brew install python-tk
```

**Solução Windows:**
Geralmente o Tkinter já vem instalado com Python

---

### Avisos sobre versões de bibliotecas

**Solução:** Atualize as bibliotecas

```bash
pip install --upgrade flask pandas scikit-learn matplotlib seaborn numpy
```

---

### Erro de encoding em arquivos CSV

**Causa:** Encoding incompatível

**Solução:** O sistema tenta múltiplos encodings automaticamente (UTF-8, Latin-1, CP1252). Se persistir, verifique o encoding do arquivo:

```python
# Teste o encoding manualmente
import pandas as pd
df = pd.read_csv('arquivo.csv', encoding='utf-8')  # ou 'latin-1' ou 'cp1252'
```

---

## Equipe

Este projeto foi desenvolvido por estudantes do curso de **Tecnologia em Ciência de Dados** da Universidade Presbiteriana Mackenzie:

| Nome | RA |
|------|-----|
| **Aline A. Ferreira** | 10433718 |
| **Karen Santos Souza** | 10342208 |
| **Natallia Rodrigues de Oliveira** | 10444681 |
| **Rafael Ferreira Eloi** | 10442962 |

---

## Documentação

### Documentação Técnica Completa

O projeto conta com documentação técnica detalhada disponível no repositório:

**Arquivo:** `PROJETO_APLICADO_III_Documento_Tecnico.docx`

Este documento contém:
- Fundamentação teórica do projeto
- Metodologia detalhada
- Análise exploratória dos dados
- Descrição completa dos algoritmos implementados
- Resultados experimentais
- Discussão e conclusões
- Referências bibliográficas

### Navegação da Wiki

- **Home**: Descrição geral e objetivos
- **Pipeline de Execução**: Detalhamento técnico do fluxo
- **Resultados Obtidos**: Métricas e análises
- **Conclusão e Trabalhos Futuros**: Perspectivas do projeto
- **Estrutura e Reprodutibilidade**: Guia de reprodução

---

## Contexto Acadêmico

- **Instituição:** Universidade Presbiteriana Mackenzie
- **Curso:** Tecnologia em Ciência de Dados
- **Disciplina:** Projeto Aplicado III
- **Período:** 2024/2025
- **Ambiente de Desenvolvimento:** Google Colab / Jupyter Notebook

---

## Metodologia

### Abordagem de Desenvolvimento

1. **Análise de Requisitos**: Definição do escopo e objetivos
2. **Coleta de Dados**: Obtenção e validação dos datasets
3. **Exploração de Dados**: Análise exploratória e estatísticas
4. **Modelagem**: Implementação e otimização do KNN
5. **Avaliação**: Métricas e validação do modelo
6. **Implementação**: Desenvolvimento da interface web
7. **Testes**: Validação end-to-end do sistema
8. **Documentação**: Elaboração de documentação técnica

### Técnicas Aplicadas

- **Aprendizado Supervisionado**: K-Nearest Neighbors
- **Validação Cruzada**: Estratificada (3-fold)
- **Normalização**: StandardScaler
- **Tratamento de Dados**: SimpleImputer
- **Encoding**: LabelEncoder
- **Avaliação**: Múltiplas métricas e visualizações

---

## Conclusão e Trabalhos Futuros

### Conclusão

O Sistema de Recomendação de Médicos Especialistas desenvolvido atingiu os objetivos propostos, apresentando:

- Alta acurácia (>99%) na identificação de doenças
- Interface intuitiva e acessível
- Tempo de resposta adequado para uso em produção
- Robustez e generalização comprovadas

### Trabalhos Futuros

Possíveis melhorias e extensões:

1. **Ampliação do Dataset**: Inclusão de mais doenças e sintomas
2. **Deep Learning**: Experimentação com redes neurais
3. **NLP**: Processamento de descrições textuais de sintomas
4. **Mobile**: Desenvolvimento de aplicativo móvel
5. **API RESTful**: Criação de API para integração
6. **Multilíngue**: Suporte para múltiplos idiomas
7. **Explicabilidade**: Implementação de LIME/SHAP
8. **Deploy**: Hospedagem em nuvem (AWS, Azure, GCP)

---

## Licença

Este projeto é desenvolvido para fins **acadêmicos** como parte do Projeto Aplicado III do curso de Tecnologia em Ciência de Dados da Universidade Presbiteriana Mackenzie.

© 2024-2025 - Todos os direitos reservados aos autores.

---

## Contato

Para dúvidas, sugestões ou contribuições:

- Reporte problemas na aba "Issues" do repositório GitHub
- Entre em contato através dos professores da disciplina
- Consulte a documentação completa no Wiki do projeto

---

## Agradecimentos

Agradecemos aos professores da disciplina de Projeto Aplicado III pelo suporte, orientação e feedback durante o desenvolvimento deste projeto.

Agradecemos também à Universidade Presbiteriana Mackenzie pela infraestrutura e recursos disponibilizados.

---

<div align="center">

**Desenvolvido com dedicação por estudantes do Mackenzie**

![Mackenzie](https://img.shields.io/badge/Mackenzie-Ciência%20de%20Dados-C8102E?style=for-the-badge)

</div>
