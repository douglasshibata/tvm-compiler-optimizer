# 📘 Tutorial de Operação e Instalação: Pipeline de Otimização e Compilação de Modelos de Deep Learning (Apache TVM)

Guia passo a passo para configuração, inicialização, execução de testes e resolução de problemas no projeto **Pipeline de Otimização e Compilação de Modelos de Deep Learning (Apache TVM)** (`tvm-compiler-optimizer`).

---

## 🛠️ 1. Visão Geral e Arquitetura

Este diretório contém a estrutura completa de uma arquitetura corporativa para compilação, otimização e auto-tuning de modelos de Deep Learning usando o **Apache TVM (TVM Unity / Relax / TensorIR)** para implantação em hardware heterogêneo. --- O Apache TVM transforma grafos de computação de alto nível originados em frameworks como PyTorch, ONNX e ...

- **Tecnologias Principais:** C++17 / CMake | Python 3.11+
- **Diretório do Módulo:** `tvm-compiler-optimizer`

---

## 📋 2. Pré-requisitos do Sistema

Antes de iniciar a configuração local, certifique-se de ter os seguintes componentes instalados em seu ambiente de desenvolvimento:

- **Git:** Para clonagem e versionamento.
- **Ambiente de Runtime:** Ferramentas correspondentes à stack do projeto (C++17 / CMake | Python 3.11+):
  - Node.js 18+ / Python 3.11+ / Go 1.24+ / Java 21 LTS / Flutter SDK / CMake.
- **Docker Engine & Docker Compose:** Recomendado para execução de containers e dependências.

---

## 🚀 3. Passo a Passo de Setup e Instalação

### Passo 1: Acessar a Pasta do Projeto
```bash
cd tvm-compiler-optimizer
```

### Passo 2: Instalação de Dependências
```bash
mkdir -p build && cd build && cmake .. && make
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Subindo Infraestrutura via Docker Compose
Se o ambiente dispõe de containerização local, execute os containers integrados:

```bash
# Iniciar serviços auxiliares em background
docker compose up -d --build

# Monitorar logs de execução
docker compose logs -f
```

---

## ⚙️ 4. Configuração do Ambiente

Configure o arquivo de variáveis de ambiente (`.env` ou configurações de projeto) localizado no diretório raiz de `tvm-compiler-optimizer`:

```ini
# Configurações Gerais
ENVIRONMENT=development
LOG_LEVEL=info

# URLs de Conexão e Serviços
DATABASE_URL=sqlite:///./app.db
REDIS_URL=redis://localhost:6379/0
```

---

## 💻 5. Guia de Execução

### Inicializando a Aplicação
Inicie a aplicação ou serviço utilizando o comando correspondente:

```bash
./build/app_bin # ou binário compilado equivalente
python3 main.py # ou uvicorn/streamlit/fastapi app
```

---

## 🧪 6. Testes e Validação

Para validar o funcionamento do módulo e garantir que não há regressões, execute a suíte de testes:

```bash
ctest --output-on-failure
python3 -m pytest tests/ -v
```

---

## 🔍 7. Troubleshooting e Resolução de Problemas

1. **Porta em uso ou Conflito de Rede:**
   - Cancele processos pendentes na porta utilizada ou utilize `docker compose down` e suba novamente.
2. **Erros de Dependências Ausentes:**
   - Certifique-se de estar executando os comandos dentro do diretório `tvm-compiler-optimizer` e com as versões corretas das linguagens.
3. **Falhas na Suíte de Testes:**
   - Verifique se as variáveis de ambiente necessárias foram inicializadas corretamente.
