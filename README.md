# Pipeline de Otimização e Compilação de Modelos de Deep Learning (Apache TVM)

Este diretório contém a estrutura completa de uma arquitetura corporativa para compilação, otimização e auto-tuning de modelos de Deep Learning usando o **Apache TVM (TVM Unity / Relax / TensorIR)** para implantação em hardware heterogêneo.

---

## 1. Arquitetura do Compilador e Fluxo de Compilação TVM

O Apache TVM transforma grafos de computação de alto nível originados em frameworks como PyTorch, ONNX e TensorFlow em código de máquina otimizado de ultra-baixa latência para múltiplos targets de hardware.

### Fluxo Interno de Compilação (Diagrama ASCII)

```text
                                  +------------------------------------+
                                  |    Framework Model (PyTorch/ONNX)  |
                                  +------------------------------------+
                                                     |
                                                     | (Importação)
                                                     v
                                  +------------------------------------+
                                  |   Importador de Grafo (Relax IR)   |
                                  +------------------------------------+
                                                     |
                                                     v
                                  +------------------------------------+
                                  |     Otimizações de Alto Nível      |
                                  |  - Operator Fusion (Fusão de Ops)  |
                                  |  - Constant Folding (Simplificação)|
                                  |  - Dead Code Elimination / Layout  |
                                  +------------------------------------+
                                                     |
                                                     | (Lowering/Abaixamento)
                                                     v
                                  +------------------------------------+
                                  |     Otimizações de Baixo Nível     |
                                  |       (TensorIR / TIR Kernels)     |
                                  |  - Loop Tiling & Unrolling         |
                                  |  - Vectorization (Vetorização)     |
                                  |  - Thread Binding (GPU Thread Map) |
                                  |  - Shared Memory Scratchpads       |
                                  +------------------------------------+
                                                     |       ^
                                                     |       | Real-time Target Feedback
                                                     |       | (MetaSchedule Engine)
                                                     v       |
                                  +------------------------------------+
                                  |       Auto-Tuning de Kernels       |
                                  |  - Medição empírica no hardware    |
                                  |  - Evolutionary Search (Espaço)    |
                                  |  - Cost Model Predictor            |
                                  +------------------------------------+
                                                     |
                                                     | (Codegen / Geração de Código)
                                                     v
                                  +------------------------------------+
                                  |       Geração de Código Nativo     |
                                  |  - LLVM (CPU x86/ARM/AArch64)      |
                                  |  - CUDA C / Metal / OpenCL (GPU)   |
                                  +------------------------------------+
                                                     |
                                                     | (Build / Linkagem Estática/Dinâmica)
                                                     v
                                  +------------------------------------+
                                  |  Inference Runtime (.so / .bin / C)|
                                  |  - Zero-overhead TVM C++ Runtime   |
                                  |  - PackedFunc APIs & Dynamic Load  |
                                  +------------------------------------+
```

---

### Explicação Prática das Abstrações Fundamentais

1. **Relay e Relax (IR de Alto Nível):**
   - **Relay** é a representação intermediária (IR) funcional tradicional de grafos de computação do TVM. Ela representa fluxos de dados estáticos e dinâmicos, mas possui limitações para expressar estruturas de controle dinâmicas complexas encontradas em LLMs modernos.
   - **Relax (Relaxing Graph Abstractions)** é a evolução do Relay no ecossistema **TVM Unity**. O Relax introduz suporte nativo a tensores dinâmicos de forma flexível (completamente integrado ao Python), chamadas a funções com efeitos colaterais e representação unificada de subgrafos executados em acelerações proprietárias ou customizadas. Ela simplifica o "lowering" direto de grafos para TensorIR de forma incremental.

2. **TensorIR (IR de Baixo Nível para Kernels):**
   - TensorIR é a IR de baixo nível baseada em loops imperativos que expressa o comportamento matemático detalhado dos operadores de tensores (kernels).
   - Ela possui primitivas ricas para agrupar blocos computacionais (`T.block`), unificar estruturas de loops e expressar transformações cruciais como loop tiling (ladrilhamento), vetorização, unrolling de laços e mapeamento para registradores de hardware ou memória compartilhada (Scratchpads).

3. **MetaSchedule (Engine de Auto-Tuning):**
   - MetaSchedule é a próxima geração do AutoTVM e do AutoScheduler (Ansor). Trata-se de uma engine programável que explora de forma autônoma o vasto espaço de busca de transformações no TensorIR.
   - Usando algoritmos evolutivos combinados com modelos de custo baseados em aprendizado de máquina, o MetaSchedule decide os melhores tamanhos de tiling, estratégias de vetorização e paralelismo ideais para o hardware específico de execução, gerando códigos mais rápidos do que bibliotecas otimizadas escritas manualmente.

4. **TVM Module e PackedFunc:**
   - O resultado final da compilação do TVM é encapsulado em um **TVM Module** (`tvm::runtime::Module`). Este módulo contém os kernels nativos compilados e os metadados associados.
   - A comunicação entre o host (seja Python ou C++) e o módulo compilado é baseada na abstração **PackedFunc** (`tvm::runtime::PackedFunc`). Um PackedFunc é uma função uniformemente tipada que permite a passagem dinâmica de argumentos heterogêneos (tipos primitivos, tensores `DLTensor`, objetos complexos) sem overhead de compilação ou checagem de tipos estáticos pesados, servindo como a "cola universal" do runtime do TVM.

---

## 2. Guia Operacional, Compilação Cruzada e Embedded (microTVM)

### Compilação Cruzada (Cross-Compilation)
Para implantar modelos otimizados em sistemas embarcados robustos (como ARM Cortex-A em Raspberry Pi, NVIDIA Jetson Nano ou dispositivos móveis Android) diretamente a partir de um host x86 com capacidade de desenvolvimento:

1. **Definição do Target de Destino:**
   No TVM, especificamos o hardware através de triplas LLVM ou targets explícitos.
   ```python
   # Exemplo de target para ARM AArch64 (Raspberry Pi 4)
   target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
   ```
2. **Exportação do Módulo:**
   Após a compilação, exportamos o módulo gerando o arquivo com o formato binário correto do target de destino.
   ```python
   lib.export_library("model_arm64.so", cc="aarch64-linux-gnu-g++")
   ```

### microTVM: Execução Bare-metal / Microcontroladores
Para microcontroladores sem sistema operacional (como ARM Cortex-M0/M3/M4/M7, RISC-V, ESP32) ou DSPs de ultra-baixo consumo:

- **Estratégia sem Sistema Operacional:** O microTVM remove completamente a dependência de alocação de memória dinâmica ou sistemas de arquivos. Todo o runtime é reduzido a uma biblioteca estática autônoma em C de poucos kilobytes.
- **AOT (Ahead-of-Time) Executor:** O grafo de execução do modelo é gerado estaticamente em C em tempo de compilação, mapeando buffers de entrada/saída de forma estática para memória interna Sram ou Flash.
- **Workflow do microTVM:**
  1. Compilação do modelo com target especializado: `target = tvm.target.Target("c -keys=cpu")`.
  2. Uso do `tvm.micro.export_model_library_format(module, "model.tar")` para exportar os fontes em C, parâmetros estáticos e headers do runtime estático.
  3. Integração do código gerado no build system do firmware (CMake, Make, PlatformIO) e gravação no dispositivo embarcado.

### Estratégias de Quantização (Post-Training Quantization - PTQ)
A quantização para precisão de inteiros (`INT8`) acelera drasticamente a inferência ao substituir operações caras de ponto flutuante de 32 bits (`FP32`) por instruções aritméticas de inteiros altamente eficientes disponíveis em CPUs modernas (AVX-512, NEON) e microcontroladores (instruções DSP).

1. **Calibração (Calibration):**
   - Um conjunto pequeno de dados representativos (sem rótulos) é passado pelo modelo em precisão FP32.
   - O TVM analisa as faixas de ativação (mínimo e máximo) de cada tensor intermediário e computa os fatores de escala (*scale*) e pontos de deslocamento (*zero-point*).
2. **Transformação de Layout e Inserção de Operadores `Quantize/Dequantize`:**
   - O otimizador de grafos do TVM converte operadores matemáticos pesados (como `conv2d` ou `dense`) para versões quantizadas que realizam contas diretamente em INT8.
   - Aplica transformações para fundir as etapas de quantização, minimizando tráfego de dados de memória.
3. **Execução Nativa Otimizada:**
   - Durante o codegen, o TVM gera instruções assembly vetoriais específicas da plataforma (ex: `dp4a` em GPUs NVIDIA, instruções de multiplicação acumulada vetorial NEON em ARM) para atingir acelerações de até 4x em relação a implementações convencionais.
