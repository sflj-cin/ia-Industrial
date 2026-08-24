# ia-Industrial
Estudo de Caso referente ao processo seletivo SENAI/PE – Cód. 164 (Pesquisador Industrial II, IA Industrial)

# Desafio IA Industrial — SENAI/PE Cód. 164 (Pesquisador Industrial II)

Prova de conceito de monitoramento inteligente de motores elétricos rotativos a partir
de sinais de vibração/acústica — subset do **UOEMD-VAFCVS** (University of Ottawa
Electric Motor Dataset — Vibration and Acoustic Faults under Constant and Variable
Speed Conditions), SEHRI; DUMOND; BOUCHARD (2024), licença CC-BY 4.0.

Este repositório contém um **notebook único, autocontido, executável de ponta a ponta
no Google Colab**, cobrindo os dois módulos do desafio:

- **Módulo 1** (obrigatório): EDA (tempo e frequência), engenharia de features,
  classificação das 4 classes de condição do motor, com validação sem vazamento de
  dados.
- **Módulo 2** (bônus): softsensor de temperatura, com otimização de hiperparâmetros
  via validação cruzada aninhada.

Toda decisão técnica não trivial no notebook é acompanhada de equação e referência
bibliográfica em formato ABNT — a lista completa está na Seção 11 do próprio notebook.
Este README resume e contextualiza essas decisões; a Seção "Referências" ao final
reproduz apenas as citações diretamente referidas aqui.

---

## Estrutura do repositório

```
.
├── README.md
├── motor_fault_monitoring_poc_v3.ipynb
└── requirements.txt
```

Deliberadamente enxuto: como o notebook é autocontido (instala dependências, faz
upload/extração do dataset, executa o pipeline completo e produz todas as figuras
inline), não há necessidade de módulos `.py` separados — a "arquitetura" do pipeline
está na organização das 11 seções do próprio notebook, cada uma com uma
responsabilidade clara (dados → EDA tempo → EDA frequência → features → validação →
diagnóstico de vazamento → classificação → avaliação → softsensor → conclusões →
referências).

## Como reproduzir

### Opção 1 — Google Colab (recomendado)

1. Abra `motor_fault_monitoring_poc_v3.ipynb` no [Google Colab](https://colab.research.google.com/).
2. Execute todas as células em ordem (*Ambiente de execução → Executar tudo*). A
   primeira célula instala as dependências (Seção 0); quando a célula de carregamento
   de dados rodar, um seletor de upload abrirá — envie o `.zip` com os 16 CSVs do
   dataset (fornecido pelo desafio).
3. O notebook detecta automaticamente o ambiente Colab (`google.colab`) e conduz o
   fluxo de upload/extração; fora do Colab, ele procura uma pasta `data/` já existente
   com os 16 CSVs (ver Opção 2).
4. Tempo de execução completo: poucos minutos em CPU (o gargalo é I/O dos 16 CSVs de
   ~20MB cada; o laço diagnóstico leave-one-file-out da Seção 8.3, com 16×2 = 32
   treinos adicionais, e a busca de hiperparâmetros aninhada da Seção 9, com
   4×15×3 = 180 ajustes de modelo, são as etapas mais custosas, ainda assim segundos a
   poucos minutos).

### Opção 2 — Ambiente local

```bash
pip install -r requirements.txt
mkdir data && unzip dataset_desafio.zip -d data   # os 16 CSVs devem ficar direto em ./data
jupyter notebook motor_fault_monitoring_poc_v3.ipynb
# Executar tudo (Kernel -> Restart & Run All)
```

Fora do Colab, o notebook detecta a ausência do módulo `google.colab` e usa `./data`
como origem dos dados diretamente, sem tentar abrir o seletor de upload.

---

## 1. Escolhas técnicas e justificativas

### 1.1 Stack

| Biblioteca | Versão | Papel | Justificativa |
|------------|--------|-------|---------------|
| `numpy` | ≥1.24 | Cálculo numérico | Base universal, sem overhead |
| `pandas` | ≥2.0 | Manipulação de DataFrames | Leitura eficiente de CSV grandes |
| `scipy.signal` | ≥1.11 | Análise espectral industrial | Welch PSD com janela de Hann — padrão ISO para vibração |
| `scipy.stats` | ≥1.11 | Estatísticas (kurtosis, skew) | Indicadores clássicos de falha em rolamentos |
| `scikit-learn` | ≥1.3 | Modelos e validação | RF e GBM com LOGO-CV, pipelines reproduzíveis |
| `optuna` | ≥3.0 | HPO para softsensor | TPE sampler (>2× mais eficiente que Random Search) |
| `matplotlib`/`seaborn` | ≥3.7/0.12 | Visualizações | Qualidade publicável, integração com Colab |

**Por que não Deep Learning?** Com apenas 16 arquivos (160 janelas de 1 s), qualquer rede neural profunda overfittaria nos dados de treino. Features clássicas + modelos ensemble são a escolha correta para datasets industriais pequenos com alta interpretabilidade requerida.

| Componente | Escolha | Por quê |
|---|---|---|
| Manipulação de dados | pandas + NumPy | Padrão de mercado; cada CSV (420 mil linhas) cabe confortavelmente em memória em `float32`. |
| Análise espectral | `scipy.signal.welch`, `scipy.signal.coherence`, `scipy.signal.spectrogram` | Estimadores espectrais estatisticamente mais robustos que uma FFT única — ver Seção 1.2. |
| Modelos de classificação | `RandomForestClassifier` (scikit-learn) + `XGBClassifier` (XGBoost) | Ensemble de bagging (BREIMAN, 2001) e de boosting (CHEN; GUESTRIN, 2016) — mecanismos de regularização distintos, o que torna a comparação informativa. Ambos dispensam normalização de features e expõem importância nativa, útil para auditoria (Seção 2.4). Com apenas 304 janelas de treino, uma abordagem *end-to-end* de deep learning tenderia a sobreajustar (LEI et al., 2020) — ver Seção 3.2. |
| Otimização de hiperparâmetros (Módulo 2) | Optuna (TPE sampler) | Busca bayesiana eficiente em espaço de hiperparâmetros contínuo/pequeno, superior a grid search com poucas avaliações disponíveis (AKIBA et al., 2019) — ver Seção 5. |
| Armazenamento intermediário | Parquet (`pyarrow`) | A tabela de features (304 × 87) é reutilizada por várias seções do notebook sem recomputar a extração, que é o passo mais custoso do pipeline. |

### 1.2 Estratégia de janelamento e engenharia de features

**Janelamento.** Cada arquivo (10s @ 42kHz = 420.000 amostras) é dividido em janelas
de **1 segundo (42.000 amostras) com 50% de sobreposição** (hop = 21.000 amostras) →
19 janelas/arquivo → **304 janelas totais** (76 por classe, balanceado por
construção). A sobreposição aumenta a densidade de amostras de treino ao custo de
correlação entre janelas vizinhas — por isso o agrupamento por **arquivo/condição**
(não por janela) é obrigatório na validação (Seção 1.3). A premissa de que o sinal é
suficientemente estacionário dentro de cada janela de 1s (e ao longo dos 10s do
arquivo) foi verificada empiricamente via espectrograma STFT (notebook, Seção 3.4):
coeficiente de variação temporal da energia em banda (0–500Hz) de **0,059** — baixo,
consistente com a hipótese de estacionariedade que sustenta tanto o cálculo da PSD de
Welch (média de segmentos assumindo mesmo processo subjacente) quanto o tratamento das
19 janelas de um arquivo como amostras comparáveis.

**87 features por janela** (21 por canal × 4 canais + 3 features de correlação
cruzada entre sensores):

*Domínio do tempo* (10/canal): RMS, desvio-padrão, média absoluta, pico, pico-a-pico,
fator de crista, fator de forma, fator de impulso, curtose (Fisher), assimetria —
vocabulário clássico de indicadores de condição em análise de vibração (RANDALL,
2011).

*Domínio da frequência* (11/canal), calculadas sobre a densidade espectral de potência
estimada pelo **método de Welch** — divide o sinal em segmentos sobrepostos, aplica
janela de Hann a cada um e promedia os periodogramas resultantes, reduzindo a
variância da estimativa em relação a uma FFT única ao custo de resolução espectral
($\Delta f = F_s/M \approx 10{,}25$ Hz com `nperseg=4096`) — WELCH (1967); HARRIS
(1978) para a justificativa da janela de Hann. Incluem centroide espectral, entropia
espectral, rolloff85, energia em 5 bandas fixas, e **3 features de order tracking**
(amplitude nas harmônicas 1×/2×/3× da frequência de rotação nominal do VFD,
normalizada pela energia total) — segue a recomendação do dicionário de dados do
desafio de expressar features em múltiplos da frequência de rotação em vez de bandas
absolutas, cuja necessidade é demonstrada empiricamente no notebook (Seção 3.3:
mesma falha R_U comparada a 15Hz e 45Hz — a harmônica migra proporcionalmente à
velocidade; uma banda absoluta fixa perderia a assinatura ao mudar de condição).

*Correlação cruzada entre sensores* (3 features): correlação de Pearson
accel2×accel3 e accel2×microfone, e **coerência espectral** accel2×accel3 avaliada na
1ª ordem de rotação (via `scipy.signal.coherence`, mesma família de estimador de
Welch aplicada ao espectro cruzado) — motivada pela descrição do dicionário de dados
de que B_R (rotor empenado) pode gerar vibração axial e radial acopladas. Na análise
de correlação univariada com o rótulo (notebook, Seção 4.5), `corr_accel2_mic` aparece
como a **2ª feature mais correlacionada** das 87 (|r|=0,324, atrás apenas da curtose de
accel1), o que dá algum suporte empírico a essa hipótese — com a ressalva de que
correlação univariada não captura interações multivariadas e não substitui a
importância nativa dos modelos (GUYON; ELISSEEFF, 2003).

### 1.3 Estratégia de validação e justificativa

O dataset tem **exatamente 1 arquivo por combinação (classe, velocidade, carga)** — 4
classes × 2 velocidades × 2 cargas = 16 arquivos. Isso implica que qualquer split que
respeite o agrupamento por arquivo (necessário para não vazar dados — KAUFMAN et al.,
2012) deixa estruturalmente pelo menos uma combinação inteiramente fora do treino.

Adotamos **leave-one-condition-out**: 4 folds, cada um retirando **todas as 4 classes
de uma combinação (velocidade, carga)** para teste, treinando com os outros 12
arquivos. Isso mede diretamente a pergunta relevante em manutenção preditiva: *o
modelo generaliza para uma condição operacional nunca vista, para qualquer classe?* —
mais rigoroso que a alternativa mais comum (*leave-one-file-out*, 16 folds), na qual o
treino ainda contém a mesma condição operacional do teste para as outras 3 classes.
Usamos leave-one-file-out apenas como **diagnóstico complementar** (notebook, Seção
8.3), não como validação oficial — ver nota de rigor abaixo.

**Nota de rigor sobre agregação de métricas em CV agrupada.** Uma métrica multiclasse
"macro" (como F1-macro) só é bem definida quando o fold de teste contém exemplos de
todas as classes de interesse. No esquema oficial (leave-one-condition-out), cada fold
retirado contém as 4 classes — logo F1-macro por fold é válido, e o valor agregado
final é calculado juntando as predições de **todos os folds antes de computar a
métrica uma única vez** (SOKOLOVA; LAPALME, 2009), evitando problemas de média-de-médias
em folds pequenos. Já no diagnóstico leave-one-file-out (Seção 8.3 do notebook), cada
fold de teste contém apenas 1 classe (o arquivo pertence a uma única classe) — nesse
caso, F1-macro sobre as 4 classes é matematicamente mal-definido (as 3 classes
ausentes teriam recall/F1 tratados como zero por convenção, subestimando o desempenho
em até 4×). Por isso, para esse diagnóstico específico, usamos **recall da classe
verdadeira** (equivalente a acurácia do fold, bem definido quando há uma única classe
presente) em vez de F1-macro — uma correção metodológica deliberada, não um detalhe
de implementação.

**Diagnóstico de vazamento (notebook, Seção 6).** Para quantificar o efeito de um erro
metodológico comum, comparamos com um split ingênuo por **janela**
(`StratifiedKFold`, 5 folds, ignorando o arquivo de origem):

| Modelo | F1-macro (split de janelas, **com** vazamento) | F1-macro (leave-one-condition-out, **sem** vazamento) |
|---|---|---|
| RandomForest | **1,000** | 0,588 |
| XGBoost | **1,000** | 0,645 |

Ambos os modelos atingem F1-macro = 1,00 quando o split ignora a estrutura de
agrupamento — a métrica reportada seria quase o dobro do valor real. É o achado
metodológico mais importante deste desafio.

---

## 2. Análise dos resultados

### 2.1 Por que F1-macro em vez de acurácia simples

Acurácia simples é enganosa em diagnóstico multiclasse mesmo com classes balanceadas
em número de janelas, como aqui: ela pondera cada *janela* igualmente, não cada
*classe*, então um modelo que acerta perfeitamente a classe majoritária e erra
sistematicamente uma classe minoritária de falha pode reportar acurácia alta enquanto
ignora inteiramente uma condição de falha relevante. F1-macro (média não ponderada do
F1 por classe) força o modelo a ser avaliado igualmente em todas as classes,
independente de frequência — a escolha padrão quando o objetivo é generalização
uniforme entre classes, não apenas na classe mais fácil (SOKOLOVA; LAPALME, 2009). Em
manutenção preditiva, PR-AUC (curva precisão-recall) é outra métrica frequentemente
preferível à acurácia pelo mesmo motivo — é mais sensível ao desempenho sobre a classe
positiva (falha) que a curva ROC em cenários com prevalência baixa de falha, embora
aqui, com classes balanceadas por construção, o ganho relativo de PR-AUC sobre
F1-macro seja menor do que seria em um dataset real de produção (onde a classe
saudável tipicamente domina).

### 2.2 Resultados

**Classificação — leave-one-condition-out (métrica oficial), pooled sobre os 4 folds:**

| Modelo | F1-macro | B_R (F1) | H_H (F1) | R_M (F1) | R_U (F1) |
|---|---|---|---|---|---|
| RandomForest | 0,588 | 0,27 | 1,00 | 0,55 | 0,53 |
| XGBoost | **0,645** | 0,50 | 0,89 | 0,72 | 0,47 |

**Holdout oficial (teste = 45Hz + com carga), relatório de classificação completo:**

| Modelo | Acurácia | F1-macro | B_R (F1) | H_H (F1) | R_M (F1) | R_U (F1) |
|---|---|---|---|---|---|---|
| RandomForest | 0,750 | 0,660 | 0,00 | 1,00 | 0,95 | 0,69 |
| XGBoost | 0,724 | 0,644 | 0,00 | 1,00 | 0,92 | 0,66 |

**Achado mais notável:** H_H (saudável) é recuperada quase perfeitamente em qualquer
condição e em ambos os esquemas de avaliação — plausível, já que sua assinatura (baixa
amplitude, ausência de periodicidade síncrona) não depende de qual harmônica de falha
domina. B_R (rotor empenado) é a classe mais difícil de forma consistente: no
agregado leave-one-condition-out chega a F1=0,27–0,50 (melhora considerável frente a
uma versão anterior deste protótipo, que usava periodograma bruto em vez de PSD de
Welch e não tinha features de correlação cruzada, e obtinha F1=0,00 para B_R em
qualquer condição), mas no holdout específico de 45Hz+com carga cai a F1=0,00 nos dois
modelos — evidência de que a assinatura de B_R é particularmente sensível à condição
operacional específica de treino/teste, consistente com apenas 1 arquivo por classe
por condição não sustentar generalização robusta (RANDALL; ANTONI, 2011, sobre a
sensibilidade de assinaturas de defeito de rotor à condição de carga/velocidade). O
diagnóstico leave-one-file-out por arquivo (notebook, Seção 8.3) reforça esse padrão:
arquivos de B_R têm recall=0 na maioria dos casos, exceto B_R_3_0 (XGBoost, recall=1,0)
— sinal de que o modelo às vezes acerta B_R quando a condição de teste tem um "vizinho"
suficientemente parecido no treino, mas não de forma confiável.

**Auditoria de importância de features:** tanto pela correlação univariada com o
rótulo quanto pela importância Gini do modelo treinado com 100% dos dados (notebook,
Seções 4.5 e 8.5), as features mais relevantes concentram-se nos canais de eixo
(`accel2`) e em features de baixa frequência/order tracking — plausível fisicamente.
Ressalva: a importância Gini de árvores tem viés documentado a favor de features
contínuas e correlacionadas entre si (STROBL et al., 2007), então o ranking deve ser
lido como indicativo.

### 2.3 Trade-off falso positivo × falso negativo

Em manutenção preditiva, os dois erros têm custos assimétricos:

- **Falso positivo** (prever falha em motor saudável) → parada não planejada de um
  ativo saudável, custo de inspeção desnecessária. Custo geralmente operacional e
  recorrente.
- **Falso negativo** (deixar passar uma falha real) → risco de falha catastrófica,
  parada não planejada pior, risco de segurança. Custo geralmente maior, com cauda
  pesada.

Na prática, a maioria dos programas de manutenção preditiva tolera mais falsos
positivos que falsos negativos — prefere-se investigar um alarme infundado a deixar
passar uma falha real. Isso favorece otimizar **recall por classe de falha** (ou
calibrar threshold via PR-AUC) em vez de F1 balanceado puro. Olhando a tabela de 2.2
sob essa ótica: RandomForest tem recall de R_U (desbalanceamento) alto (1,00 no
holdout) às custas de precisão baixa (0,53) — no contexto de custo assimétrico acima,
esse é o tipo de erro *preferível* (excesso de alarmes de desbalanceamento, não falhas
não detectadas). Já o recall de B_R = 0 no holdout é o cenário mais preocupante sob
essa lente — o modelo, como está, deixaria passar toda ocorrência de rotor empenado
nessa condição específica. Não implementamos calibração formal de threshold neste
protótipo — o volume de dados (1 arquivo por classe/condição) não sustenta uma curva
de calibração confiável; é o próximo passo natural com mais dados (Seção 4).

### 2.4 Limitações identificadas

**Da solução:**

- 304 janelas totais, fortemente correlacionadas dentro de cada arquivo (janelas com
  50% de sobreposição) — o "N efetivo" para generalização está mais próximo de 16
  (arquivos) que de 304.
- A frequência de rotação usada no order tracking é a nominal do VFD, não a mecânica
  real do eixo — o escorregamento do motor de indução (visível no deslocamento do pico
  dominante do espectro em relação à ordem nominal, notebook Seção 3.3) não é
  corrigido por ausência de tacômetro/encoder no dataset.
- Sem calibração de limiar de decisão nem análise formal de custo assimétrico
  aplicada (Seção 2.3) — decisão relevante em produção mas que exigiria mais dados por
  classe/condição.
- A importância de features (Seção 2.2) está sujeita ao viés de importância Gini
  documentado por STROBL et al. (2007); a correlação univariada (Seção 1.2) não
  captura interações entre features (GUYON; ELISSEEFF, 2003).
- O diagnóstico leave-one-file-out (por arquivo) usa um esquema de treino mais
  permissivo que a validação oficial — seus valores são estruturalmente mais
  otimistas e não substituem a Seção 2.2.

**Do dataset:**

- Apenas 1 arquivo por combinação (classe, velocidade, carga) — sem repetição
  independente que separe "assinatura da classe de falha" de "particularidades
  daquele ensaio/motor específico".
- Apenas 2 níveis de velocidade e 2 de carga — insuficiente para modelar a relação
  contínua entre condição operacional e expressão espectral da falha (empiricamente
  não trivial, Seção 3.3 do notebook).
- Ausência de RPM medido diretamente, limitando a precisão do order tracking.
- 10s por gravação é curto para dinâmica térmica de mais longo prazo (relevante para
  o Módulo 2, Seção 4).

---

## 3. Evolução para um sistema real

### 3.1 De batch para streaming

O notebook é inteiramente batch: lê arquivos completos, computa features por janela
offline. Evoluindo para streaming: (i) **aquisição contínua** — substituir a leitura
de CSV por um buffer circular alimentado por um driver de aquisição (ex.: NI DAQ,
como o NI USB-6212 mencionado no dicionário de dados, ou um gateway Modbus/OPC-UA);
(ii) **janelas deslizantes** — mesma lógica de janela+hop aplicada a um buffer
contínuo, disparando inferência incremental; o custo de extração de features por
janela (~poucos ms de CPU neste protótipo) é tranquilamente viável dentro do
orçamento de tempo real (~1 predição a cada 0,5s por sensor, dado o hop de 50%); (iii)
**concept drift** — o próprio resultado da Seção 2.2 (sensibilidade de generalização à
condição operacional, sobretudo para B_R) é evidência direta de que a fronteira de
decisão do modelo depende da condição operacional — em produção, isso se manifestaria
como drift aparente sempre que o motor mudasse de regime de carga/velocidade, mesmo
sem mudança real no estado de saúde. Mitigação: monitorar a distribuição das features
de entrada segmentada por condição operacional conhecida (não apenas a saída do
modelo), com testes estatísticos (ex.: Kolmogorov-Smirnov) disparando retreino
programado.

```
BATCH (esta PoC):
  CSV → Feature Extraction → Modelo → Relatório
  Latência: minutos | Uso: auditoria, manutenção programada

STREAMING (produção):
  DAQ (42kHz) → Buffer circular (1s) → Feature Extraction → Modelo → Alarme
  Latência: ~1s | Ferramentas: Apache Kafka + consumer Python ou MQTT + edge processor
```

**Janela deslizante em streaming:**
```python
# Pseudocódigo — buffer circular com hop de 0.5s (50% sobreposição)
buffer = collections.deque(maxlen=42000)
while True:
    sample = daq.read()
    buffer.append(sample)
    if len(buffer) == 42000:
        features = extract_features(np.array(buffer))
        label = model.predict([features])[0]
        if label != "H_H":
            send_alarm(label, confidence=model.predict_proba([features]).max())
```

**Concept Drift:**
- Monitorar distribuição das features com ADWIN ou Page-Hinkley test
- Retreinamento automático quando KS-test p < 0.05 por N janelas consecutivas
- Manter janela deslizante de dados rotulados recentes (últimas 4 semanas)

### 3.2 Feature engineering clássica vs. deep learning end-to-end

Features clássicas (tempo/frequência/order tracking) foram escolhidas deliberadamente
em vez de uma abordagem end-to-end (CNN 1D sobre a forma de onda bruta, ou sobre
espectrograma via CNN 2D): com 304 janelas (efetivamente ~16 instâncias
independentes), treinar uma rede do zero levaria a overfitting severo (LEI et al.,
2020); features físicas interpretáveis permitem auditoria de sanidade (Seção 2.2);
custo de inferência é trivial (viável em edge, Seção 3.3). Deep learning se justifica
com um dataset ordens de magnitude maior (múltiplos motores, múltiplas instâncias por
classe/condição) — aí consegue aprender assinaturas que a engenharia manual não
antecipou, especialmente para falhas sutis/combinadas.

### 3.3 Deploy em edge (bônus)

Custo de inferência de árvores já treinadas (RandomForest/XGBoost) é da ordem de
microssegundos; extração de features de uma janela de 1s (4 canais) é da ordem de
poucos milissegundos de CPU single-thread. Viável em gateway industrial (ex.:
Raspberry Pi/Jetson) junto ao PLC. Trade-offs: reduz banda de telemetria (envia
features/alarmes, não sinal bruto de 42kHz) e latência de detecção, mas dificulta
re-treino centralizado (exige pipeline OTA) e exige monitoramento de drift
descentralizado (Seção 3.1).

---

**Hardware-alvo:** Raspberry Pi 4 (4GB RAM) ou NVIDIA Jetson Nano

```python
# Export para ONNX (inferência sem scikit-learn em produção)
from sklearn2pmml import sklearn2pmml
# ou via sklearn-onnx:
from skl2onnx import convert_sklearn
model_onnx = convert_sklearn(rf_final, "RF", [("input", FloatTensorType([None, n_features]))])
```

**Trade-offs edge:**
- RAM limitada → RF com ≤100 árvores, profundidade ≤10
- Sem scipy em sistemas embarcados → implementar Welch PSD com numpy puro (FFT nativa)
- Conectividade intermitente → buffer local de alarmes com timestamp para sincronização

---

## 4. Softsensors em ambiente industrial (bônus)

**Quando faz sentido:** quando o sensor físico é caro, inacessível (ex.: dentro de um
enrolamento), tem manutenção cara em ambientes agressivos, ou quando se quer
redundância analítica para detectar falha do próprio sensor físico (comparar leitura
real vs. predição do softsensor sinaliza sensor divergente).

**Resultado deste protótipo (notebook, Seção 9).** O alvo é a temperatura **média da
janela** (a variância intra-arquivo, ~0,01–0,09°C, é desprezível frente à variância
entre arquivos, ~0,5–2,4°C — quase toda a variância do alvo é um offset por ensaio, não
um efeito capturado pelos sinais vibro-acústicos de 1s). Os hiperparâmetros foram
otimizados via **CV aninhada** (Optuna/TPE no laço interno, `GroupKFold` por arquivo
restrito ao treino de cada fold externo, nunca consultando o teste externo durante a
busca — VARMA; SIMON, 2006, sobre o viés otimista de reutilizar o mesmo esquema de CV
para seleção de hiperparâmetros e para relatar a métrica final):

| Condição de teste (fold externo) | MAE (°C) | R² |
|---|---|---|
| 15Hz, sem carga | 2,163 | -0,779 |
| 15Hz, com carga | 1,508 | -13,199 |
| 45Hz, sem carga | 1,412 | -0,258 |
| 45Hz, com carga | 1,240 | -8,289 |
| **Agregado (pooled)** | **1,581** | **-0,595** |
| Baseline ingênuo (média do treino) | 1,390 | — |

O softsensor **não supera** o baseline ingênuo, e R² é negativo em todos os folds —
resultado honesto, não um artefato de tuning otimista (a CV aninhada elimina essa
hipótese). A causa raiz é estrutural: quase toda a variância do alvo é um offset por
arquivo que o modelo nunca observa para a condição de teste no esquema
leave-one-condition-out, e não há, nas features vibro-acústicas de 1s, informação
suficiente para extrapolá-lo. Alternativa de projeto: gravações mais longas (minutos a
horas) com alvo formulado como **delta de temperatura** relativo ao início da
gravação, removendo o offset por ensaio que hoje domina a variância.

**Monitoramento de drift e retreinamento em produção:** quando o softsensor tem um
sensor físico redundante disponível, o próprio erro contra a leitura real é o sinal
primário de drift. Quando o softsensor substitui o sensor físico, monitorar a
distribuição das features de entrada (mesma lógica da Seção 3.1) e disparar retreino
programado sempre que a planta mudar de regime operacional de forma sustentada — dado
o achado da Seção 2.2 (sensibilidade à condição operacional), provavelmente necessário
com mais frequência do que se imaginaria à primeira vista.

---

## 5. Otimização e algoritmos evolutivos (bônus)

**Onde brilham:** problemas de otimização combinatória ou multi-objetivo com espaço de
busca não-convexo/não-diferenciável ou função objetivo cara de avaliar — comuns em
manutenção industrial. Exemplo concreto: **planejamento de manutenção** — decidir quais
ativos parar e quando, respeitando janelas de produção, disponibilidade de equipe e
dependências entre ativos, é um problema combinatório de escalonamento com restrições;
Algoritmos Genéticos lidam bem com representações discretas (ordem de tarefas,
alocação binária) e função objetivo não-diferenciável (custo de parada + risco de
falha, combinando previsões de múltiplos modelos como os deste desafio). Outro
exemplo: **otimização multi-objetivo de setpoints** de um conjunto de motores para
minimizar simultaneamente consumo energético e risco de falha estimado — objetivos
conflitantes, para os quais algoritmos evolutivos multi-objetivo (ex.: NSGA-II)
produzem diretamente uma fronteira de Pareto.

**Quando são overkill:** para otimização de hiperparâmetros de um modelo de ML (como o
Módulo 2 deste desafio), o espaço de busca é contínuo/quase-contínuo, de dimensão
baixa a moderada, e cada avaliação (treinar+validar) é razoavelmente cara — regime
onde **otimização bayesiana** (Optuna, usada aqui) é superior: modela a superfície de
resposta e escolhe o próximo ponto de forma informada, precisando de muito menos
avaliações que um algoritmo evolutivo para convergir (AKIBA et al., 2019). Usamos 15
trials por fold externo (60 no total) — número inviável para um GA convergir com
qualidade equivalente no mesmo orçamento de avaliações. Grid search, por sua vez, seria
ineficiente aqui pela dimensionalidade do espaço (4 hiperparâmetros contínuos/inteiros)
— cresce exponencialmente com o número de dimensões. Algoritmos evolutivos voltam a
ser preferíveis quando o espaço é discreto/combinatório, multimodal de forma severa, ou
quando o problema é genuinamente multi-objetivo sem uma forma natural de escalarizar
os objetivos em uma única métrica — nenhuma dessas condições se aplica à otimização de
hiperparâmetros de árvores de decisão deste desafio.

---

## Referências

AKIBA, Takuya et al. **Optuna**: a next-generation hyperparameter optimization
framework. In: PROCEEDINGS OF THE 25TH ACM SIGKDD INTERNATIONAL CONFERENCE ON
KNOWLEDGE DISCOVERY & DATA MINING, 25., 2019, Anchorage. **Anais [...]**. New York:
ACM, 2019. p. 2623-2631.

BREIMAN, Leo. Random forests. **Machine Learning**, [s. l.], v. 45, n. 1, p. 5-32,
2001.

CHEN, Tianqi; GUESTRIN, Carlos. XGBoost: a scalable tree boosting system. In:
PROCEEDINGS OF THE 22ND ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND
DATA MINING, 22., 2016, San Francisco. **Anais [...]**. New York: ACM, 2016.
p. 785-794.

GUYON, Isabelle; ELISSEEFF, André. An introduction to variable and feature selection.
**Journal of Machine Learning Research**, [s. l.], v. 3, p. 1157-1182, 2003.

HARRIS, Fredric J. On the use of windows for harmonic analysis with the discrete
Fourier transform. **Proceedings of the IEEE**, [s. l.], v. 66, n. 1, p. 51-83, 1978.

KAUFMAN, Shachar et al. Leakage in data mining: formulation, detection, and
avoidance. **ACM Transactions on Knowledge Discovery from Data**, New York, v. 6,
n. 4, art. 15, p. 1-21, dez. 2012.

LEI, Yaguo et al. Applications of machine learning to machine fault diagnosis: a
review and roadmap. **Mechanical Systems and Signal Processing**, [s. l.], v. 138,
art. 106587, 2020.

RANDALL, Robert Bond. **Vibration-based condition monitoring**: industrial, aerospace
and automotive applications. Chichester: John Wiley & Sons, 2011.

RANDALL, Robert Bond; ANTONI, Jérôme. Rolling element bearing diagnostics — a
tutorial. **Mechanical Systems and Signal Processing**, [s. l.], v. 25, n. 2,
p. 485-520, 2011.

SEHRI, Mohamed; DUMOND, Philippe; BOUCHARD, Martin. University of Ottawa constant and
variable speed electric motor vibration and acoustic fault signature dataset. **Data
in Brief**, [s. l.], v. 53, art. 109327, 2024.

SOKOLOVA, Marina; LAPALME, Guy. A systematic analysis of performance measures for
classification tasks. **Information Processing & Management**, [s. l.], v. 45, n. 4,
p. 427-437, 2009.

STROBL, Carolin et al. Bias in random forest variable importance measures:
illustrations, sources and a solution. **BMC Bioinformatics**, London, v. 8, art. 25,
2007.

VARMA, Sudhir; SIMON, Richard. Bias in error estimation when using cross-validation
for model selection. **BMC Bioinformatics**, London, v. 7, art. 91, 2006.

WELCH, Peter D. The use of fast Fourier transform for the estimation of power
spectra: a method based on time averaging over short, modified periodograms. **IEEE
Transactions on Audio and Electroacoustics**, [s. l.], v. 15, n. 2, p. 70-73, jun.
1967.

*(Lista completa de referências, incluindo as citadas apenas no notebook — NYQUIST,
1928; SHANNON, 1949; OPPENHEIM; SCHAFER, 2010; PEDREGOSA et al., 2011; VIRTANEN et
al., 2020 — está na Seção 11 de `motor_fault_monitoring_poc_v3.ipynb`.)*
