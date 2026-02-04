# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Deacu Octavian-Stefan  
**Data:** 04.02.2026  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
project-name/
├── README.md
├── docs/
│   └── datasets/          # descriere seturi de date, surse, diagrame
├── data/
│   ├── raw/               # date brute
│   ├── processed/         # date curățate și transformate
│   ├── train/             # set de instruire
│   ├── validation/        # set de validare
│   └── test/              # set de testare
├── src/
│   ├── preprocessing/     # funcții pentru preprocesare
│   ├── data_acquisition/  # generare / achiziție date (dacă există)
│   └── neural_network/    # implementarea RN (în etapa următoare)
├── config/                # fișiere de configurare
└── requirements.txt       # dependențe Python (dacă aplicabil)
```

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** [Descriere sursă date - ex: senzori robot, dataset public, simulare]
* **Modul de achiziție:** ☐ Senzori reali / ☐ Simulare / ☐ Fișier extern / ☐ Generare programatică
* **Perioada / condițiile colectării:** [Ex: Noiembrie 2024 - Ianuarie 2025, condiții experimentale specifice]

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** [Ex: 15,000]
* **Număr de caracteristici (features):** [Ex: 12]
* **Tipuri de date:** ☐ Numerice / ☐ Categoriale / ☐ Temporale / ☐ Imagini
* **Format fișiere:** ☐ CSV / ☐ TXT / ☐ JSON / ☐ PNG / ☐ Altele: [...]

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| flange | categorial | - | - | - |
| sleeve | categorial | – | - | - |
| shaft | categorial | - | - | - |
| FLANGE_OD | numeric | mm | Diametrul exterior (flange) | - |
| FLANGE_THK | numeric | mm | Grosime (flange) | - |
| BORE_D | numeric | mm | Diametrul gaurii din mijloc (flange) | - |
| BOLT_D | numeric | mm | Diametrul gaurilor exterioare | - |
| BOLT_COUNT | numeric | - | Numarul de gauri exterioare | - |
| BOLT_DEPTH | numeric | mm | Adancimea gaurilor exterioare | - |
| BOLT_PCD | numeric | mm | Diametrul cercului in jurul caruia sunt puse gaurile exterioare | - |
| SLEEVE_OD | numeric | mm | Diametrul exterior (sleeve) | - |
| SLEEVE_L | numeric | mm | Grosime (sleeve) | - |
| BORE_D | numeric | mm | Diametrul gaurii din mijloc (sleeve) | - |
| CHAMFER_L | numeric | mm | Dimensiunea filetului (sleeve) | - |
| CHAMFER_ANG | numeric | deg | Unghiul filetului (sleeve) | - |
| SHAFT_OD | numeric | mm | Diametrul exterior (shaft) | - |
| SHAFT_L | numeric | mm | Lungimea (shaft) | - |
| BORE_D | numeric | mm | Diametrul gaurii din mijloc (shaft) | - |
| CHAMFER_L | numeric | mm | Lungimea filetului (shaft) | - |
| CHAMFER_ANG | numeric | deg | Unghiul filetului (shaft) | - |
**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard**
* **Min–max și quartile**
* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* Cele 3 clase pot reprezenta piese cu dimensiuni eronate. Spre exemplu gaura din mijloc poate avea diametru mai mare decat intreaga piesa.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
 
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* * Flange: FLANGE_OD > BORE_D, FLANGE_OD - BOLT_D > BOLT_PCD > BORE_D + BOLT_D  
  * Sleeve: SLEEVE_OD > BORE_D, CHAMFER_L < BORE_D
  * Shaft: SHAFT_OD > BORE_D, CHAMFER_L < BORE_D

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 80% – train
* 10% – validation
* 10% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [ ✔ ] Structură repository configurată
- [ ✔ ] Dataset analizat (EDA realizată)
- [ ✔ ] Date preprocesate
- [ ✔ ] Seturi train/val/test generate
- [ ✔ ] Documentație actualizată în README + `data/README.md`

---
