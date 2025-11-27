# Proiect-RN-Generare-piese-CAD

# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Deacu Octavian-Stefan]  
**Data:** [20.11.2025]  

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

* **Origine:** [dataset public]
* **Modul de achiziție:** ☐ Fisier extern si generare programatica
* **Perioada / condițiile colectării:** [Ex: Noiembrie 2024 - Ianuarie 2025, condiții experimentale specifice]

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 1000
* **Număr de caracteristici (features):** 6
* **Tipuri de date:** ☐ Numerice / ☐ Categoriale / ☐ Temporale
* **Format fișiere:** ☐ CSV / ☐ TXT / ☐ JSON/ ☐ SLDPRT

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** |  **Tip**   | **Unitate** | **Descriere** | **Domeniu valori** |
|--------------------|------------|-------------|---------------|--------------------|
|       shaft        | categorial |      -      |      ...      |         -          |
|      bearing       | categorial |      -      |      ...      |         -          |
|     gearwheel      | categorial |      -      |      ...      |         -          |
|   bearing_flanged  | categorial |      -      |      ...      |         -          |
|   bearing_sleeve   | categorial |      -      |      ...      |         -          |
|      coupling      | categorial |      -      |      ...      |         -          |
|    fisier_sldprt   |  temporal  |      -      |      ...      |         -          |
|       shaft        | categorial |      -      |      ...      |         -          |

**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic
    Verificarea si validarea integritatii datelor (extensia corecta, lipsa erorilor)
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

* [exemplu] O piesa este necotata
* [exemplu] Un fisier introdus nu are extensia sldprt
* [exemplu] Schita/Schitele unei piese nu este/sunt constransa/constranse

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile
* **Conversia pieselor fără „feature tree” → etichetate separat**

### 4.2 Transformarea caracteristicilor

| **Caracteristică** |   **Tip**   | **Procesare** |
|--------------------|-------------|---------------|
|    bounding_box    |   numeric   |    min-max    |
|    face_number     |   numeric   | standardizare |
|    edge_number     |   numeric   |  log-scaling  |
|    feature_number  |   numeric   |  normalizare  |
|   categorie_piesa  |   numeric   |  normalizare  |



* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test
* Stratificare dupa clasa piesei

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

- [ ] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
