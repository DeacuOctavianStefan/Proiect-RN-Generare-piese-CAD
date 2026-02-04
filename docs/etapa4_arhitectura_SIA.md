# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Deacu Octavian-Stefan  
**Link Repository GitHub**
**Data:** 03.02.2026  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Realizarea unei modelari rapide a unei piese | Piesa este generata în < 30 secunde | RN + Web Service |
| Modificarea unei piese | Se poate creea o clasa noua pentru piesa respectiva apoi se pot face modificari rapide | Data Logging + RN + UI |

**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

#### Cum se calculează 40%:

**Exemplu 1 - Dataset DOAR public în Etapa 3:**
```
Etapa 3: Ați folosit 10,000 samples dintr-o sursa externa (ex: Kaggle)
Etapa 4: Trebuie să generați/achiziționați date astfel încât:
  
Opțiune A: Adăugați 6,666 samples noi → Total 16,666 (6,666/16,666 = 40%)
Opțiune B: Păstrați 6,000 publice + 4,000 generate → Total 10,000 (4,000/10,000 = 40%)
```

**Exemplu 2 - Dataset parțial original în Etapa 3:**
```
Etapa 3: Ați avut deja 3,000 samples generate + 7,000 publice = 10,000 total
Etapa 4: 3,000 samples existente numără ca "originale"
        Dacă 3,000/10,000 = 30% < 40% → trebuie să generați încă ~1,700 samples
        pentru a ajunge la 4,700/10,000 = 47% > 40% ✓
```

**Exemplu 3 - Dataset complet original:**
```
Etapa 3-4: Generați toate datele (simulare, senzori proprii, etichetare manuală - varianta recomandata)
           → 100% original ✓ (depășește cu mult 40% - FOARTE BINE!)
```

#### Tipuri de contribuții acceptate (exemple din inginerie):

Alegeți UNA sau MAI MULTE dintre variantele de mai jos și **demonstrați clar în repository**:

| **Tip contribuție** | **Exemple concrete din inginerie** | **Dovada minimă cerută** |
|---------------------|-------------------------------------|--------------------------|
| **Date sintetice prin metode avansate** | • Simulări CAD | Cod implementare metodă + exemple before/after + justificare hiperparametri + validare pe subset real |

#### Declarație obligatorie în README:

Scrieți clar în acest README (Secțiunea 2):

```markdown
### Contribuția originală la setul de date:

**Total observații finale:** 6000 (după Etapa 3 + Etapa 4)
**Observații originale:** 6000 (100%)

**Tipul contribuției:**
[] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[X] Date sintetice prin metode avansate  

**Descriere detaliată:**
  Am pornit de la un template ideal pentru fiecare clasa/familie, apoi am facut un script care genereaza variatii ale parametrilor acestora si am facut cate 2000 de variante pentru fiecare familie. Acestea au fost impartite pe clasa astfel: 1600 train, 200 test, 200 validation, apoi am mai facut un script pentru generarea de prompt-uri variate
**Locația codului:** `src/data_acquisition/generate_valid_sweep.py , src/data_acquisition/generate_prompt_variants.py`
**Locația datelor:** `data/raw/`

**Dovezi:**
- Grafic comparativ: `docs/generated_vs_real.png`
- Setup experimental: `docs/acquisition_setup.jpg` (dacă aplicabil)
- Tabel statistici: `docs/data_statistics.csv`
```

#### Exemple pentru "contribuție originală":
-Simulări fizice realiste cu ecuații și parametri justificați  
-Date reale achiziționate cu senzori proprii (setup documentat)  
-Augmentări avansate cu justificare fizică (ex: simulare perspective camera industrială)  


#### Atenție - Ce NU este considerat "contribuție originală":

- Augmentări simple (rotații, flips, crop) pe date publice  
- Aplicare filtre standard (Gaussian blur, contrast) pe imagini publice  
- Normalizare/standardizare (aceasta e preprocesare, nu generare)  
- Subset dintr-un dataset public (ex: selectat 40% din ImageNet)


---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**
```
IDLE → ACQUIRE_DATA → PREPROCESS → INFERENCE → DISPLAY/ACT → LOG → [ERROR] → STOP
                ↑______________________________________________|
```

**Exemple concrete per domeniu de inginerie:**

#### A. Monitorizare continuă proces industrial (vibrații motor, temperaturi, presiuni):
```
IDLE → START_ACQUISITION → COLLECT_SENSOR_DATA → BUFFER_CHECK → 
PREPROCESS (filtrare, FFT) → RN_INFERENCE → THRESHOLD_CHECK → 
  ├─ [Normal] → LOG_RESULT → UPDATE_DASHBOARD → COLLECT_SENSOR_DATA (loop)
  └─ [Anomalie] → TRIGGER_ALERT → NOTIFY_OPERATOR → LOG_INCIDENT → 
                  COLLECT_SENSOR_DATA (loop)
       ↓ [User stop / Emergency]
     SAFE_SHUTDOWN → STOP
```

#### B. Clasificare imagini defecte producție (suduri, suprafețe, piese):
```
IDLE → WAIT_TRIGGER (senzor trecere piesă) → CAPTURE_IMAGE → 
VALIDATE_IMAGE (blur check, brightness) → 
  ├─ [Valid] → PREPROCESS (resize, normalize) → RN_INFERENCE → 
              CLASSIFY_DEFECT → 
                ├─ [OK] → LOG_OK → CONVEYOR_PASS → IDLE
                └─ [DEFECT] → LOG_DEFECT → TRIGGER_REJECTION → IDLE
  └─ [Invalid] → ERROR_IMAGE_QUALITY → RETRY_CAPTURE (max 3×) → IDLE
       ↓ [Shift end]
     GENERATE_REPORT → STOP
```

#### C. Predicție traiectorii robot mobil (AGV, AMR în depozit):
```
IDLE → LOAD_MAP → RECEIVE_TARGET → PLAN_PATH → 
VALIDATE_PATH (obstacle check) →
  ├─ [Clear] → EXECUTE_SEGMENT → ACQUIRE_SENSORS (LIDAR, IMU) → 
              RN_PREDICT_NEXT_STATE → UPDATE_TRAJECTORY → 
                ├─ [Target reached] → STOP_AT_TARGET → LOG_MISSION → IDLE
                └─ [In progress] → EXECUTE_SEGMENT (loop)
  └─ [Obstacle detected] → REPLAN_PATH → VALIDATE_PATH
       ↓ [Emergency stop / Battery low]
     SAFE_STOP → LOG_STATUS → STOP
```

#### D. Predicție consum energetic (turbine eoliene, procese batch):
```
IDLE → LOAD_HISTORICAL_DATA → ACQUIRE_CURRENT_CONDITIONS 
(vânt, temperatură, demand) → PREPROCESS_FEATURES → 
RN_FORECAST (24h ahead) → VALIDATE_FORECAST (sanity checks) →
  ├─ [Valid] → DISPLAY_FORECAST → UPDATE_CONTROL_STRATEGY → 
              LOG_PREDICTION → WAIT_INTERVAL (1h) → 
              ACQUIRE_CURRENT_CONDITIONS (loop)
  └─ [Invalid] → ERROR_FORECAST → USE_FALLBACK_MODEL → LOG_ERROR → 
                ACQUIRE_CURRENT_CONDITIONS (loop)
       ↓ [User request report]
     GENERATE_DAILY_REPORT → STOP
```

**Notă pentru proiecte simple:**
Chiar dacă aplicația voastră este o clasificare simplă (user upload → classify → display), trebuie să modelați fluxul ca un State Machine. Acest exercițiu vă învață să gândiți modular și să anticipați toate stările posibile (inclusiv erori).

**Legendă obligatorie (scrieți în README):**
```markdown
### Justificarea State Machine-ului ales:

Am ales arhitectura [descrieți tipul: monitorizare continuă / clasificare la senzor / 
predicție batch / control în timp real] pentru că proiectul nostru [explicați nevoia concretă 
din tabelul Secțiunea 1].

Stările principale sunt:
1. [STARE_1]: [ce se întâmplă aici - ex: "achiziție 1000 samples/sec de la accelerometru"]
2. [STARE_2]: [ce se întâmplă aici - ex: "calcul FFT și extragere 50 features frecvență"]
3. [STARE_3]: [ce se întâmplă aici - ex: "inferență RN cu latență < 50ms"]
...

Tranzițiile critice sunt:
- [STARE_A] → [STARE_B]: [când se întâmplă - ex: "când buffer-ul atinge 1024 samples"]
- [STARE_X] → [ERROR]: [condiții - ex: "când senzorul nu răspunde > 100ms"]

Starea ERROR este esențială pentru că [explicați ce erori pot apărea în contextul 
aplicației voastre industriale - ex: "senzorul se poate deconecta în mediul industrial 
cu vibrații și temperatură variabilă, trebuie să gestionăm reconnect automat"].

Bucla de feedback [dacă există] funcționează astfel: [ex: "rezultatul inferenței 
actualizează parametrii controlerului PID pentru reglarea vitezei motorului"].
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | - | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/train_llm_json_with_metrics.py` sau folder dedicat | - | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit | - | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [ ✔ ] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [ ✔ ] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [ ✔ ] Include minimum 40% date originale în dataset-ul final
- [ ✔ ] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [ ✔ ] Arhitectură RN definită și compilată fără erori
- [ ✔ ] Model poate fi salvat și reîncărcat
- [ ✔ ] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ✔ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [ ✔ ] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [ ✔ ] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
Proiect-RN-Generare-piese-CAD-main/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/ 
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [ ✔ ] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [ ✔ ] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [ ✔ ] Cod generare/achiziție date funcțional și documentat
- [ ✔ ] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [ ✔ ] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [ ✔ ] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [ ✔ ] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [ ✔ ] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [ ✔ ] Produce minimum 40% date originale din dataset-ul final
- [ ✔ ] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [ ✔ ] Documentație în `src/data_acquisition/README.md` cu:
  - [ ✔ ] Metodă de generare/achiziție explicată
  - [ ✔ ] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [ ✔ ] Justificare relevanță date pentru problema voastră
- [ ✔ ] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [ ✔ ] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [ ✔ ] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [ ✔ ] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [ ✔ ] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [ ✔ ] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


