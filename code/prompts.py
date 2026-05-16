

# classification
system_prompt_for_classification = """
You are an expert at classifying text into predefined categories. """

response_schema_for_classification = {
    "title": "TextClassificationResult",
    "type": "object",
    "description": "Result schema for classifying text into predefined categories",
    "properties": {
        "type_legislation": {
            "type": "object",
            "description": "The category that best fits the provided text",
            "properties": {
                "name": {"type": "string", "description": "Name of the category"},
                "confidence": {"type": "number", "description": "A confidence score between 0 and 100 indicating the certainty of the classification"}
            },
        },
        "type_category": {
            "type": "object",
            "description": "The category that best fits the provided text",
            "properties": {
                "name": {"type": "string", "description": "Name of the category"},
                "confidence": {"type": "number", "description": "A confidence score between 0 and 100 indicating the certainty of the classification"}
            },
        }
    },
    "required": ["type_legislation", "type_category"]
}



# open domain detection
system_prompt_for_odd = """
Extrahuj TYPY ENTIT (uzlov) a TYPY VZTAHOV z pravneho alebo financneho textu.

# PRAVIDLA
- Extrahuj iba TYPY, nie instancie
- Pouzi vseobecne, ale rozlisitelne typy
- Bez diakritiky
- Uzly: PascalCase | Vztahy: UPPER_SNAKE_CASE
- Nehalucinuj, zahrn len to, co dava zmysel v pravnom kontexte
- Vyber spravny vyznam slova (napr. sluzba != sluha)
- Pouzivaj jedine Slovensky jazyk.

# COVERAGE
Zachyt VSETKO relevantne:
- pravne akty: Zakon, Paragraf, Odsek, Rozhodnutie
- subjekty: Osoba, PravnickaOsoba, Organ, SpravcaDane
- financne: Dan, Poplatok, Prijem, Zavazok
- abstrakcie: Povinnost, Pravo, Narok, Sankcia, Lehota
- mechanizmy: Koeficient, Vypocet, Sadzba
- cas: ZdanovacieObdobie, KalendarnyRok

# LOGIKA
- Zachyt aj implicitne koncepty
- Ak je pojem centralny (opakujuci sa), zahrn ho
- Minimalizuj duplicity
- Pouzi konzistentne nazvy

# VZTAHY
- Pouzi vseobecne vztahy (napr. UPRAVUJE, MA_POVINNOST, MA_PRAVO)
- Zachyt aj ciel alebo protistranu akcie
"""


response_schema_for_odd = {
            "title": "DocumentOpenDomainDetectionResult",
            "type": "object",
            "description": "Result schema for open domain detection of entities and relationships from text",
            "properties": {
                "node_types": {
                    "type": "array",
                    "description": "List of canonical node type labels",
                    "items": {
                        "type": "string",
                        "description": "The label/type of the entity (e.g., Person, Organization)"
                    }
                },
                "relationship_types": {
                    "type": "array",
                    "description": "List of canonical relationship types",
                    "items": {
                        "type": "string",
                        "description": "The type of relationship between nodes (e.g., WORKS_FOR, LOCATED_IN)"
                    }
                }
            },
            "required": ["node_types", "relationship_types"]
        }




# schema refinement
system_prompt_for_schema_refinement = """
# ROLE
Si expertný ontológ a dátový inžinier špecializujúci sa na znalostné grafy. Tvojou úlohou je vyčistiť, normalizovať a deduplikovať schému entít a vzťahov.

# CIELE
1. **Deduplikácia a sémantické zjednotenie**: Zlúč synonymá a sémanticky blízke typy do jedného kanonického typu.
2. **Normalizácia formátu**: 
    - Uzly: PascalCase (napr. DanovePriznanie).
    - Vzťahy: UPPER_SNAKE_CASE (napr. MA_POVINNOST).
    - **STRIKTNÉ PRAVIDLO**: Odstráň všetku diakritiku (á->a, č->c, atď.) zo všetkých názvov.
3. **Ontologická integrita (KRITICKÉ)**:
    - NIKDY nezlučuj entity z rôznych kategorií: Aktér (Osoba) vs. Proces/Dokument (Ziadost), Miesto (Kraj) vs. Politický útvar (Stat).
    - NIKDY nezlučuj protichodné vzťahy (VSTUPUJE_DO vs. VYSTUPUJE_Z).
4. **Generalizácia**: Príliš špecifické uzly nahraď všeobecnejšími (napr. "Sluha" -> "Osoba"), ak to neznižuje zrozumiteľnosť právneho/biznisového kontextu.

# PRAVIDLÁ PRE ZÁKLADNÚ ONTOLÓGIU
- Ak je poskytnutá Základná ontológia, má prednosť v pomenovaní.
- Ak surový typ (Open Domain) zodpovedá typu v základnej ontológii, mapuj ho naň.
- Ak je surový typ unikátny a dôležitý, pridaj ho ako nový typ.
"""


response_schema_for_schema_refinement = {
            "title": "DocumentSchemaRefinementResult",
            "type": "object",
            "description": "Result schema for schema refinement of entities and relationships from text",
            "properties": {
                "node_types": {
                    "type": "array",
                    "description": "List of canonical node type labels",
                    "items": {
                        "type": "string",
                        "description": "The label/type of the entity (e.g., Person, Organization)"
                    }
                },
                "relationship_types": {
                    "type": "array",
                    "description": "List of canonical relationship types",
                    "items": {
                        "type": "string",
                        "description": "The type of relationship between nodes (e.g., WORKS_FOR, LOCATED_IN)"
                    }
                },
                "merge_log": {
                    "type": "object",
                    "description": "Log of merged types mapping canonical types to their original variants",
                    "properties": {
                        "node_types": {
                            "type": "object",
                            "description": "Mapping of canonical node types to lists of original types that were merged",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "relationship_types": {
                            "type": "object",
                            "description": "Mapping of canonical relationship types to lists of original types that were merged",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": ["node_types", "relationship_types"]
                }
            },
            "required": ["node_types", "relationship_types", "merge_log"]
        }

# - "podľa odseku 1" means odsek of this paragraph
### CURRENT LEGAL CONTEXT
# Use this as authoritative context for all local references:
# {path_str if path_str else "UNKNOWN"}

# If context is UNKNOWN, do not invent paragraph numbers.
# If text says "podla odseku 1", interpret it inside CURRENT LEGAL CONTEXT.
# Canonical legal IDs:
# - Paragraf: "Paragraf § X"
# - Odsek with known paragraph: "Paragraf § X Odsek Y"
# - Odsek without known paragraph: "Odsek Y"
# - Pismeno with known paragraph/odsek: "Paragraf § X Odsek Y Pismeno a)"

# - add concise, explicit semantic clarification into relationship properties (`MA_NAROK_NA` → `add: vratenie dane`), use key `add`


# schema driven extraction
system_prompt_for_sde = """
Si expert na extrakciu znalostneho grafu zo slovenskeho pravno-financneho textu (NER + RE).
Striktne dodrziavas poskytnutu schemu a extrahujes IBA to, co je preukazatelne z textu.

### ZAKLADNE PRAVIDLA
- IBA typy zo schemy. Ziadne premenovania.
- Slovensky jazyk, BEZ DIAKRITIKY vo vsetkych id (a->a, c->c, s->s, z->z, y->y).
- Preferuj specificky typ pred genericky, ALE generic typ (TYKA_SA, UPRAVUJE, OBSAHUJE)
  je casto SPRAVNA volba - pouzi ho ked sa hodi semanticky.
- Zlozite vety rozkladaj na multi-hop trojice.

###############################################
### GRANULARITA UZLOV
###############################################

Paragraf:                                      "Paragraf § 16"
Odsek vnutri AKTUALNEHO paragrafu (kontext)    "Odsek 3" (lokalna forma)
Odsek z INEHO paragrafu ("§ 8 ods. 3")         "Paragraf § 8 Odsek 3"
Pismeno lokalne:                               "Odsek 2 Pismeno a)"
Pismeno cross-paragraph:                       "Paragraf § 54 Odsek 2 Pismeno a)"

Strana N = cislo strany, NIKDY nie Paragraf.

### QUALIFIER SUFFIX (povinne, ak entita ziska vyznam az cez kvalifikator)
"oslobodenie od dane podla odseku 3"      -> "Oslobodenie Od Dane Podla Odseku 3"
"obrat podla odseku 15"                   -> "Obrat Podla Odseku 15"
"pohonne latky ine ako v odseku 12"       -> "Pohonne Latky Ine Ako V Odseku 12"
NIKDY len hole "Oslobodenie Od Dane" ak text dava kvalifikator.

### ID FORMAT
- Title Case, oddelene medzerou
- Sumy/hodnoty: verbatim z textu, zachovaj jednotky a velke/male pismena tak ako v texte:
    "430 eur na osobu"      -> "430 eur na osobu"
    "suma zlavy"            -> "Suma Zlavy"
- Aktor nody pouzi tak ako v texte: "Platitel", "Ziadatel", "Zdanitelna Osoba",
  "Danovy Urad", "Financne Riaditelstvo", "Colny Urad" (kazdy je samostatny uzol)

###############################################
### VZTAHY - PRAVIDLO EVIDENCIE
###############################################
Pre KAZDY vztah do pola 'evidence' vypisz doslovny fragment textu (5-15 slov),
ktory ho podporuje. Bez doslovneho fragmentu hranu NEVYTVOR.

ROZPOCET HRAN: ciel ~1.0x pocet uzlov (gold dataset ma priemer 0.9x).
NEHALUCINUJ aby si dosiahol cisla; ale ak je hrana v texte explicitne -> ZAHRN ju.

TVRDY LIMIT: pocet hran <= 1.2x pocet uzlov. Ak by si prekrocil, ZMAZ najslabsie
hrany BEZ explicitneho textoveho triggera. Hrany s evidence ako "kontext"
alebo "vyplyva" NIE SU explicitne -> ZMAZ.

###############################################
### KATALOG VZTAHOV (s realnymi gold patternami)
###############################################

================================================================================
TYKA_SA  --- thematic "is about" konektor (gold: 162x, druhy najcastejsi!)
================================================================================
Pouzij ked aktivita/doklad/zaznam/podmienka/oprava JE O nejakom predmete.
Trigger: "tyka sa", "v veci", "ohladom", "o" + akuzativ, ale aj IMPLICITNE
ked je vztah jasne thematicky.

GOLD VZORY (PAR sirec_typ + ciel_typ):
  Cinnost     -[TYKA_SA]-> Tovar          "Dodanie Tovaru" sa tyka tovaru
  Cinnost     -[TYKA_SA]-> Dan            "Vratenie Dane Z DPH"
  Cinnost     -[TYKA_SA]-> Osoba          "Ina Preprava Tovaru" sa tyka inej osoby
  Cinnost     -[TYKA_SA]-> Cinnost        "Uplatnenie Vratenej Dane V Ziadosti"
  Cinnost     -[TYKA_SA]-> SpravcaDane    "Zmena Cinnosti" -> Danovy Urad
  Rozhodnutie -[TYKA_SA]-> Cinnost        "Rozhodnutie O Nepovoleni Uplatnovania"
  Doklad      -[TYKA_SA]-> Cinnost/Osoba  "Dovozny Doklad" sa tyka dovozu
  Podmienka   -[TYKA_SA]-> PravnickaOsoba "Dodanie Pre Pravnicku Osobu Ktora..."
  Oprava      -[TYKA_SA]-> Dan            "Uprava Odpocitanej Dane"
  Zaznam      -[TYKA_SA]-> Hodnota        "Udaje O Aktualnom Schodku..."
  Hodnota     -[TYKA_SA]-> Tovar
  PravnyPredpis -[TYKA_SA]-> Oznamenie    "Ustanovenia Colnych Predpisov..."

NEPOUZI MA_OBSAH ked ide o thematicke spojenie. MA_OBSAH = "ma textovy obsah".

================================================================================
JE_PODLA  --- pravne grounding (gold: 233x, najcastejsi!)
================================================================================
Pouzij ked entita ziska validitu/zaklad od konkretneho paragrafu/odseku/pismena.
Trigger: "podla § X", "podla odseku X", "v zmysle § X", "podla zakona c. X".

SMER: Entita -> Paragraf/Odsek/Pismeno

GOLD VZORY:
  ZdanitelnaOsoba -[JE_PODLA]-> Pismeno      14x! "platitel podla odseku 1 pism. g)"
  Cinnost         -[JE_PODLA]-> Pismeno      "odoslanie podla odseku 1 pism. a)"
  Oprava          -[JE_PODLA]-> Odsek/Paragraf
  IdentifikacneCislo -[JE_PODLA]-> Paragraf
  Rozhodnutie     -[JE_PODLA]-> Odsek
  OslobodenieOdDane -[JE_PODLA]-> Paragraf   "oslobodenie podla § 30"
  Obrat           -[JE_PODLA]-> Odsek        "obrat podla odseku 15"
  Tovar           -[JE_PODLA]-> Odsek        s qualifier suffixom

NEPOUZI ODKAZUJE_NA pre JE_PODLA pripady. ODKAZUJE_NA je medzi DVOMI Odsekmi.

================================================================================
OBSAHUJE  --- strukturalna hierarchia (gold: 72x)
================================================================================
Pouzij PRE VSETKY strukturalne containmenty, ktore su v texte viditelne:
  Zakon       -[OBSAHUJE]-> Paragraf
  Paragraf    -[OBSAHUJE]-> Odsek
  Odsek       -[OBSAHUJE]-> Pismeno
  Priloha     -[OBSAHUJE]-> Ziadost/Tlacivo

GOLD VZORY:
  "Zakon O Dani Z Pridanej Hodnoty" -[OBSAHUJE]-> "Paragraf § 61"
  "Paragraf § 84a"                  -[OBSAHUJE]-> "Paragraf § 84a Odsek 2"
  "Priloha C 4"                     -[OBSAHUJE]-> "Ziadost O Vratenie Dane..."
  "Paragraf § 8 Odsek 1"            -[OBSAHUJE]-> "Paragraf § 8 Odsek 1 Pismeno c)"

PRAVIDLO: ak v texte vidis Paragraf a aj Odseky v nom, vytvor OBSAHUJE.
Ak vidis Zakon a Paragraf -> OBSAHUJE.
NEBOJ sa vytvarat strukturalne hrany aj ked obsah Odseku detailne nerozoberas.

ALTERNATIVY:
  MA_ODSEK    Paragraf -> Odsek    (synonym OBSAHUJE; pouzi ked text doslova "obsahuje odseky" alebo "ma odsek")
  MA_PISMENO  Odsek -> Pismeno     (analogicky)

================================================================================
UPRAVUJE  --- regulacia (gold: 36x)
================================================================================
Pouzij ked Paragraf/Odsek/Zakon REGULUJE/USTANOVUJE temu.
Trigger: "upravuje", "ustanovuje", "predmet", "tento paragraf upravuje".

GOLD VZORY:
  Paragraf -[UPRAVUJE]-> Konanie/Cinnost/Pravo/Povinnost
  Odsek    -[UPRAVUJE]-> Tema regulacie

NEZAKAZ UPRAVUJE! Je legitimnym typom. Ale netreba ho pouzivat pre KAZDY pojem
v paragrafe -- skor pre HLAVNE temy.

================================================================================
VYMEDZUJE / DEFINUJE / ROZUMIE_SA  --- definicie (gold: VYMEDZUJE 25x, ROZUMIE_SA 17x)
================================================================================
VYMEDZUJE  - text DOSLOVA "vymedzuje", "vymedzuje sa". Paragraf -> pojem.
DEFINUJE   - text DOSLOVA "definuje", "je definovany".
ROZUMIE_SA - text DOSLOVA "rozumie sa", "na ucely tohto zakona sa rozumie".

NEPOUZIVAJ VYMEDZUJE ako catch-all! Ak text nepise jedno z tychto sloves
doslova, MOZNO sa hodi UPRAVUJE alebo TYKA_SA alebo MA_NAZOV.

================================================================================
JE_TYPOM  --- taxonomia "X has subtype Y"
================================================================================
SMER: Nadradeny (vseobecny pojem) -> Podradeny (specificky subtyp).
Pouzi ked text vymenuje typy/druhy: "X moze byt: a) ..., b) ..., c) ..."

GOLD VZORY:
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Vycvikove Vozidlo
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Predvadzacie Alebo Testovacie...
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Nahradne Osobne Motorove Vozidlo

ROZDIEL OD POVAZUJE_SA_ZA:
  JE_TYPOM       = taxonomia "X je druh/typ Y"; X je hyponymum Y
  POVAZUJE_SA_ZA = legalna fikcia "X sa povazuje za Y"; klasifikacia entity A do kategorie B
                   "Vydavky sa povazuju za prechodne polozky"

ALTERNATIVY:
  JE_DRUHOM   reverzny smer: "X je druhom Y" -> Specific -> General
              (pouzi len ak text doslova "je druhom")
  ZAHRNUJE    "X zahrna: Y, Z" -> X (sup) -> Y/Z (sub)
              casto pouzivane ekvivalentne k JE_TYPOM, ak je tema mnozinova

================================================================================
NEVZTAHUJE_SA_NA / NEPLATI_PRE / NEMA_NAROK_NA  --- negacie (gold: 42x)
================================================================================
SMER: Pravidlo/Koncept -> Vyluceny objekt.

NEVZTAHUJE_SA_NA  "nevztahuje sa na", "nezahrna sa", "neprihliada sa"
  "Do zakladu dane sa nezahrna zaloha"
  -> Zaklad Dane -[NEVZTAHUJE_SA_NA]-> Zaloha
  NIE Zaloha -[NEVZTAHUJE_SA_NA]-> Zaklad Dane!

NEPLATI_PRE       "neplati pre", "tieto ustanovenia neplatia"
NEMA_NAROK_NA     "nema narok na vratenie"
NESPLNA_PODMIENKY "nesplna podmienky"

================================================================================
VZTAHUJE_SA_NA / APLIKUJE_SA_NA  (gold: VZTAHUJE_SA_NA 81x; APLIKUJE_SA_NA 0x!)
================================================================================
PRAVIDLO: VZDY pouzi VZTAHUJE_SA_NA. NIKDY APLIKUJE_SA_NA (gold ho nepouziva).

SMER: Pravidlo/Koncept -> Aplikovany objekt
GOLD VZORY:
  Prevadzkovanie Autoskoly -[VZTAHUJE_SA_NA]-> Osobne Motorove Vozidlo
  Nahradne Vozidlo         -[VZTAHUJE_SA_NA]-> Zakaznik Platitela

================================================================================
JE_OSLOBODENE_OD / OSLOBODZUJE_OD  (gold: JE_OSLOBODENE_OD)
================================================================================
SMER: Cinnost/Dodanie/Tovar/PravnickaOsoba -> Dan

================================================================================
NACHADZA_SA_V vs MA_MIESTO  --- lokacia
================================================================================
NACHADZA_SA_V  STATIC location entity:
  Sidlo NACHADZA_SA_V Clensky Stat
  Elektronicky Portal NACHADZA_SA_V Clensky Stat
  Tovar NACHADZA_SA_V Tuzemsko

MA_MIESTO      action's site / dodanie's location:
  Podanie MA_MIESTO Elektronicky Portal
  Dodanie MA_MIESTO Tuzemsko

MA_MIESTO_DODANIA  specific: dodanie -> Lokacia (clensky stat / treti stat)

================================================================================
MA_PODMIENKU  --- conditions (gold: 71x, tretia najcastejsia!)
================================================================================
Trigger: "ak", "za podmienky", "podmienkou je", "v pripade ze".
SMER: Pravidlo/Cinnost -> Podmienka

GOLD pattern: ked je v texte explicitna podmienka, vytvor Podmienka uzol
a spoji ho MA_PODMIENKU.

================================================================================
VYCHADZA_Z  --- odvodenie/zdroj (gold: 45x)
================================================================================
Pouzi pre vypocty, derivacie, zdroje hodnot.
Trigger: "vychadza z", "rozdiel medzi", "vypocita sa z", "zaklada sa na".

GOLD VZORY:
  Vypocet/Suma     -[VYCHADZA_Z]-> Suma/Hodnota/Doklad
  Cinnost          -[VYCHADZA_Z]-> Doklad/Uzemie
  Zaznam           -[VYCHADZA_Z]-> Doklad   "Udaje Z Faktury" -> "Faktura"
  Lehota           -[VYCHADZA_Z]-> Datum    "Lehota najmenej 15 dni" -> "Den Dorucenia"

================================================================================
ZAHRNUJE  --- vyclenenie typov (gold: 26x globalne)
================================================================================
Pouzij ked text doslova "zahrna", "patri sem", "obsahuje druhy":
  "Osobne motorove vozidlo zahrna: vycvikove, predvadzacie, nahradne"
  -> Osobne Motorove Vozidlo -[ZAHRNUJE]-> Vycvikove Vozidlo
  -> Osobne Motorove Vozidlo -[ZAHRNUJE]-> Predvadzacie Vozidlo
ROVNOCENNE k JE_TYPOM ked text pouziva slovo "zahrna" miesto "je typom".
SMER: Nadradeny -> Clenovia

JE_SUCASTOU   Cast -> Celok (reverz; text "je sucastou", "patri do")
PATRI_DO      Specific -> Kategoria

================================================================================
JE_DRUHOM  --- reverzny smer k JE_TYPOM (gold: 5x)
================================================================================
Trigger: text DOSLOVA "je druhom", "je druhom napriklad".
SMER: Specific -> General  (opacny od JE_TYPOM)
  "Vycvikove vozidlo je druhom osobneho motoroveho vozidla"
  -> Vycvikove Vozidlo -[JE_DRUHOM]-> Osobne Motorove Vozidlo

================================================================================
MA_VLASTNOST  --- kvalitativny atribut (gold: 17x)
================================================================================
Pouzij ked entita ma kvalitativnu vlastnost / sposob pouzitia.
Trigger: "vylucne", "uplne", "vyhradne", "uziva sa ako"
GOLD VZORY:
  Predvadzacie Vozidlo -[MA_VLASTNOST]-> Vylucne Pouzitie
  Nahradne Vozidlo     -[MA_VLASTNOST]-> Vylucne Pouzitie
  Tovar                -[MA_VLASTNOST]-> Vlastnost

ROZDIEL OD MA_PODMIENKU:
  MA_VLASTNOST  inherentny atribut entity
  MA_PODMIENKU  podmienka pre uplatnenie pravidla

================================================================================
NASTAVA_PRI / VZNIKA_PRI  --- udalost spustaca (gold: NASTAVA_PRI 17x)
================================================================================
NASTAVA_PRI:  "X nastava pri Y" -> X (vysledok) -> Y (spustac)
VZNIKA_PRI:   "X vznika pri Y" -> analogicky

================================================================================
MA_NAZOV  --- nadpis paragrafu (gold pattern, casto missed)
================================================================================
Ked text zacina "§ 48d  Oslobodenie od dane v osobitnom sklade":
  Paragraf § 48d -[MA_NAZOV]-> Oslobodenie Od Dane V Osobitnom Sklade

NIE VYMEDZUJE alebo UPRAVUJE pre nadpis paragrafu - to je MA_NAZOV.

================================================================================
ATRIBUTY MA_*  (poradie podla frekvencie v gold)
================================================================================
MA_MIESTO (34)    actor/object -> miesto akcie
MA_DATUM (33)     entita -> konkretny datum / Datum uzol
MA_LEHOTU (32)    pravidlo/akcia -> Lehota
MA_OBDOBIE (28)   pravidlo -> ZdanovacieObdobie / KalendarnyRok
MA_UCEL (22)      akcia -> ucel
MA_SUMU (20)      koncept -> Suma; "vo vyske", "suma"
MA_OBSAH (18)     dokument -> obsahuje text/Zaznam (NIE pre thematic - to je TYKA_SA!)
MA_VLASTNOST (17) entita -> Hodnota/atribut "vylucne pouzitie"
MA_DOKLAD (10)    akcia -> Doklad (pouzi STRIDMO, nie pre kazdy doklad)
MA_DOVOD (14)     Rozhodnutie -> Dovod
MA_HODNOTU        koncept -> bezrozmerna Hodnota
MA_SADZBU         dan -> SadzbaDane
MA_NAZOV          paragraf/dokument -> nazov
MA_STATUS         entita -> Status
MA_POVINNOST (39) entita -> Povinnost
MA_PRAVO          entita -> Pravo
MA_NAROK_NA       entita -> Narok

================================================================================
ACTOR -> ACTION (povinne dekomponovat actor-vety)
================================================================================
VYKONAVA (40)  actor -> akcia (genericky)
PODAVA         actor -> Ziadost/Odvolanie/Vyhlasenie
PREDKLADA      actor -> Doklad (predkladanie)
DORUCUJE (14)  actor -> Doklad/Oznamenie (dorucenie)
VYDAVA         organ -> Rozhodnutie/Opatrenie
OZNAMUJE       actor -> Oznamenie (nie VYKONAVA!)
PRIJIMA        actor -> objekt
NADOBUDA       Platitel/ZdanitelnaOsoba -> Tovar/Majetok
DODAVA         dodavatel -> Tovar
POSKYTUJE      poskytovatel -> Sluzba/Nahradne Vozidlo (NIE -> Zakaznik!)
ZAPLATI        Platitel -> Dan/Suma
USKUTOCNUJE    actor -> Cinnost
REGISTRUJE     Danovy Urad -> Subjekt
PRIDELUJE      Danovy Urad -> IdentifikacneCislo
VIES_ZAZNAMY_O actor -> Hodnota/Cinnost
UCHOVAVA       actor -> Doklad
ZRUSUJE (14)   actor -> Povolenie/Rozhodnutie (POUZI ZRUSUJE, NIE "RUSI")
OPRAVUJE       actor -> Hodnota/Vypocet

================================================================================
ZIVOTNY CYKLUS
================================================================================
PLATI_OD       pravidlo -> Datum
PLATI_DO       pravidlo -> Datum
MA_UCINOK      Rozhodnutie -> Hodnota
MA_ODKLADNY_UCINOK
ZANIKA         entita -> "Zanik" event
PRECHADZA_NA   Pravo/Povinnost -> Novy nositel (PravnyNastupca)

###############################################
### SMER VZTAHOV (najcastejsie obracane)
###############################################
TYKA_SA           Aktivita/Doklad/Zaznam -> Subject Matter
JE_PODLA          Entita -> Paragraf/Odsek
OBSAHUJE          Vyssi struktura -> Nizsia
ODKAZUJE_NA       Odsek A -> Odsek B (krizovy odkaz vnutri zakona)
                  (POUZI len ked je v texte "podla odseku X" a oba su Odsek nody)
NEVZTAHUJE_SA_NA  Pravidlo -> Vyluceny objekt
VZTAHUJE_SA_NA    Pravidlo -> Aplikovany objekt
JE_OSLOBODENE_OD  Cinnost/Tovar -> Dan
POSKYTUJE         Poskytovatel -> Sluzba/Vec (NIE Sluzba -> Prijemca!)
POVAZUJE_SA_ZA    Vec A -> Klasifikovana B
JE_TYPOM          Nadradeny -> Podradeny (taxonomia)
VYCHADZA_Z        Odvodena -> Zdroj
MA_*              Vlastnik -> Hodnota

###############################################
### FEW-SHOT PRIKLADY (vsetky z realnych gold edges)
###############################################

PRIKLAD 1 - TYKA_SA + struktura:
Text: "§ 67 (3) Doklad o oprave dane sa tyka opravy zakladu dane.
       Doklad obsahuje zaznam o povodnom doklade."
Triples:
  Paragraf § 67   -[OBSAHUJE]->   Odsek 3
       evidence: "§ 67 (3)"
  Paragraf § 67   -[MA_NAZOV]->   Doklad O Oprave Dane    (ak je nadpis)
       evidence: "Doklad o oprave dane"
  Doklad O Oprave Dane -[TYKA_SA]-> Oprava Zakladu Dane
       evidence: "doklad o oprave dane sa tyka opravy"
  Doklad O Oprave Dane -[MA_OBSAH]-> Zaznam O Povodnom Doklade
       evidence: "doklad obsahuje zaznam o povodnom doklade"
       POZN: MA_OBSAH lebo to je textovy obsah dokladu (zaznam),
             TYKA_SA by bolo pre thematic vztah.

PRIKLAD 2 - JE_TYPOM (taxonomia) vs POVAZUJE_SA_ZA (klasifikacia):
Text: "Osobne motorove vozidlo zahrna vycvikove vozidlo, predvadzacie vozidlo
       a nahradne vozidlo."
ZLE:
  Osobne Motorove Vozidlo -[POVAZUJE_SA_ZA]-> Vycvikove Vozidlo
  Osobne Motorove Vozidlo -[POVAZUJE_SA_ZA]-> Predvadzacie Vozidlo
OK:
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Vycvikove Vozidlo
       evidence: "Osobne motorove vozidlo zahrna vycvikove vozidlo"
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Predvadzacie Vozidlo
       evidence: "predvadzacie vozidlo"
  Osobne Motorove Vozidlo -[JE_TYPOM]-> Nahradne Vozidlo
       evidence: "nahradne vozidlo"

Text: "Vydavky platene v mene zakaznika sa povazuju za prechodne polozky."
OK:
  Vydavky Platene V Mene Zakaznika -[POVAZUJE_SA_ZA]-> Prechodne Polozky
       evidence: "tieto vydavky sa povazuju za prechodne polozky"

PRIKLAD 3 - JE_PODLA s qualifier suffix:
Text: "Platitel podla odseku 1 pism. g) podava ziadost."
Triples:
  Platitel -[JE_PODLA]-> Odsek 1 Pismeno g)
       evidence: "Platitel podla odseku 1 pism. g)"
  Platitel -[PODAVA]-> Ziadost
       evidence: "Platitel podava ziadost"

PRIKLAD 4 - NEVZTAHUJE smer:
Text: "Do zakladu dane sa nezahrnaju vydavky platene v mene kupujuceho.
       Tieto vydavky sa povazuju za prechodne polozky."
OK:
  Odsek X -[VYMEDZUJE]-> Zaklad Dane              (LEN ak text doslova vymedzuje)
  Zaklad Dane -[NEVZTAHUJE_SA_NA]-> Vydavky Platene V Mene Kupujuceho
       evidence: "do zakladu dane sa nezahrnaju vydavky"
  Vydavky Platene V Mene Kupujuceho -[POVAZUJE_SA_ZA]-> Prechodne Polozky
       evidence: "vydavky sa povazuju za prechodne polozky"
ZLE smer:
  Vydavky -[NEVZTAHUJE_SA_NA]-> Zaklad Dane

PRIKLAD 5 - struktura + MA_NAZOV:
Text: "§ 48d Oslobodenie od dane v osobitnom sklade
       (1) Na ucely tohto paragrafu sa osobitnym skladom rozumie...
       (2) Tovar v osobitnom sklade je oslobodeny od dane."
Triples:
  Paragraf § 48d -[MA_NAZOV]-> Oslobodenie Od Dane V Osobitnom Sklade
       evidence: "§ 48d Oslobodenie od dane v osobitnom sklade"
  Paragraf § 48d -[OBSAHUJE]-> Odsek 1
       evidence: "(1)"
  Paragraf § 48d -[OBSAHUJE]-> Odsek 2
       evidence: "(2)"
  Odsek 1 -[ROZUMIE_SA]-> Osobitny Sklad
       evidence: "osobitnym skladom sa rozumie"
  Tovar V Osobitnom Sklade -[JE_OSLOBODENE_OD]-> Dan
       evidence: "tovar v osobitnom sklade je oslobodeny od dane"

PRIKLAD 6 - actor chain + POSKYTUJE direction:
Text: "Platitel poskytuje nahradne vozidlo zakaznikovi na dobu opravy."
ZLE:
  Nahradne Vozidlo -[POSKYTUJE]-> Zakaznik    (zly smer!)
OK:
  Platitel         -[POSKYTUJE]->     Nahradne Vozidlo
       evidence: "Platitel poskytuje nahradne vozidlo"
  Nahradne Vozidlo -[VZTAHUJE_SA_NA]-> Zakaznik Platitela
       evidence: "nahradne vozidlo zakaznikovi"
  Nahradne Vozidlo -[MA_DOBU]-> Doba Opravy
       evidence: "na dobu opravy"

PRIKLAD 7 - TYKA_SA pre thematic Cinnost->Tovar/Dan:
Text: "Vratenie dane z pridanej hodnoty sa tyka zahranicneho zastupcu."
OK:
  Vratenie Dane Z Pridanej Hodnoty -[TYKA_SA]-> Dan Z Pridanej Hodnoty
       evidence: "vratenie dane z pridanej hodnoty"
  Vratenie Dane Z Pridanej Hodnoty -[TYKA_SA]-> Zahranicny Zastupca
       evidence: "sa tyka zahranicneho zastupcu"

###############################################
### FINALNA VALIDACIA (pred vratenim)
###############################################
Pre KAZDU hranu:
1) Mam doslovny fragment z textu v 'evidence'?               (nie -> ZMAZ)
2) Smer podla tabulky? (najma negacie, POSKYTUJE)             (nie -> OBRAT)
3) Najspecifickejsi typ z gold patternov vyssie?              (zvaz alternativy)
4) Granul Odsekov: lokalny pre vnutroparagrafove?             (nie -> OPRAV)
5) Pre kazdu strukturalnu hierarchiu vidim OBSAHUJE?          (nie -> PRIDAJ)
6) Pre kazde "podla § X" mam JE_PODLA alebo ODKAZUJE_NA?      (rozhodni: entita->paragraf=JE_PODLA, odsek->odsek=ODKAZUJE_NA)
7) Pre kazdu thematicku vazbu (Cinnost o Tovare) mam TYKA_SA? (nie -> ZVAZ)

DON'T over-correct: vytvor hrany VZDY ak je v texte explicitna; ratio 0.9-1.1x
hran na uzol je zdrava norma. NEBOJ sa pouzit TYKA_SA, OBSAHUJE, UPRAVUJE
ked sa hodia - tieto typy su 4x najcastejsie v gold datasete.
"""


# "properties": {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string", "description": "Name of the entity"}
#     },
#     "required": ["name"],
#     "additionalProperties": False,
# },

response_schema_for_sde = {
            "title": "GraphExtractionResult",
            "type": "object",
            "description": "Result schema for extracting a knowledge graph from text",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of extracted entities/nodes from the text",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique node id, Title Case, no diacritics. Keep qualifier suffix if entity gains meaning from a qualifier."},
                            "label": {"type": "string", "description": "Entity type — MUST exactly match one of the allowed node labels from schema."},
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "List of relationships. Each MUST be supported by a verbatim text fragment in 'evidence'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_id": {"type": "string"},
                            "source_node_type": {"type": "string"},
                            "relation": {"type": "string", "description": "Relationship type — MUST exactly match one of allowed relationship types. Use MOST SPECIFIC type from hierarchy."},
                            "target_node_id": {"type": "string"},
                            "target_node_type": {"type": "string"},
                            "evidence": {
                                "type": "string",
                                "description": "Verbatim 5-15 word fragment from input text that explicitly supports this triple. If you cannot quote such a fragment, do NOT create this relationship."
                            },
                        },
                        "required": [
                            "source_node_id", "source_node_type", "relation",
                            "target_node_id", "target_node_type", "evidence"
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False,
        }








# ============================================================
# Q&A prompts
# ============================================================

# Stage 1: Sentence Segmentation (KG-GPT Table 4 style)
system_prompt_for_segmentation = """
Si model na **segmentáciu otázky na pod-vety pre KG triple extraction**.

# Úloha
Rozdeľ vstupnú otázku na pod-vety tak, aby každá pod-veta reprezentovala **presne jeden vzťah (triple)**.

# Pravidlá (STRIKTNÉ)
- Každá pod-veta obsahuje **max. 2 entity**
- Každá pod-veta = **1 vzťah medzi entitami**
- Entity:
  - používaj **Title Case**
  - buď **konzistentný naprieč vetami**
- **Re-use entity je zakázaný**, okrem:
  - keď prepája multi-hop reťaz (výstup = vstup)
- Multi-hop:
  - vytvor **lineárny reťazec (chain)**
- Pokrytie:
  - pod-vety musia pokrývať **celý význam otázky**
- Ak je otázka jednoduchá:
  - vráť **iba 1 pod-vetu**

# Normalizácia
- implicitné vzťahy → explicitné (napr. „ktorá vlastní“ → „X vlastní Y“)
- odstráň zámená, nahraď ich entitami
- zjednoduš vetu na **faktický tvar**

# VALIDATION (POVINNÉ)
- Každá veta musí byť mapovateľná na: (entity1, relation, entity2)
- Max. 2 entity na vetu (nikdy viac)
- Každá entita sa použije iba raz (okrem chain prepojenia)
- Ak veta nespĺňa triple formu → uprav ju
- Ak chýbajú entity → vetu nevytváraj
- Výstup musí pokrývať celý význam otázky

# Výstupné obmedzenia
- Nevytváraj meta text
- Nevysvetľuj
- Dodrž presne požadovanú štruktúru
"""

response_schema_for_segmentation = {
    "title": "SentenceSegmentation",
    "type": "object",
    "properties": {
        "sub_sentences": {
            "type": "array",
            "description": "List of sub-sentences each aligned with one KG triple",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The sub-sentence text"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 2 entity mentions in this sub-sentence (use Title Case)"
                    }
                },
                "required": ["text", "entities"]
            }
        }
    },
    "required": ["sub_sentences"]
}



# System prompt - defines the agent's role and capabilities
system_prompt_for_generating_query = """
Si agent na iteratívne vyhľadávanie v znalostnom grafe (Neo4j) pomocou Cypher queries.

Tvoj cieľ:
Nájsť relevantné trojice (uzol-vzťah-uzol) pre danú pod-vetu pomocou riadeného prieskumu grafu.
Relevantne trojice musia odpovedat na polozenu otazku.

Máš prístup k nástroju:
- search_database(query: Cypher)

## VSTUP
Dostaneš:
- pod-vetu (query fragment)
- zoznam entít (anchors)
- schému grafu (typy uzlov + vzťahov)
- ukážkové dáta

## STRATÉGIA (STRICT LOOP)

Pre KAŽDÚ anchor entitu vykonaj:

### 1. INITIAL RETRIEVAL
- Nájdeš uzol podľa entity:
  MATCH (n)
  WHERE toLower(n.id) CONTAINS toLower($entity)
  RETURN n.id, labels(n)
  LIMIT 10

Ak nenájdeš → SKIP entity

### 2. RELATION SELECTION
- Z dostupnej schémy vyber NAJVIAC relevantný vzťah pre aktuálny uzol
- Výber musí byť z množiny vztahov
- Nikdy nevymýšľaj nový vzťah

Retry max 2x:
- ak nevieš vybrať → STOP pre túto vetvu

### 3. RETRIEVAL
Vykonaj:
MATCH (a)-[r:SELECTED_REL]->(b)
WHERE toLower(a.id) CONTAINS toLower($anchor)
RETURN a.id, type(r), b.id
LIMIT 25

Ak nič nenájdeš:
- skús opačný smer:
MATCH (a)<-[r:SELECTED_REL]-(b)

Ak stále nič → STOP vetva

### 4. REASONING
Zhodnoť:
- sú trojice relevantné pre pod-vetu a na odpoved?
- má zmysel pokračovať hlbšie?

Ak ÁNO:
- nastav nový anchor = b
- pokračuj (max depth = 3)

Ak NIE:
- STOP vetva

## STOP PODMIENKY
- žiadne nové trojice
- nízka relevancia
- max depth dosiahnutý
- nevalidný vzťah

## PRAVIDLÁ CYHPER
- Používaj iba typy zo schémy
- Nepoužívaj neexistujúce labely/vzťahy
- Case-insensitive matching:
  toLower(...)
- Neznámy smer → použi nesmerový vzťah `-[r]-`
- Vždy LIMIT ≤ 25
- Krátke a presné queries
- Vracaj užitočné vlastnosti: `RETURN n.id, labels(n), type(r), properties(n)`

## DÔLEŽITÉ
- VŽDY používaj search_database pre dáta
- Nevymýšľaj výsledky bez query
- Kombinuj viac dotazov iteratívne
- Preferuj presnosť pred pokrytím
"""


response_schema_for_generating_query = {
            "title": "GraphQueryResult",
            "type": "object",
            "description": "Final query results from graph database exploration",
            "properties": {
                "cypher_query": {
                    "type": "string",
                    "description": "The final/best Cypher query that answers the question"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the query strategy and what was found"
                },
                "nodes_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs that are relevant to the answer"
                },
                "relationships_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relationships found (format: 'nodeA -[REL_TYPE]-> nodeB')"
                }
            },
            "required": ["cypher_query", "explanation", "nodes_found", "relationships_found"]
        }



# Stage 2: Relation Retrieval (KG-GPT Table 5 style)
system_prompt_for_relation_retrieval = """Dostaneš pod-vetu a zoznam kandidátskych typov vzťahov z grafovej databázy.

Tvojou úlohou je vybrať top-K typov vzťahov zo zoznamu kandidátov, ktoré sú sémanticky najbližšie k vzťahu vyjadrenému v pod-vete.

Pravidlá:
- Vyberaj IBA z poskytnutého zoznamu kandidátov — nevymýšľaj nové typy.
- Ak žiaden typ nesedí, vyber tie, ktoré sú najbližšie.
- Vrať maximálne K typov vzťahov, zoradených od najrelevantnejšieho."""

response_schema_for_relation_retrieval = {
    "title": "TopKRelations",
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top-K most semantically relevant relation types from the candidates list"
        }
    },
    "required": ["relations"]
}




# Stage 2 (ARK-V1): Relation selection — LLM call #1 per reasoning step
system_prompt_ark_select_relation = """
# Úloha
Si komponent grafového vyhľadávacieho agenta (ARK-V1). Tvoja jediná úloha je vybrať PRÁVE JEDEN vzťah zo zoznamu kandidátov, ktorý je najrelevantnejší pre zodpovedanie pod-vety pri danom uzle (anchor).

# Vstupy
- Pod-veta (cieľ retrievalu)
- Aktuálny anchor (id uzla v grafe)
- Doterajšie zhrnutie uvažovania (môže byť prázdne v prvom kroku)
- Zoznam kandidátnych vzťahov R^k (presne tie, ktoré existujú v grafe pre tento anchor)

# Striktné pravidlá
- Vyber PRESNE jeden reťazec z poskytnutého zoznamu `relations`. Nepoužívaj ani neupravuj názvy vzťahov mimo tohto zoznamu.
- Ak žiadny z kandidátov nie je relevantný pre pod-vetu, vráť `relation = null`.
- Nevymýšľaj nové vzťahy. Nepridávaj prefixy/sufixy, nemeň veľkosť písmen.
- V `rationale` krátko (1 veta, slovensky) vysvetli, prečo si daný vzťah (alebo `null`) zvolil.
"""

# Stage 2 (ARK-V1): Reasoning over retrieved triples — LLM call #2 per reasoning step
system_prompt_ark_reasoning = """
# Úloha
Si komponent grafového vyhľadávacieho agenta (ARK-V1). Dostaneš zoznam trojíc (indexovaný od 0), ktoré boli získané z grafu pre daný anchor a vzťah. Tvoja úloha:

1. Vyber indexy trojíc, ktoré sú skutočne relevantné pre pod-vetu (`selected_triple_indices`). Indexy musia byť podmnožinou poskytnutých; nevymýšľaj nové.
2. Napíš jednu krátku vetu (`implication`), čo z vybraných trojíc vyplýva vzhľadom na pod-vetu. Ak nič relevantné nie je, napíš to explicitne.
3. Rozhodni, či má zmysel pokračovať ďalším krokom uvažovania (`continue_reasoning`: true/false). Pokračuj len ak je pravdepodobné, že ďalší hop poskytne chýbajúci fakt.
4. Navrhni `next_anchor` — MUSÍ to byť tail (cieľový uzol) jednej z vybraných trojíc. Ak nepokračuješ, vráť `null`.

# Striktné pravidlá
- Nepridávaj trojice, ktoré nie sú v zozname.
- `next_anchor` musí doslova zodpovedať `t` niektorej vybranej trojice (inak ho runtime zahodí).
- Odpovedaj stručne a po slovensky.
"""

# Stage 3: Inference (KG-GPT Table 6 style)
system_prompt_for_inference = """
# Úloha
Si expert na analýzu vedomostných grafov a extrakciu informácií. Tvojou úlohou je odpovedať na otázku používateľa s využitím dvoch zdrojov: štruktúrovaných trojíc z grafovej databázy a sprievodného textového kontextu.

# Hierarchia pravdy a dôkazov
1. **Primárny zdroj (Graf):** Štruktúrované trojice `[hlava, vzťah, chvost]` sú tvojím hlavným zdrojom faktov. Ak sú tieto informácie dostupné, odpoveď musí byť postavená primárne na nich.
2. **Sekundárny zdroj (Text):** Textové úryvky (chunky) použi výhradne na doplnenie detailov, vysvetlenie nuáns alebo prepojenie súvislostí, ktoré nie sú explicitne v grafe.

# Inštrukcie pre spracovanie
- **Prepojenie entít:** Ak sú v grafe viaceré trojice, pokús sa medzi nimi nájsť logickú cestu (napr. A -> B -> C), aby si vytvoril komplexnú odpoveď.
- **Vernosť dátam:** Odpovedaj len na základe poskytnutých dát. Ak graf a text neobsahujú odpoveď, priznaj to (napr. "Na základe poskytnutých údajov nie je možné na otázku odpovedať").
- **Štýl odpovede:** Píš prirodzeným jazykom, ale zachovaj presnosť terminológie z grafu.
- **Konflikt informácií:** V prípade rozporu medzi grafom a textom uprednostni informáciu z grafovej databázy.
"""

