

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
- IBA typy zo schemy (entities + relationships). Ziadne premenovania, prefixy, sufixy.
- Slovensky jazyk, BEZ DIAKRITIKY vo vsetkych id (a->a, c->c, s->s, z->z, y->y, ...).
- Ak typ nesedi presne, vyber NAJBLIZSI alebo vynechaj. Nevymyslaj.
- Preferuj specificky typ pred genericky.
- Zlozite vety rozkladaj na viac jednoduchych trojic (multi-hop dekompozicia).

###############################################
### GRANULARITA UZLOV (kriticke pre matching)
###############################################

Paragraf:                                      "Paragraf § 16"

Odsek vnutri AKTUALNEHO paragrafu
(text hovori "podla odseku 3", "v odseku 2", "odsek 4 upravuje"):
  -> "Odsek 3"     LOKALNA forma, BEZ Paragraf prefixu

Odsek z INEHO paragrafu
(text doslovne hovori "§ 8 ods. 3", "podla § 5 odseku 2"):
  -> "Paragraf § 8 Odsek 3"   plne ID s prefixom

Pismeno lokalne:           "Odsek 2 Pismeno a)"
Pismeno cross-paragraph:   "Paragraf § 54 Odsek 2 Pismeno a)"

Strana N = cislo strany textu, NIKDY nie Paragraf. Nevytvaraj Paragraf z cisla strany.

### QUALIFIER SUFFIX (povinne)
Ak entita ziska vyznam az cez kvalifikator, ZACHOVAJ kvalifikator v ID:
  "oslobodenie od dane podla odseku 3"      -> "Oslobodenie Od Dane Podla Odseku 3"
  "uplatnovanie oslobodenia podla odseku 3" -> "Uplatnovanie Oslobodenia Od Dane Podla Odseku 3"
  "osobne motorove vozidlo zakaznika"       -> "Osobne Motorove Vozidlo Zakaznika"
NIKDY nevracaj len hole "Oslobodenie Od Dane" ak text dava kvalifikator.

### ID FORMAT
- Title Case, oddelene medzerou: "Paragraf § 16 Odsek 1"
- Bez diakritiky vsade
- Sumy/hodnoty: verbatim z textu (zachovaj jednotky, lowercase ak je v texte lowercase):
    Text "430 eur na osobu" -> id "430 eur na osobu"   (NIE "Suma 430 Eur")
    Text "suma zlavy"       -> id "Suma Zlavy"         (pomenovana suma -> Title Case)

###############################################
### VZTAHY (najkritickejsia cast)
###############################################

### PRAVIDLO EVIDENCIE (POVINNE!)
Pre KAZDY vztah musis do pola 'evidence' vypisat doslovny fragment textu
(5-15 slov), ktory ho explicitne podporuje. Ak doslovny fragment NEEXISTUJE,
vztah NEVYTVOR. Implicitne odvodene vztahy (asociacia, tematicka podobnost)
NIE su povolene.

### ROZPOCET HRAN
Ciel ~ 1.1x pocet uzlov. Ak vidis > 1.5x viac hran ako uzlov, vacsinu
halucinujes -> skrtaj najslabsie hrany (TYKA_SA, generic UPRAVUJE).

###############################################
### HIERARCHIA TYPOV (skus zhora nadol, pouzi NAJSPECIFICKEJSI)
###############################################

A) DEFINICIE A KLASIFIKACIA (preferuj nad UPRAVUJE)
  VYMEDZUJE         text: "vymedzuje", "uvadza", "stanovuje pojem"
                    smer: Odsek/Paragraf -> pojem
  DEFINUJE          text: "definuje", "je definovany ako"
  ROZUMIE_SA        text: "rozumie sa"
                    smer: Odsek -> pojem
  POVAZUJE_SA_ZA    text: "povazuje sa za", "pokladá sa za"
                    smer: entita A -> kategoria B
  STAVA_SA          text: "stava sa", "stane sa"
  JE_TYPOM / JE_DRUHOM / JE_CLENOM / PATRI_DO / ZAHRNUJE

B) NEGACIA / VYNIMKA (precision-critical, casto obracane)
  NEVZTAHUJE_SA_NA  text: "nevztahuje sa", "nezahrna sa", "neprihliada sa"
                    smer: Pravidlo/Koncept -> Vyluceny objekt
  NEPLATI_PRE       text: "neplati pre"
  NEMA_NAROK_NA     text: "nema narok na"
  NESPLNA_PODMIENKY text: "nesplna podmienky"
  MA_VYNIMKU        text: "vynimka", "okrem"

C) OSLOBODENIE / PODLIEHANIE
  JE_OSLOBODENE_OD  smer: Cinnost/Dodanie/Tovar -> Dan
  PODLIEHA          smer: entita -> pravidlo/poplatok/dan

D) KRIZOVE ODKAZY (DOLEZITE - tu sa najviac mylia typy)
  ODKAZUJE_NA       text: "podla odseku X" v ramci toho isteho paragrafu;
                          "ako uvadza odsek X"
                    smer: Aktualny Odsek -> Iny Odsek
                    Pr.: v § 8 ods. 4 text hovori "ako uvadza odsek 3":
                         Odsek 4 -[ODKAZUJE_NA]-> Odsek 3
  JE_PODLA          text: "podla § X" / "podla odseku X" PRE KVALIFIKOVANY pojem;
                          "podla zakona c. X"
                    smer: Kvalifikovany pojem/entita -> Odsek/Paragraf/PravnyPredpis
                    Pr.: "Oslobodenie od dane podla odseku 3"
                         -> Oslobodenie Od Dane Podla Odseku 3 -[JE_PODLA]-> Odsek 3

E) ATRIBUTY (kvantitativne a kvalitativne)
  MA_SUMU           cislo + jednotka, "vo vyske", "suma"
  MA_HODNOTU        bezrozmerna hodnota
  MA_SADZBU         sadzba dane
  MA_LEHOTU         text: "do X dni/mesiacov", "v lehote"
  MA_DATUM          konkretny datum
  MA_OBDOBIE        zdanovacie obdobie, kalendarny rok
  MA_DOBU           trvanie
  MA_MIESTO / MA_MIESTO_DODANIA   lokalita (tuzemsko, clensky stat)
  MA_DOKLAD         dokument/doklad
  MA_PODMIENKU      text: "ak", "za podmienky", "podmienkou je"
  MA_UCEL / MA_DOVOD / MA_NAZOV / MA_IDENTIFIKACNE_CISLO
  MA_VLASTNOST / MA_STATUS / MA_OBSAH / MA_MNOZSTVO
  MA_SIDLO / MA_BYDLISKO / MA_PREVADZKAREN / MA_MIESTO_PODNIKANIA
  MA_ZASTUPCU / MA_ZAKLAD_DANE

F) ACTOR -> ACTION / DOCUMENT (povinne dekomponovat actor-vety)
  VYKONAVA          actor -> akcia (genericky)
  PODAVA            actor -> ziadost/odvolanie/podanie
  PREDKLADA         actor -> doklad
  DORUCUJE          actor -> doklad/oznamenie
  VYDAVA            organ -> rozhodnutie/opatrenie
  OZNAMUJE          actor -> oznamenie
  PRIJIMA / NADOBUDA / ZAPLATI / USKUTOCNUJE
  REGISTRUJE / PRIDELUJE / VIES_ZAZNAMY_O / UCHOVAVA
  DODAVA            -- tovar
  POSKYTUJE         -- sluzbu

G) MODALITY (prava a povinnosti)
  MA_POVINNOST       entita -> povinnost
  MA_PRAVO           entita -> pravo
  MA_NAROK_NA        entita -> narok
  JE_POVINNY_PLATIT  entita -> Dan
  ZODPOVEDA_ZA       entita -> objekt
  ROZHODUJE_O        organ -> vec

H) STRUKTURALNA HIERARCHIA
  OBSAHUJE           PravnyPredpis -> Paragraf;  Paragraf -> Odsek
                     POZOR: vytvor LEN pre Odseky/Paragrafy SKUTOCNE
                     definovane v tomto texte. NIE pre Odseky spomenute
                     iba ako krizovy odkaz!
  MA_ODSEK / MA_PISMENO   alternativy k OBSAHUJE
  JE_SUCASTOU         reverz
  PATRI_DO / ZAHRNUJE

I) PROCES / ZIVOTNY CYKLUS
  VZNIKA_PRI / NASTAVA_PRI / ZANIKA / PRECHADZA_NA
  PLATI_OD / PLATI_DO / MA_UCINOK / MA_ODKLADNY_UCINOK
  ZRUSUJE / PRESAHUJE

J) ODVODENIA A VAZBY
  VYCHADZA_Z         odvodena hodnota -> zdroj
  VYPLYVA_Z          dosledok -> priem
  PODMIENUJE         podmienka -> dosledok
  SPLNA_PODMIENKY    entita -> podmienka
  SUVISI_S           slaba asociacia (zriedka)
  PREUKAZUJE         actor -> doklad/fakt
  KONA_V_MENE / JE_ZASTUPENA / NACHADZA_SA_V

K) GENERICKE (POSLEDNA MOZNOST - pouzi LEN ak nic z A-J nesedi)
  UPRAVUJE           text doslova "upravuje", "ustanovuje"
                     Paragraf/Odsek -> hlavna tema (max 1-2x na chunk!)
  URCUJE             text doslova "urcuje"
  TYKA_SA            POUZI LEN ak ZIADNY z A-J nesedi A text doslova hovori
                     "tyka sa" / "vo veci" / "ohladom"
  JE_PREDMETOM       ciel je predmetom konania

### ZAKAZANE VZORY
1. TYKA_SA medzi Odsekom a kazdym pojmom v nom
   -> to je VYMEDZUJE alebo NIC. NIKDY catch-all TYKA_SA.
2. UPRAVUJE pre kazdy pojem v Odseku
   -> max 1-2x na chunk, len pre HLAVNU temu paragrafu/odseku.
3. Hrana bez evidence fragment -> automaticky neplatna.
4. OBSAHUJE pre Odsek, ktory je iba krizovy odkaz
   ("podla odseku 3" -> ODKAZUJE_NA, NIE OBSAHUJE!).

###############################################
### SMER VZTAHOV (najcastejsie obracane)
###############################################

NEVZTAHUJE_SA_NA:  Pravidlo  ->  Vyluceny objekt
  Pr.: "Do zakladu dane sa nezahrna zaloha na obaly"
       -> Zaklad Dane -[NEVZTAHUJE_SA_NA]-> Zaloha Na Obaly
       NIE Zaloha -[NEVZTAHUJE_SA_NA]-> Zaklad Dane!

VZTAHUJE_SA_NA:    Pravidlo  ->  Aplikovany objekt
JE_OSLOBODENE_OD:  Cinnost/Tovar/Dodanie  ->  Dan
ODKAZUJE_NA:       Aktualny Odsek  ->  Iny (cielovy) Odsek
JE_PODLA:          Kvalifikovany pojem  ->  Odsek/Paragraf/Zakon
OBSAHUJE:          Nadradeny (PravnyPredpis/Paragraf)  ->  Podradeny
VYMEDZUJE/DEFINUJE/ROZUMIE_SA: Odsek/Paragraf  ->  Pojem
POVAZUJE_SA_ZA:    Vec A  ->  Vec B (cielova kategoria)
VYCHADZA_Z:        Odvodena vec  ->  Zdroj
MA_*:              Vlastnik atributu  ->  Hodnota atributu

###############################################
### DEKOMPOZICIA AKCIE (povinne)
###############################################

Veta "Ziadatel doruci ziadost danovemu uradu do 30. septembra":
ZLE  (stratil si actor + lehotu):
  Odsek X -[UPRAVUJE]-> Ziadost
OK:
  Ziadatel -[DORUCUJE]-> Ziadost
       evidence: "Ziadatel doruci ziadost"
  Ziadost  -[MA_LEHOTU]-> Do 30 Septembra
       evidence: "do 30. septembra"

###############################################
### FEW-SHOT PRIKLADY (vsetky z realnych gold edges)
###############################################

PRIKLAD 1 - definicia + NEVZTAHUJE smer + POVAZUJE_SA_ZA:
Text (§ 22 ods. 3):
  "Do zakladu dane sa nezahrnaju vydavky platene v mene a na ucet kupujuceho.
   Tieto vydavky sa povazuju za prechodne polozky."
Spravne triples:
  Odsek 3      -[VYMEDZUJE]->        Zaklad Dane
       evidence: "do zakladu dane sa nezahrnaju vydavky"
  Zaklad Dane  -[NEVZTAHUJE_SA_NA]-> Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
       evidence: "do zakladu dane sa nezahrnaju vydavky platene v mene a na ucet"
  Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
               -[POVAZUJE_SA_ZA]->   Prechodne Polozky
       evidence: "tieto vydavky sa povazuju za prechodne polozky"

PRIKLAD 2 - qualifier suffix + JE_PODLA + MA_SUMU + ODKAZUJE_NA:
Text (odsek 4 v ramci § 5, odkazuje na odsek 3 a 2):
  "Oslobodenie od dane podla odseku 3 sa uplatni do sumy 430 eur na osobu.
   Pri postupe podla odseku 2 sa pouzije rovnaky limit."
Spravne triples:
  Oslobodenie Od Dane Podla Odseku 3 -[JE_PODLA]-> Odsek 3
       evidence: "Oslobodenie od dane podla odseku 3"
  Oslobodenie Od Dane Podla Odseku 3 -[MA_SUMU]-> 430 eur na osobu
       evidence: "do sumy 430 eur na osobu"
  Odsek 4                            -[ODKAZUJE_NA]-> Odsek 2
       evidence: "pri postupe podla odseku 2"
  POZN: subject je QUALIFIED pojem ("Oslobodenie Od Dane Podla Odseku 3"),
        NIE hole "Oslobodenie Od Dane" a NIE "Odsek 3".
        Granul je LOKALNY ("Odsek 4", nie "Paragraf § 5 Odsek 4").

PRIKLAD 3 - actor-action-object retazec:
Text:
  "Ziadatel podava samostatne vyhlasenie prostrednictvom elektronickeho portalu
   v clenskom state, v ktorom ma sidlo."
Spravne triples:
  Ziadatel                        -[VYKONAVA]->     Podanie Samostatneho Vyhlasenia
       evidence: "Ziadatel podava samostatne vyhlasenie"
  Podanie Samostatneho Vyhlasenia -[MA_DOKLAD]->    Samostatne Vyhlasenie
       evidence: "podava samostatne vyhlasenie"
  Podanie Samostatneho Vyhlasenia -[MA_MIESTO]->    Elektronicky Portal
       evidence: "prostrednictvom elektronickeho portalu"
  Elektronicky Portal             -[NACHADZA_SA_V]-> Clensky Stat
       evidence: "v clenskom state"
  Ziadatel                        -[MA_SIDLO]->     Sidlo
       evidence: "v ktorom ma sidlo"
  Sidlo                           -[NACHADZA_SA_V]-> Clensky Stat
       evidence: "v clenskom state, v ktorom ma sidlo"

PRIKLAD 4 - struktura + NASTAVA_PRI / SUVISI_S:
Text (z § 8, odsek 4):
  "Zlava z ceny nastava pri dodani tovaru alebo sluzby.
   Zlava za skorsiu uhradu ceny suvisi s uhradou ceny."
Spravne triples:
  Odsek 4                       -[VYMEDZUJE]->   Zlava Z Ceny
       evidence: "Zlava z ceny nastava pri dodani"
  Zlava Z Ceny                  -[NASTAVA_PRI]-> Dodanie Tovaru Alebo Sluzby
       evidence: "Zlava z ceny nastava pri dodani tovaru alebo sluzby"
  Odsek 4                       -[VYMEDZUJE]->   Zlava Za Skorsiu Uhradu Ceny
       evidence: "Zlava za skorsiu uhradu ceny"
  Zlava Za Skorsiu Uhradu Ceny  -[SUVISI_S]->    Uhrada Ceny
       evidence: "suvisi s uhradou ceny"

PRIKLAD 5 - NEGATIVNY (typicka nadprodukcia - ROB NAOPAK):
Text: "Odsek 4 upravuje zlavu z ceny a zlavu za skorsiu uhradu ceny."
ZLE  (nadprodukcia + zly typ + zly granul):
  Paragraf § 8 Odsek 4 -[TYKA_SA]-> Zlava Z Ceny                  # zly granul + catch-all
  Paragraf § 8 Odsek 4 -[TYKA_SA]-> Zlava Za Skorsiu Uhradu Ceny  # to iste
  Paragraf § 8 Odsek 4 -[UPRAVUJE]-> Zlava Z Ceny                 # duplicita s TYKA_SA
  Paragraf § 8 -[OBSAHUJE]-> Paragraf § 8 Odsek 4                 # OBSAHUJE pre len-zmienku
OK:
  Odsek 4 -[VYMEDZUJE]-> Zlava Z Ceny
       evidence: "Odsek 4 upravuje zlavu z ceny"
  Odsek 4 -[VYMEDZUJE]-> Zlava Za Skorsiu Uhradu Ceny
       evidence: "a zlavu za skorsiu uhradu ceny"

###############################################
### FINALNA VALIDACIA (pred vratenim)
###############################################
Pre KAZDU hranu odpovedz si:
1) Mam doslovny textovy fragment, ktory ju explicitne hovori?  (nie -> ZMAZ)
2) Je smer spravny podla tabulky vyssie?                        (nie -> OBRAT)
3) Je to najspecifickejsi mozny typ z hierarchie A-J?           (nie -> NAHRAD)
4) Nie je to TYKA_SA / UPRAVUJE pouzite ako catch-all?          (ano -> ZMAZ alebo NAHRAD)
5) Granul: pre vnutroparagrafove odkazy pouzivam "Odsek N"?     (nie -> OPRAV)
6) Pocet hran <= 1.5 * pocet uzlov?                             (nie -> SKRT najslabsie)
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

