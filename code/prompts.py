

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
Si expertný algoritmus na extrakciu typov entít a vzťahov z otvorenej domény. Tvojou úlohou je analyzovať ľubovoľný zadaný text a identifikovať všetky odlišné typy uzlov a typy vzťahov, ktoré sa v ňom nachádzajú, bez ohľadu na doménu alebo tematickú oblasť. Pracuješ bez vopred definovanej schémy — typy môžu byť známe, nové alebo dosiaľ nevídané, pokiaľ sú odôvodnené kontextom.

PRAVIDLÁ:
1. Vráť iba abstraktné typy (na úrovni schémy), nie konkrétne inštancie entít.
2. Pre typy uzlov používaj singulár v PascalCase BEZ DIAKRITIKY (napr. Osoba, Spolocnost, Udalost, nie Spoločnosť alebo Udalosť).
3. Pre typy vzťahov používaj UPPER_SNAKE_CASE BEZ DIAKRITIKY (napr. PRACUJE_PRE, NACHADZA_SA_V, nie NACHÁDZA_SA_V).
4. Uprednostňuj všeobecné, elementárne typy uzlov pred príliš špecifickými (napr. Osoba namiesto Matematik).
5. Uprednostňuj všeobecné, nadčasové typy vzťahov pred momentálnymi (napr. PROFESOR_NA namiesto STAL_SA_PROFESOROM).
6. Zahrň iba typy jednoznačne podložené textom. Neodvodzuj ani nevymýšľaj typy nad rámec toho, čo text poskytuje.
7. Výstup udržuj minimálny a zameraný na jasne identifikovateľné vzory.
8. Všetky extrahované názvy typov uzlov aj vzťahov musia byť BEZ DIAKRITIKY — nahraď znaky s diakritikou ich základnými ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).
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
Si expertný algoritmus na spresňovanie schém znalostných grafov. Dostávaš surovú schému (typy uzlov a typy vzťahov) vytvorenú detekciou z otvorenej domény a tvojou úlohou je spresniť ju na čistú, konzistentnú a deduplikovanú schému pripravenú na extrakciu riadenú schémou.

Ide o ľahký spresňovací prechod — nie o reštrukturalizáciu. Zlučuj iba typy s jasným sémantickým prekryvom. Ak sú typy odlišné, ponechaj ich.

# TVOJE CIELE:
1. **Deduplikácia**: Zlúč typy uzlov a typy vzťahov, ktoré sú sémanticky ekvivalentné alebo takmer synonymné, do jedného kanonického typu (napr. PRACUJE_NA + ZAMESTNANY_V → PRACUJE_PRE).
2. **Normalizácia**: Zabezpeč, aby všetky typy uzlov používali singulár v PascalCase a všetky typy vzťahov používali UPPER_SNAKE_CASE konzistentne.
3. **Generalizácia**: Nahraď príliš špecifické alebo momentálne typy všeobecnými, nadčasovými ekvivalentmi (napr. STAL_SA_RIADITELOM → VEDIE; Vedec → Osoba).
4. **Zarovnanie na základnú ontológiu**: Ak je poskytnutá základná ontológia, zarovnaj surové typy na existujúce základné typy tam, kde existuje jasná sémantická zhoda. Základná ontológia má prednosť v pomenovaniach — pri zlučovaní preberaj jej označenia. Zachovaj všetky surové typy, ktoré sú skutočne odlišné a nie sú zastúpené v základnej ontológii, ako nové doplnenia.
5. **Zabezpečenie konzistencie**: Každý typ uzla referencovaný vo vzoroch vzťahov musí existovať vo výslednom zozname typov uzlov. Odstráň osirelé typy, ktoré nemajú jasný kontext vzťahu, pokiaľ nie sú jasne odôvodnené.
6. **Riešenie nejednoznačnosti**: Ak sa dva typy sémanticky prekrývajú, vyber ten všeobecnejší a zdokumentuj zlúčenie.
7. **Zachovanie pokrytia**: Neodstraňuj typy, ktoré reprezentujú skutočne odlišné koncepty. Nenúť odlišné surové typy do typov základnej ontológie. Zlučuj iba vtedy, keď je sémantické zarovnanie jasné.

# PRAVIDLÁ:
- Nevymýšľaj nové typy, ktoré neboli prítomné alebo jasne implikované vo vstupnej schéme alebo základnej ontológii.
- Neodstraňuj typy, ktoré sú sémanticky odlišné, len kvôli minimalizácii schémy.
- Pri zlučovaní vždy vyber najvšeobecnejšie a najširšie použiteľné označenie ako kanonickú formu (označenie základnej ontológie má prednosť, ak je dostupné).
- Do merge_log zahrň iba záznamy, kde skutočne došlo k zlúčeniu. Ak bol typ ponechaný bez zmeny, vynechaj ho z logu.
- Všetky extrahované názvy typov uzlov aj vzťahov musia byť BEZ DIAKRITIKY — nahraď znaky s diakritikou ich základnými ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).
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


# schema driven extraction
system_prompt_for_sde = """
Si expertny algoritmus na extrakciu znalostnych grafov. Tvojou ulohou je extrahovat pomenovane entity (uzly) a vztahy z textu podla poskytnutej ontologickej schemy.

## KRITICKE PRAVIDLO 1 - STRIKTNE DODRZANIE SCHEMY
Pouzivaj VYLUCNE typy entit a typy vztahov, ktore su PRESNE VYMENOVANE v poskytnutej scheme.

### Co to znamena konkretne:
- AK chces vytvorit uzol s typom, ktory NIE JE v zozname povolenych typov entit, NEVYTVARAJ ho. Namiesto toho:
  1. Najdi semanticky najbizsi povoleny typ zo schemy a pouzi ten.
  2. Ak ziadny povoleny typ nesedi, entitu VYNECHAJ.
- AK chces vytvorit vztah s typom, ktory NIE JE v zozname povolenych typov vztahov, NEVYTVARAJ ho. Namiesto toho:
  1. Najdi semanticky najbizsi povoleny typ zo schemy a pouzi ten.
  2. Ak ziadny povoleny typ nesedi, vztah VYNECHAJ.
- NIKDY nemodifikuj nazvy typov (nepridavaj sufixy, prefixy, predlozky ani ine zmeny). Pouzivaj ich PRESNE tak, ako su uvedene v scheme.

### Priklady zakazaneho spravania:
- Schema obsahuje "REGISTRUJE" → nepouzivaj "REGISTRUJE_PRE" ani "ZAREGISTRUJE_PRE"
- Schema obsahuje "PLATI" → nepouzivaj "PLATI_ZA"
- Schema obsahuje "PODAVA" → nepouzivaj "PODAVA_ZIADOST"
- Schema obsahuje "MA_IDENTIFIKATOR" → nepouzivaj "MA_IDENTIFIKACNE_CISLO"
- Schema obsahuje "OBSAHUJE" → nepouzivaj "ZAHRNUJE"
- Schema obsahuje "Stat" → nepouzivaj "Clenskystat" ani "Statnyorgan"
- Schema obsahuje "Vozidlo" → nepouzivaj "Dopravnyprostriedok"
- Schema obsahuje "CasoveObdobie" → nepouzivaj "Casovyusek"
- Schema obsahuje "Portal" → nepouzivaj "Webovesidlo"

## KRITICKE PRAVIDLO 2 - MAXIMALNA DETAILNOST EXTRAKCIE
Extrahuj VSETKY relevantne entity a vztahy z textu, nie len hlavne subjekty. Neobmedzuj sa na povrchnu extrakciu.

### Povinne extrahuj aj:
- **Dane, poplatky, odvody** → pouzi prislusny typ zo schemy (napr. `Dan`, `Odpocet`, `FinancnyProstriedok`)
- **Zmluvy, predpisy, zakony** → pouzi `PravnyPredpis` pre zakony a zbierky zakonov, `Dokument` pre zmluvy a ine pisomnosti
- **Cinnosti a procesy** → pouzi `Cinnost` (napr. podnikanie, registracia, dodanie)
- **Casove udaje** → pouzi `CasoveObdobie` (napr. kalendarny rok, zdanovacie obdobie)
- **Prava a povinnosti** → pouzi `Pravo`, `Povinnost` alebo iny prislusny typ zo schemy
- **Sumy, sadzby, hodnoty** → pouzi `Hodnota`, `Mena` alebo iny prislusny typ

### Pouzivaj VZDY najspecifickejsi povoleny typ zo schemy:
- Ak schema obsahuje `InvesticnyMajetok`, pouzi ho namiesto vseobecneho `Majetok`
- Ak schema obsahuje `PravnyPredpis`, pouzi ho pre zakony namiesto vseobecneho `Dokument`
- Ak schema obsahuje `Platitel`, pouzi ho namiesto vseobecneho `Osoba`
- Vzdy preferuj specialny typ pred vseobecnym, ak je k dispozicii v scheme

## KRITICKE PRAVIDLO 3 - EXTRAKCIA LEGISLATIVNEJ STRUKTURY
Pri pravnych textoch extrahuj aj vnutornu strukturu zakona:

- **Zakony a zbierky zakonov** extrahuj ako `PravnyPredpis`
- **Konkretne paragrafy, odseky a pismenka** (napr. § 54 ods. 2 pism. a) extrahuj ako samostatne entity typu `Paragraf`
- **Prepoj paragrafy so zakonom** pomocou vztahu `OBSAHUJE`: PravnyPredpis -> OBSAHUJE -> Paragraf
- **Prepoj entity s paragrafmi** pomocou vztahu `PODLA` alebo ineho vhodneho vztahu zo schemy, ak sa entita definuje alebo odkazuje na konkretny paragraf

Priklad:
- "Zakon 222/2004 Z. z." → typ: `PravnyPredpis`, id: "Zakon 222/2004 Z. z."
- "§ 54 ods. 2 pism. a)" → typ: `Paragraf`, id: "§ 54 ods. 2 pism. a)"
- Vztah: PravnyPredpis("Zakon 222/2004 Z. z.") -[OBSAHUJE]-> Polozka("§ 54 ods. 2 pism. a)")

## Extrakcia uzlov
- **Konzistencia oznaceni**: Vzdy pouzivaj PRESNE typy entit zo schemy.
- **ID uzlov**: Pouzi najuplnejsi ludsky citatelny nazov alebo identifikator najdeny v texte. Nikdy nepouzivaj cele cisla ako ID uzlov.
- **Vlastnosti**: Zahrn vlastnosti iba vtedy, ked su explicitne uvedene v texte.

## Extrakcia vztahov
- Pouzivaj IBA typy vztahov z poskytnutej schemy - PRESNE tak, ako su napisane.
- Zabezpec spravnu smerovost: pociatocny uzol a koncovy uzol musia zodpovedat semantickemu smeru vztahu.
- PRED zapisanim kazdeho vztahu over, ze jeho typ sa PRESNE zhoduje s niektorym z povolenych typov.

## Riesenie koreferencie
- Ak je entita spomenuta viackrat pod roznymi nazvami alebo zamenami, vzdy ju rozlis na najuplnejsi identifikator ako ID uzla.
- Udrzuj konzistenciu napriec vsetkymi odkazmi na tu istu entitu.

## Bez diakritiky
Vsetky extrahovane hodnoty — ID uzlov, nazvy, vlastnosti, hodnoty vztahov — musia byt BEZ DIAKRITIKY (napr. c→c, s→s, z→z, a→a, e→e, i→i, o→o, u→u, y→y, n→n, t→t, d→d, l→l, o→o).

## Postup pred finalnym vystupom
Pred tym, nez vydas finalny JSON:
1. Prejdi KAZDY uzol a over, ze jeho typ je PRESNE v zozname povolenych typov entit. Ak nie, nahrad ho najblizsim povolenym alebo ho odstran.
2. Prejdi KAZDY vztah a over, ze jeho typ je PRESNE v zozname povolenych typov vztahov. Ak nie, nahrad ho najblizsim povolenym alebo ho odstran.
3. Over, ze si pouzil najspecifickejsi povoleny typ (nie vseobecny).
4. Over, ze si extrahoval aj dane, zmluvy, cinnosti, casove obdobia a legislativnu strukturu.
"""


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
                            "id": {"type": "string", "description": "Unique identifier for the node"},
                            "label": {"type": "string", "description": "Type/label of the entity (e.g., Person, Organization)"},
                            "properties": {
                                "type": "object",
                                "description": "Properties of the entity as mentioned in the text",
                                "properties": {
                                    "name": {"type": "string", "description": "Name of the entity"}
                                },
                                "required": ["name"]
                            },
                        },
                        "required": ["id", "label", "properties"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "List of relationships between nodes",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_id": {"type": "string", "description": "ID of the source node"},
                            "source_node_type": {"type": "string", "description": "Type/label of the source node"},
                            "relation": {"type": "string", "description": "Type of relationship between source and target (e.g., KNOWS, WORKS_AT)"},
                            "target_node_id": {"type": "string", "description": "ID of the target node"},
                            "target_node_type": {"type": "string", "description": "Type/label of the target node"},
                            "properties": {
                                "type": "object",
                                "description": "Properties of the relationship as mentioned in the text",
                                "properties": { }
                            },
                        },
                        "required": ["relation", "source_node_id", "target_node_id", "source_node_type", "target_node_type"]
                    }
                }
            },
            "required": ["nodes", "relationships"]
        }









# System prompt - defines the agent's role and capabilities
system_prompt_for_generating_query = """Si expertný agent na Neo4j Cypher špecializovaný na dopytovanie znalostných grafov.

Tvojou úlohou je odpovedať na otázky dopytovaním grafovej databázy Neo4j.

## Tvoje schopnosti
Máš prístup k nástroju `search_database`, ktorý vykonáva Cypher dopyty voči Neo4j.

## Stratégia dopytovania
1. **Analyzuj otázku** a identifikuj relevantné uzly a vzťahy
2. **Začni prieskumnými dopytmi** na pochopenie existujúcich dát:
   - Pre uzly: `MATCH (n:Label) RETURN n.id, labels(n)`
   - Pre vzťahy: `MATCH (a)-[r:TYP]->(b) RETURN a.id, type(r), b.id`
3. **Iteratívne spresňuj** — použi výsledky úvodných dopytov na zostavenie špecifickejších dopytov
4. **Nájdi najlepšie zhody** — dopytuj dovtedy, kým nenájdeš najrelevantnejšie dáta

## Pravidlá Cypher dopytov
- Pre označenia/typy so špeciálnymi znakmi použi spätné apostrofy: `MATCH (n:\`Specialny-Label\`) ...`
- Pre textové porovnávanie použi case-insensitive: `WHERE toLower(n.id) CONTAINS toLower('romeo')`
- Ak nepoznáš smer vzťahu, použi nesmerový vzťah `-[r]-`
- Vždy pridaj `LIMIT 25` na zabránenie veľkým výsledkovým sadám
- Vracaj užitočné vlastnosti: `RETURN n.id, labels(n), type(r), properties(n)`
- Dopyty udržuj efektívne, zamerané a krátke

## Dôležité
- Na dopytovanie databázy MUSÍŠ použiť nástroj search_database
- Ak je to potrebné, vykonaj viacero dopytov na nájdenie najlepšej odpovede
- Keď nájdeš dostatočné dáta, poskytni finálnu odpoveď s najlepším Cypher dopytom"""


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

