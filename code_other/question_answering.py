
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

















# ====================================================================================================

        
        
    # ---------------- KG-GPT QUERYING METHODS ----------------

    def segment_question(self, question: str) -> List[SubSentence]:
        """Stage 1 (KG-GPT): Break question into sub-sentences each aligned with one KG triple.

        Args:
            question: The user's natural language question

        Returns:
            List of SubSentence objects, each with text and up to 2 entity mentions
        """
        print("\nStage 1: Sentence segmentation...")
        
        
        
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=response_schema_for_segmentation),  # type: ignore[arg-type]
            system_prompt=system_prompt_for_segmentation
        )
        response = agent.invoke({"messages": [{"role": "user", "content": f"Otázka: {question}"}]})

        # structured_response is already a dict when using ProviderStrategy
        data = cast(dict, response["structured_response"])

        sub_sentences = [
            SubSentence(text=s.get("text", "").strip(), entities=s.get("entities", []) or [])
            for s in data.get("sub_sentences", [])
            if s.get("text")
        ]

        if not sub_sentences:
            sub_sentences = [SubSentence(text=question, entities=[])]


        print(f"  → {len(sub_sentences)} sub-sentence(s): {[s.text for s in sub_sentences]}")
        return sub_sentences
    
    

    def search_agent_retrieve(self, sub_sentence: str, entities: list[str]) -> tuple[list[tuple], list[str]]:
        """ARK-V1-inspired Stage 2: Agent with search_database tool iteratively queries
        the graph to find evidence triples for a single sub-sentence.

        Args:
            sub_sentence: Sub-sentence text from Stage 1 (one implied KG triple)
            entities: Entity mentions from segmentation (Title Case)

        Returns:
            (evidence_triples, node_ids) — triples as (head, rel, tail) tuples and
            node IDs for chunk retrieval, both compatible with Stage 3
        """
        import re

        graph = self.graph

        @tool
        def search_database(cypher_query: str) -> str:
            """Execute a Cypher query against the Neo4j graph database.

            Args:
                cypher_query: A valid Cypher query string to execute

            Returns:
                JSON string of query results, or error message if query fails
            """
            try:
                is_read_only_cypher(cypher_query)
                results = graph.query(cypher_query)
                serialized = serialize_for_json(results)
                return json.dumps(serialized, ensure_ascii=False, indent=2)[:4000]
            except Exception as e:
                return f"Query error: {e}"

        schema = self.get_graph_schema()
        sample_schema = self.get_sample_graph_schema()

        user_prompt = f"""
        Pod-veta:
        {sub_sentence}

        Entity (anchors):
        {entities}

        Schéma grafu:
        - Uzly: {schema.nodes}
        - Vzťahy: {schema.relationships}

        Ukážkové dáta:
        {sample_schema}

        Úloha:
        Nájdi všetky relevantné trojice (uzol-vzťah-uzol) súvisiace s pod-vetou.

        Postup:
        1. Začni od entít (anchors)
        2. Iteratívne vyberaj vzťahy zo schémy
        3. Dopytuj graf pomocou Cypher
        4. Rozhoduj, či pokračovať alebo zastaviť

        Obmedzenia:
        - Používaj iba vzťahy zo schémy
        - Nevymýšľaj dáta
        - Max hĺbka = 3
        - Max retries pre výber vzťahu = 2
        """
        

        agent = create_agent(
            model=self.openai_client,
            tools=[search_database],
            response_format=ToolStrategy(schema=response_schema_for_generating_query), # type: ignore[arg-type]
            system_prompt=system_prompt_for_generating_query,
        )

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        result = response["structured_response"]

        node_ids: list[str] = list(result.get("nodes_found", []))
        triples: list[tuple] = []

        for rel_str in result.get("relationships_found", []):
            match = re.match(r"^(.+?)\s*-\[(.+?)\]->\s*(.+)$", rel_str.strip())
            if match:
                head, rel, tail = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                triples.append((head, rel, tail))
                if head not in node_ids:
                    node_ids.append(head)
                if tail not in node_ids:
                    node_ids.append(tail)

        if not triples and result.get("cypher_query"):
            try:
                rows = self.graph.query(result["cypher_query"])
                for row in rows:
                    h = row.get("head") or row.get("a.id", "")
                    r = row.get("rel") or row.get("type(r)", "")
                    t = row.get("tail") or row.get("b.id", "")
                    if h and r and t:
                        triples.append((h, r, t))
            except Exception:
                pass

        return triples, node_ids



    # ---------------- ARK-V1 Stage 2 (state-machine retrieval) ----------------
    
    def _resolve_anchor(self, name: str) -> Optional[str]:
        if not name:
            return None
        
        rows = self.graph.query( "MATCH (n) WHERE toLower(n.id) = toLower($name) RETURN n.id AS id LIMIT 1", {"name": name})
        if rows:
            return rows[0]["id"]
        
        rows = self.graph.query( "MATCH (n) WHERE toLower(n.id) CONTAINS toLower($name) RETURN n.id AS id LIMIT 1", {"name": name})
        return rows[0]["id"] if rows else None
    

    def _retrieve_relations(self, anchor: str) -> list[str]:
        q_out = "MATCH (a)-[r]->() WHERE toLower(a.id) = toLower($anchor) RETURN DISTINCT type(r) AS rel"
        q_in = "MATCH (a)<-[r]-() WHERE toLower(a.id) = toLower($anchor) RETURN DISTINCT type(r) AS rel"
        
        is_read_only_cypher(q_out)
        is_read_only_cypher(q_in)
        
        rels: list[str] = []
        seen: set[str] = set()
        
        for q in (q_out, q_in):
            for row in self.graph.query(q, {"anchor": anchor}):
                
                r = row.get("rel")
                if r and r not in seen:
                    seen.add(r)
                    rels.append(r)
                    
        return rels


    def _retrieve_triples(self, anchor: str, rel: str) -> list[tuple]:
        # f-string for rel type is safe: rel is validated to be in R^k returned by graph.
        TRIPLE_LIMIT = 25
        
        q_out = f"MATCH (a)-[r:`{rel}`]->(t) WHERE toLower(a.id) = toLower($anchor) RETURN a.id AS h, type(r) AS r, t.id AS t LIMIT {TRIPLE_LIMIT}"
        q_in = f"MATCH (a)<-[r:`{rel}`]-(t) WHERE toLower(a.id) = toLower($anchor) RETURN t.id AS h, type(r) AS r, a.id AS t LIMIT {TRIPLE_LIMIT}"
        
        is_read_only_cypher(q_out)
        is_read_only_cypher(q_in)
        
        rows = self.graph.query(q_out, {"anchor": anchor})
        if not rows:
            rows = self.graph.query(q_in, {"anchor": anchor})
            
        out: list[tuple] = []
        for row in rows:
            
            h, r, t = row.get("h"), row.get("r"), row.get("t")
            if h and r and t:
                out.append((h, r, t))
                
        return out


    def _select_relation(self, sub_sentence: str, anchor: str, summary: str, relations: list[str]) -> Optional[str]:
        C_MAX = 2
        
        class RelationChoice(BaseModel):
            relation: Optional[str] = Field(
                default=None,
                description="Exactly one relation string from the provided list, or null if none relevant.",
            )
            rationale: str = Field(default="", description="Brief Slovak justification.")
            
            
        model = self.gemini_client_flash.with_structured_output(
            schema=RelationChoice, method="json_schema"
        )
        
        last_err = ""
        for attempt in range(C_MAX + 1):
            user = (
                f"Pod-veta: {sub_sentence}\n"
                f"Anchor: {anchor}\n"
                f"Doterajšie zhrnutie: {summary or '(prázdne)'}\n"
                f"Kandidátne vzťahy (R^k): {relations}\n"
                + (f"\nPredchádzajúci pokus bol neplatný: {last_err}\n" if last_err else "")
                + "Vyber presne jeden vzťah z R^k alebo null."
            )
            
            try:
                resp = cast(RelationChoice, model.invoke([
                    ("system", system_prompt_ark_select_relation),
                    ("human", user),
                ]))
                
            except Exception as e:
                last_err = f"LLM error: {e}"
                continue
            
            
            chosen = resp.relation
            
            if chosen is None:
                return None
            if chosen in relations:
                return chosen
            
            last_err = f"'{chosen}' nie je v zozname R^k."
            
        return None


    def _reason_step(self, sub_sentence: str, anchor: str, rel: str, triples: list[tuple], summary: str) -> ReasoningStep:
        
        model = self.gemini_client_flash.with_structured_output(
            schema=ReasoningStep, method="json_schema"
        )
        
        numbered = "\n".join(f"[{i}] {h} -[{r}]-> {t}" for i, (h, r, t) in enumerate(triples))
        
        user = (
            f"Pod-veta: {sub_sentence}\n"
            f"Anchor: {anchor}\n"
            f"Vybraný vzťah: {rel}\n"
            f"Doterajšie zhrnutie: {summary or '(prázdne)'}\n"
            f"Trojice:\n{numbered}\n"
            "Vyber relevantné indexy, zhrň implikáciu, rozhodni o pokračovaní a navrhni next_anchor (tail vybratých trojíc) alebo null."
        )
        
        try:
            resp = model.invoke([
                ("system", system_prompt_ark_reasoning),
                ("human", user),
            ])
            
        except Exception as e:
            print(f"  [ark-v1] reasoning LLM error: {e}")
            return ReasoningStep()

        if not isinstance(resp, ReasoningStep):
            print(f"  [ark-v1] reasoning unexpected response type: {type(resp).__name__}")
            return ReasoningStep()
        
        return resp
    
    

    def ark_v1_retrieve(self, sub_sentence: str, entities: list[str]) -> tuple[list[tuple], list[str]]:
        """ARK-V1 (Klein & Ohnemus 2025) state-machine retrieval for a single sub-sentence.

        For each anchor entity we perform up to K_MAX reasoning steps. Each step:
        1) enumerate relations actually present in the graph for the anchor,
        2) LLM picks one (validated against the candidate list, Cmax retries),
        3) enumerate triples (anchor)-[rel]-(*) from the graph,
        4) LLM selects relevant indices, summarizes implication, optionally advances
           the anchor to a tail of a selected triple for the next hop.

        The per-step prompt is rebuilt from scratch (system + Q + sub_sentence +
        running summary + current candidates) so context size stays ~constant in k.
        """
        K_MAX = 3

        evidence: list[tuple] = []
        node_ids: list[str] = []
        summary = ""

        for raw_anchor in (entities or [])[:2]:
            anchor = self._resolve_anchor(raw_anchor)
            
            if anchor is None:
                print(f"  [ark-v1] anchor '{raw_anchor}' not found in graph, skipping")
                continue

            for k in range(K_MAX):
                
                relations = self._retrieve_relations(anchor)
                if not relations:
                    break
                
                rel = self._select_relation(sub_sentence, anchor, summary, relations)
                if rel is None:
                    break
                
                triples = self._retrieve_triples(anchor, rel)
                if not triples:
                    break

                step = self._reason_step(sub_sentence, anchor, rel, triples, summary)
                valid_idx = [i for i in step.selected_triple_indices if 0 <= i < len(triples)]
                selected = [triples[i] for i in valid_idx]
                
                evidence.extend(selected)
                for h, _, t in selected:
                    
                    if h not in node_ids:
                        node_ids.append(h)
                        
                    if t not in node_ids:
                        node_ids.append(t)
                        

                print(f"  [ark-v1] Step {k+1}: anchor={anchor} rel={rel} selected={len(selected)}/{len(triples)} → {step.implication}")
                summary += f"\nKrok {k+1} (anchor={anchor}, rel={rel}): {step.implication}"

                if not step.continue_reasoning:
                    break
                
                tails = {t for (_, _, t) in selected}
                if step.next_anchor and step.next_anchor in tails:
                    anchor = step.next_anchor
                else:
                    break

        # dedup triples while preserving order
        seen = set()
        dedup_triples: list[tuple] = []
        
        for tr in evidence:
            if tr not in seen:
                seen.add(tr)
                dedup_triples.append(tr)
                
        return dedup_triples, node_ids



    def get_chunks_from_nodes(self, node_ids: List[str]) -> List[str]:
        """Retrieve text chunks connected to given nodes via IN_CHUNK relationship.

            in: node_ids: List[str]
            out: List[str] - chunk texts
        """
        if not node_ids:
            return []

        try:
            results = self.graph.query("""
                MATCH (n)-[:IN_CHUNK]->(c:Chunk)
                WHERE n.id IN $node_ids
                RETURN DISTINCT c.text AS text, c.page AS page, n.id AS source_node
                ORDER BY c.page
            """, {"node_ids": node_ids})

            chunks = [record['text'] for record in results if record.get('text')]
            print(f"\nNajdenych {len(chunks)} textovych usekov spojenych s uzlami.\n")
            return chunks

        except Exception as e:
            print(f"Chyba pri ziskavani chunkov: {e}")
            return []
        
        

    def answer(self, question: str, evidence_triples: List[tuple], chunks: List[str] = []) -> str:
        """Stage 3 (KG-GPT): Derive a natural language answer from evidence triples and chunk context.

        Args:
            question: The original user question
            evidence_triples: List of (head, relation, tail) tuples from graph retrieval
            chunks: Text chunks connected to retrieved nodes via IN_CHUNK

        Returns:
            Natural language answer string
        """
        print("\n\nStage 3: Inference...\n\n")

        linearized = [[h, r, t] for h, r, t in evidence_triples]
        chunk_text = "\n---\n".join(chunks) if chunks else "Žiadne textové úseky neboli nájdené."

        user_prompt = f"""
        ## Vstupné údaje pre analýzu

        ### 1. Otázka používateľa
        {question}

        ### 2. Dôkazy z grafovej databázy (Primárne)
        Nasledujúce trojice reprezentujú overené fakty v štruktúre [Subjekt, Relácia, Objekt]:
        {json.dumps(linearized, ensure_ascii=False, indent=2)}

        ### 3. Textový kontext z dokumentov (Doplnkové)
        Tento text slúži na pochopenie širšieho kontextu a detailov:
        {chunk_text}

        ---
        **Inštrukcia:** Na základe vyššie uvedených údajov vypracuj stručnú, ale výstižnú odpoveď. Zameraj sa na to, aby odpoveď priamo adresovala otázku a prioritne využívala fakty z grafových trojíc.
        """

        class InferenceAnswer(BaseModel):
            """Natural language answer derived from evidence triples and chunk context."""
            answer: str = Field(description="Natural language answer grounded in the evidence triples and chunk context")

        structured_model = self.gemini_client_flash.with_structured_output(
            schema=InferenceAnswer, method="json_schema"
        )

        response = structured_model.invoke([
            ("system", system_prompt_for_inference),
            ("human", user_prompt),
        ])

        return cast(InferenceAnswer, response).answer
    
    

    # ---------------- INTERACTIVE QUESTIONING ----------------

    def query(self, question: str | None = None, verbose: bool = True) -> str:
        """KG-GPT 3-stage pipeline: Segment → Retrieve → Infer.

        Args:
            question: the user question. If None, read from stdin (interactive mode).
            verbose: print per-stage progress.

        Returns:
            The final answer string (also printed in interactive mode).
        """
        interactive = question is None
        if interactive:
            question = input("Enter your question: ")
            if question == '-h':
                print("Help Instructions: \n - To exit, type 'exit' \n - To view graph schema, type '-s' \n")
                return ""
            elif question == '-s':
                schema = self.get_graph_schema()
                print(schema)
                return str(schema)
            elif question.lower() == 'exit':
                print("Exiting...")
                return ""

        # Stage 1: Sentence Segmentation
        sub_sentences = self.segment_question(question)

        # Stage 2: Graph Retrieval (per sub-sentence)
        all_triples: List[tuple] = []
        all_node_ids: List[str] = []

        for sub in sub_sentences:
            if verbose:
                print(f"\nStage 2 (ARK-V1): Retrieving evidence for: '{sub.text}'")
            triples, node_ids = self.ark_v1_retrieve(sub.text, sub.entities)
            if verbose:
                print(f"  ARK-V1 found {len(triples)} triples, {len(node_ids)} nodes")
            all_triples.extend(triples)
            all_node_ids.extend(node_ids)

        # Retrieve source text chunks via IN_CHUNK
        chunks = self.get_chunks_from_nodes(list(dict.fromkeys(all_node_ids)))

        # Stage 3: Inference
        final_answer = self.answer(question, all_triples, chunks)
        
        if interactive:
            print(f"\n{final_answer}")
        return final_answer