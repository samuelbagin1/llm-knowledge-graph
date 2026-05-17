chunk: 0
path: ['§ 1']
path_as_text: Paragraf § 1
text: Tento zákon upravuje daň z pridanej hodnoty (ďalej len „daň“).

relationships:
  - Tento Zakon -> [OBSAHUJE] -> Paragraf § 1
  - Tento Zakon -> [UPRAVUJE] -> Dan Z Pridanej Hodnoty

nodes:
  PravnyPredpis: Tento Zakon
  Paragraf: Paragraf § 1
  Dan: Dan Z Pridanej Hodnoty


---

chunk: 12
path: ['§ 3', '3']
path_as_text: Paragraf § 3 Odsek 3
text: (3) Vykonávanie činnosti na základe pracovnoprávneho vzťahu, štátnozamestnaneckého pomeru, služobného pomeru alebo iného obdobného vzťahu, keď fyzická osoba je povinná dodržiavať pokyny alebo príkazy, čím sa vytvára stav podriadenosti a nadriadenosti z hľadiska podmienok vykonávanej činnosti a jej odmeňovania, sa nepovažuje za nezávislé vykonávanie činnosti podľa odseku 1.

relations:
  Paragraf § 3 -> [OBSAHUJE] -> Paragraf § 3 Odsek 3
  Paragraf § 3 -> [OBSAHUJE] -> Paragraf § 3 Odsek 1
  Paragraf § 3 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 3 Odsek 1
  Paragraf § 3 Odsek 3 -> [UPRAVUJE] -> Nezavisle Vykonavanie Cinnosti
  Vykonavanie Cinnosti Na Zaklade Pracovnopravneho Vztahu Statnozamestnaneckeho Pomeru Sluzobneho Pomeru Alebo Ineho Obdobneho Vztahu -> [NEVZTAHUJE_SA_NA] -> Nezavisle Vykonavanie Cinnosti
  Fyzicka Osoba -> [MA_POVINNOST] -> Povinnost Dodrziavat Pokyny Alebo Prikazy
  Stav Podriadenosti A Nadriadenosti -> [VYPLYVA_Z] -> Povinnost Dodrziavat Pokyny Alebo Prikazy
  Stav Podriadenosti A Nadriadenosti -> [VZTAHUJE_SA_NA] -> Podmienky Vykonavanej Cinnosti A Jej Odmenovania

nodes:
  Paragraf: Paragraf § 3
  Odsek: Paragraf § 3 Odsek 3
  Odsek: Paragraf § 3 Odsek 1
  Konanie: Vykonavanie Cinnosti Na Zaklade Pracovnopravneho Vztahu Statnozamestnaneckeho Pomeru Sluzobneho Pomeru Alebo Ineho Obdobneho Vztahu
  Konanie: Nezavisle Vykonavanie Cinnosti
  Osoba: Fyzicka Osoba
  Povinnost: Povinnost Dodrziavat Pokyny Alebo Prikazy
  Status: Stav Podriadenosti A Nadriadenosti
  Podmienka: Podmienky Vykonavanej Cinnosti A Jej Odmenovania


---

chunk: 31
path: ['§ 4', '4', 'b)']
path_as_text: Paragraf § 4 Odsek 4 Pismeno b)
text: (4) Ak nie je v odseku 10 písm. a) ustanovené inak, daňový úrad zaregistruje b) platiteľa, ktorý bol povinný podať žiadosť o registráciu pre daň podľa odseku 2 písm. b) až e), alebo platiteľa podľa odseku 8 písm. a), pridelí mu identifikačné číslo pre daň a vydá rozhodnutie o registrácii pre daň najneskôr do desiatich dní odo dňa doručenia žiadosti o registráciu pre daň z dôvodu, že  1. zdaniteľná osoba sa stala platiteľom podľa odseku 1 písm. b) alebo písm. i) alebo podľa odseku 8 písm. a), pričom identifikačné číslo pre daň nadobúda platnosť dňom, keď sa zdaniteľná osoba stala platiteľom, 2. zdaniteľná osoba sa stala platiteľom podľa odseku 1 písm. c) až h) a doručenia dokladov podľa odseku 3, pričom identifikačné číslo pre daň nadobúda platnosť dňom, keď sa zdaniteľná osoba stala platiteľom.

relations:
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 4
  Paragraf § 4 Odsek 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 4 Pismeno b)
  Paragraf § 4 Odsek 4 Pismeno b) -> [OBSAHUJE] -> Paragraf § 4 Odsek 4 Pismeno b) Bod 1
  Paragraf § 4 Odsek 4 Pismeno b) -> [OBSAHUJE] -> Paragraf § 4 Odsek 4 Pismeno b) Bod 2
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 10 Pismeno a)
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 2 Pismeno b)
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 2 Pismeno c)
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 2 Pismeno d)
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 2 Pismeno e)
  Paragraf § 4 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 8 Pismeno a)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 1 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno b)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 1 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno i)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 1 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 8 Pismeno a)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno c)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno d)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno e)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno f)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno g)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno h)
  Paragraf § 4 Odsek 4 Pismeno b) Bod 2 -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 3
  Platitel -> [MA_POVINNOST] -> Povinnost Podat Ziadost O Registraciu Pre Dan
  Povinnost Podat Ziadost O Registraciu Pre Dan -> [VZTAHUJE_SA_NA] -> Ziadost O Registraciu Pre Dan
  Ziadost O Registraciu Pre Dan -> [VZTAHUJE_SA_NA] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Danovy Urad -> [REGISTRUJE] -> Platitel
  Platitel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Registracii Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [ROZHODUJE_O] -> Registracia Pre Dan
  Danovy Urad -> [MA_LEHOTU] -> Lehota Desat Dni Od Dorucenia Ziadosti O Registraciu Pre Dan
  Danovy Urad -> [PRIJIMA] -> Ziadost O Registraciu Pre Dan
  Zdanitelna Osoba -> [JE_TYPOM] -> Platitel
  Identifikacne Cislo Pre Dan -> [MA_STATUS] -> Platnost Identifikacneho Cisla Pre Dan
  Platnost Identifikacneho Cisla Pre Dan -> [MA_DATUM] -> Den Ked Sa Zdanitelna Osoba Stala Platitelom
  Doklady Podla Paragraf § 4 Odsek 3 -> [VYPLYVA_Z] -> Paragraf § 4 Odsek 3

nodes:
  Paragraf: Paragraf § 4
  Odsek: Paragraf § 4 Odsek 4
  Pismeno: Paragraf § 4 Odsek 4 Pismeno b)
  Bod: Paragraf § 4 Odsek 4 Pismeno b) Bod 1
  Bod: Paragraf § 4 Odsek 4 Pismeno b) Bod 2
  Odsek: Paragraf § 4 Odsek 10
  Pismeno: Paragraf § 4 Odsek 10 Pismeno a)
  Odsek: Paragraf § 4 Odsek 2
  Pismeno: Paragraf § 4 Odsek 2 Pismeno b)
  Pismeno: Paragraf § 4 Odsek 2 Pismeno c)
  Pismeno: Paragraf § 4 Odsek 2 Pismeno d)
  Pismeno: Paragraf § 4 Odsek 2 Pismeno e)
  Odsek: Paragraf § 4 Odsek 8
  Pismeno: Paragraf § 4 Odsek 8 Pismeno a)
  Odsek: Paragraf § 4 Odsek 1
  Pismeno: Paragraf § 4 Odsek 1 Pismeno b)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno c)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno d)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno e)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno f)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno g)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno h)
  Pismeno: Paragraf § 4 Odsek 1 Pismeno i)
  Odsek: Paragraf § 4 Odsek 3
  Organizacia: Danovy Urad
  Subjekt: Platitel
  Subjekt: Zdanitelna Osoba
  Povinnost: Povinnost Podat Ziadost O Registraciu Pre Dan
  Ziadost: Ziadost O Registraciu Pre Dan
  Registracia: Registracia Pre Dan
  Zaznam: Identifikacne Cislo Pre Dan
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Dan: Dan
  Lehota: Lehota Desat Dni Od Dorucenia Ziadosti O Registraciu Pre Dan
  Dokument: Doklady Podla Paragraf § 4 Odsek 3
  Status: Platnost Identifikacneho Cisla Pre Dan
  Datum: Den Ked Sa Zdanitelna Osoba Stala Platitelom


---

chunk: 45
path: ['§ 4', '14', 'a)']
path_as_text: Paragraf § 4 Odsek 14 Pismeno a)
text: (14) Na účely tohto zákona sa a) bydliskom rozumie adresa trvalého pobytu fyzickej osoby v tuzemsku a u fyzickej osoby, ktorá nemá trvalý pobyt v tuzemsku, sa bydliskom rozumie trvalé miesto jej pobytu v zahraničí,

relations:
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 4
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 14
  Paragraf § 4 Odsek 14 -> [OBSAHUJE] -> Paragraf § 4 Odsek 14 Pismeno a)
  Paragraf § 4 Odsek 14 Pismeno a) -> [DEFINUJE] -> Bydlisko
  Bydlisko -> [VZTAHUJE_SA_NA] -> Tento Zakon
  Bydlisko -> [JE_TYPOM] -> Adresa Trvaleho Pobytu
  Adresa Trvaleho Pobytu -> [VZTAHUJE_SA_NA] -> Fyzicka Osoba
  Adresa Trvaleho Pobytu -> [NACHADZA_SA_V] -> Tuzemsko
  Bydlisko -> [JE_TYPOM] -> Trvale Miesto Pobytu
  Trvale Miesto Pobytu -> [VZTAHUJE_SA_NA] -> Fyzicka Osoba Bez Trvaleho Pobytu V Tuzemsku
  Trvale Miesto Pobytu -> [NACHADZA_SA_V] -> Zahranicie
  Fyzicka Osoba Bez Trvaleho Pobytu V Tuzemsku -> [NESPLNA_PODMIENKY] -> Trvaly Pobyt V Tuzemsku

nodes:
  PravnyPredpis: Tento Zakon
  Paragraf: Paragraf § 4
  Odsek: Paragraf § 4 Odsek 14
  Pismeno: Paragraf § 4 Odsek 14 Pismeno a)
  Adresa: Bydlisko
  Adresa: Adresa Trvaleho Pobytu
  Adresa: Trvale Miesto Pobytu
  Osoba: Fyzicka Osoba
  Osoba: Fyzicka Osoba Bez Trvaleho Pobytu V Tuzemsku
  Podmienka: Trvaly Pobyt V Tuzemsku
  Lokacia: Tuzemsko
  Lokacia: Zahranicie


---

chunk: 62
path: ['§ 4b', '6']
path_as_text: Paragraf § 4b Odsek 6
text: (6) Ak sa člen skupiny rozhodne vystúpiť zo skupiny alebo musí vystúpiť zo skupiny z dôvodu neplnenia podmienok podľa § 4a, zástupca skupiny je povinný bezodkladne podať žiadosť o zmenu registrácie skupiny; ak je vystupujúcim členom skupiny zástupca skupiny, žiadosť musí obsahovať aj označenie člena skupiny, ktorý bol určený členmi skupiny ako nový zástupca skupiny. Daňový úrad vydá bezodkladne rozhodnutie o zmene registrácie skupiny, proti ktorému nemožno podať odvolanie; účinky zmeny registrácie skupiny nastávajú v deň uvedený v rozhodnutí, ktorý nesmie byť neskorší ako 30. deň odo dňa podania žiadosti o zmenu registrácie skupiny. Daňový úrad, ktorý je miestne príslušný pre vystupujúceho člena skupiny, zaregistruje vystupujúceho člena skupiny za samostatného platiteľa ku dňu, keď nastali účinky zmeny registrácie skupiny a pridelí mu identifikačné číslo pre daň; proti tomuto rozhodnutiu nemožno podať odvolanie. Práva a povinnosti skupiny vyplývajúce z tohto zákona prechádzajú na zdaniteľnú osobu, ktorá vystúpila zo skupiny, dňom, keď nastali účinky zmeny registrácie skupiny, a to v rozsahu, v akom sa vzťahujú na plnenia uskutočnené a prijaté touto zdaniteľnou osobou.

relations:
  Paragraf § 4b -> [OBSAHUJE] -> Paragraf § 4b Odsek 6
  Paragraf § 4b Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 4a

  Clen Skupiny -> [PATRI_DO] -> Skupina
  Vystupujuci Clen Skupiny -> [JE_TYPOM] -> Clen Skupiny
  Vystupujuci Clen Skupiny -> [PATRI_DO] -> Skupina
  Vystupujuci Clen Skupiny -> [NESPLNA_PODMIENKY] -> Podmienky Podla Paragrafu § 4a
  Podmienky Podla Paragrafu § 4a -> [ODKAZUJE_NA] -> Paragraf § 4a

  Zastupca Skupiny -> [MA_POVINNOST] -> Povinnost Podat Ziadost O Zmenu Registracie Skupiny
  Povinnost Podat Ziadost O Zmenu Registracie Skupiny -> [VZTAHUJE_SA_NA] -> Ziadost O Zmenu Registracie Skupiny
  Povinnost Podat Ziadost O Zmenu Registracie Skupiny -> [MA_LEHOTU] -> Lehota Bezodkladne
  Zastupca Skupiny -> [PODAVA] -> Ziadost O Zmenu Registracie Skupiny
  Ziadost O Zmenu Registracie Skupiny -> [VZTAHUJE_SA_NA] -> Zmena Registracie Skupiny

  Ziadost O Zmenu Registracie Skupiny -> [OBSAHUJE] -> Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny
  Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny -> [VZTAHUJE_SA_NA] -> Novy Zastupca Skupiny
  Clenovia Skupiny -> [URCUJE] -> Novy Zastupca Skupiny

  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Zmene Registracie Skupiny
  Rozhodnutie O Zmene Registracie Skupiny -> [ROZHODUJE_O] -> Zmena Registracie Skupiny
  Rozhodnutie O Zmene Registracie Skupiny -> [NEMA_NAROK_NA] -> Odvolanie
  Danovy Urad -> [MA_LEHOTU] -> Lehota Bezodkladne

  Zmena Registracie Skupiny -> [MA_STATUS] -> Ucinok Zmeny Registracie Skupiny
  Ucinok Zmeny Registracie Skupiny -> [MA_DATUM] -> Den Uvedeny V Rozhodnuti
  Den Uvedeny V Rozhodnuti -> [VYPLYVA_Z] -> Rozhodnutie O Zmene Registracie Skupiny
  Ucinok Zmeny Registracie Skupiny -> [MA_LEHOTU] -> Lehota Najneskor 30 Dni Od Podania Ziadosti O Zmenu Registracie Skupiny

  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [VZTAHUJE_SA_NA] -> Vystupujuci Clen Skupiny
  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [REGISTRUJE] -> Vystupujuci Clen Skupiny
  Vystupujuci Clen Skupiny -> [MA_STATUS] -> Samostatny Platitel
  Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela -> [VZTAHUJE_SA_NA] -> Vystupujuci Clen Skupiny
  Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela -> [MA_DATUM] -> Ucinok Zmeny Registracie Skupiny
  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [VYDAVA] -> Rozhodnutie O Registracii Za Samostatneho Platitela
  Rozhodnutie O Registracii Za Samostatneho Platitela -> [ROZHODUJE_O] -> Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela
  Rozhodnutie O Registracii Za Samostatneho Platitela -> [NEMA_NAROK_NA] -> Odvolanie
  Vystupujuci Clen Skupiny -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Dan

  Prava A Povinnosti Skupiny Vyplyvajuce Z Tohto Zakona -> [VYPLYVA_Z] -> Tento Zakon
  Prava A Povinnosti Skupiny Vyplyvajuce Z Tohto Zakona -> [PRECHADZA_NA] -> Zdanitelna Osoba Ktora Vystupila Zo Skupiny
  Prava A Povinnosti Skupiny Vyplyvajuce Z Tohto Zakona -> [MA_DATUM] -> Ucinok Zmeny Registracie Skupiny
  Prava A Povinnosti Skupiny Vyplyvajuce Z Tohto Zakona -> [VZTAHUJE_SA_NA] -> Plnenia Uskutocnene A Prijate Zdanitelnou Osobou
  Plnenia Uskutocnene A Prijate Zdanitelnou Osobou -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba Ktora Vystupila Zo Skupiny

nodes:
  Paragraf: Paragraf § 4b
  Odsek: Paragraf § 4b Odsek 6
  Paragraf: Paragraf § 4a
  Subjekt: Clen Skupiny
  Subjekt: Vystupujuci Clen Skupiny
  Subjekt: Zastupca Skupiny
  Subjekt: Novy Zastupca Skupiny
  Subjekt: Clenovia Skupiny
  Subjekt: Skupina
  Podmienka: Podmienky Podla Paragrafu § 4a
  Povinnost: Povinnost Podat Ziadost O Zmenu Registracie Skupiny
  Ziadost: Ziadost O Zmenu Registracie Skupiny
  Registracia: Zmena Registracie Skupiny
  Zaznam: Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny
  Organizacia: Danovy Urad
  Rozhodnutie: Rozhodnutie O Zmene Registracie Skupiny
  Pravo: Odvolanie
  Status: Ucinok Zmeny Registracie Skupiny
  Datum: Den Uvedeny V Rozhodnuti
  Lehota: Lehota Bezodkladne
  Lehota: Lehota Najneskor 30 Dni Od Podania Ziadosti O Zmenu Registracie Skupiny
  Organizacia: Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny
  Registracia: Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela
  Status: Samostatny Platitel
  Rozhodnutie: Rozhodnutie O Registracii Za Samostatneho Platitela
  Zaznam: Identifikacne Cislo Pre Dan
  Dan: Dan
  Povinnost: Prava A Povinnosti Skupiny Vyplyvajuce Z Tohto Zakona
  PravnyPredpis: Tento Zakon
  Osoba: Zdanitelna Osoba Ktora Vystupila Zo Skupiny
  Sluzba: Plnenia Uskutocnene A Prijate Zdanitelnou Osobou


---

chunk: 78
path: ['§ 5', '1', 'c)']
path_as_text: Paragraf § 5 Odsek 1 Pismeno c)
text: (1) Zdaniteľná osoba, ktorá nemá v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo sa v tuzemsku obvykle nezdržiava (ďalej len „zahraničná osoba“), sa stáva platiteľom c) nadobudnutím tovaru v tuzemsku z iného členského štátu, ktoré je predmetom dane podľa § 2, ak nejde o malý podnik zahraničnej osoby, ktorý uplatňuje oslobodenie od dane podľa § 68f ods. 2, a ak nejde o nadobudnutie tovaru v tuzemsku z iného členského štátu, ktoré sa považuje za zdanené podľa § 45 ods. 2.

relations:
  Paragraf § 5 -> [OBSAHUJE] -> Paragraf § 5 Odsek 1
  Paragraf § 5 Odsek 1 -> [OBSAHUJE] -> Paragraf § 5 Odsek 1 Pismeno c)
  Paragraf § 68f -> [OBSAHUJE] -> Paragraf § 68f Odsek 2
  Paragraf § 45 -> [OBSAHUJE] -> Paragraf § 45 Odsek 2

  Paragraf § 5 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 2
  Paragraf § 5 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 68f Odsek 2
  Paragraf § 5 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 45 Odsek 2
  Paragraf § 5 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Vznik Statusu Platitela Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu

  Zahranicna Osoba -> [JE_TYPOM] -> Zdanitelna Osoba
  Zahranicna Osoba -> [MA_PODMIENKU] -> Nema Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Obvykle Zdrziavanie V Tuzemsku

  Zahranicna Osoba -> [NADOBUDA] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Zahranicna Osoba -> [MA_STATUS] -> Platitel
  Platitel -> [VYPLYVA_Z] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu

  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [NACHADZA_SA_V] -> Tuzemsko
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Tovar Z Ineho Clenskeho Statu
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [JE_PREDMETOM_DANE] -> Dan
  Dan -> [JE_PODLA] -> Paragraf § 2

  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Nejde O Maly Podnik Zahranicnej Osoby Uplatnujuci Oslobodenie Od Dane Podla Paragraf § 68f Odsek 2
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Nejde O Nadobudnutie Tovaru Povazovane Za Zdanene Podla Paragraf § 45 Odsek 2

  Maly Podnik Zahranicnej Osoby -> [JE_TYPOM] -> Zahranicna Osoba
  Maly Podnik Zahranicnej Osoby -> [MA_PRAVO] -> Oslobodenie Od Dane
  Oslobodenie Od Dane -> [VYPLYVA_Z] -> Paragraf § 68f Odsek 2

  Nadobudnutie Tovaru Povazovane Za Zdanene -> [VYPLYVA_Z] -> Paragraf § 45 Odsek 2

nodes:
  Paragraf: Paragraf § 5
  Odsek: Paragraf § 5 Odsek 1
  Pismeno: Paragraf § 5 Odsek 1 Pismeno c)
  Paragraf: Paragraf § 2
  Paragraf: Paragraf § 68f
  Odsek: Paragraf § 68f Odsek 2
  Paragraf: Paragraf § 45
  Odsek: Paragraf § 45 Odsek 2

  Subjekt: Zdanitelna Osoba
  Subjekt: Zahranicna Osoba
  Subjekt: Maly Podnik Zahranicnej Osoby

  Status: Platitel
  Stat: Tuzemsko
  Stat: Iny Clensky Stat

  Konanie: Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Konanie: Vznik Statusu Platitela Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu

  Tovar: Tovar
  Tovar: Tovar Z Ineho Clenskeho Statu
  Dan: Dan
  Pravo: Oslobodenie Od Dane

  Podmienka: Nema Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Obvykle Zdrziavanie V Tuzemsku
  Podmienka: Nejde O Maly Podnik Zahranicnej Osoby Uplatnujuci Oslobodenie Od Dane Podla Paragraf § 68f Odsek 2
  Podmienka: Nejde O Nadobudnutie Tovaru Povazovane Za Zdanene Podla Paragraf § 45 Odsek 2

  Status: Nadobudnutie Tovaru Povazovane Za Zdanene



---

chunk: 93
path: ['§ 5a']
path_as_text: Paragraf § 5a
text: Na účely tohto zákona sa za platiteľa, ktorý má pridelené identifikačné číslo pre daň podľa § 4, § 4b, § 4c alebo § 5, považuje platiteľ, ktorý a) sa stal platiteľom po doručení rozhodnutia o registrácii pre daň podľa § 4, a to počnúc dňom, keď sa stal platiteľom; ak platiteľ nesplnil oznamovaciu povinnosť podľa § 4 ods. 5, tak počnúc 1. januárom kalendárneho roka nasledujúceho po kalendárnom roku, v ktorom presiahol obrat podľa § 4 ods. 1 písm. a),  b) sa stal platiteľom pred doručením rozhodnutia o registrácii pre daň podľa § 4 alebo § 5, a to počnúc dňom doručenia tohto rozhodnutia alebo c) je skupinou, počnúc dňom, ku ktorému daňový úrad vykoná registráciu skupiny.

relations:
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 5a

  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 5
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1
  Paragraf § 4 Odsek 1 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1 Pismeno a)

  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 4
  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 4b
  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 4c
  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 5
  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 5
  Paragraf § 5a -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 1 Pismeno a)

  Paragraf § 5a -> [DEFINUJE] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel S Identifikacnym Cislom Pre Dan -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacne Cislo Pre Dan -> [JE_PODLA] -> Paragraf § 4
  Identifikacne Cislo Pre Dan -> [JE_PODLA] -> Paragraf § 4b
  Identifikacne Cislo Pre Dan -> [JE_PODLA] -> Paragraf § 4c
  Identifikacne Cislo Pre Dan -> [JE_PODLA] -> Paragraf § 5

  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 -> [MA_PODMIENKU] -> Stal Sa Platitelom Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 -> [MA_DATUM] -> Den Ked Sa Stal Platitelom

  Rozhodnutie O Registracii Pre Dan -> [ROZHODUJE_O] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Rozhodnutie O Registracii Pre Dan -> [JE_PODLA] -> Paragraf § 4

  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 -> [MA_PODMIENKU] -> Nesplnenie Oznamovacej Povinnosti Podla Paragraf § 4 Odsek 5
  Nesplnenie Oznamovacej Povinnosti Podla Paragraf § 4 Odsek 5 -> [VYPLYVA_Z] -> Oznamovacia Povinnost
  Oznamovacia Povinnost -> [JE_PODLA] -> Paragraf § 4 Odsek 5
  Nesplnenie Oznamovacej Povinnosti Podla Paragraf § 4 Odsek 5 -> [MA_PODMIENKU] -> Presiahnutie Obratu Podla Paragraf § 4 Odsek 1 Pismeno a)
  Presiahnutie Obratu Podla Paragraf § 4 Odsek 1 Pismeno a) -> [VZTAHUJE_SA_NA] -> Obrat
  Presiahnutie Obratu Podla Paragraf § 4 Odsek 1 Pismeno a) -> [JE_PODLA] -> Paragraf § 4 Odsek 1 Pismeno a)
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 -> [MA_DATUM] -> 1. Januar Kalendarneho Roka Nasledujuceho Po Kalendarnom Roku Presiahnutia Obratu

  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5 -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5 -> [MA_PODMIENKU] -> Stal Sa Platitelom Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5
  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5 -> [MA_DATUM] -> Den Dorucenia Rozhodnutia
  Rozhodnutie O Registracii Pre Dan -> [JE_PODLA] -> Paragraf § 5

  Skupina -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Skupina -> [MA_PODMIENKU] -> Je Skupinou
  Danovy Urad -> [REGISTRUJE] -> Skupina
  Registracia Skupiny -> [VZTAHUJE_SA_NA] -> Skupina
  Registracia Skupiny -> [MA_DATUM] -> Den Ku Ktoremu Danovy Urad Vykona Registraciu Skupiny
  Skupina -> [MA_DATUM] -> Den Ku Ktoremu Danovy Urad Vykona Registraciu Skupiny

nodes:
  PravnyPredpis: Tento Zakon
  Paragraf: Paragraf § 5a
  Paragraf: Paragraf § 4
  Paragraf: Paragraf § 4b
  Paragraf: Paragraf § 4c
  Paragraf: Paragraf § 5
  Odsek: Paragraf § 4 Odsek 5
  Odsek: Paragraf § 4 Odsek 1
  Pismeno: Paragraf § 4 Odsek 1 Pismeno a)

  Subjekt: Platitel S Identifikacnym Cislom Pre Dan
  Subjekt: Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4
  Subjekt: Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5
  Organizacia: Skupina
  Organizacia: Danovy Urad

  Dan: Dan
  Zaznam: Identifikacne Cislo Pre Dan
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Registracia: Registracia Pre Dan
  Registracia: Registracia Skupiny
  Povinnost: Oznamovacia Povinnost
  Obrat: Obrat

  Podmienka: Stal Sa Platitelom Po Doruceni Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4
  Podmienka: Nesplnenie Oznamovacej Povinnosti Podla Paragraf § 4 Odsek 5
  Podmienka: Presiahnutie Obratu Podla Paragraf § 4 Odsek 1 Pismeno a)
  Podmienka: Stal Sa Platitelom Pred Dorucenim Rozhodnutia O Registracii Pre Dan Podla Paragraf § 4 Alebo Paragraf § 5
  Podmienka: Je Skupinou

  Datum: Den Ked Sa Stal Platitelom
  Datum: Den Dorucenia Rozhodnutia
  Datum: 1. Januar Kalendarneho Roka Nasledujuceho Po Kalendarnom Roku Presiahnutia Obratu
  Datum: Den Ku Ktoremu Danovy Urad Vykona Registraciu Skupiny


---

chunk: 103
path: ['§ 6a', '2']
path_as_text: Paragraf § 6a Odsek 2
text: (2) Ak zdaniteľná osoba spĺňa podmienky na registráciu podľa § 5 a je registrovaná podľa § 4, považuje sa za platiteľa registrovaného podľa § 5 odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava; túto skutočnosť je povinná oznámiť daňovému úradu do desiatich dní odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava.

relations:
  Paragraf § 6a -> [OBSAHUJE] -> Paragraf § 6a Odsek 2
  Paragraf § 6a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 5
  Paragraf § 6a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 4

  Zdanitelna Osoba -> [SPLNA_PODMIENKY] -> Podmienky Na Registraciu Podla Paragraf § 5
  Podmienky Na Registraciu Podla Paragraf § 5 -> [JE_PODLA] -> Paragraf § 5

  Zdanitelna Osoba -> [MA] -> Registracia Podla Paragraf § 4
  Registracia Podla Paragraf § 4 -> [JE_PODLA] -> Paragraf § 4

  Zdanitelna Osoba -> [MA_STATUS] -> Platitel Registrovany Podla Paragraf § 5
  Platitel Registrovany Podla Paragraf § 5 -> [JE_PODLA] -> Paragraf § 5
  Platitel Registrovany Podla Paragraf § 5 -> [MA_PODMIENKU] -> Podmienky Na Registraciu Podla Paragraf § 5
  Platitel Registrovany Podla Paragraf § 5 -> [MA_PODMIENKU] -> Registracia Podla Paragraf § 4
  Platitel Registrovany Podla Paragraf § 5 -> [MA_DATUM] -> Den Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania

  Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Sidlo V Tuzemsku
  Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Miesto Podnikania V Tuzemsku
  Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Prevadzkaren V Tuzemsku
  Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Bydlisko V Tuzemsku
  Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Miesto Obvykleho Zdrziavania V Tuzemsku

  Sidlo V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Podnikania V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Prevadzkaren V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Bydlisko V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Obvykleho Zdrziavania V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko

  Zdanitelna Osoba -> [MA_POVINNOST] -> Povinnost Oznamenia Skutocnosti Danovemu Uradu
  Povinnost Oznamenia Skutocnosti Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Oznamenie Skutocnosti Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania
  Povinnost Oznamenia Skutocnosti Danovemu Uradu -> [MA_LEHOTU] -> Lehota Do Desiatich Dni
  Lehota Do Desiatich Dni -> [VZTAHUJE_SA_NA] -> Den Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania

  Zdanitelna Osoba -> [OZNAMUJE] -> Oznamenie Skutocnosti Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania
  Oznamenie Skutocnosti Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania
  Oznamenie Skutocnosti Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania -> [VZTAHUJE_SA_NA] -> Danovy Urad

nodes:
  Paragraf: Paragraf § 6a
  Odsek: Paragraf § 6a Odsek 2
  Paragraf: Paragraf § 5
  Paragraf: Paragraf § 4

  Subjekt: Zdanitelna Osoba
  Subjekt: Platitel Registrovany Podla Paragraf § 5
  Organizacia: Danovy Urad

  Registracia: Registracia Podla Paragraf § 5
  Registracia: Registracia Podla Paragraf § 4

  Podmienka: Podmienky Na Registraciu Podla Paragraf § 5
  Dovod: Skutocnost Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania

  Stat: Tuzemsko
  Adresa: Sidlo V Tuzemsku
  Adresa: Miesto Podnikania V Tuzemsku
  Lokacia: Prevadzkaren V Tuzemsku
  Adresa: Bydlisko V Tuzemsku
  Lokacia: Miesto Obvykleho Zdrziavania V Tuzemsku

  Povinnost: Povinnost Oznamenia Skutocnosti Danovemu Uradu
  Oznamenie: Oznamenie Skutocnosti Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania
  Lehota: Lehota Do Desiatich Dni
  Datum: Den Prestania Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkarne Bydlisko Alebo Miesto Obvykleho Zdrziavania


---

chunk: 124
path: ['§ 8', '4', 'h)']
path_as_text: Paragraf § 8 Odsek 4 Pismeno h)
text: (4) Za dodanie tovaru sa považuje aj premiestnenie tovaru, ktorý je vo vlastníctve zdaniteľnej osoby, z tuzemska do iného členského štátu, ak je tento tovar odoslaný alebo prepravený ňou alebo na jej účet do iného členského štátu na účely jej podnikania. Takéto premiestnenie sa považuje za dodanie tovaru za protihodnotu okrem premiestnenia tovaru, ktoré spĺňa podmienky režimu call-off stock podľa § 8a, a okrem premiestnenia tovaru h) na dočasné použitie na obdobie nepresahujúce 24 mesiacov na území iného členského štátu, v ktorom by sa dovoz toho istého tovaru z územia tretieho štátu považoval za prepustený do režimu dočasné použitie s úplným oslobodením od dovozného cla,

relations:
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno h)
  Paragraf § 8 Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 8a

  Paragraf § 8 Odsek 4 -> [UPRAVUJE] -> Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [JE_TYPOM] -> Dodanie Tovaru
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [JE_TYPOM] -> Dodanie Tovaru Za Protihodnotu
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Tovar Vo Vlastnictve Zdanitelnej Osoby
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Odoslanie Alebo Preprava Tovaru Zdanitelnou Osobou Alebo Na Jej Ucet
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Ucel Podnikania Zdanitelnej Osoby

  Tovar Vo Vlastnictve Zdanitelnej Osoby -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Vo Vlastnictve Zdanitelnej Osoby -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tuzemsko
  Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat

  Premiestnenie Tovaru Splnajuce Podmienky Rezimu Call-Off Stock Podla Paragraf § 8a -> [NEVZTAHUJE_SA_NA] -> Dodanie Tovaru Za Protihodnotu
  Premiestnenie Tovaru Splnajuce Podmienky Rezimu Call-Off Stock Podla Paragraf § 8a -> [MA_PODMIENKU] -> Podmienky Rezimu Call-Off Stock
  Podmienky Rezimu Call-Off Stock -> [JE_PODLA] -> Paragraf § 8a

  Paragraf § 8 Odsek 4 Pismeno h) -> [UPRAVUJE] -> Premiestnenie Tovaru Na Docasne Pouzitie
  Premiestnenie Tovaru Na Docasne Pouzitie -> [NEVZTAHUJE_SA_NA] -> Dodanie Tovaru Za Protihodnotu
  Premiestnenie Tovaru Na Docasne Pouzitie -> [VZTAHUJE_SA_NA] -> Tovar
  Premiestnenie Tovaru Na Docasne Pouzitie -> [MA_PODMIENKU] -> Docasne Pouzitie Tovaru
  Docasne Pouzitie Tovaru -> [MA_OBDOBIE] -> Obdobie Nepresahujuce 24 Mesiacov
  Docasne Pouzitie Tovaru -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Premiestnenie Tovaru Na Docasne Pouzitie -> [MA_PODMIENKU] -> Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu Povazovany Za Prepusteny Do Rezimu Docasne Pouzitie S Uplnym Oslobodenim Od Dovozneho Cla

  Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu -> [VZTAHUJE_SA_NA] -> Treti Stat
  Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu -> [MA_STATUS] -> Rezim Docasne Pouzitie
  Rezim Docasne Pouzitie -> [MA_PRAVO] -> Uplne Oslobodenie Od Dovozneho Cla
  Uplne Oslobodenie Od Dovozneho Cla -> [OSLOBODZUJE_OD] -> Dovozne Clo

nodes:
  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 4
  Pismeno: Paragraf § 8 Odsek 4 Pismeno h)
  Paragraf: Paragraf § 8a

  Subjekt: Zdanitelna Osoba
  Stat: Tuzemsko
  Stat: Iny Clensky Stat
  Stat: Treti Stat

  Tovar: Tovar
  Konanie: Dodanie Tovaru
  Konanie: Dodanie Tovaru Za Protihodnotu
  Konanie: Premiestnenie Tovaru Vo Vlastnictve Zdanitelnej Osoby Z Tuzemska Do Ineho Clenskeho Statu
  Konanie: Premiestnenie Tovaru Splnajuce Podmienky Rezimu Call-Off Stock Podla Paragraf § 8a
  Konanie: Premiestnenie Tovaru Na Docasne Pouzitie
  Konanie: Docasne Pouzitie Tovaru
  Konanie: Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu

  Podmienka: Tovar Vo Vlastnictve Zdanitelnej Osoby
  Podmienka: Odoslanie Alebo Preprava Tovaru Zdanitelnou Osobou Alebo Na Jej Ucet
  Podmienka: Ucel Podnikania Zdanitelnej Osoby
  Podmienka: Podmienky Rezimu Call-Off Stock
  Podmienka: Dovoz Toho Isteho Tovaru Z Uzemia Tretieho Statu Povazovany Za Prepusteny Do Rezimu Docasne Pouzitie S Uplnym Oslobodenim Od Dovozneho Cla

  Obdobie: Obdobie Nepresahujuce 24 Mesiacov
  Status: Rezim Docasne Pouzitie
  Pravo: Uplne Oslobodenie Od Dovozneho Cla
  Dan: Dovozne Clo


---

chunk: 126
path: ['§ 8', '5']
path_as_text: Paragraf § 8 Odsek 5
text: (5) Okamihom, keď sa prestane plniť niektorá z podmienok podľa odseku 4 písm. a) až i), považuje sa takéto premiestnenie tovaru za dodanie tovaru za protihodnotu.

relations:
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 5
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno a)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno b)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno c)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno d)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno e)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno f)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno g)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno h)
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno i)

  Paragraf § 8 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 8 Odsek 4
  Paragraf § 8 Odsek 5 -> [UPRAVUJE] -> Premiestnenie Tovaru Povazovane Za Dodanie Tovaru Za Protihodnotu

  Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i) -> [JE_PODLA] -> Paragraf § 8 Odsek 4
  Prestanie Plnenia Niektorej Z Podmienok Podla Paragraf § 8 Odsek 4 Pismeno a) Az i) -> [VZTAHUJE_SA_NA] -> Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)

  Premiestnenie Tovaru -> [MA_PODMIENKU] -> Prestanie Plnenia Niektorej Z Podmienok Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)
  Premiestnenie Tovaru -> [MA_DATUM] -> Okamih Prestania Plnenia Niektorej Z Podmienok
  Premiestnenie Tovaru -> [JE_TYPOM] -> Dodanie Tovaru Za Protihodnotu

  Dodanie Tovaru Za Protihodnotu -> [VYPLYVA_Z] -> Prestanie Plnenia Niektorej Z Podmienok Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)

nodes:
  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 5
  Odsek: Paragraf § 8 Odsek 4
  Pismeno: Paragraf § 8 Odsek 4 Pismeno a)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno b)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno c)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno d)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno e)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno f)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno g)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno h)
  Pismeno: Paragraf § 8 Odsek 4 Pismeno i)

  Konanie: Premiestnenie Tovaru
  Konanie: Premiestnenie Tovaru Povazovane Za Dodanie Tovaru Za Protihodnotu
  Konanie: Dodanie Tovaru Za Protihodnotu

  Podmienka: Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)
  Dovod: Prestanie Plnenia Niektorej Z Podmienok Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)
  Datum: Okamih Prestania Plnenia Niektorej Z Podmienok


---

chunk: 149
path: ['§ 9a', '1', 'b)']
path_as_text: Paragraf § 9a Odsek 1 Pismeno b)
text: (1) Na účely tohto zákona je b) jednoúčelovým poukazom poukaz, pri ktorom je v čase jeho vystavenia známe miesto dodania tovaru alebo miesto dodania služby, na ktoré sa poukaz vzťahuje, a daň splatná z tohto tovaru alebo služby,

relations:
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 9a
  Paragraf § 9a -> [OBSAHUJE] -> Paragraf § 9a Odsek 1
  Paragraf § 9a Odsek 1 -> [OBSAHUJE] -> Paragraf § 9a Odsek 1 Pismeno b)

  Paragraf § 9a Odsek 1 Pismeno b) -> [DEFINUJE] -> Jednoucelovy Poukaz
  Jednoucelovy Poukaz -> [JE_TYPOM] -> Poukaz

  Jednoucelovy Poukaz -> [MA_PODMIENKU] -> Zname Miesto Dodania Tovaru Alebo Miesto Dodania Sluzby V Case Vystavenia Poukazu
  Jednoucelovy Poukaz -> [MA_PODMIENKU] -> Znama Dan Splatna Z Tovaru Alebo Sluzby V Case Vystavenia Poukazu

  Jednoucelovy Poukaz -> [VZTAHUJE_SA_NA] -> Tovar
  Jednoucelovy Poukaz -> [VZTAHUJE_SA_NA] -> Sluzba

  Zname Miesto Dodania Tovaru Alebo Miesto Dodania Sluzby V Case Vystavenia Poukazu -> [MA_DATUM] -> Cas Vystavenia Poukazu
  Zname Miesto Dodania Tovaru Alebo Miesto Dodania Sluzby V Case Vystavenia Poukazu -> [VZTAHUJE_SA_NA] -> Miesto Dodania Tovaru
  Zname Miesto Dodania Tovaru Alebo Miesto Dodania Sluzby V Case Vystavenia Poukazu -> [VZTAHUJE_SA_NA] -> Miesto Dodania Sluzby

  Miesto Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Dodania Sluzby -> [VZTAHUJE_SA_NA] -> Sluzba

  Znama Dan Splatna Z Tovaru Alebo Sluzby V Case Vystavenia Poukazu -> [MA_DATUM] -> Cas Vystavenia Poukazu
  Znama Dan Splatna Z Tovaru Alebo Sluzby V Case Vystavenia Poukazu -> [VZTAHUJE_SA_NA] -> Dan Splatna Z Tovaru Alebo Sluzby
  Dan Splatna Z Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Tovar
  Dan Splatna Z Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Sluzba

nodes:
  PravnyPredpis: Tento Zakon
  Paragraf: Paragraf § 9a
  Odsek: Paragraf § 9a Odsek 1
  Pismeno: Paragraf § 9a Odsek 1 Pismeno b)

  Dokument: Jednoucelovy Poukaz
  Dokument: Poukaz

  Podmienka: Zname Miesto Dodania Tovaru Alebo Miesto Dodania Sluzby V Case Vystavenia Poukazu
  Podmienka: Znama Dan Splatna Z Tovaru Alebo Sluzby V Case Vystavenia Poukazu

  Datum: Cas Vystavenia Poukazu
  Lokacia: Miesto Dodania Tovaru
  Lokacia: Miesto Dodania Sluzby
  Tovar: Tovar
  Sluzba: Sluzba
  Dan: Dan Splatna Z Tovaru Alebo Sluzby


---

chunk: 155
path: ['§ 9a', '6']
path_as_text: Paragraf § 9a Odsek 6
text: (6) Ak prevod viacúčelového poukazu uskutoční iná zdaniteľná osoba ako dodávateľ tovaru alebo dodávateľ služby podľa odseku 5, každá služba dodaná v súvislosti s prevodom viacúčelového poukazu touto inou zdaniteľnou osobou, ako napríklad distribučná služba alebo propagačná služba, je samostatne predmetom dane.

relations:
  Paragraf § 9a -> [OBSAHUJE] -> Paragraf § 9a Odsek 6
  Paragraf § 9a -> [OBSAHUJE] -> Paragraf § 9a Odsek 5
  Paragraf § 9a Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 9a Odsek 5

  Paragraf § 9a Odsek 6 -> [UPRAVUJE] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou

  Prevod Viacuceloveho Poukazu -> [VZTAHUJE_SA_NA] -> Viacucelovy Poukaz
  Prevod Viacuceloveho Poukazu -> [MA_PODMIENKU] -> Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5

  Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5 -> [VZTAHUJE_SA_NA] -> Ina Zdanitelna Osoba
  Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5 -> [VZTAHUJE_SA_NA] -> Dodavatel Tovaru
  Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5 -> [VZTAHUJE_SA_NA] -> Dodavatel Sluzby

  Dodavatel Tovaru -> [DODAVA] -> Tovar
  Dodavatel Sluzby -> [DODAVA] -> Sluzba

  Ina Zdanitelna Osoba -> [DODAVA] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou -> [SUVISI_S] -> Prevod Viacuceloveho Poukazu
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou -> [MA_PODMIENKU] -> Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou -> [JE_PREDMETOM_DANE] -> Dan
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou -> [MA_STATUS] -> Samostatne Predmetom Dane

  Distribucna Sluzba -> [JE_TYPOM] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou
  Propagacna Sluzba -> [JE_TYPOM] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou

nodes:
  Paragraf: Paragraf § 9a
  Odsek: Paragraf § 9a Odsek 6
  Odsek: Paragraf § 9a Odsek 5

  Subjekt: Ina Zdanitelna Osoba
  Subjekt: Dodavatel Tovaru
  Subjekt: Dodavatel Sluzby

  Konanie: Prevod Viacuceloveho Poukazu
  Dokument: Viacucelovy Poukaz

  Podmienka: Prevod Viacuceloveho Poukazu Uskutocneny Inou Zdanitelnou Osobou Ako Dodavatel Tovaru Alebo Dodavatel Sluzby Podla Paragraf § 9a Odsek 5

  Tovar: Tovar
  Sluzba: Sluzba
  Sluzba: Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu Inou Zdanitelnou Osobou
  Sluzba: Distribucna Sluzba
  Sluzba: Propagacna Sluzba

  Dan: Dan
  Status: Samostatne Predmetom Dane



---

chunk: 172
path: ['§ 11', '7']
path_as_text: Paragraf § 11 Odsek 7
text: (7) Nadobúdateľ podľa odseku 4 písm. b) sa môže rozhodnúť, že bude zdaňovať nadobudnutie tovaru pred dosiahnutím hodnoty 14 000 eur a toto svoje rozhodnutie oznámi písomne daňovému úradu pri podaní žiadosti o registráciu pre daň (§ 7). Zdaňovanie nadobudnutia tovaru je nadobúdateľ povinný uplatňovať najmenej po dobu dvoch kalendárnych rokov.

relations:
  Paragraf § 11 -> [OBSAHUJE] -> Paragraf § 11 Odsek 7
  Paragraf § 11 -> [OBSAHUJE] -> Paragraf § 11 Odsek 4
  Paragraf § 11 Odsek 4 -> [OBSAHUJE] -> Paragraf § 11 Odsek 4 Pismeno b)
  Paragraf § 11 Odsek 7 -> [ODKAZUJE_NA] -> Paragraf § 11 Odsek 4 Pismeno b)
  Paragraf § 11 Odsek 7 -> [ODKAZUJE_NA] -> Paragraf § 7

  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [MA_PRAVO] -> Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Rozhodnutie O Zdanovani Nadobudnutia Tovaru -> [VZTAHUJE_SA_NA] -> Zdanovanie Nadobudnutia Tovaru
  Rozhodnutie O Zdanovani Nadobudnutia Tovaru -> [MA_PODMIENKU] -> Pred Dosiahnutim Hodnoty 14 000 Eur

  Pred Dosiahnutim Hodnoty 14 000 Eur -> [MA_HODNOTU] -> Hodnota 14 000 Eur

  Zdanovanie Nadobudnutia Tovaru -> [VZTAHUJE_SA_NA] -> Nadobudnutie Tovaru
  Nadobudnutie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [OZNAMUJE] -> Pisomne Oznamenie Rozhodnutia Danovemu Uradu
  Pisomne Oznamenie Rozhodnutia Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Pisomne Oznamenie Rozhodnutia Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Danovy Urad
  Pisomne Oznamenie Rozhodnutia Danovemu Uradu -> [MA_PODMIENKU] -> Pri Podani Ziadosti O Registraciu Pre Dan Podla Paragraf § 7

  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [PODAVA] -> Ziadost O Registraciu Pre Dan
  Ziadost O Registraciu Pre Dan -> [VZTAHUJE_SA_NA] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Registracia Pre Dan -> [JE_PODLA] -> Paragraf § 7

  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [MA_POVINNOST] -> Povinnost Uplatnovat Zdanovanie Nadobudnutia Tovaru Najmenej Dva Kalendarne Roky
  Povinnost Uplatnovat Zdanovanie Nadobudnutia Tovaru Najmenej Dva Kalendarne Roky -> [VZTAHUJE_SA_NA] -> Zdanovanie Nadobudnutia Tovaru
  Povinnost Uplatnovat Zdanovanie Nadobudnutia Tovaru Najmenej Dva Kalendarne Roky -> [MA_OBDOBIE] -> Najmenej Dva Kalendarne Roky

nodes:
  Paragraf: Paragraf § 11
  Odsek: Paragraf § 11 Odsek 7
  Odsek: Paragraf § 11 Odsek 4
  Pismeno: Paragraf § 11 Odsek 4 Pismeno b)
  Paragraf: Paragraf § 7

  Subjekt: Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b)

  Rozhodnutie: Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Konanie: Nadobudnutie Tovaru
  Povinnost: Zdanovanie Nadobudnutia Tovaru
  Povinnost: Povinnost Uplatnovat Zdanovanie Nadobudnutia Tovaru Najmenej Dva Kalendarne Roky

  Tovar: Tovar
  Suma: Hodnota 14 000 Eur
  Podmienka: Pred Dosiahnutim Hodnoty 14 000 Eur
  Podmienka: Pri Podani Ziadosti O Registraciu Pre Dan Podla Paragraf § 7

  Oznamenie: Pisomne Oznamenie Rozhodnutia Danovemu Uradu
  Organizacia: Danovy Urad
  Ziadost: Ziadost O Registraciu Pre Dan
  Registracia: Registracia Pre Dan
  Dan: Dan
  Obdobie: Najmenej Dva Kalendarne Roky


---

chunk: 186
path: ['§ 13', '1', 'a)']
path_as_text: Paragraf § 13 Odsek 1 Pismeno a)
text: (1) Miestom dodania tovaru, a) ak je dodanie tovaru spojené s odoslaním alebo prepravou tovaru, je miesto, kde sa tovar nachádza v čase, keď sa odoslanie alebo preprava tovaru osobe, ktorej má byť tovar dodaný, začína uskutočňovať, s výnimkou podľa písmena b), odseku 2 a § 14,

relations:
  Paragraf § 13 -> [OBSAHUJE] -> Paragraf § 13 Odsek 1
  Paragraf § 13 Odsek 1 -> [OBSAHUJE] -> Paragraf § 13 Odsek 1 Pismeno a)
  Paragraf § 13 Odsek 1 -> [OBSAHUJE] -> Paragraf § 13 Odsek 1 Pismeno b)
  Paragraf § 13 -> [OBSAHUJE] -> Paragraf § 13 Odsek 2

  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 13 Odsek 1 Pismeno b)
  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 13 Odsek 2
  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 14

  Paragraf § 13 Odsek 1 Pismeno a) -> [URCUJE] -> Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru

  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [MA_PODMIENKU] -> Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru
  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [VYPLYVA_Z] -> Miesto Kde Sa Tovar Nachadza V Case Zacatia Odoslania Alebo Prepravy Tovaru Osobe Ktorej Ma Byt Tovar Dodany
  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [NEVZTAHUJE_SA_NA] -> Vynimka Podla Paragraf § 13 Odsek 1 Pismeno b)
  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [NEVZTAHUJE_SA_NA] -> Vynimka Podla Paragraf § 13 Odsek 2
  Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru -> [NEVZTAHUJE_SA_NA] -> Vynimka Podla Paragraf § 14

  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Dodanie Tovaru -> [MA_PODMIENKU] -> Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru

  Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru

  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Odoslanie Tovaru -> [MA_DATUM] -> Cas Zacatia Odoslania Alebo Prepravy Tovaru

  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Preprava Tovaru -> [MA_DATUM] -> Cas Zacatia Odoslania Alebo Prepravy Tovaru

  Tovar -> [NACHADZA_SA_V] -> Miesto Kde Sa Tovar Nachadza V Case Zacatia Odoslania Alebo Prepravy Tovaru Osobe Ktorej Ma Byt Tovar Dodany
  Miesto Kde Sa Tovar Nachadza V Case Zacatia Odoslania Alebo Prepravy Tovaru Osobe Ktorej Ma Byt Tovar Dodany -> [MA_DATUM] -> Cas Zacatia Odoslania Alebo Prepravy Tovaru

nodes:
  Paragraf: Paragraf § 13
  Odsek: Paragraf § 13 Odsek 1
  Pismeno: Paragraf § 13 Odsek 1 Pismeno a)
  Pismeno: Paragraf § 13 Odsek 1 Pismeno b)
  Odsek: Paragraf § 13 Odsek 2
  Paragraf: Paragraf § 14

  Konanie: Dodanie Tovaru
  Konanie: Odoslanie Tovaru
  Konanie: Preprava Tovaru

  Tovar: Tovar
  Osoba: Osoba Ktorej Ma Byt Tovar Dodany

  Lokacia: Miesto Dodania Tovaru Pri Dodani Tovaru Spojenom S Odoslanim Alebo Prepravou Tovaru
  Lokacia: Miesto Kde Sa Tovar Nachadza V Case Zacatia Odoslania Alebo Prepravy Tovaru Osobe Ktorej Ma Byt Tovar Dodany

  Podmienka: Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru
  Podmienka: Vynimka Podla Paragraf § 13 Odsek 1 Pismeno b)
  Podmienka: Vynimka Podla Paragraf § 13 Odsek 2
  Podmienka: Vynimka Podla Paragraf § 14

  Datum: Cas Zacatia Odoslania Alebo Prepravy Tovaru


---

chunk: 195
path: ['§ 13a', '2']
path_as_text: Paragraf § 13a Odsek 2
text: (2) Na účely odseku 1 je prostrednou osobou dodávateľ, ktorý v reťazci dodaní nie je prvým dodávateľom a ktorý odosiela alebo prepravuje tovar alebo na účet ktorého je tovar odoslaný alebo prepravený treťou osobou.

relations:
  Paragraf § 13a -> [OBSAHUJE] -> Paragraf § 13a Odsek 1
  Paragraf § 13a -> [OBSAHUJE] -> Paragraf § 13a Odsek 2
  Paragraf § 13a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 13a Odsek 1

  Paragraf § 13a Odsek 2 -> [DEFINUJE] -> Prostredna Osoba

  Prostredna Osoba -> [JE_TYPOM] -> Dodavatel
  Prostredna Osoba -> [PATRI_DO] -> Retazec Dodani
  Prostredna Osoba -> [MA_PODMIENKU] -> Nie Je Prvym Dodavatelom V Retazci Dodani
  Prostredna Osoba -> [MA_PODMIENKU] -> Odosiela Alebo Prepravuje Tovar
  Prostredna Osoba -> [MA_PODMIENKU] -> Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby

  Nie Je Prvym Dodavatelom V Retazci Dodani -> [VZTAHUJE_SA_NA] -> Prvy Dodavatel
  Nie Je Prvym Dodavatelom V Retazci Dodani -> [VZTAHUJE_SA_NA] -> Retazec Dodani

  Odosiela Alebo Prepravuje Tovar -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Odosiela Alebo Prepravuje Tovar -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby -> [VZTAHUJE_SA_NA] -> Tretia Osoba
  Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby -> [VZTAHUJE_SA_NA] -> Prostredna Osoba
  Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby -> [VZTAHUJE_SA_NA] -> Preprava Tovaru

nodes:
  Paragraf: Paragraf § 13a
  Odsek: Paragraf § 13a Odsek 1
  Odsek: Paragraf § 13a Odsek 2

  Osoba: Prostredna Osoba
  Subjekt: Dodavatel
  Subjekt: Prvy Dodavatel
  Osoba: Tretia Osoba

  Konanie: Retazec Dodani
  Konanie: Odoslanie Tovaru
  Konanie: Preprava Tovaru

  Tovar: Tovar

  Podmienka: Nie Je Prvym Dodavatelom V Retazci Dodani
  Podmienka: Odosiela Alebo Prepravuje Tovar
  Podmienka: Tovar Je Odoslany Alebo Prepraveny Tretou Osobou Na Ucet Prostrednej Osoby


---

chunk: 217
path: ['§ 16', '7', 'c)']
path_as_text: Paragraf § 16 Odsek 7 Pismeno c)
text: (7) Na účely odsekov 5 a 6 je c) miestom skončenia prepravy tovaru miesto, kde sa preprava tovaru skutočne skončí.

relations:
  Paragraf § 16 -> [OBSAHUJE] -> Paragraf § 16 Odsek 5
  Paragraf § 16 -> [OBSAHUJE] -> Paragraf § 16 Odsek 6
  Paragraf § 16 -> [OBSAHUJE] -> Paragraf § 16 Odsek 7
  Paragraf § 16 Odsek 7 -> [OBSAHUJE] -> Paragraf § 16 Odsek 7 Pismeno c)

  Paragraf § 16 Odsek 7 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 16 Odsek 5
  Paragraf § 16 Odsek 7 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 16 Odsek 6
  Paragraf § 16 Odsek 7 Pismeno c) -> [DEFINUJE] -> Miesto Skoncenia Prepravy Tovaru

  Miesto Skoncenia Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Miesto Skoncenia Prepravy Tovaru -> [VYPLYVA_Z] -> Miesto Kde Sa Preprava Tovaru Skutocne Skonci

  Miesto Kde Sa Preprava Tovaru Skutocne Skonci -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

nodes:
  Paragraf: Paragraf § 16
  Odsek: Paragraf § 16 Odsek 5
  Odsek: Paragraf § 16 Odsek 6
  Odsek: Paragraf § 16 Odsek 7
  Pismeno: Paragraf § 16 Odsek 7 Pismeno c)

  Lokacia: Miesto Skoncenia Prepravy Tovaru
  Lokacia: Miesto Kde Sa Preprava Tovaru Skutocne Skonci

  Sluzba: Preprava Tovaru
  Tovar: Tovar


---

chunk: 218
path: ['§ 16', '8']
path_as_text: Paragraf § 16 Odsek 8
text: (8) Miestom dodania doplnkových služieb pri preprave, napríklad nakladanie, vykladanie, manipulácia a podobné služby, ak sú tieto služby dodané osobe inej ako zdaniteľnej osobe, je  miesto, kde sa tieto služby fyzicky vykonajú.

relations:
  Paragraf § 16 -> [OBSAHUJE] -> Paragraf § 16 Odsek 8

  Paragraf § 16 Odsek 8 -> [URCUJE] -> Miesto Dodania Doplnkovych Sluzieb Pri Preprave Dodanych Osobe Inej Ako Zdanitelnej Osobe

  Miesto Dodania Doplnkovych Sluzieb Pri Preprave Dodanych Osobe Inej Ako Zdanitelnej Osobe -> [VZTAHUJE_SA_NA] -> Doplnkove Sluzby Pri Preprave
  Miesto Dodania Doplnkovych Sluzieb Pri Preprave Dodanych Osobe Inej Ako Zdanitelnej Osobe -> [MA_PODMIENKU] -> Dodanie Doplnkovych Sluzieb Pri Preprave Osobe Inej Ako Zdanitelnej Osobe
  Miesto Dodania Doplnkovych Sluzieb Pri Preprave Dodanych Osobe Inej Ako Zdanitelnej Osobe -> [VYPLYVA_Z] -> Miesto Fyzickeho Vykonania Doplnkovych Sluzieb Pri Preprave

  Doplnkove Sluzby Pri Preprave -> [SUVISI_S] -> Preprava
  Doplnkove Sluzby Pri Preprave -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba

  Dodanie Doplnkovych Sluzieb Pri Preprave Osobe Inej Ako Zdanitelnej Osobe -> [VZTAHUJE_SA_NA] -> Doplnkove Sluzby Pri Preprave
  Dodanie Doplnkovych Sluzieb Pri Preprave Osobe Inej Ako Zdanitelnej Osobe -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba

  Miesto Fyzickeho Vykonania Doplnkovych Sluzieb Pri Preprave -> [VZTAHUJE_SA_NA] -> Doplnkove Sluzby Pri Preprave

  Nakladanie -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Vykladanie -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Manipulacia -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Podobne Sluzby -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave

nodes:
  Paragraf: Paragraf § 16
  Odsek: Paragraf § 16 Odsek 8

  Sluzba: Doplnkove Sluzby Pri Preprave
  Sluzba: Nakladanie
  Sluzba: Vykladanie
  Sluzba: Manipulacia
  Sluzba: Podobne Sluzby
  Sluzba: Preprava

  Osoba: Osoba Ina Ako Zdanitelna Osoba

  Konanie: Dodanie Doplnkovych Sluzieb Pri Preprave Osobe Inej Ako Zdanitelnej Osobe

  Lokacia: Miesto Dodania Doplnkovych Sluzieb Pri Preprave Dodanych Osobe Inej Ako Zdanitelnej Osobe
  Lokacia: Miesto Fyzickeho Vykonania Doplnkovych Sluzieb Pri Preprave


---

chunk: 241
path: ['§ 16a', '1', 'b)']
path_as_text: Paragraf § 16a Odsek 1 Pismeno b)
text: (1) Miestom dodania tovaru pri predaji tovaru na diaľku na území Európskej únie je miesto, kde sa odoslanie alebo preprava tovaru začína, a miestom dodania pri dodaní telekomunikačných služieb, služieb rozhlasového vysielania a televízneho vysielania a elektronických služieb, ktoré sú dodané osobe inej ako zdaniteľnej osobe, je miesto, kde má dodávateľ služby sídlo, miesto podnikania alebo prevádzkareň, a ak nemá sídlo, miesto podnikania alebo prevádzkareň, miestom dodania služby je jeho bydlisko alebo miesto, kde sa obvykle zdržiava, ak b) tovar sa odosiela alebo prepravuje do iného členského štátu, ako je členský štát podľa písmena a), alebo služba sa dodáva osobe, ktorá má sídlo, bydlisko alebo miesto, kde sa obvykle zdržiava, v inom členskom štáte, ako je členský štát podľa písmena a) a

relations:
  Paragraf § 16a -> [OBSAHUJE] -> Paragraf § 16a Odsek 1
  Paragraf § 16a Odsek 1 -> [OBSAHUJE] -> Paragraf § 16a Odsek 1 Pismeno b)
  Paragraf § 16a Odsek 1 -> [OBSAHUJE] -> Paragraf § 16a Odsek 1 Pismeno a)
  Paragraf § 16a Odsek 1 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 16a Odsek 1 Pismeno a)

  Paragraf § 16a Odsek 1 Pismeno b) -> [UPRAVUJE] -> Podmienka Odoslania Alebo Prepravy Tovaru Do Ineho Clenskeho Statu Ako Clensky Stat Podla Pismena a)
  Paragraf § 16a Odsek 1 Pismeno b) -> [UPRAVUJE] -> Podmienka Dodania Sluzby Osobe V Inom Clenskom State Ako Clensky Stat Podla Pismena a)

  Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [VYPLYVA_Z] -> Miesto Zacatia Odoslania Alebo Prepravy Tovaru
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Zacatia Odoslania Alebo Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Odoslanie Alebo Preprava Tovaru

  Odoslanie Alebo Preprava Tovaru -> [MA_PODMIENKU] -> Tovar Odosielany Alebo Prepravovany Do Ineho Clenskeho Statu Ako Clensky Stat Podla Pismena a)
  Tovar Odosielany Alebo Prepravovany Do Ineho Clenskeho Statu Ako Clensky Stat Podla Pismena a) -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat Ako Clensky Stat Podla Pismena a)
  Iny Clensky Stat Ako Clensky Stat Podla Pismena a) -> [VZTAHUJE_SA_NA] -> Clensky Stat Podla Pismena a)

  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Telekomunikacne Sluzby
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzby Rozhlasoveho Vysielania A Televizneho Vysielania
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Elektronicke Sluzby
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [MA_PODMIENKU] -> Sluzba Dodana Osobe Inej Ako Zdanitelnej Osobe
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [MA_PODMIENKU] -> Sluzba Dodana Osobe So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)

  Sluzba Dodana Osobe Inej Ako Zdanitelnej Osobe -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba
  Sluzba Dodana Osobe So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a) -> [VZTAHUJE_SA_NA] -> Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)
  Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a) -> [NACHADZA_SA_V] -> Iny Clensky Stat Ako Clensky Stat Podla Pismena a)

  Miesto Dodania Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb
  Miesto Dodania Sluzby -> [VYPLYVA_Z] -> Sidlo Miesto Podnikania Alebo Prevadzkaren Dodavatela Sluzby
  Miesto Dodania Sluzby -> [MA_PODMIENKU] -> Dodavatel Sluzby Ma Sidlo Miesto Podnikania Alebo Prevadzkaren

  Dodavatel Sluzby -> [MA_ADRESU] -> Sidlo Dodavatela Sluzby
  Dodavatel Sluzby -> [MA_ADRESU] -> Miesto Podnikania Dodavatela Sluzby
  Dodavatel Sluzby -> [MA_ADRESU] -> Prevadzkaren Dodavatela Sluzby

  Miesto Dodania Sluzby Podla Bydliska Alebo Miesta Obvykleho Zdrziavania Dodavatela Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb
  Miesto Dodania Sluzby Podla Bydliska Alebo Miesta Obvykleho Zdrziavania Dodavatela Sluzby -> [MA_PODMIENKU] -> Dodavatel Sluzby Nema Sidlo Miesto Podnikania Alebo Prevadzkaren
  Miesto Dodania Sluzby Podla Bydliska Alebo Miesta Obvykleho Zdrziavania Dodavatela Sluzby -> [VYPLYVA_Z] -> Bydlisko Alebo Miesto Obvykleho Zdrziavania Dodavatela Sluzby

  Dodavatel Sluzby -> [MA_ADRESU] -> Bydlisko Dodavatela Sluzby
  Dodavatel Sluzby -> [NACHADZA_SA_V] -> Miesto Kde Sa Dodavatel Sluzby Obvykle Zdrziava

nodes:
  Paragraf: Paragraf § 16a
  Odsek: Paragraf § 16a Odsek 1
  Pismeno: Paragraf § 16a Odsek 1 Pismeno b)
  Pismeno: Paragraf § 16a Odsek 1 Pismeno a)

  Lokacia: Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie
  Lokacia: Miesto Zacatia Odoslania Alebo Prepravy Tovaru
  Lokacia: Miesto Dodania Sluzby
  Lokacia: Miesto Dodania Sluzby Podla Bydliska Alebo Miesta Obvykleho Zdrziavania Dodavatela Sluzby
  Lokacia: Miesto Kde Sa Dodavatel Sluzby Obvykle Zdrziava

  Konanie: Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Konanie: Odoslanie Alebo Preprava Tovaru
  Konanie: Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb

  Tovar: Tovar
  Sluzba: Telekomunikacne Sluzby
  Sluzba: Sluzby Rozhlasoveho Vysielania A Televizneho Vysielania
  Sluzba: Elektronicke Sluzby

  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Osoba: Dodavatel Sluzby
  Osoba: Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)

  Adresa: Sidlo Dodavatela Sluzby
  Adresa: Miesto Podnikania Dodavatela Sluzby
  Adresa: Prevadzkaren Dodavatela Sluzby
  Adresa: Bydlisko Dodavatela Sluzby

  Stat: Iny Clensky Stat Ako Clensky Stat Podla Pismena a)
  Stat: Clensky Stat Podla Pismena a)

  Podmienka: Tovar Odosielany Alebo Prepravovany Do Ineho Clenskeho Statu Ako Clensky Stat Podla Pismena a)
  Podmienka: Sluzba Dodana Osobe Inej Ako Zdanitelnej Osobe
  Podmienka: Sluzba Dodana Osobe So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)
  Podmienka: Dodavatel Sluzby Ma Sidlo Miesto Podnikania Alebo Prevadzkaren
  Podmienka: Dodavatel Sluzby Nema Sidlo Miesto Podnikania Alebo Prevadzkaren
  Podmienka: Sidlo Miesto Podnikania Alebo Prevadzkaren Dodavatela Sluzby
  Podmienka: Bydlisko Alebo Miesto Obvykleho Zdrziavania Dodavatela Sluzby


---

chunk: 248
path: ['§ 17', '4', 'a)']
path_as_text: Paragraf § 17 Odsek 4 Pismeno a)
text: (4) Miestom nadobudnutia tovaru z iného členského štátu pri trojstrannom obchode podľa § 45 je miesto podľa odseku 1, ak a) prvý odberateľ preukáže, že tovar nadobudol na účely následného dodania tovaru v členskom štáte, v ktorom sa skončí odoslanie alebo preprava tovaru, a druhý odberateľ je osobou identifikovanou pre daň v členskom štáte, v ktorom sa skončí odoslanie alebo preprava tovaru, a je osobou povinnou platiť daň,

relations:
  Paragraf § 17 -> [OBSAHUJE] -> Paragraf § 17 Odsek 1
  Paragraf § 17 -> [OBSAHUJE] -> Paragraf § 17 Odsek 4
  Paragraf § 17 Odsek 4 -> [OBSAHUJE] -> Paragraf § 17 Odsek 4 Pismeno a)

  Paragraf § 17 Odsek 4 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 45
  Paragraf § 17 Odsek 4 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 17 Odsek 1

  Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode -> [VZTAHUJE_SA_NA] -> Trojstranny Obchod
  Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode -> [VYPLYVA_Z] -> Miesto Podla Paragraf § 17 Odsek 1
  Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode -> [MA_PODMIENKU] -> Podmienky Podla Paragraf § 17 Odsek 4 Pismeno a)

  Trojstranny Obchod -> [JE_PODLA] -> Paragraf § 45

  Podmienky Podla Paragraf § 17 Odsek 4 Pismeno a) -> [VZTAHUJE_SA_NA] -> Preukazanie Nadobudnutia Tovaru Prvym Odberatelom Na Ucely Nasledneho Dodania Tovaru
  Podmienky Podla Paragraf § 17 Odsek 4 Pismeno a) -> [VZTAHUJE_SA_NA] -> Identifikacia Druheho Odberatela Pre Dan V Clenskom State Skoncenia Odoslania Alebo Prepravy Tovaru
  Podmienky Podla Paragraf § 17 Odsek 4 Pismeno a) -> [VZTAHUJE_SA_NA] -> Povinnost Druheho Odberatela Platit Dan

  Prvy Odberatel -> [NADOBUDA] -> Tovar
  Preukazanie Nadobudnutia Tovaru Prvym Odberatelom Na Ucely Nasledneho Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Prvy Odberatel
  Preukazanie Nadobudnutia Tovaru Prvym Odberatelom Na Ucely Nasledneho Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preukazanie Nadobudnutia Tovaru Prvym Odberatelom Na Ucely Nasledneho Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Nasledne Dodanie Tovaru

  Nasledne Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Nasledne Dodanie Tovaru -> [NACHADZA_SA_V] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru

  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [NACHADZA_SA_V] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru

  Druhy Odberatel -> [JE_TYPOM] -> Osoba Identifikovana Pre Dan
  Osoba Identifikovana Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacia Druheho Odberatela Pre Dan V Clenskom State Skoncenia Odoslania Alebo Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Druhy Odberatel
  Identifikacia Druheho Odberatela Pre Dan V Clenskom State Skoncenia Odoslania Alebo Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru

  Druhy Odberatel -> [MA_POVINNOST] -> Povinnost Platit Dan
  Povinnost Platit Dan -> [VZTAHUJE_SA_NA] -> Dan

nodes:
  Paragraf: Paragraf § 17
  Odsek: Paragraf § 17 Odsek 1
  Odsek: Paragraf § 17 Odsek 4
  Pismeno: Paragraf § 17 Odsek 4 Pismeno a)
  Paragraf: Paragraf § 45

  Lokacia: Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode
  Lokacia: Miesto Podla Paragraf § 17 Odsek 1

  Konanie: Trojstranny Obchod
  Konanie: Nasledne Dodanie Tovaru
  Konanie: Odoslanie Alebo Preprava Tovaru

  Subjekt: Prvy Odberatel
  Subjekt: Druhy Odberatel
  Osoba: Osoba Identifikovana Pre Dan

  Tovar: Tovar
  Stat: Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru
  Dan: Dan
  Povinnost: Povinnost Platit Dan

  Podmienka: Podmienky Podla Paragraf § 17 Odsek 4 Pismeno a)
  Podmienka: Preukazanie Nadobudnutia Tovaru Prvym Odberatelom Na Ucely Nasledneho Dodania Tovaru
  Podmienka: Identifikacia Druheho Odberatela Pre Dan V Clenskom State Skoncenia Odoslania Alebo Prepravy Tovaru
  Podmienka: Povinnost Druheho Odberatela Platit Dan


---

chunk: 264
path: ['§ 19', '7']
path_as_text: Paragraf § 19 Odsek 7
text: (7) Pri dodaní tovaru prostredníctvom predajných automatov, prípadne iných obdobných prístrojov uvádzaných do chodu mincami, bankovkami, známkami alebo inými platobnými prostriedkami nahrádzajúcimi peniaze vzniká daňová povinnosť dňom, keď sa vyberú peniaze alebo známky z prístroja alebo iným spôsobom sa zistí výška obratu.

relations:
  Paragraf § 19 -> [OBSAHUJE] -> Paragraf § 19 Odsek 7

  Paragraf § 19 Odsek 7 -> [UPRAVUJE] -> Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov

  Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov
  Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov -> [MA_DATUM] -> Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu Inym Sposobom

  Dodanie Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov -> [VZTAHUJE_SA_NA] -> Predajne Automaty Alebo Obdobne Pristroje
  Dodanie Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov -> [MA_PODMIENKU] -> Pristroje Uvadzane Do Chodu Mincami Bankovkami Znamkami Alebo Inymi Platobnymi Prostriedkami Nahradzajucimi Peniaze

  Pristroje Uvadzane Do Chodu Mincami Bankovkami Znamkami Alebo Inymi Platobnymi Prostriedkami Nahradzajucimi Peniaze -> [VZTAHUJE_SA_NA] -> Mince Bankovky Znamky Alebo Ine Platobne Prostriedky Nahradzajuce Peniaze

  Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu Inym Sposobom -> [VYPLYVA_Z] -> Vybratie Penazi Alebo Znamok Z Pristroja
  Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu Inym Sposobom -> [VYPLYVA_Z] -> Zistenie Vysky Obratu Inym Sposobom

  Vybratie Penazi Alebo Znamok Z Pristroja -> [VZTAHUJE_SA_NA] -> Peniaze Alebo Znamky
  Vybratie Penazi Alebo Znamok Z Pristroja -> [VZTAHUJE_SA_NA] -> Predajne Automaty Alebo Obdobne Pristroje

  Zistenie Vysky Obratu Inym Sposobom -> [VZTAHUJE_SA_NA] -> Vyska Obratu

nodes:
  Paragraf: Paragraf § 19
  Odsek: Paragraf § 19 Odsek 7

  Povinnost: Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov
  Konanie: Dodanie Tovaru Prostrednictvom Predajnych Automatov Alebo Obdobnych Pristrojov
  Tovar: Tovar

  Majetok: Predajne Automaty Alebo Obdobne Pristroje
  Podmienka: Pristroje Uvadzane Do Chodu Mincami Bankovkami Znamkami Alebo Inymi Platobnymi Prostriedkami Nahradzajucimi Peniaze

  Platba: Mince Bankovky Znamky Alebo Ine Platobne Prostriedky Nahradzajuce Peniaze
  Platba: Peniaze Alebo Znamky
  Obrat: Vyska Obratu

  Datum: Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu Inym Sposobom
  Konanie: Vybratie Penazi Alebo Znamok Z Pristroja
  Konanie: Zistenie Vysky Obratu Inym Sposobom


---

chunk: 279
path: ['§ 21', '4']
path_as_text: Paragraf § 21 Odsek 4
text: (4) Ak daňová povinnosť pri dovoze tovaru vznikne podľa odseku 1 písm. c), daň sa zníži o sumu dane zaplatenej pri prepustení tovaru do voľného obehu vrátane konečného použitia alebo pri prepustení do colného režimu dočasné použitie s čiastočným oslobodením od dovozného cla alebo o sumu dane priznanej podľa § 84a ods. 3.

relations:
  Paragraf § 21 -> [OBSAHUJE] -> Paragraf § 21 Odsek 4
  Paragraf § 21 -> [OBSAHUJE] -> Paragraf § 21 Odsek 1
  Paragraf § 21 Odsek 1 -> [OBSAHUJE] -> Paragraf § 21 Odsek 1 Pismeno c)
  Paragraf § 84a -> [OBSAHUJE] -> Paragraf § 84a Odsek 3

  Paragraf § 21 Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 21 Odsek 1 Pismeno c)
  Paragraf § 21 Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 84a Odsek 3
  Paragraf § 21 Odsek 4 -> [UPRAVUJE] -> Znizenie Dane Pri Dovoze Tovaru

  Znizenie Dane Pri Dovoze Tovaru -> [MENI] -> Dan
  Znizenie Dane Pri Dovoze Tovaru -> [MA_PODMIENKU] -> Vznik Danovej Povinnosti Pri Dovoze Tovaru Podla Paragraf § 21 Odsek 1 Pismeno c)

  Vznik Danovej Povinnosti Pri Dovoze Tovaru Podla Paragraf § 21 Odsek 1 Pismeno c) -> [VZTAHUJE_SA_NA] -> Danova Povinnost Pri Dovoze Tovaru
  Vznik Danovej Povinnosti Pri Dovoze Tovaru Podla Paragraf § 21 Odsek 1 Pismeno c) -> [VYPLYVA_Z] -> Paragraf § 21 Odsek 1 Pismeno c)

  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dan
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Znizenie Dane Pri Dovoze Tovaru -> [MA_SUMU] -> Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [VYPLYVA_Z] -> Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [VZTAHUJE_SA_NA] -> Tovar
  Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [MA_STATUS] -> Volny Obeh Vratane Konecneho Pouzitia

  Znizenie Dane Pri Dovoze Tovaru -> [MA_SUMU] -> Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [VYPLYVA_Z] -> Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [VZTAHUJE_SA_NA] -> Tovar
  Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [MA_STATUS] -> Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [OSLOBODZUJE_OD] -> Dovozne Clo

  Znizenie Dane Pri Dovoze Tovaru -> [MA_SUMU] -> Suma Dane Priznanej Podla Paragraf § 84a Odsek 3
  Suma Dane Priznanej Podla Paragraf § 84a Odsek 3 -> [VYPLYVA_Z] -> Paragraf § 84a Odsek 3

nodes:
  Paragraf: Paragraf § 21
  Odsek: Paragraf § 21 Odsek 4
  Odsek: Paragraf § 21 Odsek 1
  Pismeno: Paragraf § 21 Odsek 1 Pismeno c)
  Paragraf: Paragraf § 84a
  Odsek: Paragraf § 84a Odsek 3

  Povinnost: Danova Povinnost Pri Dovoze Tovaru
  Konanie: Dovoz Tovaru
  Konanie: Znizenie Dane Pri Dovoze Tovaru
  Konanie: Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Konanie: Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla

  Tovar: Tovar
  Dan: Dan
  Dan: Dovozne Clo

  Suma: Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Suma: Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Suma: Suma Dane Priznanej Podla Paragraf § 84a Odsek 3

  Status: Volny Obeh Vratane Konecneho Pouzitia
  Status: Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla

  Podmienka: Vznik Danovej Povinnosti Pri Dovoze Tovaru Podla Paragraf § 21 Odsek 1 Pismeno c)


---

chunk: 287
path: ['§ 22', '3']
path_as_text: Paragraf § 22 Odsek 3
text: (3) Do základu dane podľa odseku 1 sa nezahŕňajú výdavky platené v mene a na účet kupujúceho alebo zákazníka, ktoré dodávateľ požaduje od kupujúceho alebo zákazníka (ďalej len „prechodné položky“). Pri dodaní tovaru v zálohovaných obaloch sa do základu dane podľa odseku 1 nezahŕňa záloha na zálohované obaly, ktoré sú dodané spolu s tovarom. Pri dodaní nápoja v zálohovanom jednorazovom obale na nápoje6abd) sa do základu dane podľa odseku 1 nezahŕňa záloh na tento obal.

relations:
  Paragraf § 22 -> [OBSAHUJE] -> Paragraf § 22 Odsek 3
  Paragraf § 22 -> [OBSAHUJE] -> Paragraf § 22 Odsek 1
  Paragraf § 22 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 22 Odsek 1

  Zaklad Dane Podla Paragraf § 22 Odsek 1 -> [JE_PODLA] -> Paragraf § 22 Odsek 1

  Paragraf § 22 Odsek 3 -> [DEFINUJE] -> Prechodne Polozky
  Prechodne Polozky -> [JE_TYPOM] -> Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
  Prechodne Polozky -> [MA_PODMIENKU] -> Vydavky Pozadovane Dodavatelom Od Kupujuceho Alebo Zakaznika
  Prechodne Polozky -> [VZTAHUJE_SA_NA] -> Kupujuci
  Prechodne Polozky -> [VZTAHUJE_SA_NA] -> Zakaznik
  Vydavky Pozadovane Dodavatelom Od Kupujuceho Alebo Zakaznika -> [VZTAHUJE_SA_NA] -> Dodavatel
  Vydavky Pozadovane Dodavatelom Od Kupujuceho Alebo Zakaznika -> [VZTAHUJE_SA_NA] -> Kupujuci
  Vydavky Pozadovane Dodavatelom Od Kupujuceho Alebo Zakaznika -> [VZTAHUJE_SA_NA] -> Zakaznik

  Zaklad Dane Podla Paragraf § 22 Odsek 1 -> [NEVZTAHUJE_SA_NA] -> Prechodne Polozky

  Dodanie Tovaru V Zalohovanych Obaloch -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru V Zalohovanych Obaloch -> [VZTAHUJE_SA_NA] -> Zalohovane Obaly
  Zalohovane Obaly -> [SUVISI_S] -> Tovar
  Zaklad Dane Podla Paragraf § 22 Odsek 1 -> [NEVZTAHUJE_SA_NA] -> Zaloha Na Zalohovane Obaly
  Zaloha Na Zalohovane Obaly -> [VZTAHUJE_SA_NA] -> Zalohovane Obaly

  Dodanie Napoja V Zalohovanom Jednorazovom Obale Na Napoje -> [VZTAHUJE_SA_NA] -> Napoj
  Dodanie Napoja V Zalohovanom Jednorazovom Obale Na Napoje -> [VZTAHUJE_SA_NA] -> Zalohovany Jednorazovy Obal Na Napoje
  Zaklad Dane Podla Paragraf § 22 Odsek 1 -> [NEVZTAHUJE_SA_NA] -> Zaloha Na Zalohovany Jednorazovy Obal Na Napoje
  Zaloha Na Zalohovany Jednorazovy Obal Na Napoje -> [VZTAHUJE_SA_NA] -> Zalohovany Jednorazovy Obal Na Napoje

nodes:
  Paragraf: Paragraf § 22
  Odsek: Paragraf § 22 Odsek 3
  Odsek: Paragraf § 22 Odsek 1

  Dan: Zaklad Dane Podla Paragraf § 22 Odsek 1

  Suma: Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
  Suma: Prechodne Polozky
  Podmienka: Vydavky Pozadovane Dodavatelom Od Kupujuceho Alebo Zakaznika

  Subjekt: Kupujuci
  Subjekt: Zakaznik
  Subjekt: Dodavatel

  Konanie: Dodanie Tovaru V Zalohovanych Obaloch
  Tovar: Tovar
  Tovar: Zalohovane Obaly
  Platba: Zaloha Na Zalohovane Obaly

  Konanie: Dodanie Napoja V Zalohovanom Jednorazovom Obale Na Napoje
  Tovar: Napoj
  Tovar: Zalohovany Jednorazovy Obal Na Napoje
  Platba: Zaloha Na Zalohovany Jednorazovy Obal Na Napoje


---

chunk: 310
path: ['§ 24', '3']
path_as_text: Paragraf § 24 Odsek 3
text: (3) Prvé miesto určenia v tuzemsku podľa odseku 2 písm. b) je miesto uvedené v nákladnom liste alebo inom sprievodnom dokumente sprevádzajúcom dovážaný tovar do tuzemska. Ak takéto miesto nie je uvedené, za prvé miesto určenia v tuzemsku sa považuje miesto prvej prekládky tovaru v tuzemsku.

relations:
  Paragraf § 24 -> [OBSAHUJE] -> Paragraf § 24 Odsek 3
  Paragraf § 24 -> [OBSAHUJE] -> Paragraf § 24 Odsek 2
  Paragraf § 24 Odsek 2 -> [OBSAHUJE] -> Paragraf § 24 Odsek 2 Pismeno b)
  Paragraf § 24 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 24 Odsek 2 Pismeno b)

  Paragraf § 24 Odsek 3 -> [DEFINUJE] -> Prve Miesto Urcenia V Tuzemsku
  Prve Miesto Urcenia V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko

  Prve Miesto Urcenia V Tuzemsku -> [VYPLYVA_Z] -> Miesto Uvedene V Nakladnom Liste Alebo Inom Sprievodnom Dokumente Sprevadzajucom Dovazany Tovar Do Tuzemska
  Miesto Uvedene V Nakladnom Liste Alebo Inom Sprievodnom Dokumente Sprevadzajucom Dovazany Tovar Do Tuzemska -> [VZTAHUJE_SA_NA] -> Nakladny List
  Miesto Uvedene V Nakladnom Liste Alebo Inom Sprievodnom Dokumente Sprevadzajucom Dovazany Tovar Do Tuzemska -> [VZTAHUJE_SA_NA] -> Iny Sprievodny Dokument
  Miesto Uvedene V Nakladnom Liste Alebo Inom Sprievodnom Dokumente Sprevadzajucom Dovazany Tovar Do Tuzemska -> [NACHADZA_SA_V] -> Tuzemsko

  Nakladny List -> [SUVISI_S] -> Dovazany Tovar Do Tuzemska
  Iny Sprievodny Dokument -> [SUVISI_S] -> Dovazany Tovar Do Tuzemska
  Dovazany Tovar Do Tuzemska -> [VZTAHUJE_SA_NA] -> Tovar
  Dovazany Tovar Do Tuzemska -> [VZTAHUJE_SA_NA] -> Tuzemsko

  Prve Miesto Urcenia V Tuzemsku -> [MA_PODMIENKU] -> Neuvedenie Miesta V Nakladnom Liste Alebo Inom Sprievodnom Dokumente
  Prve Miesto Urcenia V Tuzemsku -> [VYPLYVA_Z] -> Miesto Prvej Prekladky Tovaru V Tuzemsku

  Miesto Prvej Prekladky Tovaru V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Prvej Prekladky Tovaru V Tuzemsku -> [VZTAHUJE_SA_NA] -> Dovazany Tovar Do Tuzemska

nodes:
  Paragraf: Paragraf § 24
  Odsek: Paragraf § 24 Odsek 3
  Odsek: Paragraf § 24 Odsek 2
  Pismeno: Paragraf § 24 Odsek 2 Pismeno b)

  Lokacia: Prve Miesto Urcenia V Tuzemsku
  Lokacia: Miesto Uvedene V Nakladnom Liste Alebo Inom Sprievodnom Dokumente Sprevadzajucom Dovazany Tovar Do Tuzemska
  Lokacia: Miesto Prvej Prekladky Tovaru V Tuzemsku

  Stat: Tuzemsko
  Dokument: Nakladny List
  Dokument: Iny Sprievodny Dokument
  Tovar: Tovar
  Tovar: Dovazany Tovar Do Tuzemska

  Podmienka: Neuvedenie Miesta V Nakladnom Liste Alebo Inom Sprievodnom Dokumente


---

chunk: 320
path: ['§ 25', '5', 'a)']
path_as_text: Paragraf § 25 Odsek 5 Pismeno a)
text: (5) Ak pri dovoze tovaru vznikne daňová povinnosť v tuzemsku právnickej osobe z iného členského štátu, ktorá nie je zdaniteľnou osobou, colný orgán vráti tejto osobe daň zaplatenú pri dovoze, ak a) ide o tovar odoslaný alebo prepravený z územia tretieho štátu a miestom určenia tovaru je iný členský štát ako tuzemsko a

relations:
  Paragraf § 25 -> [OBSAHUJE] -> Paragraf § 25 Odsek 5
  Paragraf § 25 Odsek 5 -> [OBSAHUJE] -> Paragraf § 25 Odsek 5 Pismeno a)

  Paragraf § 25 Odsek 5 Pismeno a) -> [UPRAVUJE] -> Vratenie Dane Zaplatenej Pri Dovoze

  Vratenie Dane Zaplatenej Pri Dovoze -> [VZTAHUJE_SA_NA] -> Dan Zaplatena Pri Dovoze
  Vratenie Dane Zaplatenej Pri Dovoze -> [VZTAHUJE_SA_NA] -> Colny Organ
  Vratenie Dane Zaplatenej Pri Dovoze -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Z Ineho Clenskeho Statu
  Pravnicka Osoba Z Ineho Clenskeho Statu -> [MA_NAROK_NA] -> Vratenie Dane Zaplatenej Pri Dovoze

  Vratenie Dane Zaplatenej Pri Dovoze -> [MA_PODMIENKU] -> Vznik Danovej Povinnosti Pri Dovoze Tovaru V Tuzemsku Pravnickej Osobe Z Ineho Clenskeho Statu Ktora Nie Je Zdanitelnou Osobou
  Vratenie Dane Zaplatenej Pri Dovoze -> [MA_PODMIENKU] -> Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu
  Vratenie Dane Zaplatenej Pri Dovoze -> [MA_PODMIENKU] -> Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko

  Vznik Danovej Povinnosti Pri Dovoze Tovaru V Tuzemsku Pravnickej Osobe Z Ineho Clenskeho Statu Ktora Nie Je Zdanitelnou Osobou -> [VZTAHUJE_SA_NA] -> Danova Povinnost Pri Dovoze Tovaru
  Vznik Danovej Povinnosti Pri Dovoze Tovaru V Tuzemsku Pravnickej Osobe Z Ineho Clenskeho Statu Ktora Nie Je Zdanitelnou Osobou -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Z Ineho Clenskeho Statu
  Vznik Danovej Povinnosti Pri Dovoze Tovaru V Tuzemsku Pravnickej Osobe Z Ineho Clenskeho Statu Ktora Nie Je Zdanitelnou Osobou -> [NACHADZA_SA_V] -> Tuzemsko

  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Pravnicka Osoba Z Ineho Clenskeho Statu -> [PATRI_DO] -> Iny Clensky Stat
  Pravnicka Osoba Z Ineho Clenskeho Statu -> [MA_STATUS] -> Nie Je Zdanitelnou Osobou

  Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu -> [VZTAHUJE_SA_NA] -> Uzemie Tretieho Statu

  Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat
  Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko -> [NEVZTAHUJE_SA_NA] -> Tuzemsko

nodes:
  Paragraf: Paragraf § 25
  Odsek: Paragraf § 25 Odsek 5
  Pismeno: Paragraf § 25 Odsek 5 Pismeno a)

  Organizacia: Colny Organ
  Osoba: Pravnicka Osoba Z Ineho Clenskeho Statu
  Status: Nie Je Zdanitelnou Osobou

  Povinnost: Danova Povinnost Pri Dovoze Tovaru
  Konanie: Dovoz Tovaru
  Pravo: Vratenie Dane Zaplatenej Pri Dovoze

  Dan: Dan Zaplatena Pri Dovoze
  Tovar: Tovar

  Stat: Tuzemsko
  Stat: Iny Clensky Stat
  Lokacia: Uzemie Tretieho Statu

  Podmienka: Vznik Danovej Povinnosti Pri Dovoze Tovaru V Tuzemsku Pravnickej Osobe Z Ineho Clenskeho Statu Ktora Nie Je Zdanitelnou Osobou
  Podmienka: Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu
  Podmienka: Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko


---

chunk: 333
path: ['§ 25a', '4', 'b)']
path_as_text: Paragraf § 25a Odsek 4 Pismeno b)
text: (4) Platiteľ nemôže vykonať opravu základu dane pri nevymožiteľnej pohľadávke podľa odseku 2, ak b) tovar alebo služba bola dodaná odberateľovi (dlžníkovi) po vyhlásení konkurzu na majetok odberateľa (dlžníka) alebo

relations:
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 4
  Paragraf § 25a Odsek 4 -> [OBSAHUJE] -> Paragraf § 25a Odsek 4 Pismeno b)
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 2
  Paragraf § 25a Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 25a Odsek 2

  Paragraf § 25a Odsek 4 Pismeno b) -> [UPRAVUJE] -> Zakaz Opravy Zakladu Dane Pri Nevymozitelnej Pohladavke

  Platitel -> [NEMA_NAROK_NA] -> Oprava Zakladu Dane Pri Nevymozitelnej Pohladavke
  Oprava Zakladu Dane Pri Nevymozitelnej Pohladavke -> [VZTAHUJE_SA_NA] -> Nevymozitelna Pohladavka
  Oprava Zakladu Dane Pri Nevymozitelnej Pohladavke -> [JE_PODLA] -> Paragraf § 25a Odsek 2
  Oprava Zakladu Dane Pri Nevymozitelnej Pohladavke -> [MA_PODMIENKU] -> Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika

  Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika -> [VZTAHUJE_SA_NA] -> Sluzba
  Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika -> [VZTAHUJE_SA_NA] -> Odberatel Dlznik
  Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika -> [MA_DATUM] -> Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika

  Konkurz Na Majetok Odberatela Dlznika -> [VZTAHUJE_SA_NA] -> Majetok Odberatela Dlznika
  Majetok Odberatela Dlznika -> [PATRI_DO] -> Odberatel Dlznik

nodes:
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 4
  Pismeno: Paragraf § 25a Odsek 4 Pismeno b)
  Odsek: Paragraf § 25a Odsek 2

  Subjekt: Platitel
  Subjekt: Odberatel Dlznik

  Konanie: Oprava Zakladu Dane Pri Nevymozitelnej Pohladavke
  Konanie: Zakaz Opravy Zakladu Dane Pri Nevymozitelnej Pohladavke
  Konanie: Konkurz Na Majetok Odberatela Dlznika

  Pohladavka: Nevymozitelna Pohladavka
  Tovar: Tovar
  Sluzba: Sluzba
  Majetok: Majetok Odberatela Dlznika

  Podmienka: Tovar Alebo Sluzba Bola Dodana Odberatelovi Dlznikovi Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika
  Datum: Po Vyhlaseni Konkurzu Na Majetok Odberatela Dlznika


---

chunk: 356
path: ['§ 25a', '10', 'e)']
path_as_text: Paragraf § 25a Odsek 10 Pismeno e)
text: (10) Opravný doklad podľa odseku 7 písm. b) musí obsahovať e) sumu, ktorú platiteľ prijal v súvislosti s nevymožiteľnou pohľadávkou podľa odseku 2 alebo jej časťou, a z toho sumu prislúchajúcej dane,

relations:
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 10
  Paragraf § 25a Odsek 10 -> [OBSAHUJE] -> Paragraf § 25a Odsek 10 Pismeno e)
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 7
  Paragraf § 25a Odsek 7 -> [OBSAHUJE] -> Paragraf § 25a Odsek 7 Pismeno b)
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 2

  Paragraf § 25a Odsek 10 Pismeno e) -> [ODKAZUJE_NA] -> Paragraf § 25a Odsek 7 Pismeno b)
  Paragraf § 25a Odsek 10 Pismeno e) -> [ODKAZUJE_NA] -> Paragraf § 25a Odsek 2
  Paragraf § 25a Odsek 10 Pismeno e) -> [URCUJE] -> Obsah Opravneho Dokladu Podla Paragraf § 25a Odsek 7 Pismeno b)

  Opravny Doklad -> [JE_PODLA] -> Paragraf § 25a Odsek 7 Pismeno b)
  Opravny Doklad -> [OBSAHUJE] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Opravny Doklad -> [OBSAHUJE] -> Suma Prisluchajucej Dane

  Platitel -> [PRIJIMA] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou

  Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou -> [SUVISI_S] -> Nevymozitelna Pohladavka
  Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou -> [SUVISI_S] -> Cast Nevymozitelnej Pohladavky

  Nevymozitelna Pohladavka -> [JE_PODLA] -> Paragraf § 25a Odsek 2
  Cast Nevymozitelnej Pohladavky -> [JE_SUCASTOU] -> Nevymozitelna Pohladavka

  Suma Prisluchajucej Dane -> [VZTAHUJE_SA_NA] -> Prisluchajuca Dan
  Suma Prisluchajucej Dane -> [VYPLYVA_Z] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou

nodes:
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 10
  Pismeno: Paragraf § 25a Odsek 10 Pismeno e)
  Odsek: Paragraf § 25a Odsek 7
  Pismeno: Paragraf § 25a Odsek 7 Pismeno b)
  Odsek: Paragraf § 25a Odsek 2

  Dokument: Opravny Doklad
  Subjekt: Platitel

  Pohladavka: Nevymozitelna Pohladavka
  Pohladavka: Cast Nevymozitelnej Pohladavky

  Suma: Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Suma: Suma Prisluchajucej Dane
  Dan: Prisluchajuca Dan

  Zaznam: Obsah Opravneho Dokladu Podla Paragraf § 25a Odsek 7 Pismeno b)


---

chunk: 361
path: ['§ 25a', '13']
path_as_text: Paragraf § 25a Odsek 13
text: (13) Ak platiteľ opravil základ dane podľa odseku 3 a nesplnil povinnosť podľa odseku 11 písm. a), oprava základu dane uvedená v daňovom priznaní za zdaňovacie obdobie, v ktorom vykonal opravu základu dane, sa neuzná.

relations:
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 13
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 3
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 11
  Paragraf § 25a Odsek 11 -> [OBSAHUJE] -> Paragraf § 25a Odsek 11 Pismeno a)

  Paragraf § 25a Odsek 13 -> [ODKAZUJE_NA] -> Paragraf § 25a Odsek 3
  Paragraf § 25a Odsek 13 -> [ODKAZUJE_NA] -> Paragraf § 25a Odsek 11 Pismeno a)
  Paragraf § 25a Odsek 13 -> [UPRAVUJE] -> Neuznanie Opravy Zakladu Dane Uvedenej V Danovom Priznani

  Platitel -> [MA] -> Oprava Zakladu Dane Podla Paragraf § 25a Odsek 3
  Oprava Zakladu Dane Podla Paragraf § 25a Odsek 3 -> [JE_PODLA] -> Paragraf § 25a Odsek 3

  Platitel -> [MA_POVINNOST] -> Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a)
  Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a) -> [JE_PODLA] -> Paragraf § 25a Odsek 11 Pismeno a)
  Platitel -> [NESPLNA_PODMIENKY] -> Splnenie Povinnosti Podla Paragraf § 25a Odsek 11 Pismeno a)

  Neuznanie Opravy Zakladu Dane Uvedenej V Danovom Priznani -> [MA_PODMIENKU] -> Oprava Zakladu Dane Podla Paragraf § 25a Odsek 3
  Neuznanie Opravy Zakladu Dane Uvedenej V Danovom Priznani -> [MA_PODMIENKU] -> Nesplnenie Povinnosti Podla Paragraf § 25a Odsek 11 Pismeno a)

  Oprava Zakladu Dane Uvedena V Danovom Priznani -> [JE_SUCASTOU] -> Danove Priznanie
  Oprava Zakladu Dane Uvedena V Danovom Priznani -> [MA_STATUS] -> Neuznana Oprava Zakladu Dane
  Oprava Zakladu Dane Uvedena V Danovom Priznani -> [VYPLYVA_Z] -> Neuznanie Opravy Zakladu Dane Uvedenej V Danovom Priznani

  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie V Ktorom Platitel Vykonal Opravu Zakladu Dane
  Oprava Zakladu Dane Podla Paragraf § 25a Odsek 3 -> [MA_OBDOBIE] -> Zdanovacie Obdobie V Ktorom Platitel Vykonal Opravu Zakladu Dane

nodes:
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 13
  Odsek: Paragraf § 25a Odsek 3
  Odsek: Paragraf § 25a Odsek 11
  Pismeno: Paragraf § 25a Odsek 11 Pismeno a)

  Subjekt: Platitel

  Konanie: Oprava Zakladu Dane Podla Paragraf § 25a Odsek 3
  Konanie: Oprava Zakladu Dane Uvedena V Danovom Priznani
  Konanie: Neuznanie Opravy Zakladu Dane Uvedenej V Danovom Priznani

  Povinnost: Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a)
  Podmienka: Splnenie Povinnosti Podla Paragraf § 25a Odsek 11 Pismeno a)
  Podmienka: Nesplnenie Povinnosti Podla Paragraf § 25a Odsek 11 Pismeno a)

  DanovePriznanie: Danove Priznanie
  ZdanovacieObdobie: Zdanovacie Obdobie V Ktorom Platitel Vykonal Opravu Zakladu Dane

  Status: Neuznana Oprava Zakladu Dane


---

chunk: 379
path: ['§ 27', '4']
path_as_text: Paragraf § 27 Odsek 4
text: (4) Na účely správneho zatriedenia tovaru do číselného kódu podľa prílohy č. 7 sa použije záväzná informácia o nomenklatúrnom zatriedení tovaru vydaná colným orgánom podľa osobitného predpisu.6b)

relations:
  Paragraf § 27 -> [OBSAHUJE] -> Paragraf § 27 Odsek 4
  Paragraf § 27 Odsek 4 -> [ODKAZUJE_NA] -> Priloha C. 7
  Paragraf § 27 Odsek 4 -> [ODKAZUJE_NA] -> Osobitny Predpis

  Paragraf § 27 Odsek 4 -> [UPRAVUJE] -> Spravne Zatriedenie Tovaru Do Ciselneho Kodu Podla Prilohy C. 7

  Spravne Zatriedenie Tovaru Do Ciselneho Kodu Podla Prilohy C. 7 -> [VZTAHUJE_SA_NA] -> Tovar
  Spravne Zatriedenie Tovaru Do Ciselneho Kodu Podla Prilohy C. 7 -> [VZTAHUJE_SA_NA] -> Ciselny Kod Podla Prilohy C. 7
  Ciselny Kod Podla Prilohy C. 7 -> [JE_PODLA] -> Priloha C. 7

  Spravne Zatriedenie Tovaru Do Ciselneho Kodu Podla Prilohy C. 7 -> [VYPLYVA_Z] -> Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru
  Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru -> [VZTAHUJE_SA_NA] -> Nomenklaturne Zatriedenie Tovaru
  Nomenklaturne Zatriedenie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Colny Organ -> [VYDAVA] -> Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru
  Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru -> [JE_PODLA] -> Osobitny Predpis

nodes:
  Paragraf: Paragraf § 27
  Odsek: Paragraf § 27 Odsek 4

  Tovar: Tovar
  Zaznam: Ciselny Kod Podla Prilohy C. 7
  Priloha: Priloha C. 7

  Konanie: Spravne Zatriedenie Tovaru Do Ciselneho Kodu Podla Prilohy C. 7
  Dokument: Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru
  Zaznam: Nomenklaturne Zatriedenie Tovaru

  Organizacia: Colny Organ
  PravnyPredpis: Osobitny Predpis


---

chunk: 402
path: ['§ 34']
path_as_text: Paragraf § 34
text: Oslobodené od dane sú kultúrne služby a dodanie tovarov úzko s nimi súvisiacich, ak sú poskytované a) právnickou osobou zriadenou zákonom,14) b) právnickou osobou zriadenou Ministerstvom kultúry Slovenskej republiky, vyšším územným celkom alebo obcou podľa osobitného predpisu,15) c) právnickou osobou alebo fyzickou osobou, ktorá spĺňa jednu podmienku alebo viac podmienok podľa § 30 ods. 2.

relations:
  Paragraf § 30 -> [OBSAHUJE] -> Paragraf § 30 Odsek 2
  Paragraf § 34 -> [ODKAZUJE_NA] -> Paragraf § 30 Odsek 2
  Paragraf § 34 -> [ODKAZUJE_NA] -> Osobitny Predpis
  Paragraf § 34 -> [UPRAVUJE] -> Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane

  Kulturne Sluzby -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Dodanie Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Dodanie Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami -> [VZTAHUJE_SA_NA] -> Tovary Uzko Suvisiace S Kulturnymi Sluzbami
  Tovary Uzko Suvisiace S Kulturnymi Sluzbami -> [SUVISI_S] -> Kulturne Sluzby

  Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane -> [VZTAHUJE_SA_NA] -> Kulturne Sluzby
  Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane -> [VZTAHUJE_SA_NA] -> Dodanie Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami
  Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane -> [MA_PODMIENKU] -> Poskytovanie Pravnickou Osobou Zriadenou Zakonom
  Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane -> [MA_PODMIENKU] -> Poskytovanie Pravnickou Osobou Zriadenou Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou Podla Osobitneho Predpisu
  Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane -> [MA_PODMIENKU] -> Poskytovanie Pravnickou Alebo Fyzickou Osobou Splnajucou Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2

  Poskytovanie Pravnickou Osobou Zriadenou Zakonom -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Zriadena Zakonom
  Poskytovanie Pravnickou Osobou Zriadenou Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou Podla Osobitneho Predpisu -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou
  Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou -> [JE_PODLA] -> Osobitny Predpis

  Poskytovanie Pravnickou Alebo Fyzickou Osobou Splnajucou Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2 -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2
  Poskytovanie Pravnickou Alebo Fyzickou Osobou Splnajucou Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2 -> [VZTAHUJE_SA_NA] -> Fyzicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2

  Pravnicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2 -> [SPLNA_PODMIENKY] -> Jedna Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2
  Fyzicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2 -> [SPLNA_PODMIENKY] -> Jedna Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2
  Jedna Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2 -> [JE_PODLA] -> Paragraf § 30 Odsek 2

nodes:
  Paragraf: Paragraf § 34
  Paragraf: Paragraf § 30
  Odsek: Paragraf § 30 Odsek 2

  Dan: Dan
  Sluzba: Kulturne Sluzby
  Tovar: Tovary Uzko Suvisiace S Kulturnymi Sluzbami
  Konanie: Dodanie Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami

  Pravo: Oslobodenie Kulturnych Sluzieb A Dodania Tovarov Uzko Suvisiacich S Kulturnymi Sluzbami Od Dane

  Podmienka: Poskytovanie Pravnickou Osobou Zriadenou Zakonom
  Podmienka: Poskytovanie Pravnickou Osobou Zriadenou Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou Podla Osobitneho Predpisu
  Podmienka: Poskytovanie Pravnickou Alebo Fyzickou Osobou Splnajucou Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2
  Podmienka: Jedna Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2

  Osoba: Pravnicka Osoba Zriadena Zakonom
  Osoba: Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky Vyssim Uzemnym Celkom Alebo Obcou
  Osoba: Pravnicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2
  Osoba: Fyzicka Osoba Splnajuca Jednu Alebo Viac Podmienok Podla Paragraf § 30 Odsek 2

  Organizacia: Ministerstvo Kultury Slovenskej Republiky
  Organizacia: Vyssi Uzemny Celok
  Organizacia: Obec
  PravnyPredpis: Osobitny Predpis


---

chunk: 425
path: ['§ 39', '1', 'c)']
path_as_text: Paragraf § 39 Odsek 1 Pismeno c)
text: (1) Oslobodené od dane sú: c) činnosti týkajúce sa vkladov a bežných účtov vrátane ich sprostredkovania,

relations:
  Paragraf § 39 -> [OBSAHUJE] -> Paragraf § 39 Odsek 1
  Paragraf § 39 Odsek 1 -> [OBSAHUJE] -> Paragraf § 39 Odsek 1 Pismeno c)

  Paragraf § 39 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane

  Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Vkladov
  Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Beznych Uctov
  Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane -> [VZTAHUJE_SA_NA] -> Sprostredkovanie Cinnosti Tykajucich Sa Vkladov
  Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane -> [VZTAHUJE_SA_NA] -> Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov

  Cinnosti Tykajuce Sa Vkladov -> [VZTAHUJE_SA_NA] -> Vklady
  Cinnosti Tykajuce Sa Beznych Uctov -> [VZTAHUJE_SA_NA] -> Bezne Ucty

  Sprostredkovanie Cinnosti Tykajucich Sa Vkladov -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Vkladov
  Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Beznych Uctov

  Cinnosti Tykajuce Sa Vkladov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Cinnosti Tykajuce Sa Beznych Uctov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Sprostredkovanie Cinnosti Tykajucich Sa Vkladov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov -> [JE_OSLOBODENE_OD_DANE] -> Dan

nodes:
  Paragraf: Paragraf § 39
  Odsek: Paragraf § 39 Odsek 1
  Pismeno: Paragraf § 39 Odsek 1 Pismeno c)

  Pravo: Oslobodenie Cinnosti Tykajucich Sa Vkladov A Beznych Uctov Vratane Ich Sprostredkovania Od Dane

  Sluzba: Cinnosti Tykajuce Sa Vkladov
  Sluzba: Cinnosti Tykajuce Sa Beznych Uctov
  Sluzba: Sprostredkovanie Cinnosti Tykajucich Sa Vkladov
  Sluzba: Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov

  Ucet: Vklady
  BankovyUcet: Bezne Ucty
  Dan: Dan


---

chunk: 443
path: ['§ 43', '1', 'b)']
path_as_text: Paragraf § 43 Odsek 1 Pismeno b)
text: (1) Oslobodené od dane je dodanie tovaru, ktorý je odoslaný alebo prepravený z tuzemska do iného členského štátu predávajúcim alebo nadobúdateľom alebo treťou osobou na účet predávajúceho alebo na účet nadobúdateľa, ak b) nadobúdateľ podľa písmena a) je identifikovaný pre daň v inom členskom štáte a oznámil svoje identifikačné číslo pre daň pridelené v inom členskom štáte dodávateľovi.

relations:
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 1
  Paragraf § 43 Odsek 1 -> [OBSAHUJE] -> Paragraf § 43 Odsek 1 Pismeno a)
  Paragraf § 43 Odsek 1 -> [OBSAHUJE] -> Paragraf § 43 Odsek 1 Pismeno b)
  Paragraf § 43 Odsek 1 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 1 Pismeno a)

  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Dodanie Tovaru -> [MA_PODMIENKU] -> Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu
  Dodanie Tovaru -> [MA_PODMIENKU] -> Podmienka Nadobudatela Podla Paragraf § 43 Odsek 1 Pismeno b)

  Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tuzemsko
  Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat
  Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu -> [MA_PODMIENKU] -> Odoslanie Alebo Preprava Predavajucim Nadobudatelom Alebo Tretou Osobou Na Ucet Predavajuceho Alebo Nadobudatela

  Predavajuci -> [DODAVA] -> Tovar
  Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a) -> [NADOBUDA] -> Tovar
  Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a) -> [JE_PODLA] -> Paragraf § 43 Odsek 1 Pismeno a)

  Odoslanie Alebo Preprava Predavajucim Nadobudatelom Alebo Tretou Osobou Na Ucet Predavajuceho Alebo Nadobudatela -> [VZTAHUJE_SA_NA] -> Predavajuci
  Odoslanie Alebo Preprava Predavajucim Nadobudatelom Alebo Tretou Osobou Na Ucet Predavajuceho Alebo Nadobudatela -> [VZTAHUJE_SA_NA] -> Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a)
  Odoslanie Alebo Preprava Predavajucim Nadobudatelom Alebo Tretou Osobou Na Ucet Predavajuceho Alebo Nadobudatela -> [VZTAHUJE_SA_NA] -> Tretia Osoba

  Podmienka Nadobudatela Podla Paragraf § 43 Odsek 1 Pismeno b) -> [VZTAHUJE_SA_NA] -> Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a)
  Podmienka Nadobudatela Podla Paragraf § 43 Odsek 1 Pismeno b) -> [VZTAHUJE_SA_NA] -> Identifikacia Pre Dan V Inom Clenskom State
  Podmienka Nadobudatela Podla Paragraf § 43 Odsek 1 Pismeno b) -> [VZTAHUJE_SA_NA] -> Oznamenie Identifikacneho Cisla Pre Dan Dodavatelovi

  Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a) -> [MA] -> Identifikacia Pre Dan V Inom Clenskom State
  Identifikacia Pre Dan V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacia Pre Dan V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat

  Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a) -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State
  Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat

  Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a) -> [OZNAMUJE] -> Oznamenie Identifikacneho Cisla Pre Dan Dodavatelovi
  Oznamenie Identifikacneho Cisla Pre Dan Dodavatelovi -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State
  Oznamenie Identifikacneho Cisla Pre Dan Dodavatelovi -> [VZTAHUJE_SA_NA] -> Dodavatel

nodes:
  Paragraf: Paragraf § 43
  Odsek: Paragraf § 43 Odsek 1
  Pismeno: Paragraf § 43 Odsek 1 Pismeno a)
  Pismeno: Paragraf § 43 Odsek 1 Pismeno b)

  Konanie: Dodanie Tovaru
  Tovar: Tovar
  Dan: Dan

  Stat: Tuzemsko
  Stat: Iny Clensky Stat

  Osoba: Predavajuci
  Osoba: Nadobudatel Podla Paragraf § 43 Odsek 1 Pismeno a)
  Osoba: Tretia Osoba
  Osoba: Dodavatel

  Registracia: Identifikacia Pre Dan V Inom Clenskom State
  Zaznam: Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State
  Oznamenie: Oznamenie Identifikacneho Cisla Pre Dan Dodavatelovi

  Podmienka: Tovar Odoslany Alebo Prepraveny Z Tuzemska Do Ineho Clenskeho Statu
  Podmienka: Odoslanie Alebo Preprava Predavajucim Nadobudatelom Alebo Tretou Osobou Na Ucet Predavajuceho Alebo Nadobudatela
  Podmienka: Podmienka Nadobudatela Podla Paragraf § 43 Odsek 1 Pismeno b)


---

chunk: 448
path: ['§ 43', '5', 'b)']
path_as_text: Paragraf § 43 Odsek 5 Pismeno b)
text: (5) Platiteľ je povinný preukázať, že sú splnené podmienky oslobodenia od dane podľa odsekov 1 až 4 b) dokladom o odoslaní tovaru, ak prepravu tovaru zabezpečí dodávateľ alebo odberateľ poštovým podnikom, alebo kópiou dokladu o preprave tovaru, v ktorom je potvrdené odberateľom alebo osobou ním poverenou prevzatie tovaru v inom členskom štáte, ak prepravu tovaru zabezpečí dodávateľ alebo odberateľ osobou inou ako poštovým podnikom; ak platiteľ takú kópiu dokladu o preprave tovaru nemá, prevzatie tovaru v inom členskom štáte je povinný preukázať iným dokladom,

relations:
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 5
  Paragraf § 43 Odsek 5 -> [OBSAHUJE] -> Paragraf § 43 Odsek 5 Pismeno b)
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 1
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 2
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 3
  Paragraf § 43 -> [OBSAHUJE] -> Paragraf § 43 Odsek 4

  Paragraf § 43 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 1
  Paragraf § 43 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 2
  Paragraf § 43 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 3
  Paragraf § 43 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 4

  Platitel -> [MA_POVINNOST] -> Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4
  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4 -> [VZTAHUJE_SA_NA] -> Podmienky Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4
  Podmienky Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4 -> [VZTAHUJE_SA_NA] -> Dan

  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4 -> [MA_PODMIENKU] -> Preukazanie Dokladom O Odoslani Tovaru Pri Preprave Tovaru Postovym Podnikom
  Preukazanie Dokladom O Odoslani Tovaru Pri Preprave Tovaru Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Doklad O Odoslani Tovaru
  Preukazanie Dokladom O Odoslani Tovaru Pri Preprave Tovaru Postovym Podnikom -> [MA_PODMIENKU] -> Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Postovym Podnikom

  Doklad O Odoslani Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Dodavatel
  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Odberatel
  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Postovy Podnik

  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4 -> [MA_PODMIENKU] -> Preukazanie Kopiou Dokladu O Preprave Tovaru Pri Preprave Osobou Inou Ako Postovym Podnikom
  Preukazanie Kopiou Dokladu O Preprave Tovaru Pri Preprave Osobou Inou Ako Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Kopia Dokladu O Preprave Tovaru
  Preukazanie Kopiou Dokladu O Preprave Tovaru Pri Preprave Osobou Inou Ako Postovym Podnikom -> [MA_PODMIENKU] -> Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Osobou Inou Ako Postovym Podnikom

  Kopia Dokladu O Preprave Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Kopia Dokladu O Preprave Tovaru -> [OBSAHUJE] -> Potvrdenie Prevzatia Tovaru V Inom Clenskom State Odberatelom Alebo Osobou Poverenou Odberatelom
  Potvrdenie Prevzatia Tovaru V Inom Clenskom State Odberatelom Alebo Osobou Poverenou Odberatelom -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru V Inom Clenskom State
  Potvrdenie Prevzatia Tovaru V Inom Clenskom State Odberatelom Alebo Osobou Poverenou Odberatelom -> [VZTAHUJE_SA_NA] -> Odberatel
  Potvrdenie Prevzatia Tovaru V Inom Clenskom State Odberatelom Alebo Osobou Poverenou Odberatelom -> [VZTAHUJE_SA_NA] -> Osoba Poverena Odberatelom
  Osoba Poverena Odberatelom -> [KONA_V_MENE] -> Odberatel

  Prevzatie Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Tovar
  Prevzatie Tovaru V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat

  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Osobou Inou Ako Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Dodavatel
  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Osobou Inou Ako Postovym Podnikom -> [VZTAHUJE_SA_NA] -> Odberatel
  Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Osobou Inou Ako Postovym Podnikom -> [NEVZTAHUJE_SA_NA] -> Postovy Podnik

  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4 -> [MA_PODMIENKU] -> Preukazanie Prevzatia Tovaru V Inom Clenskom State Inym Dokladom Ak Platitel Nema Kopiu Dokladu O Preprave Tovaru
  Preukazanie Prevzatia Tovaru V Inom Clenskom State Inym Dokladom Ak Platitel Nema Kopiu Dokladu O Preprave Tovaru -> [VZTAHUJE_SA_NA] -> Iny Doklad
  Preukazanie Prevzatia Tovaru V Inom Clenskom State Inym Dokladom Ak Platitel Nema Kopiu Dokladu O Preprave Tovaru -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru V Inom Clenskom State
  Preukazanie Prevzatia Tovaru V Inom Clenskom State Inym Dokladom Ak Platitel Nema Kopiu Dokladu O Preprave Tovaru -> [MA_PODMIENKU] -> Platitel Nema Kopiu Dokladu O Preprave Tovaru

nodes:
  Paragraf: Paragraf § 43
  Odsek: Paragraf § 43 Odsek 5
  Pismeno: Paragraf § 43 Odsek 5 Pismeno b)
  Odsek: Paragraf § 43 Odsek 1
  Odsek: Paragraf § 43 Odsek 2
  Odsek: Paragraf § 43 Odsek 3
  Odsek: Paragraf § 43 Odsek 4

  Subjekt: Platitel
  Subjekt: Dodavatel
  Subjekt: Odberatel
  Organizacia: Postovy Podnik
  Osoba: Osoba Poverena Odberatelom

  Povinnost: Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4
  Podmienka: Podmienky Oslobodenia Od Dane Podla Paragraf § 43 Odseky 1 Az 4
  Podmienka: Preukazanie Dokladom O Odoslani Tovaru Pri Preprave Tovaru Postovym Podnikom
  Podmienka: Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Postovym Podnikom
  Podmienka: Preukazanie Kopiou Dokladu O Preprave Tovaru Pri Preprave Osobou Inou Ako Postovym Podnikom
  Podmienka: Preprava Tovaru Zabezpecena Dodavatelom Alebo Odberatelom Osobou Inou Ako Postovym Podnikom
  Podmienka: Preukazanie Prevzatia Tovaru V Inom Clenskom State Inym Dokladom Ak Platitel Nema Kopiu Dokladu O Preprave Tovaru
  Podmienka: Platitel Nema Kopiu Dokladu O Preprave Tovaru

  Dokument: Doklad O Odoslani Tovaru
  Dokument: Kopia Dokladu O Preprave Tovaru
  Dokument: Iny Doklad

  Tovar: Tovar
  Sluzba: Preprava Tovaru
  Konanie: Prevzatie Tovaru V Inom Clenskom State
  Zaznam: Potvrdenie Prevzatia Tovaru V Inom Clenskom State Odberatelom Alebo Osobou Poverenou Odberatelom

  Stat: Iny Clensky Stat
  Dan: Dan


---

chunk: 471
path: ['§ 45', '4', 'b)']
path_as_text: Paragraf § 45 Odsek 4 Pismeno b)
text: (4) Zo záznamov vedených na určenie dane musí byť zrejmé b) u druhého odberateľa, ak použije pri trojstrannom obchode identifikačné číslo pre daň pridelené v tuzemsku, základ dane, suma dane a názov alebo meno a adresa prvého odberateľa.

relations:
  Paragraf § 45 -> [OBSAHUJE] -> Paragraf § 45 Odsek 4
  Paragraf § 45 Odsek 4 -> [OBSAHUJE] -> Paragraf § 45 Odsek 4 Pismeno b)

  Paragraf § 45 Odsek 4 Pismeno b) -> [UPRAVUJE] -> Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela

  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [VZTAHUJE_SA_NA] -> Dan
  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [VZTAHUJE_SA_NA] -> Druhy Odberatel
  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [MA_PODMIENKU] -> Druhy Odberatel Pouzije Pri Trojstrannom Obchode Identifikacne Cislo Pre Dan Pridelene V Tuzemsku

  Druhy Odberatel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Tuzemsko

  Druhy Odberatel Pouzije Pri Trojstrannom Obchode Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Druhy Odberatel
  Druhy Odberatel Pouzije Pri Trojstrannom Obchode Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Trojstranny Obchod
  Druhy Odberatel Pouzije Pri Trojstrannom Obchode Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Pridelene V Tuzemsku

  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [OBSAHUJE] -> Zaklad Dane
  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [OBSAHUJE] -> Suma Dane
  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [OBSAHUJE] -> Nazov Alebo Meno Prveho Odberatela
  Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela -> [OBSAHUJE] -> Adresa Prveho Odberatela

  Prvy Odberatel -> [MA_IDENTIFIKATOR] -> Nazov Alebo Meno Prveho Odberatela
  Prvy Odberatel -> [MA_ADRESU] -> Adresa Prveho Odberatela

nodes:
  Paragraf: Paragraf § 45
  Odsek: Paragraf § 45 Odsek 4
  Pismeno: Paragraf § 45 Odsek 4 Pismeno b)

  Zaznam: Zaznamy Vedene Na Urcenie Dane U Druheho Odberatela
  Dan: Dan

  Subjekt: Druhy Odberatel
  Subjekt: Prvy Odberatel
  Konanie: Trojstranny Obchod

  Zaznam: Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Stat: Tuzemsko

  Podmienka: Druhy Odberatel Pouzije Pri Trojstrannom Obchode Identifikacne Cislo Pre Dan Pridelene V Tuzemsku

  Suma: Zaklad Dane
  Suma: Suma Dane
  Zaznam: Nazov Alebo Meno Prveho Odberatela
  Adresa: Adresa Prveho Odberatela


---

chunk: 484
path: ['§ 47', '6']
path_as_text: Paragraf § 47 Odsek 6
text: (6) Oslobodené od dane sú služby vrátane prepravných a s nimi súvisiacich doplnkových služieb, iné ako služby oslobodené od dane podľa § 28 až 41, ktoré sú priamo spojené s vývozom tovaru a s tovarom pod colným opatrením podľa § 18 ods. 2.

relations:
  Paragraf § 47 -> [OBSAHUJE] -> Paragraf § 47 Odsek 6
  Paragraf § 18 -> [OBSAHUJE] -> Paragraf § 18 Odsek 2

  Paragraf § 47 Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 18 Odsek 2
  Paragraf § 47 Odsek 6 -> [ODKAZUJE_NA] -> Paragrafy § 28 Az § 41
  Paragraf § 47 Odsek 6 -> [UPRAVUJE] -> Oslobodenie Sluzieb Priamo Spojenych S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim Od Dane

  Oslobodenie Sluzieb Priamo Spojenych S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim Od Dane -> [VZTAHUJE_SA_NA] -> Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim
  Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim -> [NEVZTAHUJE_SA_NA] -> Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41

  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 28
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 29
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 30
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 31
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 32
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 33
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 34
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 35
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 36
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 37
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 38
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 39
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 40
  Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41 -> [ODKAZUJE_NA] -> Paragraf § 41


  Prepravne Sluzby -> [JE_TYPOM] -> Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim
  Doplnkove Sluzby Suvisiace S Prepravnymi Sluzbami -> [JE_TYPOM] -> Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim
  Doplnkove Sluzby Suvisiace S Prepravnymi Sluzbami -> [SUVISI_S] -> Prepravne Sluzby

  Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim -> [SUVISI_S] -> Vyvoz Tovaru
  Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim -> [SUVISI_S] -> Tovar Pod Colnym Opatrenim Podla Paragraf § 18 Odsek 2

  Vyvoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Pod Colnym Opatrenim Podla Paragraf § 18 Odsek 2 -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Pod Colnym Opatrenim Podla Paragraf § 18 Odsek 2 -> [JE_PODLA] -> Paragraf § 18 Odsek 2

nodes:
  Paragraf: Paragraf § 47
  Odsek: Paragraf § 47 Odsek 6
  Paragraf: Paragraf § 18
  Odsek: Paragraf § 18 Odsek 2
  Paragraf: Paragrafy § 28 Az § 41
  Paragraf: Paragraf § 28
  Paragraf: Paragraf § 29
  Paragraf: Paragraf § 30
  Paragraf: Paragraf § 31
  Paragraf: Paragraf § 32
  Paragraf: Paragraf § 33
  Paragraf: Paragraf § 34
  Paragraf: Paragraf § 35
  Paragraf: Paragraf § 36
  Paragraf: Paragraf § 37
  Paragraf: Paragraf § 38
  Paragraf: Paragraf § 39
  Paragraf: Paragraf § 40
  Paragraf: Paragraf § 41


  Dan: Dan

  Pravo: Oslobodenie Sluzieb Priamo Spojenych S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim Od Dane

  Sluzba: Sluzby Priamo Spojene S Vyvozom Tovaru A Tovarom Pod Colnym Opatrenim
  Sluzba: Prepravne Sluzby
  Sluzba: Doplnkove Sluzby Suvisiace S Prepravnymi Sluzbami
  Sluzba: Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41

  Konanie: Vyvoz Tovaru
  Tovar: Tovar
  Tovar: Tovar Pod Colnym Opatrenim Podla Paragraf § 18 Odsek 2


---

chunk: 494
path: ['§ 47', '13', 'b)']
path_as_text: Paragraf § 47 Odsek 13 Pismeno b)
text: (13) Oslobodené od dane je bezodplatné dodanie tovaru formou daru poskytnutého na základe  písomnej darovacej zmluvy uzatvorenej medzi platiteľom a Ministerstvom vnútra Slovenskej republiky na účel vývozu tovaru mimo územia Európskej únie ako súčasť humanitárnej činnosti a dobročinnej činnosti. Ministerstvo vnútra Slovenskej republiky za každý kalendárny rok do 15. januára nasledujúceho kalendárneho roka predloží finančnému riaditeľstvu b) zoznam evidenčných čísiel colných vyhlásení o vývoze tovaru darovaného platiteľom za príslušný kalendárny rok.

relations:
  Paragraf § 47 -> [OBSAHUJE] -> Paragraf § 47 Odsek 13
  Paragraf § 47 Odsek 13 -> [OBSAHUJE] -> Paragraf § 47 Odsek 13 Pismeno b)

  Paragraf § 47 Odsek 13 -> [UPRAVUJE] -> Oslobodenie Bezodplatneho Dodania Tovaru Formou Daru Od Dane
  Bezodplatne Dodanie Tovaru Formou Daru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Bezodplatne Dodanie Tovaru Formou Daru -> [VZTAHUJE_SA_NA] -> Darovany Tovar
  Bezodplatne Dodanie Tovaru Formou Daru -> [MA_PODMIENKU] -> Pisomna Darovacia Zmluva Uzatvorena Medzi Platitelom A Ministerstvom Vnutra Slovenskej Republiky
  Bezodplatne Dodanie Tovaru Formou Daru -> [MA_PODMIENKU] -> Ucel Vyvozu Tovaru Mimo Uzemia Europskej Unie Ako Sucast Humanitarnej Cinnosti A Dobrocinnej Cinnosti

  Pisomna Darovacia Zmluva Uzatvorena Medzi Platitelom A Ministerstvom Vnutra Slovenskej Republiky -> [VZTAHUJE_SA_NA] -> Platitel
  Pisomna Darovacia Zmluva Uzatvorena Medzi Platitelom A Ministerstvom Vnutra Slovenskej Republiky -> [VZTAHUJE_SA_NA] -> Ministerstvo Vnutra Slovenskej Republiky
  Pisomna Darovacia Zmluva Uzatvorena Medzi Platitelom A Ministerstvom Vnutra Slovenskej Republiky -> [VZTAHUJE_SA_NA] -> Darovany Tovar

  Ucel Vyvozu Tovaru Mimo Uzemia Europskej Unie Ako Sucast Humanitarnej Cinnosti A Dobrocinnej Cinnosti -> [VZTAHUJE_SA_NA] -> Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [VZTAHUJE_SA_NA] -> Darovany Tovar
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [VZTAHUJE_SA_NA] -> Uzemie Europskej Unie
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [JE_SUCASTOU] -> Humanitarna Cinnost
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [JE_SUCASTOU] -> Dobrocinna Cinnost

  Platitel -> [DODAVA] -> Darovany Tovar

  Paragraf § 47 Odsek 13 Pismeno b) -> [UPRAVUJE] -> Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom

  Ministerstvo Vnutra Slovenskej Republiky -> [MA_POVINNOST] -> Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu
  Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu -> [VZTAHUJE_SA_NA] -> Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom
  Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu -> [VZTAHUJE_SA_NA] -> Financne Riaditelstvo
  Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu -> [MA_OBDOBIE] -> Kazdy Kalendarny Rok
  Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu -> [MA_LEHOTU] -> Do 15. Januara Nasledujuceho Kalendarneho Roka

  Ministerstvo Vnutra Slovenskej Republiky -> [PREDKLADA] -> Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom

  Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom -> [VZTAHUJE_SA_NA] -> Evidencne Cisla Colnych Vyhlaseni
  Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom -> [VZTAHUJE_SA_NA] -> Colne Vyhlasenie O Vyvoze Tovaru
  Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom -> [MA_OBDOBIE] -> Prislusny Kalendarny Rok

  Colne Vyhlasenie O Vyvoze Tovaru -> [VZTAHUJE_SA_NA] -> Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Colne Vyhlasenie O Vyvoze Tovaru -> [VZTAHUJE_SA_NA] -> Darovany Tovar
  Darovany Tovar -> [VYPLYVA_Z] -> Bezodplatne Dodanie Tovaru Formou Daru

nodes:
  Paragraf: Paragraf § 47
  Odsek: Paragraf § 47 Odsek 13
  Pismeno: Paragraf § 47 Odsek 13 Pismeno b)

  Pravo: Oslobodenie Bezodplatneho Dodania Tovaru Formou Daru Od Dane
  Konanie: Bezodplatne Dodanie Tovaru Formou Daru
  Dan: Dan

  Zmluva: Pisomna Darovacia Zmluva Uzatvorena Medzi Platitelom A Ministerstvom Vnutra Slovenskej Republiky
  Subjekt: Platitel
  Organizacia: Ministerstvo Vnutra Slovenskej Republiky
  Organizacia: Financne Riaditelstvo

  Tovar: Darovany Tovar
  Konanie: Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Lokacia: Uzemie Europskej Unie
  Konanie: Humanitarna Cinnost
  Konanie: Dobrocinna Cinnost

  Podmienka: Ucel Vyvozu Tovaru Mimo Uzemia Europskej Unie Ako Sucast Humanitarnej Cinnosti A Dobrocinnej Cinnosti

  Povinnost: Povinnost Predlozit Zoznam Evidencnych Cisiel Colnych Vyhlaseni Financnemu Riaditelstvu
  Dokument: Zoznam Evidencnych Cisiel Colnych Vyhlaseni O Vyvoze Tovaru Darovaneho Platitelom
  Zaznam: Evidencne Cisla Colnych Vyhlaseni
  Dokument: Colne Vyhlasenie O Vyvoze Tovaru

  Obdobie: Kazdy Kalendarny Rok
  Obdobie: Prislusny Kalendarny Rok
  Lehota: Do 15. Januara Nasledujuceho Kalendarneho Roka


---

chunk: 517
path: ['§ 48', '2', 'w)']
path_as_text: Paragraf § 48 Odsek 2 Pismeno w)
text: (2) Tovar, ktorý je prepustený do colného režimu voľný obeh s oslobodením od cla podľa osobitného predpisu,22) je oslobodený od dane, ak ide o w) rôzne dokumenty a predmety,

relations:
  Paragraf § 48 -> [OBSAHUJE] -> Paragraf § 48 Odsek 2
  Paragraf § 48 Odsek 2 -> [OBSAHUJE] -> Paragraf § 48 Odsek 2 Pismeno w)
  Paragraf § 48 Odsek 2 -> [ODKAZUJE_NA] -> Osobitny Predpis

  Paragraf § 48 Odsek 2 Pismeno w) -> [UPRAVUJE] -> Oslobodenie Roznych Dokumentov A Predmetov Od Dane

  Rozne Dokumenty A Predmety -> [JE_TYPOM] -> Tovar
  Rozne Dokumenty A Predmety -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Rozne Dokumenty A Predmety -> [MA_PODMIENKU] -> Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu

  Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu -> [VZTAHUJE_SA_NA] -> Tovar
  Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu -> [MA_STATUS] -> Colny Rezim Volny Obeh
  Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu -> [MA_PODMIENKU] -> Oslobodenie Od Cla Podla Osobitneho Predpisu
  Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu -> [JE_PODLA] -> Osobitny Predpis

  Oslobodenie Od Cla Podla Osobitneho Predpisu -> [OSLOBODZUJE_OD] -> Clo
  Oslobodenie Od Cla Podla Osobitneho Predpisu -> [JE_PODLA] -> Osobitny Predpis

nodes:
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 2
  Pismeno: Paragraf § 48 Odsek 2 Pismeno w)

  Tovar: Tovar
  Tovar: Rozne Dokumenty A Predmety

  Dan: Dan
  Dan: Clo

  Pravo: Oslobodenie Roznych Dokumentov A Predmetov Od Dane
  Podmienka: Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu
  Podmienka: Oslobodenie Od Cla Podla Osobitneho Predpisu

  Status: Colny Rezim Volny Obeh
  PravnyPredpis: Osobitny Predpis


---

chunk: 525
path: ['§ 48', '5', 'a)']
path_as_text: Paragraf § 48 Odsek 5 Pismeno a)
text: (5) Oslobodený od dane je dovoz tovaru a) osobami, ktoré požívajú výsady a imunity podľa medzinárodného práva,23) ak sa na tento dovoz vzťahuje oslobodenie od cla,

relations:
  Paragraf § 48 -> [OBSAHUJE] -> Paragraf § 48 Odsek 5
  Paragraf § 48 Odsek 5 -> [OBSAHUJE] -> Paragraf § 48 Odsek 5 Pismeno a)

  Paragraf § 48 Odsek 5 Pismeno a) -> [UPRAVUJE] -> Oslobodenie Dovozu Tovaru Od Dane Osobami Pozivajucimi Vysady A Imunity Podla Medzinarodneho Prava

  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava
  Dovoz Tovaru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Dovoz Tovaru -> [MA_PODMIENKU] -> Oslobodenie Od Cla Vztahujuce Sa Na Tento Dovoz

  Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava -> [MA_PRAVO] -> Vysady A Imunity Podla Medzinarodneho Prava
  Vysady A Imunity Podla Medzinarodneho Prava -> [JE_PODLA] -> Medzinarodne Pravo

  Oslobodenie Od Cla Vztahujuce Sa Na Tento Dovoz -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Oslobodenie Od Cla Vztahujuce Sa Na Tento Dovoz -> [OSLOBODZUJE_OD] -> Clo

nodes:
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 5
  Pismeno: Paragraf § 48 Odsek 5 Pismeno a)

  Konanie: Dovoz Tovaru
  Tovar: Tovar

  Pravo: Oslobodenie Dovozu Tovaru Od Dane Osobami Pozivajucimi Vysady A Imunity Podla Medzinarodneho Prava
  Dan: Dan
  Dan: Clo

  Osoba: Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava
  Pravo: Vysady A Imunity Podla Medzinarodneho Prava
  PravnyPredpis: Medzinarodne Pravo
  Pravo: Oslobodenie Od Cla Vztahujuce Sa Na Tento Dovoz


---

chunk: 540
path: ['§ 48a', '1', 'a)']
path_as_text: Paragraf § 48a Odsek 1 Pismeno a)
text: (1) Na účely tohto ustanovenia sa rozumie a) cestujúcim leteckou dopravou osoba cestujúca leteckým dopravným prostriedkom okrem dopravného prostriedku súkromného rekreačného lietania,

relations:
  Paragraf § 48a -> [OBSAHUJE] -> Paragraf § 48a Odsek 1
  Paragraf § 48a Odsek 1 -> [OBSAHUJE] -> Paragraf § 48a Odsek 1 Pismeno a)

  Paragraf § 48a Odsek 1 Pismeno a) -> [DEFINUJE] -> Cestujuci Leteckou Dopravou

  Cestujuci Leteckou Dopravou -> [JE_TYPOM] -> Osoba
  Cestujuci Leteckou Dopravou -> [MA_PODMIENKU] -> Osoba Cestujuca Leteckym Dopravnym Prostriedkom
  Cestujuci Leteckou Dopravou -> [NEVZTAHUJE_SA_NA] -> Dopravny Prostriedok Sukromneho Rekreacneho Lietania

  Osoba Cestujuca Leteckym Dopravnym Prostriedkom -> [VZTAHUJE_SA_NA] -> Osoba
  Osoba Cestujuca Leteckym Dopravnym Prostriedkom -> [VZTAHUJE_SA_NA] -> Letecky Dopravny Prostriedok
  Osoba Cestujuca Leteckym Dopravnym Prostriedkom -> [NEVZTAHUJE_SA_NA] -> Dopravny Prostriedok Sukromneho Rekreacneho Lietania

nodes:
  Paragraf: Paragraf § 48a
  Odsek: Paragraf § 48a Odsek 1
  Pismeno: Paragraf § 48a Odsek 1 Pismeno a)

  Osoba: Cestujuci Leteckou Dopravou
  Osoba: Osoba

  Podmienka: Osoba Cestujuca Leteckym Dopravnym Prostriedkom

  Vozidlo: Letecky Dopravny Prostriedok
  Vozidlo: Dopravny Prostriedok Sukromneho Rekreacneho Lietania


---

chunk: 563
path: ['§ 48b', '2']
path_as_text: Paragraf § 48b Odsek 2
text: (2) Colný úrad rozhodnutím určí výšku zabezpečenia dane a lehotu na jej zaplatenie. Proti rozhodnutiu o zabezpečení dane nie je možné podať odvolanie. Ak osoba podľa odseku 1 zabezpečenie dane nezaplatí v lehote a vo výške určenej v rozhodnutí, colný úrad oslobodenie od dane podľa § 48 ods. 3 neuplatní.

relations:
  Paragraf § 48b -> [OBSAHUJE] -> Paragraf § 48b Odsek 2
  Paragraf § 48b -> [OBSAHUJE] -> Paragraf § 48b Odsek 1
  Paragraf § 48 -> [OBSAHUJE] -> Paragraf § 48 Odsek 3

  Paragraf § 48b Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 48b Odsek 1
  Paragraf § 48b Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 48 Odsek 3

  Colny Urad -> [VYDAVA] -> Rozhodnutie O Zabezpeceni Dane
  Rozhodnutie O Zabezpeceni Dane -> [ROZHODUJE_O] -> Zabezpecenie Dane
  Rozhodnutie O Zabezpeceni Dane -> [URCUJE] -> Vyska Zabezpecenia Dane
  Rozhodnutie O Zabezpeceni Dane -> [URCUJE] -> Lehota Na Zaplatenie Zabezpecenia Dane

  Zaplatenie Zabezpecenia Dane -> [MA_LEHOTU] -> Lehota Na Zaplatenie Zabezpecenia Dane
  Zaplatenie Zabezpecenia Dane -> [MA_SUMU] -> Vyska Zabezpecenia Dane

  Osoba Podla Paragraf § 48b Odsek 1 -> [JE_PODLA] -> Paragraf § 48b Odsek 1
  Osoba Podla Paragraf § 48b Odsek 1 -> [PLATI] -> Zaplatenie Zabezpecenia Dane

  Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane -> [VZTAHUJE_SA_NA] -> Rozhodnutie O Zabezpeceni Dane
  Osoba Podla Paragraf § 48b Odsek 1 -> [NEMA_NAROK_NA] -> Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane

  Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3 -> [OSLOBODZUJE_OD] -> Dan
  Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3 -> [JE_PODLA] -> Paragraf § 48 Odsek 3

  Colny Urad -> [NEVZTAHUJE_SA_NA] -> Uplatnenie Oslobodenia Od Dane Podla Paragraf § 48 Odsek 3
  Uplatnenie Oslobodenia Od Dane Podla Paragraf § 48 Odsek 3 -> [VZTAHUJE_SA_NA] -> Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3
  Uplatnenie Oslobodenia Od Dane Podla Paragraf § 48 Odsek 3 -> [MA_PODMIENKU] -> Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti

  Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [VZTAHUJE_SA_NA] -> Osoba Podla Paragraf § 48b Odsek 1
  Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [VZTAHUJE_SA_NA] -> Zabezpecenie Dane
  Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [MA_LEHOTU] -> Lehota Na Zaplatenie Zabezpecenia Dane
  Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [MA_SUMU] -> Vyska Zabezpecenia Dane
  Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [VYPLYVA_Z] -> Rozhodnutie O Zabezpeceni Dane

nodes:
  Paragraf: Paragraf § 48b
  Odsek: Paragraf § 48b Odsek 2
  Odsek: Paragraf § 48b Odsek 1
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 3

  Organizacia: Colny Urad
  Osoba: Osoba Podla Paragraf § 48b Odsek 1

  Rozhodnutie: Rozhodnutie O Zabezpeceni Dane
  Povinnost: Zabezpecenie Dane
  Suma: Vyska Zabezpecenia Dane
  Lehota: Lehota Na Zaplatenie Zabezpecenia Dane
  Platba: Zaplatenie Zabezpecenia Dane

  Dokument: Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane

  Pravo: Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3
  Pravo: Uplatnenie Oslobodenia Od Dane Podla Paragraf § 48 Odsek 3
  Dan: Dan

  Podmienka: Nezaplatenie Zabezpecenia Dane V Lehote A Vo Vyske Urcenej V Rozhodnuti


---

chunk: 566
path: ['§ 48b', '3', 'c)']
path_as_text: Paragraf § 48b Odsek 3 Pismeno c)
text: (3) Colný úrad uvoľní zabezpečenie dane do desiatich dní od predloženia dôkazu o tom, že odoslanie alebo preprava tovaru sa skončila v inom členskom štáte okrem odseku 4. Dôkazom, že odoslanie alebo preprava tovaru sa skončila v inom členskom štáte, je doklad o prevzatí tovaru príjemcom v inom členskom štáte. Doklad o prevzatí tovaru musí obsahovať c) adresu miesta a dátum prevzatia tovaru v inom členskom štáte, ak odoslanie alebo prepravu tovaru vykoná dodávateľ, alebo adresu miesta a dátum skončenia prepravy, ak odoslanie alebo prepravu tovaru vykoná odberateľ,

relations:
  Paragraf § 48b -> [OBSAHUJE] -> Paragraf § 48b Odsek 3
  Paragraf § 48b Odsek 3 -> [OBSAHUJE] -> Paragraf § 48b Odsek 3 Pismeno c)
  Paragraf § 48b -> [OBSAHUJE] -> Paragraf § 48b Odsek 4
  Paragraf § 48b Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 48b Odsek 4

  Colny Urad -> [MA_POVINNOST] -> Povinnost Uvolnit Zabezpecenie Dane
  Povinnost Uvolnit Zabezpecenie Dane -> [VZTAHUJE_SA_NA] -> Zabezpecenie Dane
  Povinnost Uvolnit Zabezpecenie Dane -> [MA_LEHOTU] -> Lehota Do Desiatich Dni Od Predlozenia Dokazu
  Povinnost Uvolnit Zabezpecenie Dane -> [MA_PODMIENKU] -> Predlozenie Dokazu O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Povinnost Uvolnit Zabezpecenie Dane -> [NEVZTAHUJE_SA_NA] -> Vynimka Podla Paragraf § 48b Odsek 4

  Predlozenie Dokazu O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Skoncenie Odoslania Alebo Prepravy Tovaru V Inom Clenskom State

  Skoncenie Odoslania Alebo Prepravy Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Odoslanie Alebo Preprava Tovaru
  Skoncenie Odoslania Alebo Prepravy Tovaru V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

  Doklad O Prevzati Tovaru -> [JE_TYPOM] -> Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Doklad O Prevzati Tovaru -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru Prijemcom V Inom Clenskom State

  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Tovar
  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Prijemca
  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat

  Paragraf § 48b Odsek 3 Pismeno c) -> [URCUJE] -> Obsah Dokladu O Prevzati Tovaru

  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Adresa Miesta Prevzatia Tovaru V Inom Clenskom State
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Datum Prevzatia Tovaru V Inom Clenskom State
  Adresa Miesta Prevzatia Tovaru V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Datum Prevzatia Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru Prijemcom V Inom Clenskom State

  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Adresa Miesta Skoncenia Prepravy
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Datum Skoncenia Prepravy

  Adresa Miesta Prevzatia Tovaru V Inom Clenskom State -> [MA_PODMIENKU] -> Odoslanie Alebo Prepravu Tovaru Vykona Dodavatel
  Datum Prevzatia Tovaru V Inom Clenskom State -> [MA_PODMIENKU] -> Odoslanie Alebo Prepravu Tovaru Vykona Dodavatel

  Adresa Miesta Skoncenia Prepravy -> [MA_PODMIENKU] -> Odoslanie Alebo Prepravu Tovaru Vykona Odberatel
  Datum Skoncenia Prepravy -> [MA_PODMIENKU] -> Odoslanie Alebo Prepravu Tovaru Vykona Odberatel

  Odoslanie Alebo Prepravu Tovaru Vykona Dodavatel -> [VZTAHUJE_SA_NA] -> Dodavatel
  Odoslanie Alebo Prepravu Tovaru Vykona Odberatel -> [VZTAHUJE_SA_NA] -> Odberatel

nodes:
  Paragraf: Paragraf § 48b
  Odsek: Paragraf § 48b Odsek 3
  Pismeno: Paragraf § 48b Odsek 3 Pismeno c)
  Odsek: Paragraf § 48b Odsek 4

  Organizacia: Colny Urad
  Povinnost: Povinnost Uvolnit Zabezpecenie Dane
  Platba: Zabezpecenie Dane
  Lehota: Lehota Do Desiatich Dni Od Predlozenia Dokazu

  Dokument: Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Dokument: Doklad O Prevzati Tovaru
  Zaznam: Obsah Dokladu O Prevzati Tovaru

  Konanie: Predlozenie Dokazu O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Konanie: Odoslanie Alebo Preprava Tovaru
  Konanie: Skoncenie Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Konanie: Prevzatie Tovaru Prijemcom V Inom Clenskom State

  Tovar: Tovar
  Stat: Iny Clensky Stat
  Subjekt: Prijemca
  Subjekt: Dodavatel
  Subjekt: Odberatel

  Adresa: Adresa Miesta Prevzatia Tovaru V Inom Clenskom State
  Datum: Datum Prevzatia Tovaru V Inom Clenskom State
  Adresa: Adresa Miesta Skoncenia Prepravy
  Datum: Datum Skoncenia Prepravy

  Podmienka: Vynimka Podla Paragraf § 48b Odsek 4
  Podmienka: Odoslanie Alebo Prepravu Tovaru Vykona Dodavatel
  Podmienka: Odoslanie Alebo Prepravu Tovaru Vykona Odberatel


---

chunk: 586
path: ['§ 48ca', '4', 'a)']
path_as_text: Paragraf § 48ca Odsek 4 Pismeno a)
text: (4) Prevádzkovateľ colného skladu je povinný viesť záznamy v členení podľa kalendárnych mesiacov o a) množstve tovaru v metrických tonách umiestneného do colného skladu, dátume umiestnenia tovaru a osobe, pre ktorú bol tento tovar umiestnený,

relations:
  Paragraf § 48ca -> [OBSAHUJE] -> Paragraf § 48ca Odsek 4
  Paragraf § 48ca Odsek 4 -> [OBSAHUJE] -> Paragraf § 48ca Odsek 4 Pismeno a)

  Paragraf § 48ca Odsek 4 Pismeno a) -> [UPRAVUJE] -> Vedenie Zaznamov Prevadzkovatelom Colneho Skladu

  Prevadzkovatel Colneho Skladu -> [MA_POVINNOST] -> Vedenie Zaznamov Prevadzkovatelom Colneho Skladu
  Vedenie Zaznamov Prevadzkovatelom Colneho Skladu -> [VZTAHUJE_SA_NA] -> Zaznamy O Tovare Umiestnenom Do Colneho Skladu
  Zaznamy O Tovare Umiestnenom Do Colneho Skladu -> [MA_OBDOBIE] -> Kalendarny Mesiac

  Zaznamy O Tovare Umiestnenom Do Colneho Skladu -> [OBSAHUJE] -> Mnozstvo Tovaru V Metrickych Tonach
  Zaznamy O Tovare Umiestnenom Do Colneho Skladu -> [OBSAHUJE] -> Datum Umiestnenia Tovaru
  Zaznamy O Tovare Umiestnenom Do Colneho Skladu -> [OBSAHUJE] -> Osoba Pre Ktoru Bol Tovar Umiestneny

  Mnozstvo Tovaru V Metrickych Tonach -> [VZTAHUJE_SA_NA] -> Tovar Umiestneny Do Colneho Skladu

  Umiestnenie Tovaru Do Colneho Skladu -> [VZTAHUJE_SA_NA] -> Tovar Umiestneny Do Colneho Skladu
  Umiestnenie Tovaru Do Colneho Skladu -> [NACHADZA_SA_V] -> Colny Sklad
  Umiestnenie Tovaru Do Colneho Skladu -> [MA_DATUM] -> Datum Umiestnenia Tovaru
  Umiestnenie Tovaru Do Colneho Skladu -> [VZTAHUJE_SA_NA] -> Osoba Pre Ktoru Bol Tovar Umiestneny

  Tovar Umiestneny Do Colneho Skladu -> [NACHADZA_SA_V] -> Colny Sklad

nodes:
  Paragraf: Paragraf § 48ca
  Odsek: Paragraf § 48ca Odsek 4
  Pismeno: Paragraf § 48ca Odsek 4 Pismeno a)

  Subjekt: Prevadzkovatel Colneho Skladu
  Povinnost: Vedenie Zaznamov Prevadzkovatelom Colneho Skladu
  Zaznam: Zaznamy O Tovare Umiestnenom Do Colneho Skladu

  Obdobie: Kalendarny Mesiac
  Tovar: Tovar Umiestneny Do Colneho Skladu
  Lokacia: Colny Sklad
  Mnozstvo: Mnozstvo Tovaru V Metrickych Tonach
  Datum: Datum Umiestnenia Tovaru
  Osoba: Osoba Pre Ktoru Bol Tovar Umiestneny
  Konanie: Umiestnenie Tovaru Do Colneho Skladu


---

chunk: 607
path: ['§ 48d', '11', 'b)']
path_as_text: Paragraf § 48d Odsek 11 Pismeno b)
text: (11) Povolenie na prevádzkovanie osobitného skladu zaniká dňom b) vyhlásenia konkurzu alebo dňom vstupu do likvidácie,

relations:
  Paragraf § 48d -> [OBSAHUJE] -> Paragraf § 48d Odsek 11
  Paragraf § 48d Odsek 11 -> [OBSAHUJE] -> Paragraf § 48d Odsek 11 Pismeno b)

  Paragraf § 48d Odsek 11 Pismeno b) -> [UPRAVUJE] -> Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu

  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Den Vyhlasenia Konkurzu
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Den Vstupu Do Likvidacie

  Den Vyhlasenia Konkurzu -> [VZTAHUJE_SA_NA] -> Vyhlasenie Konkurzu
  Den Vstupu Do Likvidacie -> [VZTAHUJE_SA_NA] -> Vstup Do Likvidacie

nodes:
  Paragraf: Paragraf § 48d
  Odsek: Paragraf § 48d Odsek 11
  Pismeno: Paragraf § 48d Odsek 11 Pismeno b)

  Konanie: Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu
  Dokument: Povolenie Na Prevadzkovanie Osobitneho Skladu

  Datum: Den Vyhlasenia Konkurzu
  Datum: Den Vstupu Do Likvidacie
  Dovod: Vyhlasenie Konkurzu
  Dovod: Vstup Do Likvidacie

---

chunk: 609
path: ['§ 48d', '11', 'd)']
path_as_text: Paragraf § 48d Odsek 11 Pismeno d)
text: (11) Povolenie na prevádzkovanie osobitného skladu zaniká dňom d) keď prevádzkovateľ osobitného skladu prestal byť platiteľom.

relations:
  Paragraf § 48d -> [OBSAHUJE] -> Paragraf § 48d Odsek 11
  Paragraf § 48d Odsek 11 -> [OBSAHUJE] -> Paragraf § 48d Odsek 11 Pismeno d)

  Paragraf § 48d Odsek 11 Pismeno d) -> [UPRAVUJE] -> Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu

  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Osobitneho Skladu
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Den Ked Prevadzkovatel Osobitneho Skladu Prestal Byt Platitelom

  Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu -> [VYPLYVA_Z] -> Prestanie Prevadzkovatela Osobitneho Skladu Byt Platitelom
  Prestanie Prevadzkovatela Osobitneho Skladu Byt Platitelom -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Osobitneho Skladu
  Prestanie Prevadzkovatela Osobitneho Skladu Byt Platitelom -> [VZTAHUJE_SA_NA] -> Platitel
  Prestanie Prevadzkovatela Osobitneho Skladu Byt Platitelom -> [MA_DATUM] -> Den Ked Prevadzkovatel Osobitneho Skladu Prestal Byt Platitelom

nodes:
  Paragraf: Paragraf § 48d
  Odsek: Paragraf § 48d Odsek 11
  Pismeno: Paragraf § 48d Odsek 11 Pismeno d)

  Rozhodnutie: Povolenie Na Prevadzkovanie Osobitneho Skladu
  Konanie: Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu

  Subjekt: Prevadzkovatel Osobitneho Skladu
  Status: Platitel

  Dovod: Prestanie Prevadzkovatela Osobitneho Skladu Byt Platitelom
  Datum: Den Ked Prevadzkovatel Osobitneho Skladu Prestal Byt Platitelom


---

chunk: 632
path: ['§ 48e', '9', 'b)']
path_as_text: Paragraf § 48e Odsek 9 Pismeno b)
text: (9) Osoba, ktorá spôsobí, že sa tovar vyjme z daňového skladu, je povinná predtým, ako nastane táto skutočnosť, oznámiť prevádzkovateľovi daňového skladu identifikačné číslo pre daň pridelené v tuzemsku a doručiť mu b) faktúru, ktorú vyhotovila o dodaní tovaru, ak v súvislosti s dodaním tovaru dochádza k vyňatiu tovaru z daňového skladu, alebo iný doklad, ktorý preukazuje dodanie tovaru, ak faktúra nie je vyhotovená pred vyňatím tovaru z daňového skladu.

relations:
  Paragraf § 48e -> [OBSAHUJE] -> Paragraf § 48e Odsek 9
  Paragraf § 48e Odsek 9 -> [OBSAHUJE] -> Paragraf § 48e Odsek 9 Pismeno b)

  Paragraf § 48e Odsek 9 -> [UPRAVUJE] -> Povinnosti Osoby Sposobujucej Vynatie Tovaru Z Danoveho Skladu
  Paragraf § 48e Odsek 9 Pismeno b) -> [UPRAVUJE] -> Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu

  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [ZODPOVEDA_ZA] -> Vynatie Tovaru Z Danoveho Skladu
  Vynatie Tovaru Z Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Tovar
  Vynatie Tovaru Z Danoveho Skladu -> [NACHADZA_SA_V] -> Danovy Sklad

  Prevadzkovatel Danoveho Skladu -> [MA] -> Danovy Sklad

  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [MA_POVINNOST] -> Povinnost Oznamit Identifikacne Cislo Pre Dan Prevadzkovatelovi Danoveho Skladu
  Povinnost Oznamit Identifikacne Cislo Pre Dan Prevadzkovatelovi Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Povinnost Oznamit Identifikacne Cislo Pre Dan Prevadzkovatelovi Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Danoveho Skladu
  Povinnost Oznamit Identifikacne Cislo Pre Dan Prevadzkovatelovi Danoveho Skladu -> [MA_LEHOTU] -> Pred Vynatim Tovaru Z Danoveho Skladu

  Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Tuzemsko
  Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Dan

  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [MA_POVINNOST] -> Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu
  Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Danoveho Skladu
  Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu -> [MA_LEHOTU] -> Pred Vynatim Tovaru Z Danoveho Skladu

  Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu -> [MA_PODMIENKU] -> Dorucenie Faktury O Dodani Tovaru Ak V Suvislosti S Dodanim Tovaru Dochadza K Vynatiu Tovaru Z Danoveho Skladu
  Dorucenie Faktury O Dodani Tovaru Ak V Suvislosti S Dodanim Tovaru Dochadza K Vynatiu Tovaru Z Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Faktura O Dodani Tovaru
  Faktura O Dodani Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [VYDAVA] -> Faktura O Dodani Tovaru

  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [SUVISI_S] -> Vynatie Tovaru Z Danoveho Skladu

  Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu -> [MA_PODMIENKU] -> Dorucenie Ineho Dokladu Preukazujuceho Dodanie Tovaru Ak Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu
  Dorucenie Ineho Dokladu Preukazujuceho Dodanie Tovaru Ak Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Iny Doklad Preukazujuci Dodanie Tovaru
  Dorucenie Ineho Dokladu Preukazujuceho Dodanie Tovaru Ak Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu -> [MA_PODMIENKU] -> Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu
  Iny Doklad Preukazujuci Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru

nodes:
  Paragraf: Paragraf § 48e
  Odsek: Paragraf § 48e Odsek 9
  Pismeno: Paragraf § 48e Odsek 9 Pismeno b)

  Osoba: Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu
  Organizacia: Prevadzkovatel Danoveho Skladu
  Lokacia: Danovy Sklad
  Tovar: Tovar

  Konanie: Vynatie Tovaru Z Danoveho Skladu
  Konanie: Dodanie Tovaru

  Zaznam: Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Dokument: Faktura O Dodani Tovaru
  Dokument: Iny Doklad Preukazujuci Dodanie Tovaru

  Povinnost: Povinnost Oznamit Identifikacne Cislo Pre Dan Prevadzkovatelovi Danoveho Skladu
  Povinnost: Povinnost Dorucit Fakturu Alebo Iny Doklad Prevadzkovatelovi Danoveho Skladu

  Podmienka: Dorucenie Faktury O Dodani Tovaru Ak V Suvislosti S Dodanim Tovaru Dochadza K Vynatiu Tovaru Z Danoveho Skladu
  Podmienka: Dorucenie Ineho Dokladu Preukazujuceho Dodanie Tovaru Ak Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu
  Podmienka: Faktura Nie Je Vyhotovena Pred Vynatim Tovaru Z Danoveho Skladu

  Lehota: Pred Vynatim Tovaru Z Danoveho Skladu
  Stat: Tuzemsko
  Dan: Dan


---

chunk: 648
path: ['§ 49', '7', 'b)']
path_as_text: Paragraf § 49 Odsek 7 Pismeno b)
text: (7) Platiteľ nemôže odpočítať daň pri b) prechodných položkách podľa § 22 ods. 3.

relations:
  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 7
  Paragraf § 49 Odsek 7 -> [OBSAHUJE] -> Paragraf § 49 Odsek 7 Pismeno b)
  Paragraf § 22 -> [OBSAHUJE] -> Paragraf § 22 Odsek 3

  Paragraf § 49 Odsek 7 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 22 Odsek 3
  Paragraf § 49 Odsek 7 Pismeno b) -> [UPRAVUJE] -> Zakaz Odpocitania Dane Pri Prechodnych Polozkach

  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane Pri Prechodnych Polozkach
  Odpocitanie Dane Pri Prechodnych Polozkach -> [VZTAHUJE_SA_NA] -> Dan
  Odpocitanie Dane Pri Prechodnych Polozkach -> [VZTAHUJE_SA_NA] -> Prechodne Polozky

  Prechodne Polozky -> [JE_PODLA] -> Paragraf § 22 Odsek 3

nodes:
  Paragraf: Paragraf § 49
  Odsek: Paragraf § 49 Odsek 7
  Pismeno: Paragraf § 49 Odsek 7 Pismeno b)
  Paragraf: Paragraf § 22
  Odsek: Paragraf § 22 Odsek 3

  Subjekt: Platitel
  Dan: Dan

  Pravo: Odpocitanie Dane Pri Prechodnych Polozkach
  Konanie: Zakaz Odpocitania Dane Pri Prechodnych Polozkach
  Suma: Prechodne Polozky


---

chunk: 655
path: ['§ 50', '2', 'a)']
path_as_text: Paragraf § 50 Odsek 2 Pismeno a)
text: (2) Koeficient sa vypočíta ako podiel, v ktorého čitateli je hodnota bez dane dodaných tovarov a služieb za kalendárny rok, pri ktorých je daň odpočítateľná, a v ktorého menovateli je hodnota bez dane zo všetkých dodaných tovarov a služieb za kalendárny rok. Pri výpočte koeficientu sa do čitateľa ani do menovateľa koeficientu neuvádza hodnota z a) predaja podniku alebo časti podniku tvoriacej samostatnú organizačnú zložku,

relations:
  Paragraf § 50 -> [OBSAHUJE] -> Paragraf § 50 Odsek 2
  Paragraf § 50 Odsek 2 -> [OBSAHUJE] -> Paragraf § 50 Odsek 2 Pismeno a)

  Paragraf § 50 Odsek 2 -> [UPRAVUJE] -> Vypocet Koeficientu

  Vypocet Koeficientu -> [VZTAHUJE_SA_NA] -> Koeficient
  Koeficient -> [VYPLYVA_Z] -> Podiel Citatela A Menovatela Koeficientu
  Podiel Citatela A Menovatela Koeficientu -> [MA] -> Citatel Koeficientu
  Podiel Citatela A Menovatela Koeficientu -> [MA] -> Menovatel Koeficientu

  Citatel Koeficientu -> [MA_HODNOTU] -> Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna
  Menovatel Koeficientu -> [MA_HODNOTU] -> Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok

  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [VZTAHUJE_SA_NA] -> Dodanie Tovarov A Sluzieb Pri Ktorych Je Dan Odpocitatelna
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [MA_OBDOBIE] -> Kalendarny Rok
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [VZTAHUJE_SA_NA] -> Dan

  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [VZTAHUJE_SA_NA] -> Vsetky Dodane Tovary A Sluzby
  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [MA_OBDOBIE] -> Kalendarny Rok
  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [VZTAHUJE_SA_NA] -> Dan

  Citatel Koeficientu -> [NEVZTAHUJE_SA_NA] -> Hodnota Z Predaja Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku
  Menovatel Koeficientu -> [NEVZTAHUJE_SA_NA] -> Hodnota Z Predaja Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku

  Hodnota Z Predaja Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku -> [VYPLYVA_Z] -> Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku
  Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku -> [VZTAHUJE_SA_NA] -> Podnik
  Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku -> [VZTAHUJE_SA_NA] -> Cast Podniku Tvoriaca Samostatnu Organizacnu Zlozku

nodes:
  Paragraf: Paragraf § 50
  Odsek: Paragraf § 50 Odsek 2
  Pismeno: Paragraf § 50 Odsek 2 Pismeno a)

  Konanie: Vypocet Koeficientu
  Zaznam: Koeficient
  Zaznam: Podiel Citatela A Menovatela Koeficientu
  Zaznam: Citatel Koeficientu
  Zaznam: Menovatel Koeficientu

  Suma: Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna
  Suma: Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok
  Suma: Hodnota Z Predaja Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku

  Konanie: Dodanie Tovarov A Sluzieb Pri Ktorych Je Dan Odpocitatelna
  Konanie: Vsetky Dodane Tovary A Sluzby
  Konanie: Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku

  Organizacia: Podnik
  Organizacia: Cast Podniku Tvoriaca Samostatnu Organizacnu Zlozku

  Obdobie: Kalendarny Rok
  Dan: Dan


---

chunk: 678
path: ['§ 53', '1', 'c)']
path_as_text: Paragraf § 53 Odsek 1 Pismeno c)
text: (1) Ak po skončení zdaňovacieho obdobia, v ktorom platiteľ c) uplatnil odpočítanie dane, uplatnil pomerné odpočítanie dane podľa § 49 ods. 4 alebo ak platiteľ nemohol uplatniť odpočítanie dane podľa § 49 ods. 2, 3 alebo ods. 7, dôjde k prvotnému použitiu,  1. je platiteľ povinný opraviť odpočítanú daň, ak vykonal odpočítanie dane vo vyššej výške, ako by mohol vykonať na základe prvotného použitia; to neplatí, ak ide o dodanie tovaru podľa § 8 ods. 3 alebo dodanie služby podľa § 9 ods. 2, 2. platiteľ môže opraviť odpočítanú daň, ak vykonal odpočítanie dane v nižšej výške, ako by mohol vykonať na základe prvotného použitia.

relations:
  Paragraf § 53 -> [OBSAHUJE] -> Paragraf § 53 Odsek 1
  Paragraf § 53 Odsek 1 -> [OBSAHUJE] -> Paragraf § 53 Odsek 1 Pismeno c)
  Paragraf § 53 Odsek 1 Pismeno c) -> [OBSAHUJE] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 1
  Paragraf § 53 Odsek 1 Pismeno c) -> [OBSAHUJE] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 2

  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 2
  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 3
  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 4
  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 7
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 3
  Paragraf § 9 -> [OBSAHUJE] -> Paragraf § 9 Odsek 2

  Paragraf § 53 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 49 Odsek 2
  Paragraf § 53 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 49 Odsek 3
  Paragraf § 53 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 49 Odsek 4
  Paragraf § 53 Odsek 1 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 49 Odsek 7
  Paragraf § 53 Odsek 1 Pismeno c) Bod 1 -> [ODKAZUJE_NA] -> Paragraf § 8 Odsek 3
  Paragraf § 53 Odsek 1 Pismeno c) Bod 1 -> [ODKAZUJE_NA] -> Paragraf § 9 Odsek 2

  Paragraf § 53 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Oprava Odpocitanej Dane Po Prvotnom Pouziti

  Oprava Odpocitanej Dane Po Prvotnom Pouziti -> [MA_PODMIENKU] -> Prvotne Pouzitie Po Skonceni Zdanovacieho Obdobia
  Oprava Odpocitanej Dane Po Prvotnom Pouziti -> [MA_PODMIENKU] -> Uplatnenie Odpocitania Dane Alebo Pomerneho Odpocitania Dane Alebo Nemoznost Uplatnit Odpocitanie Dane

  Prvotne Pouzitie Po Skonceni Zdanovacieho Obdobia -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Platitel -> [MA] -> Uplatnene Odpocitanie Dane
  Platitel -> [MA] -> Uplatnene Pomerne Odpocitanie Dane Podla Paragraf § 49 Odsek 4
  Uplatnene Pomerne Odpocitanie Dane Podla Paragraf § 49 Odsek 4 -> [JE_PODLA] -> Paragraf § 49 Odsek 4
  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane Podla Paragraf § 49 Odsek 2 3 Alebo 7

  Odpocitanie Dane Podla Paragraf § 49 Odsek 2 3 Alebo 7 -> [JE_PODLA] -> Paragraf § 49 Odsek 2
  Odpocitanie Dane Podla Paragraf § 49 Odsek 2 3 Alebo 7 -> [JE_PODLA] -> Paragraf § 49 Odsek 3
  Odpocitanie Dane Podla Paragraf § 49 Odsek 2 3 Alebo 7 -> [JE_PODLA] -> Paragraf § 49 Odsek 7

  Platitel -> [MA_POVINNOST] -> Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske
  Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske -> [VYPLYVA_Z] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 1
  Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske -> [MA_PODMIENKU] -> Odpocitanie Dane Vo Vyssej Vyske Ako Podla Prvotneho Pouzitia
  Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske -> [NEVZTAHUJE_SA_NA] -> Dodanie Tovaru Podla Paragraf § 8 Odsek 3
  Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske -> [NEVZTAHUJE_SA_NA] -> Dodanie Sluzby Podla Paragraf § 9 Odsek 2

  Dodanie Tovaru Podla Paragraf § 8 Odsek 3 -> [JE_PODLA] -> Paragraf § 8 Odsek 3
  Dodanie Sluzby Podla Paragraf § 9 Odsek 2 -> [JE_PODLA] -> Paragraf § 9 Odsek 2

  Platitel -> [MA_PRAVO] -> Pravo Opravit Odpocitanu Dan Pri Nizsej Vyske
  Pravo Opravit Odpocitanu Dan Pri Nizsej Vyske -> [VYPLYVA_Z] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 2
  Pravo Opravit Odpocitanu Dan Pri Nizsej Vyske -> [MA_PODMIENKU] -> Odpocitanie Dane V Nizsej Vyske Ako Podla Prvotneho Pouzitia

nodes:
  Paragraf: Paragraf § 53
  Odsek: Paragraf § 53 Odsek 1
  Pismeno: Paragraf § 53 Odsek 1 Pismeno c)
  Bod: Paragraf § 53 Odsek 1 Pismeno c) Bod 1
  Bod: Paragraf § 53 Odsek 1 Pismeno c) Bod 2

  Paragraf: Paragraf § 49
  Odsek: Paragraf § 49 Odsek 2
  Odsek: Paragraf § 49 Odsek 3
  Odsek: Paragraf § 49 Odsek 4
  Odsek: Paragraf § 49 Odsek 7
  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 3
  Paragraf: Paragraf § 9
  Odsek: Paragraf § 9 Odsek 2

  Subjekt: Platitel
  ZdanovacieObdobie: Zdanovacie Obdobie

  Konanie: Oprava Odpocitanej Dane Po Prvotnom Pouziti
  Pravo: Uplatnene Odpocitanie Dane
  Pravo: Uplatnene Pomerne Odpocitanie Dane Podla Paragraf § 49 Odsek 4
  Pravo: Odpocitanie Dane Podla Paragraf § 49 Odsek 2 3 Alebo 7

  Povinnost: Povinnost Opravit Odpocitanu Dan Pri Vyssej Vyske
  Pravo: Pravo Opravit Odpocitanu Dan Pri Nizsej Vyske

  Podmienka: Prvotne Pouzitie Po Skonceni Zdanovacieho Obdobia
  Podmienka: Uplatnenie Odpocitania Dane Alebo Pomerneho Odpocitania Dane Alebo Nemoznost Uplatnit Odpocitanie Dane
  Podmienka: Odpocitanie Dane Vo Vyssej Vyske Ako Podla Prvotneho Pouzitia
  Podmienka: Odpocitanie Dane V Nizsej Vyske Ako Podla Prvotneho Pouzitia

  Konanie: Dodanie Tovaru Podla Paragraf § 8 Odsek 3
  Konanie: Dodanie Sluzby Podla Paragraf § 9 Odsek 2


---

chunk: 689
path: ['§ 53a', '2', 'a)']
path_as_text: Paragraf § 53a Odsek 2 Pismeno a)
text: (2) Odpočítanú daň zo služby vykonanej na investičnom majetku uvedenom v § 54 ods. 2 písm. a) platiteľ opraví vo výške, ktorá sa vzťahuje na obdobie, ktoré sa začína kalendárnym mesiacom, v ktorom platiteľ tento investičný majetok dodal podľa odseku 1, a končí sa uplynutím 60. kalendárneho mesiaca od uplatnenia odpočítania dane. Odpočítanú daň zo služby vykonanej na investičnom majetku uvedenom v § 54 ods. 2 písm. b) platiteľ opraví vo výške, ktorá sa vzťahuje na obdobie, ktoré sa začína kalendárnym mesiacom, v ktorom platiteľ tento investičný majetok dodal podľa odseku 1, a končí sa uplynutím 240. kalendárneho mesiaca od uplatnenia odpočítania dane. Pri oprave odpočítanej dane platiteľ zohľadní pomerné odpočítanie dane zo služby vykonanej na investičnom majetku.

relations:
  Paragraf § 53a -> [OBSAHUJE] -> Paragraf § 53a Odsek 1
  Paragraf § 53a -> [OBSAHUJE] -> Paragraf § 53a Odsek 2
  Paragraf § 54 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2
  Paragraf § 54 Odsek 2 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2 Pismeno a)
  Paragraf § 54 Odsek 2 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2 Pismeno b)

  Paragraf § 53a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 53a Odsek 1
  Paragraf § 53a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno a)
  Paragraf § 53a Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno b)

  Platitel -> [MA_POVINNOST] -> Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku

  Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Sluzba Vykonana Na Investicnom Majetku
  Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku -> [MA_PODMIENKU] -> Zohladnenie Pomerneho Odpocitania Dane Zo Sluzby Vykonanej Na Investicnom Majetku

  Sluzba Vykonana Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno a) -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a)
  Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno a)

  Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno a) -> [MA_OBDOBIE] -> Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane -> [MA_DATUM] -> Kalendarny Mesiac Dodania Investicneho Majetku Podla Paragraf § 53a Odsek 1
  Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane -> [MA_LEHOTU] -> Uplynutie 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane

  Sluzba Vykonana Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno b) -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno b)
  Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno b) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno b)

  Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno b) -> [MA_OBDOBIE] -> Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane -> [MA_DATUM] -> Kalendarny Mesiac Dodania Investicneho Majetku Podla Paragraf § 53a Odsek 1
  Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane -> [MA_LEHOTU] -> Uplynutie 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane

  Platitel -> [DODAVA] -> Investicny Majetok Podla Paragraf § 53a Odsek 1
  Investicny Majetok Podla Paragraf § 53a Odsek 1 -> [JE_PODLA] -> Paragraf § 53a Odsek 1

  Zohladnenie Pomerneho Odpocitania Dane Zo Sluzby Vykonanej Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Pomerne Odpocitanie Dane Zo Sluzby Vykonanej Na Investicnom Majetku
  Pomerne Odpocitanie Dane Zo Sluzby Vykonanej Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Sluzba Vykonana Na Investicnom Majetku

nodes:
  Paragraf: Paragraf § 53a
  Odsek: Paragraf § 53a Odsek 1
  Odsek: Paragraf § 53a Odsek 2
  Paragraf: Paragraf § 54
  Odsek: Paragraf § 54 Odsek 2
  Pismeno: Paragraf § 54 Odsek 2 Pismeno a)
  Pismeno: Paragraf § 54 Odsek 2 Pismeno b)

  Subjekt: Platitel

  Povinnost: Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku
  Povinnost: Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno a)
  Povinnost: Oprava Odpocitanej Dane Zo Sluzby Vykonanej Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno b)

  Dan: Odpocitana Dan
  Dan: Pomerne Odpocitanie Dane Zo Sluzby Vykonanej Na Investicnom Majetku

  Sluzba: Sluzba Vykonana Na Investicnom Majetku
  Sluzba: Sluzba Vykonana Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno a)
  Sluzba: Sluzba Vykonana Na Investicnom Majetku Podla Paragraf § 54 Odsek 2 Pismeno b)

  Majetok: Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a)
  Majetok: Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno b)
  Majetok: Investicny Majetok Podla Paragraf § 53a Odsek 1

  Obdobie: Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Obdobie: Obdobie Od Kalendarneho Mesiaca Dodania Investicneho Majetku Do Uplynutia 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane

  Datum: Kalendarny Mesiac Dodania Investicneho Majetku Podla Paragraf § 53a Odsek 1
  Lehota: Uplynutie 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Lehota: Uplynutie 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane

  Podmienka: Zohladnenie Pomerneho Odpocitania Dane Zo Sluzby Vykonanej Na Investicnom Majetku


---

chunk: 701
path: ['§ 54', '1', 'b)']
path_as_text: Paragraf § 54 Odsek 1 Pismeno b)
text: (1) Ak v období nasledujúcom po zdaňovacom období, v ktorom došlo k prvotnému použitiu investičného majetku, platiteľ zmení účel jeho použitia, b) má právo upraviť odpočítanú daň, ak v dôsledku tejto zmeny bola daň pri prvotnom použití tohto investičného majetku odpočítaná v nižšej výške, v akej mohla byť odpočítaná v kalendárnom roku, v ktorom došlo k zmene účelu použitia tohto investičného majetku.

relations:
  Paragraf § 54 -> [OBSAHUJE] -> Paragraf § 54 Odsek 1
  Paragraf § 54 Odsek 1 -> [OBSAHUJE] -> Paragraf § 54 Odsek 1 Pismeno b)

  Paragraf § 54 Odsek 1 Pismeno b) -> [UPRAVUJE] -> Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku

  Platitel -> [MA_PRAVO] -> Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku
  Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku -> [MA_PODMIENKU] -> Zmena Ucelu Pouzitia Investicneho Majetku V Obdobi Nasledujucom Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku
  Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku -> [MA_PODMIENKU] -> Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia

  Zmena Ucelu Pouzitia Investicneho Majetku V Obdobi Nasledujucom Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Zmena Ucelu Pouzitia Investicneho Majetku V Obdobi Nasledujucom Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku -> [MA_OBDOBIE] -> Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku

  Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Prvotne Pouzitie Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok

  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia -> [VYPLYVA_Z] -> Zmena Ucelu Pouzitia Investicneho Majetku V Obdobi Nasledujucom Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia -> [MA_SUMU] -> Nizsia Vyska Odpocitanej Dane
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia -> [MA_HODNOTU] -> Vyska Dane Ktora Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia Investicneho Majetku

  Vyska Dane Ktora Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia Investicneho Majetku -> [MA_OBDOBIE] -> Kalendarny Rok Zmeny Ucelu Pouzitia Investicneho Majetku
  Kalendarny Rok Zmeny Ucelu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Zmena Ucelu Pouzitia Investicneho Majetku

nodes:
  Paragraf: Paragraf § 54
  Odsek: Paragraf § 54 Odsek 1
  Pismeno: Paragraf § 54 Odsek 1 Pismeno b)

  Subjekt: Platitel
  Majetok: Investicny Majetok

  Pravo: Pravo Upravit Odpocitanu Dan Pri Zmene Ucelu Pouzitia Investicneho Majetku
  Dan: Odpocitana Dan

  Dovod: Zmena Ucelu Pouzitia Investicneho Majetku
  Konanie: Prvotne Pouzitie Investicneho Majetku

  ZdanovacieObdobie: Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Obdobie: Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku
  Obdobie: Kalendarny Rok Zmeny Ucelu Pouzitia Investicneho Majetku

  Podmienka: Zmena Ucelu Pouzitia Investicneho Majetku V Obdobi Nasledujucom Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku
  Podmienka: Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia

  Suma: Nizsia Vyska Odpocitanej Dane
  Suma: Vyska Dane Ktora Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Ucelu Pouzitia Investicneho Majetku


---

chunk: 724
path: ['§ 54d', '1', 'b)']
path_as_text: Paragraf § 54d Odsek 1 Pismeno b)
text: (1) Ak v období nasledujúcom po zdaňovacom období, v ktorom došlo k prvotnému použitiu investičného majetku uvedeného v § 54 ods. 2 písm. a) alebo písm. d), pri ktorom bola odpočítaná časť dane podľa § 49 ods. 5 prvej vety alebo pri ktorom nebola odpočítaná daň, platiteľ zmení rozsah použitia tohto investičného majetku na účely podnikania, ako aj na iný účel ako na podnikanie, b) má právo upraviť odpočítanú daň, ak v dôsledku tejto zmeny bola daň pri prvotnom použití tohto investičného majetku odpočítaná v nižšej výške, v akej mohla byť odpočítaná v kalendárnom roku, v ktorom došlo k zmene rozsahu použitia tohto investičného majetku na účely podnikania, ako aj na iný účel ako na podnikanie.

relations:
  Paragraf § 54d -> [OBSAHUJE] -> Paragraf § 54d Odsek 1
  Paragraf § 54d Odsek 1 -> [OBSAHUJE] -> Paragraf § 54d Odsek 1 Pismeno b)

  Paragraf § 54 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2
  Paragraf § 54 Odsek 2 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2 Pismeno a)
  Paragraf § 54 Odsek 2 -> [OBSAHUJE] -> Paragraf § 54 Odsek 2 Pismeno d)
  Paragraf § 49 -> [OBSAHUJE] -> Paragraf § 49 Odsek 5

  Paragraf § 54d Odsek 1 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno a)
  Paragraf § 54d Odsek 1 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno d)
  Paragraf § 54d Odsek 1 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 49 Odsek 5

  Paragraf § 54d Odsek 1 Pismeno b) -> [UPRAVUJE] -> Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku

  Platitel -> [MA_PRAVO] -> Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku

  Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku -> [MA_PODMIENKU] -> Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie
  Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku -> [MA_PODMIENKU] -> Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia

  Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d)
  Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie -> [VZTAHUJE_SA_NA] -> Pouzitie Investicneho Majetku Na Ucely Podnikania
  Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie -> [VZTAHUJE_SA_NA] -> Pouzitie Investicneho Majetku Na Iny Ucel Ako Na Podnikanie
  Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie -> [MA_OBDOBIE] -> Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku

  Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno a)
  Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno d)
  Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d) -> [MA_PODMIENKU] -> Odpocitana Cast Dane Podla Paragraf § 49 Odsek 5 Prvej Vety Alebo Neodpocitana Dan

  Odpocitana Cast Dane Podla Paragraf § 49 Odsek 5 Prvej Vety Alebo Neodpocitana Dan -> [VZTAHUJE_SA_NA] -> Cast Dane
  Odpocitana Cast Dane Podla Paragraf § 49 Odsek 5 Prvej Vety Alebo Neodpocitana Dan -> [VZTAHUJE_SA_NA] -> Neodpocitana Dan
  Odpocitana Cast Dane Podla Paragraf § 49 Odsek 5 Prvej Vety Alebo Neodpocitana Dan -> [JE_PODLA] -> Paragraf § 49 Odsek 5

  Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Prvotne Pouzitie Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d)

  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia -> [VYPLYVA_Z] -> Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia -> [MA_OBDOBIE] -> Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku

  Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie

nodes:
  Paragraf: Paragraf § 54d
  Odsek: Paragraf § 54d Odsek 1
  Pismeno: Paragraf § 54d Odsek 1 Pismeno b)

  Paragraf: Paragraf § 54
  Odsek: Paragraf § 54 Odsek 2
  Pismeno: Paragraf § 54 Odsek 2 Pismeno a)
  Pismeno: Paragraf § 54 Odsek 2 Pismeno d)

  Paragraf: Paragraf § 49
  Odsek: Paragraf § 49 Odsek 5

  Subjekt: Platitel
  Majetok: Investicny Majetok Podla Paragraf § 54 Odsek 2 Pismeno a) Alebo Pismeno d)

  Pravo: Pravo Upravit Odpocitanu Dan Pri Zmene Rozsahu Pouzitia Investicneho Majetku
  Dan: Odpocitana Dan
  Dan: Cast Dane
  Dan: Neodpocitana Dan

  Konanie: Prvotne Pouzitie Investicneho Majetku
  Konanie: Zmena Rozsahu Pouzitia Investicneho Majetku Na Ucely Podnikania A Na Iny Ucel Ako Na Podnikanie
  Konanie: Pouzitie Investicneho Majetku Na Ucely Podnikania
  Konanie: Pouzitie Investicneho Majetku Na Iny Ucel Ako Na Podnikanie

  ZdanovacieObdobie: Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Obdobie: Obdobie Nasledujuce Po Zdanovacom Obdobi Prvotneho Pouzitia Investicneho Majetku
  Obdobie: Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku

  Podmienka: Odpocitana Cast Dane Podla Paragraf § 49 Odsek 5 Prvej Vety Alebo Neodpocitana Dan
  Podmienka: Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske Ako Mohla Byt Odpocitana V Kalendarnom Roku Zmeny Rozsahu Pouzitia


---

chunk: 730
path: ['§ 55', '3', 'a)']
path_as_text: Paragraf § 55 Odsek 3 Pismeno a)
text: (3) Platiteľ, ktorý nesplnil registračnú povinnosť, môže v zdaňovacom období, za ktoré v dôsledku nesplnenia tejto povinnosti podáva daňové priznanie po uplynutí lehoty podľa § 78 ods. 2, uplatniť v rozsahu a za podmienok podľa § 49 až 50, § 51 ods. 1, 3 a 5 a odseku 4 právo na odpočítanie dane inej ako v odseku 1, ktorá bola a) voči nemu uplatnená iným platiteľom v tuzemsku z tovarov a služieb,

relations:
  Paragraf § 55 -> [OBSAHUJE] -> Paragraf § 55 Odsek 3
  Paragraf § 55 Odsek 3 -> [OBSAHUJE] -> Paragraf § 55 Odsek 3 Pismeno a)
  Paragraf § 78 -> [OBSAHUJE] -> Paragraf § 78 Odsek 2
  Paragraf § 51 -> [OBSAHUJE] -> Paragraf § 51 Odsek 1
  Paragraf § 51 -> [OBSAHUJE] -> Paragraf § 51 Odsek 3
  Paragraf § 51 -> [OBSAHUJE] -> Paragraf § 51 Odsek 5
  Paragraf § 55 -> [OBSAHUJE] -> Paragraf § 55 Odsek 4
  Paragraf § 55 -> [OBSAHUJE] -> Paragraf § 55 Odsek 1

  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 78 Odsek 2
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 49
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 50
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 51 Odsek 1
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 51 Odsek 3
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 51 Odsek 5
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 55 Odsek 4
  Paragraf § 55 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 55 Odsek 1

  Paragraf § 55 Odsek 3 Pismeno a) -> [UPRAVUJE] -> Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1

  Platitel -> [MA_POVINNOST] -> Registracna Povinnost
  Platitel -> [NESPLNA_PODMIENKY] -> Splnenie Registracnej Povinnosti
  Nesplnenie Registracnej Povinnosti -> [VZTAHUJE_SA_NA] -> Registracna Povinnost

  Platitel -> [PODAVA] -> Danove Priznanie
  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Danove Priznanie -> [MA_PODMIENKU] -> Podanie Danoveho Priznania Po Uplynuti Lehoty Podla Paragraf § 78 Odsek 2 V Dosledku Nesplnenia Registracnej Povinnosti
  Podanie Danoveho Priznania Po Uplynuti Lehoty Podla Paragraf § 78 Odsek 2 V Dosledku Nesplnenia Registracnej Povinnosti -> [MA_LEHOTU] -> Lehota Podla Paragraf § 78 Odsek 2
  Podanie Danoveho Priznania Po Uplynuti Lehoty Podla Paragraf § 78 Odsek 2 V Dosledku Nesplnenia Registracnej Povinnosti -> [VYPLYVA_Z] -> Nesplnenie Registracnej Povinnosti

  Platitel -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1
  Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1 -> [VZTAHUJE_SA_NA] -> Dan Ina Ako V Paragraf § 55 Odsek 1
  Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1 -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1 -> [MA_PODMIENKU] -> Rozsah A Podmienky Podla Paragraf § 49 Az § 50 Paragraf § 51 Odsek 1 3 A 5 A Paragraf § 55 Odsek 4
  Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1 -> [MA_PODMIENKU] -> Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb

  Dan Ina Ako V Paragraf § 55 Odsek 1 -> [NEVZTAHUJE_SA_NA] -> Dan Podla Paragraf § 55 Odsek 1
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Dan Ina Ako V Paragraf § 55 Odsek 1
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Platitel
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Iny Platitel
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [NACHADZA_SA_V] -> Tuzemsko
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Tovar
  Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzba

nodes:
  Paragraf: Paragraf § 55
  Odsek: Paragraf § 55 Odsek 3
  Pismeno: Paragraf § 55 Odsek 3 Pismeno a)
  Odsek: Paragraf § 55 Odsek 4
  Odsek: Paragraf § 55 Odsek 1

  Paragraf: Paragraf § 78
  Odsek: Paragraf § 78 Odsek 2
  Paragraf: Paragraf § 49
  Paragraf: Paragraf § 50
  Paragraf: Paragraf § 51
  Odsek: Paragraf § 51 Odsek 1
  Odsek: Paragraf § 51 Odsek 3
  Odsek: Paragraf § 51 Odsek 5

  Subjekt: Platitel
  Subjekt: Iny Platitel

  Povinnost: Registracna Povinnost
  Podmienka: Splnenie Registracnej Povinnosti
  Dovod: Nesplnenie Registracnej Povinnosti

  DanovePriznanie: Danove Priznanie
  ZdanovacieObdobie: Zdanovacie Obdobie
  Lehota: Lehota Podla Paragraf § 78 Odsek 2

  Pravo: Pravo Na Odpocitanie Dane Ine Ako V Paragraf § 55 Odsek 1
  Dan: Dan Ina Ako V Paragraf § 55 Odsek 1
  Dan: Dan Podla Paragraf § 55 Odsek 1

  Podmienka: Podanie Danoveho Priznania Po Uplynuti Lehoty Podla Paragraf § 78 Odsek 2 V Dosledku Nesplnenia Registracnej Povinnosti
  Podmienka: Rozsah A Podmienky Podla Paragraf § 49 Az § 50 Paragraf § 51 Odsek 1 3 A 5 A Paragraf § 55 Odsek 4
  Podmienka: Dan Uplatnena Voci Platitelovi Inym Platitelom V Tuzemsku Z Tovarov A Sluzieb

  Stat: Tuzemsko
  Tovar: Tovar
  Sluzba: Sluzba


---

chunk: 747
path: ['§ 55a', '3']
path_as_text: Paragraf § 55a Odsek 3
text: (3) Žiadateľ má nárok na vrátenie dane, ak uskutočňuje zdaniteľné obchody, pri ktorých vzniká  právo na odpočítanie dane v členskom štáte, v ktorom má sídlo, miesto podnikania, prevádzkareň, bydlisko alebo v ktorom sa obvykle zdržiava. Ak žiadateľ uskutočňuje v členskom štáte, v ktorom má sídlo, miesto podnikania, prevádzkareň, bydlisko alebo v ktorom sa obvykle zdržiava, zdaniteľné obchody, pri ktorých môže odpočítať daň, a súčasne zdaniteľné obchody, pri ktorých nemôže odpočítať daň, má nárok na vrátenie pomernej výšky dane, ktorú vypočíta podľa pravidiel platných v členskom štáte, v ktorom má sídlo, miesto podnikania, prevádzkareň, bydlisko alebo v ktorom sa obvykle zdržiava.

relations:
  Paragraf § 55a -> [OBSAHUJE] -> Paragraf § 55a Odsek 3
  Paragraf § 55a Odsek 3 -> [UPRAVUJE] -> Narok Ziadatela Na Vratenie Dane

  Ziadatel -> [MA_NAROK_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Dane -> [MA_PODMIENKU] -> Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane V Clenskom State Ziadatela

  Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane V Clenskom State Ziadatela -> [VZTAHUJE_SA_NA] -> Zdanitelne Obchody S Pravom Na Odpocitanie Dane
  Zdanitelne Obchody S Pravom Na Odpocitanie Dane -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane
  Pravo Na Odpocitanie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Zdanitelne Obchody S Pravom Na Odpocitanie Dane -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela

  Ziadatel -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela
  Ziadatel -> [MA_ADRESU] -> Sidlo
  Ziadatel -> [MA_ADRESU] -> Miesto Podnikania
  Ziadatel -> [MA] -> Prevadzkaren
  Ziadatel -> [MA_ADRESU] -> Bydlisko
  Ziadatel -> [NACHADZA_SA_V] -> Miesto Obvykleho Zdrziavania

  Sidlo -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela
  Miesto Podnikania -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela
  Prevadzkaren -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela
  Bydlisko -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela
  Miesto Obvykleho Zdrziavania -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela

  Ziadatel -> [MA_NAROK_NA] -> Vratenie Pomernej Vysky Dane
  Vratenie Pomernej Vysky Dane -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Pomernej Vysky Dane -> [MA_PODMIENKU] -> Sucasne Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane A Bez Prava Na Odpocitanie Dane V Clenskom State Ziadatela
  Vratenie Pomernej Vysky Dane -> [VYPLYVA_Z] -> Vypocet Pomernej Vysky Dane Podla Pravidiel Platnych V Clenskom State Ziadatela

  Sucasne Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane A Bez Prava Na Odpocitanie Dane V Clenskom State Ziadatela -> [VZTAHUJE_SA_NA] -> Zdanitelne Obchody S Pravom Na Odpocitanie Dane
  Sucasne Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane A Bez Prava Na Odpocitanie Dane V Clenskom State Ziadatela -> [VZTAHUJE_SA_NA] -> Zdanitelne Obchody Bez Prava Na Odpocitanie Dane
  Zdanitelne Obchody Bez Prava Na Odpocitanie Dane -> [NEMA_NAROK_NA] -> Pravo Na Odpocitanie Dane
  Zdanitelne Obchody Bez Prava Na Odpocitanie Dane -> [NACHADZA_SA_V] -> Clensky Stat Ziadatela

  Vypocet Pomernej Vysky Dane Podla Pravidiel Platnych V Clenskom State Ziadatela -> [VZTAHUJE_SA_NA] -> Pravidla Platne V Clenskom State Ziadatela
  Pravidla Platne V Clenskom State Ziadatela -> [VZTAHUJE_SA_NA] -> Clensky Stat Ziadatela

nodes:
  Paragraf: Paragraf § 55a
  Odsek: Paragraf § 55a Odsek 3

  Subjekt: Ziadatel
  Dan: Dan

  Pravo: Vratenie Dane
  Pravo: Vratenie Pomernej Vysky Dane
  Pravo: Pravo Na Odpocitanie Dane

  Konanie: Zdanitelne Obchody S Pravom Na Odpocitanie Dane
  Konanie: Zdanitelne Obchody Bez Prava Na Odpocitanie Dane
  Konanie: Vypocet Pomernej Vysky Dane Podla Pravidiel Platnych V Clenskom State Ziadatela

  Podmienka: Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane V Clenskom State Ziadatela
  Podmienka: Sucasne Uskutocnovanie Zdanitelnych Obchodov S Pravom Na Odpocitanie Dane A Bez Prava Na Odpocitanie Dane V Clenskom State Ziadatela
  Podmienka: Pravidla Platne V Clenskom State Ziadatela

  Stat: Clensky Stat Ziadatela
  Adresa: Sidlo
  Adresa: Miesto Podnikania
  Lokacia: Prevadzkaren
  Adresa: Bydlisko
  Lokacia: Miesto Obvykleho Zdrziavania


---

chunk: 770
path: ['§ 55b', '4', 'd)']
path_as_text: Paragraf § 55b Odsek 4 Pismeno d)
text: (4) Druh nadobudnutého tovaru a služieb sa vyjadruje týmito číselnými kódmi: d) poplatky za užívanie ciest a diaľnic číselným kódom 4,

relations:
  Paragraf § 55b -> [OBSAHUJE] -> Paragraf § 55b Odsek 4
  Paragraf § 55b Odsek 4 -> [OBSAHUJE] -> Paragraf § 55b Odsek 4 Pismeno d)

  Paragraf § 55b Odsek 4 -> [UPRAVUJE] -> Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb
  Paragraf § 55b Odsek 4 Pismeno d) -> [URCUJE] -> Ciselny Kod 4 Pre Poplatky Za Uzivanie Ciest A Dialnic

  Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb -> [OBSAHUJE] -> Ciselny Kod 4 Pre Poplatky Za Uzivanie Ciest A Dialnic
  Ciselny Kod 4 Pre Poplatky Za Uzivanie Ciest A Dialnic -> [VZTAHUJE_SA_NA] -> Poplatky Za Uzivanie Ciest A Dialnic
  Poplatky Za Uzivanie Ciest A Dialnic -> [MA_IDENTIFIKATOR] -> Ciselny Kod 4

nodes:
  Paragraf: Paragraf § 55b
  Odsek: Paragraf § 55b Odsek 4
  Pismeno: Paragraf § 55b Odsek 4 Pismeno d)

  Zaznam: Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb
  Zaznam: Ciselny Kod 4 Pre Poplatky Za Uzivanie Ciest A Dialnic
  Zaznam: Ciselny Kod 4

  Sluzba: Poplatky Za Uzivanie Ciest A Dialnic


---

chunk: 771
path: ['§ 55b', '4', 'e)']
path_as_text: Paragraf § 55b Odsek 4 Pismeno e)
text: (4) Druh nadobudnutého tovaru a služieb sa vyjadruje týmito číselnými kódmi: e) cestovné náklady týkajúce sa osobnej dopravy číselným kódom 5,

relations:
  Paragraf § 55b -> [OBSAHUJE] -> Paragraf § 55b Odsek 4
  Paragraf § 55b Odsek 4 -> [OBSAHUJE] -> Paragraf § 55b Odsek 4 Pismeno e)

  Paragraf § 55b Odsek 4 -> [UPRAVUJE] -> Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb
  Paragraf § 55b Odsek 4 Pismeno e) -> [URCUJE] -> Ciselny Kod 5 Pre Cestovne Naklady Tykajuce Sa Osobnej Dopravy

  Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb -> [OBSAHUJE] -> Ciselny Kod 5 Pre Cestovne Naklady Tykajuce Sa Osobnej Dopravy
  Ciselny Kod 5 Pre Cestovne Naklady Tykajuce Sa Osobnej Dopravy -> [VZTAHUJE_SA_NA] -> Cestovne Naklady Tykajuce Sa Osobnej Dopravy
  Cestovne Naklady Tykajuce Sa Osobnej Dopravy -> [MA_IDENTIFIKATOR] -> Ciselny Kod 5

nodes:
  Paragraf: Paragraf § 55b
  Odsek: Paragraf § 55b Odsek 4
  Pismeno: Paragraf § 55b Odsek 4 Pismeno e)

  Zaznam: Ciselne Kody Druhu Nadobudnuteho Tovaru A Sluzieb
  Zaznam: Ciselny Kod 5 Pre Cestovne Naklady Tykajuce Sa Osobnej Dopravy
  Zaznam: Ciselny Kod 5

  Sluzba: Cestovne Naklady Tykajuce Sa Osobnej Dopravy


---

chunk: 793
path: ['§ 55d', '8']
path_as_text: Paragraf § 55d Odsek 8
text: (8) Daňový úrad Bratislava vráti daň na účet vedený v banke v tuzemsku alebo na základe žiadosti žiadateľa na účet vedený v zahraničnej banke v inom členskom štáte, ak ju nemožno použiť podľa osobitného predpisu.27bd) Pri vrátení dane na účet vedený v zahraničnej banke v inom členskom štáte sa od sumy dane odpočítajú bankové poplatky za prevod peňažných prostriedkov.

relations:
  Paragraf § 55d -> [OBSAHUJE] -> Paragraf § 55d Odsek 8
  Paragraf § 55d Odsek 8 -> [ODKAZUJE_NA] -> Osobitny Predpis 27bd
  Paragraf § 55d Odsek 8 -> [UPRAVUJE] -> Vratenie Dane Danovym Uradom Bratislava

  Danovy Urad Bratislava -> [PLATI] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Dane -> [MA_PODMIENKU] -> Dan Nemozno Pouzit Podla Osobitneho Predpisu 27bd

  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Bankovy Ucet Vedeny V Banke V Tuzemsku
  Bankovy Ucet Vedeny V Banke V Tuzemsku -> [MA] -> Banka V Tuzemsku
  Banka V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko

  Ziadatel -> [PODAVA] -> Ziadost Ziadatela
  Vratenie Dane Na Ucet V Zahranicnej Banke V Inom Clenskom State -> [MA_PODMIENKU] -> Ziadost Ziadatela
  Vratenie Dane Na Ucet V Zahranicnej Banke V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State
  Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State -> [MA] -> Zahranicna Banka V Inom Clenskom State
  Zahranicna Banka V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat

  Vratenie Dane Na Ucet V Zahranicnej Banke V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Dane Na Ucet V Zahranicnej Banke V Inom Clenskom State -> [MA_PODMIENKU] -> Odpocitanie Bankovych Poplatkov Za Prevod Penaznych Prostriedkov Od Sumy Dane

  Odpocitanie Bankovych Poplatkov Za Prevod Penaznych Prostriedkov Od Sumy Dane -> [VZTAHUJE_SA_NA] -> Bankove Poplatky Za Prevod Penaznych Prostriedkov
  Odpocitanie Bankovych Poplatkov Za Prevod Penaznych Prostriedkov Od Sumy Dane -> [VZTAHUJE_SA_NA] -> Suma Dane
  Bankove Poplatky Za Prevod Penaznych Prostriedkov -> [VZTAHUJE_SA_NA] -> Prevod Penaznych Prostriedkov

nodes:
  Paragraf: Paragraf § 55d
  Odsek: Paragraf § 55d Odsek 8

  Organizacia: Danovy Urad Bratislava
  Dan: Dan

  Platba: Vratenie Dane
  Platba: Vratenie Dane Na Ucet V Zahranicnej Banke V Inom Clenskom State
  Platba: Prevod Penaznych Prostriedkov

  BankovyUcet: Bankovy Ucet Vedeny V Banke V Tuzemsku
  Banka: Banka V Tuzemsku
  Stat: Tuzemsko

  Osoba: Ziadatel
  Ziadost: Ziadost Ziadatela
  BankovyUcet: Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State
  Banka: Zahranicna Banka V Inom Clenskom State
  Stat: Iny Clensky Stat

  PravnyPredpis: Osobitny Predpis 27bd
  Podmienka: Dan Nemozno Pouzit Podla Osobitneho Predpisu 27bd
  Podmienka: Odpocitanie Bankovych Poplatkov Za Prevod Penaznych Prostriedkov Od Sumy Dane

  Suma: Suma Dane
  Suma: Bankove Poplatky Za Prevod Penaznych Prostriedkov


---

chunk: 812
path: ['§ 57', '2']
path_as_text: Paragraf § 57 Odsek 2
text: (2) Žiadosť o vrátenie dane môže podať zahraničná osoba z tretieho štátu aj za obdobie kalendárneho polroka, ak suma dane, ktorej vrátenie žiada, je najmenej 1 000 eur, a ak taká žiadosť bola podaná za prvý kalendárny polrok, suma dane, ktorej vrátenie žiada za druhý kalendárny polrok, je najmenej 50 eur. Žiadosť o vrátenie dane za kalendárny polrok sa podáva najneskôr v lehote podľa odseku 1.

relations:
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 1
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 2
  Paragraf § 57 Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 57 Odsek 1

  Paragraf § 57 Odsek 2 -> [UPRAVUJE] -> Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok

  Zahranicna Osoba Z Tretieho Statu -> [VZTAHUJE_SA_NA] -> Treti Stat
  Zahranicna Osoba Z Tretieho Statu -> [MA_PRAVO] -> Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok
  Zahranicna Osoba Z Tretieho Statu -> [PODAVA] -> Ziadost O Vratenie Dane Za Kalendarny Polrok

  Ziadost O Vratenie Dane Za Kalendarny Polrok -> [VZTAHUJE_SA_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Ziadost O Vratenie Dane Za Kalendarny Polrok -> [MA_OBDOBIE] -> Kalendarny Polrok
  Ziadost O Vratenie Dane Za Kalendarny Polrok -> [MA_SUMU] -> Suma Dane Najmenej 1000 Eur
  Suma Dane Najmenej 1000 Eur -> [VZTAHUJE_SA_NA] -> Dan

  Prvy Kalendarny Polrok -> [JE_TYPOM] -> Kalendarny Polrok
  Druhy Kalendarny Polrok -> [JE_TYPOM] -> Kalendarny Polrok

  Ziadost O Vratenie Dane Za Prvy Kalendarny Polrok -> [JE_TYPOM] -> Ziadost O Vratenie Dane Za Kalendarny Polrok
  Ziadost O Vratenie Dane Za Prvy Kalendarny Polrok -> [MA_OBDOBIE] -> Prvy Kalendarny Polrok
  Ziadost O Vratenie Dane Za Prvy Kalendarny Polrok -> [MA_SUMU] -> Suma Dane Najmenej 1000 Eur

  Ziadost O Vratenie Dane Za Druhy Kalendarny Polrok -> [JE_TYPOM] -> Ziadost O Vratenie Dane Za Kalendarny Polrok
  Ziadost O Vratenie Dane Za Druhy Kalendarny Polrok -> [MA_OBDOBIE] -> Druhy Kalendarny Polrok
  Ziadost O Vratenie Dane Za Druhy Kalendarny Polrok -> [MA_PODMIENKU] -> Podanie Ziadosti O Vratenie Dane Za Prvy Kalendarny Polrok
  Ziadost O Vratenie Dane Za Druhy Kalendarny Polrok -> [MA_SUMU] -> Suma Dane Najmenej 50 Eur
  Suma Dane Najmenej 50 Eur -> [VZTAHUJE_SA_NA] -> Dan

  Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok -> [VZTAHUJE_SA_NA] -> Ziadost O Vratenie Dane Za Kalendarny Polrok
  Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok -> [MA_LEHOTU] -> Lehota Podla Paragraf § 57 Odsek 1
  Lehota Podla Paragraf § 57 Odsek 1 -> [JE_PODLA] -> Paragraf § 57 Odsek 1

nodes:
  Paragraf: Paragraf § 57
  Odsek: Paragraf § 57 Odsek 1
  Odsek: Paragraf § 57 Odsek 2

  Osoba: Zahranicna Osoba Z Tretieho Statu
  Stat: Treti Stat

  Pravo: Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok
  Ziadost: Ziadost O Vratenie Dane Za Kalendarny Polrok
  Ziadost: Ziadost O Vratenie Dane Za Prvy Kalendarny Polrok
  Ziadost: Ziadost O Vratenie Dane Za Druhy Kalendarny Polrok

  Pravo: Vratenie Dane
  Dan: Dan

  Obdobie: Kalendarny Polrok
  Obdobie: Prvy Kalendarny Polrok
  Obdobie: Druhy Kalendarny Polrok

  Suma: Suma Dane Najmenej 1000 Eur
  Suma: Suma Dane Najmenej 50 Eur

  Konanie: Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok
  Konanie: Podanie Ziadosti O Vratenie Dane Za Prvy Kalendarny Polrok
  Lehota: Lehota Podla Paragraf § 57 Odsek 1


---

chunk: 816
path: ['§ 57', '5', 'a)']
path_as_text: Paragraf § 57 Odsek 5 Pismeno a)
text: (5) Zahraničná osoba z tretieho štátu musí v žiadosti o vrátenie dane vyhlásiť, že a) spĺňa podmienky podľa § 56 ods. 2,

relations:
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 5
  Paragraf § 57 Odsek 5 -> [OBSAHUJE] -> Paragraf § 57 Odsek 5 Pismeno a)
  Paragraf § 56 -> [OBSAHUJE] -> Paragraf § 56 Odsek 2

  Paragraf § 57 Odsek 5 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 56 Odsek 2
  Paragraf § 57 Odsek 5 Pismeno a) -> [UPRAVUJE] -> Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane

  Zahranicna Osoba Z Tretieho Statu -> [VZTAHUJE_SA_NA] -> Treti Stat
  Zahranicna Osoba Z Tretieho Statu -> [MA_POVINNOST] -> Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane

  Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Ziadost O Vratenie Dane
  Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Vyhlasenie O Splneni Podmienok Podla Paragraf § 56 Odsek 2

  Ziadost O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Ziadost O Vratenie Dane -> [OBSAHUJE] -> Vyhlasenie O Splneni Podmienok Podla Paragraf § 56 Odsek 2

  Vyhlasenie O Splneni Podmienok Podla Paragraf § 56 Odsek 2 -> [VZTAHUJE_SA_NA] -> Podmienky Podla Paragraf § 56 Odsek 2
  Podmienky Podla Paragraf § 56 Odsek 2 -> [JE_PODLA] -> Paragraf § 56 Odsek 2

nodes:
  Paragraf: Paragraf § 57
  Odsek: Paragraf § 57 Odsek 5
  Pismeno: Paragraf § 57 Odsek 5 Pismeno a)
  Paragraf: Paragraf § 56
  Odsek: Paragraf § 56 Odsek 2

  Osoba: Zahranicna Osoba Z Tretieho Statu
  Stat: Treti Stat

  Ziadost: Ziadost O Vratenie Dane
  Pravo: Vratenie Dane
  Dan: Dan

  Povinnost: Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane
  Zaznam: Vyhlasenie O Splneni Podmienok Podla Paragraf § 56 Odsek 2
  Podmienka: Podmienky Podla Paragraf § 56 Odsek 2


---

chunk: 839
path: ['§ 59', '6']
path_as_text: Paragraf § 59 Odsek 6
text: (6) Nárok na vrátenie dane zaniká, ak sa platiteľovi alebo poverenej osobe nepredložia doklady uvedené v odseku 3 do šiestich mesiacov od konca mesiaca, v ktorom bol tovar predaný.

relations:
  Paragraf § 59 -> [OBSAHUJE] -> Paragraf § 59 Odsek 6
  Paragraf § 59 -> [OBSAHUJE] -> Paragraf § 59 Odsek 3
  Paragraf § 59 Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 59 Odsek 3

  Paragraf § 59 Odsek 6 -> [UPRAVUJE] -> Zanik Naroku Na Vratenie Dane Pri Nepredlozeni Dokladov

  Narok Na Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Narok Na Vratenie Dane -> [ZANIKA] -> Zanik Naroku Na Vratenie Dane Pri Nepredlozeni Dokladov

  Zanik Naroku Na Vratenie Dane Pri Nepredlozeni Dokladov -> [MA_PODMIENKU] -> Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru

  Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru -> [VZTAHUJE_SA_NA] -> Doklady Uvedene V Paragraf § 59 Odsek 3
  Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru -> [VZTAHUJE_SA_NA] -> Platitel
  Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru -> [VZTAHUJE_SA_NA] -> Poverena Osoba
  Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru -> [MA_LEHOTU] -> Sest Mesiacov Od Konca Mesiaca Predaja Tovaru

  Doklady Uvedene V Paragraf § 59 Odsek 3 -> [JE_PODLA] -> Paragraf § 59 Odsek 3

  Sest Mesiacov Od Konca Mesiaca Predaja Tovaru -> [VYPLYVA_Z] -> Koniec Mesiaca V Ktorom Bol Tovar Predany
  Koniec Mesiaca V Ktorom Bol Tovar Predany -> [VZTAHUJE_SA_NA] -> Predaj Tovaru
  Predaj Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

nodes:
  Paragraf: Paragraf § 59
  Odsek: Paragraf § 59 Odsek 6
  Odsek: Paragraf § 59 Odsek 3

  Pravo: Narok Na Vratenie Dane
  Dan: Dan

  Konanie: Zanik Naroku Na Vratenie Dane Pri Nepredlozeni Dokladov
  Podmienka: Nepredlozenie Dokladov Uvedenych V Paragraf § 59 Odsek 3 Platitelovi Alebo Poverenej Osobe Do Siestich Mesiacov Od Konca Mesiaca Predaja Tovaru

  Dokument: Doklady Uvedene V Paragraf § 59 Odsek 3
  Subjekt: Platitel
  Osoba: Poverena Osoba

  Lehota: Sest Mesiacov Od Konca Mesiaca Predaja Tovaru
  Datum: Koniec Mesiaca V Ktorom Bol Tovar Predany

  Konanie: Predaj Tovaru
  Tovar: Tovar


---

chunk: 853
path: ['§ 61', '1']
path_as_text: Paragraf § 61 Odsek 1
text: (1) Osoby iných štátov, ktoré požívajú výsady a imunity podľa medzinárodného práva,23) a medzinárodné organizácie24) a ich pracovníci (ďalej len „zahraničný zástupca“) majú nárok na vrátenie dane zaplatenej v cenách tovarov a služieb určených na ich spotrebu.

relations:
  Paragraf § 61 -> [OBSAHUJE] -> Paragraf § 61 Odsek 1

  Paragraf § 61 Odsek 1 -> [DEFINUJE] -> Zahranicny Zastupca

  Osoby Inych Statov Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava -> [JE_TYPOM] -> Zahranicny Zastupca
  Medzinarodne Organizacie -> [JE_TYPOM] -> Zahranicny Zastupca
  Pracovnici Medzinarodnych Organizacii -> [JE_TYPOM] -> Zahranicny Zastupca
  Pracovnici Medzinarodnych Organizacii -> [PATRI_DO] -> Medzinarodne Organizacie

  Osoby Inych Statov Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava -> [MA_PRAVO] -> Vysady A Imunity Podla Medzinarodneho Prava
  Vysady A Imunity Podla Medzinarodneho Prava -> [JE_PODLA] -> Medzinarodne Pravo

  Zahranicny Zastupca -> [MA_NAROK_NA] -> Vratenie Dane Zaplatenej V Cenach Tovarov A Sluzieb Urcenych Na Spotrebu Zahranicneho Zastupcu
  Vratenie Dane Zaplatenej V Cenach Tovarov A Sluzieb Urcenych Na Spotrebu Zahranicneho Zastupcu -> [VZTAHUJE_SA_NA] -> Dan Zaplatena V Cenach Tovarov A Sluzieb
  Dan Zaplatena V Cenach Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Tovary Urcene Na Spotrebu Zahranicneho Zastupcu
  Dan Zaplatena V Cenach Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzby Urcene Na Spotrebu Zahranicneho Zastupcu

  Tovary Urcene Na Spotrebu Zahranicneho Zastupcu -> [VZTAHUJE_SA_NA] -> Zahranicny Zastupca
  Sluzby Urcene Na Spotrebu Zahranicneho Zastupcu -> [VZTAHUJE_SA_NA] -> Zahranicny Zastupca

nodes:
  Paragraf: Paragraf § 61
  Odsek: Paragraf § 61 Odsek 1

  Subjekt: Zahranicny Zastupca
  Osoba: Osoby Inych Statov Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava
  Organizacia: Medzinarodne Organizacie
  Osoba: Pracovnici Medzinarodnych Organizacii

  Pravo: Vysady A Imunity Podla Medzinarodneho Prava
  PravnyPredpis: Medzinarodne Pravo

  Pravo: Vratenie Dane Zaplatenej V Cenach Tovarov A Sluzieb Urcenych Na Spotrebu Zahranicneho Zastupcu
  Dan: Dan Zaplatena V Cenach Tovarov A Sluzieb
  Tovar: Tovary Urcene Na Spotrebu Zahranicneho Zastupcu
  Sluzba: Sluzby Urcene Na Spotrebu Zahranicneho Zastupcu


---

chunk: 862
path: ['§ 61', '3']
path_as_text: Paragraf § 61 Odsek 3
text: (3) Vrátenie dane sa poskytuje zahraničným zástupcom len tých štátov, ktoré také vrátenie dane alebo obdobné zvýhodnenie poskytujú osobám Slovenskej republiky. Ak iný štát také vrátenie dane alebo obdobné zvýhodnenie neposkytuje osobám Slovenskej republiky v rozsahu vrátenia dane poskytovaného Slovenskou republikou, prizná sa zahraničným zástupcom týchto štátov vrátenie dane len v takom rozsahu, ako poskytuje tento štát osobám Slovenskej republiky. Ak iný štát také vrátenie dane alebo obdobné zvýhodnenie poskytuje osobám Slovenskej republiky vo väčšom rozsahu, ako poskytuje Slovenská republika, prizná sa zahraničným zástupcom týchto štátov vrátenie dane v takom rozsahu, ako poskytuje tento štát osobám Slovenskej republiky.  Vzájomnosť podľa tohto odseku sa nevzťahuje na medzinárodné organizácie a ich pracovníkov.

relations:
  Paragraf § 61 -> [OBSAHUJE] -> Paragraf § 61 Odsek 3
  Paragraf § 61 Odsek 3 -> [UPRAVUJE] -> Vzajomnost Pri Vrateni Dane Zahranicnym Zastupcom

  Zahranicni Zastupcovia -> [MA_NAROK_NA] -> Vratenie Dane
  Vratenie Dane -> [MA_PODMIENKU] -> Vzajomnost Podla Paragraf § 61 Odsek 3
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Zahranicni Zastupcovia
  Zahranicni Zastupcovia -> [VZTAHUJE_SA_NA] -> Staty Zahranicnych Zastupcov

  Vzajomnost Podla Paragraf § 61 Odsek 3 -> [VZTAHUJE_SA_NA] -> Staty Poskytujuce Vratenie Dane Alebo Obdobne Zvyhodnenie Osobam Slovenskej Republiky
  Staty Poskytujuce Vratenie Dane Alebo Obdobne Zvyhodnenie Osobam Slovenskej Republiky -> [POSKYTUJE] -> Vratenie Dane
  Staty Poskytujuce Vratenie Dane Alebo Obdobne Zvyhodnenie Osobam Slovenskej Republiky -> [POSKYTUJE] -> Obdobne Zvyhodnenie
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Osoby Slovenskej Republiky
  Obdobne Zvyhodnenie -> [VZTAHUJE_SA_NA] -> Osoby Slovenskej Republiky

  Vratenie Dane Zahranicnym Zastupcom V Obmedzenom Rozsahu -> [MA_PODMIENKU] -> Iny Stat Neposkytuje Vratenie Dane Alebo Obdobne Zvyhodnenie V Rozsahu Poskytovanom Slovenskou Republikou
  Vratenie Dane Zahranicnym Zastupcom V Obmedzenom Rozsahu -> [MA_HODNOTU] -> Rozsah Poskytovany Inym Statom Osobam Slovenskej Republiky

  Vratenie Dane Zahranicnym Zastupcom Vo Vacsom Rozsahu -> [MA_PODMIENKU] -> Iny Stat Poskytuje Vratenie Dane Alebo Obdobne Zvyhodnenie Vo Vacsom Rozsahu Ako Slovenska Republika
  Vratenie Dane Zahranicnym Zastupcom Vo Vacsom Rozsahu -> [MA_HODNOTU] -> Rozsah Poskytovany Inym Statom Osobam Slovenskej Republiky

  Slovenska Republika -> [POSKYTUJE] -> Vratenie Dane
  Vratenie Dane Poskytovane Slovenskou Republikou -> [MA_HODNOTU] -> Rozsah Vratenia Dane Poskytovany Slovenskou Republikou

  Vzajomnost Podla Paragraf § 61 Odsek 3 -> [NEVZTAHUJE_SA_NA] -> Medzinarodne Organizacie
  Vzajomnost Podla Paragraf § 61 Odsek 3 -> [NEVZTAHUJE_SA_NA] -> Pracovnici Medzinarodnych Organizacii

nodes:
  Paragraf: Paragraf § 61
  Odsek: Paragraf § 61 Odsek 3

  Pravo: Vratenie Dane
  Pravo: Obdobne Zvyhodnenie
  Pravo: Vratenie Dane Zahranicnym Zastupcom V Obmedzenom Rozsahu
  Pravo: Vratenie Dane Zahranicnym Zastupcom Vo Vacsom Rozsahu
  Pravo: Vratenie Dane Poskytovane Slovenskou Republikou

  Subjekt: Zahranicni Zastupcovia
  Stat: Staty Zahranicnych Zastupcov
  Stat: Staty Poskytujuce Vratenie Dane Alebo Obdobne Zvyhodnenie Osobam Slovenskej Republiky
  Stat: Iny Stat
  Stat: Slovenska Republika

  Osoba: Osoby Slovenskej Republiky
  Organizacia: Medzinarodne Organizacie
  Osoba: Pracovnici Medzinarodnych Organizacii

  Podmienka: Vzajomnost Podla Paragraf § 61 Odsek 3
  Podmienka: Iny Stat Neposkytuje Vratenie Dane Alebo Obdobne Zvyhodnenie V Rozsahu Poskytovanom Slovenskou Republikou
  Podmienka: Iny Stat Poskytuje Vratenie Dane Alebo Obdobne Zvyhodnenie Vo Vacsom Rozsahu Ako Slovenska Republika

  Mnozstvo: Rozsah Vratenia Dane Poskytovany Slovenskou Republikou
  Mnozstvo: Rozsah Poskytovany Inym Statom Osobam Slovenskej Republiky


---

chunk: 885
path: ['§ 62', '3']
path_as_text: Paragraf § 62 Odsek 3
text: (3) Vrátenie dane môže zahraničný zástupca žiadať len v prípade, ak celková cena vrátane dane na jednom doklade o kúpe tovarov alebo služieb s výnimkou dokladu o kúpe pohonných látok je najmenej 33,19 eura. Ak iný štát viaže vrátenie dane osobám Slovenskej republiky na doklad o kúpe tovarov alebo služieb, na ktorom je celková cena vyššia ako 33,19 eura, môže zahraničný zástupca tohto štátu žiadať vrátenie dane z takého dokladu, na ktorom je celková cena najmenej vo výške určenej týmto štátom.

relations:
  Paragraf § 62 -> [OBSAHUJE] -> Paragraf § 62 Odsek 3
  Paragraf § 62 Odsek 3 -> [UPRAVUJE] -> Pravo Zahranicneho Zastupcu Ziadat Vratenie Dane

  Zahranicny Zastupca -> [MA_PRAVO] -> Pravo Ziadat Vratenie Dane
  Pravo Ziadat Vratenie Dane -> [VZTAHUJE_SA_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan

  Pravo Ziadat Vratenie Dane -> [MA_PODMIENKU] -> Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura
  Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura -> [VZTAHUJE_SA_NA] -> Celkova Cena Vratane Dane
  Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura -> [MA_SUMU] -> Suma 33,19 Eura
  Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura -> [NEVZTAHUJE_SA_NA] -> Doklad O Kupe Pohonnych Latok

  Doklad O Kupe Tovarov Alebo Sluzieb -> [VZTAHUJE_SA_NA] -> Tovar
  Doklad O Kupe Tovarov Alebo Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzba
  Doklad O Kupe Pohonnych Latok -> [VZTAHUJE_SA_NA] -> Pohonne Latky

  Pravo Zahranicneho Zastupcu Tohto Statu Ziadat Vratenie Dane Zo Zvyseneho Limitneho Dokladu -> [MA_PODMIENKU] -> Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura
  Pravo Zahranicneho Zastupcu Tohto Statu Ziadat Vratenie Dane Zo Zvyseneho Limitneho Dokladu -> [MA_PODMIENKU] -> Celkova Cena Na Doklade Najmenej Vo Vyske Urcenej Inym Statom
  Zahranicny Zastupca Tohto Statu -> [VZTAHUJE_SA_NA] -> Iny Stat
  Zahranicny Zastupca Tohto Statu -> [MA_PRAVO] -> Pravo Zahranicneho Zastupcu Tohto Statu Ziadat Vratenie Dane Zo Zvyseneho Limitneho Dokladu

  Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura -> [VZTAHUJE_SA_NA] -> Iny Stat
  Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura -> [VZTAHUJE_SA_NA] -> Osoby Slovenskej Republiky
  Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura -> [MA_SUMU] -> Suma 33,19 Eura

  Celkova Cena Na Doklade Najmenej Vo Vyske Urcenej Inym Statom -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Celkova Cena Na Doklade Najmenej Vo Vyske Urcenej Inym Statom -> [MA_SUMU] -> Suma Urcena Inym Statom
  Suma Urcena Inym Statom -> [VZTAHUJE_SA_NA] -> Iny Stat

nodes:
  Paragraf: Paragraf § 62
  Odsek: Paragraf § 62 Odsek 3

  Subjekt: Zahranicny Zastupca
  Subjekt: Zahranicny Zastupca Tohto Statu
  Stat: Iny Stat
  Stat: Slovenska Republika
  Osoba: Osoby Slovenskej Republiky

  Pravo: Pravo Ziadat Vratenie Dane
  Pravo: Pravo Zahranicneho Zastupcu Tohto Statu Ziadat Vratenie Dane Zo Zvyseneho Limitneho Dokladu
  Pravo: Vratenie Dane
  Dan: Dan

  Dokument: Doklad O Kupe Tovarov Alebo Sluzieb
  Dokument: Doklad O Kupe Pohonnych Latok
  Tovar: Tovar
  Sluzba: Sluzba
  Tovar: Pohonne Latky

  Suma: Celkova Cena Vratane Dane
  Suma: Suma 33,19 Eura
  Suma: Suma Urcena Inym Statom

  Podmienka: Celkova Cena Vratane Dane Na Jednom Doklade O Kupe Tovarov Alebo Sluzieb Najmenej 33,19 Eura
  Podmienka: Iny Stat Viaze Vratenie Dane Osobam Slovenskej Republiky Na Doklad S Celkovou Cenou Vyssou Ako 33,19 Eura
  Podmienka: Celkova Cena Na Doklade Najmenej Vo Vyske Urcenej Inym Statom


---

chunk: 894
path: ['§ 62aa', '5']
path_as_text: Paragraf § 62aa Odsek 5
text: (5) Ak sa prestali plniť podmienky na vrátenie dane podľa odseku 1 a rozhodnutie o vrátení dane už bolo vydané, Daňový úrad Bratislava toto rozhodnutie zruší. Ak sa prestali plniť podmienky na vrátenie dane podľa odseku 1 len čiastočne, Daňový úrad Bratislava novým rozhodnutím zruší rozhodnutie o vrátení dane a určí sumu dane, na vrátenie ktorej má Európska komisia, agentúra alebo orgán zriadený podľa práva Európskej únie nárok.

relations:
  Paragraf § 62aa -> [OBSAHUJE] -> Paragraf § 62aa Odsek 5
  Paragraf § 62aa -> [OBSAHUJE] -> Paragraf § 62aa Odsek 1
  Paragraf § 62aa Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 62aa Odsek 1

  Paragraf § 62aa Odsek 5 -> [UPRAVUJE] -> Zrusenie Rozhodnutia O Vrateni Dane

  Podmienky Na Vratenie Dane Podla Paragraf § 62aa Odsek 1 -> [VZTAHUJE_SA_NA] -> Vratenie Dane
  Podmienky Na Vratenie Dane Podla Paragraf § 62aa Odsek 1 -> [JE_PODLA] -> Paragraf § 62aa Odsek 1
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan

  Danovy Urad Bratislava -> [RUSI] -> Rozhodnutie O Vrateni Dane
  Zrusenie Rozhodnutia O Vrateni Dane -> [MA_PODMIENKU] -> Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1
  Zrusenie Rozhodnutia O Vrateni Dane -> [MA_PODMIENKU] -> Rozhodnutie O Vrateni Dane Uz Bolo Vydane

  Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1 -> [VZTAHUJE_SA_NA] -> Podmienky Na Vratenie Dane Podla Paragraf § 62aa Odsek 1

  Danovy Urad Bratislava -> [VYDAVA] -> Nove Rozhodnutie
  Nove Rozhodnutie -> [RUSI] -> Rozhodnutie O Vrateni Dane
  Nove Rozhodnutie -> [URCUJE] -> Suma Dane Na Vratenie
  Nove Rozhodnutie -> [MA_PODMIENKU] -> Ciastocne Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1

  Ciastocne Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1 -> [VZTAHUJE_SA_NA] -> Podmienky Na Vratenie Dane Podla Paragraf § 62aa Odsek 1

  Suma Dane Na Vratenie -> [VZTAHUJE_SA_NA] -> Dan

  Europska Komisia -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Agentura -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Organ Zriadeny Podla Prava Europskej Unie -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Organ Zriadeny Podla Prava Europskej Unie -> [JE_PODLA] -> Pravo Europskej Unie

nodes:
  Paragraf: Paragraf § 62aa
  Odsek: Paragraf § 62aa Odsek 5
  Odsek: Paragraf § 62aa Odsek 1

  Organizacia: Danovy Urad Bratislava
  Organizacia: Europska Komisia
  Organizacia: Agentura
  Organizacia: Organ Zriadeny Podla Prava Europskej Unie

  Rozhodnutie: Rozhodnutie O Vrateni Dane
  Rozhodnutie: Nove Rozhodnutie

  Konanie: Zrusenie Rozhodnutia O Vrateni Dane

  Podmienka: Podmienky Na Vratenie Dane Podla Paragraf § 62aa Odsek 1
  Podmienka: Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1
  Podmienka: Ciastocne Prestanie Plnenia Podmienok Na Vratenie Dane Podla Paragraf § 62aa Odsek 1
  Podmienka: Rozhodnutie O Vrateni Dane Uz Bolo Vydane

  Pravo: Vratenie Dane
  Dan: Dan
  Suma: Suma Dane Na Vratenie
  PravnyPredpis: Pravo Europskej Unie


---

chunk: 908
path: ['§ 65', '5']
path_as_text: Paragraf § 65 Odsek 5
text: (5) Ak je cestovná kancelária povinná postupovať pri odpočítaní dane podľa § 50, pri výpočte koeficientu neuvádza do čitateľa ani menovateľa služby cestovného ruchu obstarané od iných osôb.

relations:
  Paragraf § 65 -> [OBSAHUJE] -> Paragraf § 65 Odsek 5
  Paragraf § 65 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 50

  Cestovna Kancelaria -> [MA_POVINNOST] -> Povinnost Postupovat Pri Odpocitani Dane Podla Paragraf § 50
  Povinnost Postupovat Pri Odpocitani Dane Podla Paragraf § 50 -> [VZTAHUJE_SA_NA] -> Odpocitanie Dane
  Povinnost Postupovat Pri Odpocitani Dane Podla Paragraf § 50 -> [JE_PODLA] -> Paragraf § 50

  Povinnost Postupovat Pri Odpocitani Dane Podla Paragraf § 50 -> [VZTAHUJE_SA_NA] -> Vypocet Koeficientu
  Vypocet Koeficientu -> [MA] -> Citatel Koeficientu
  Vypocet Koeficientu -> [MA] -> Menovatel Koeficientu

  Citatel Koeficientu -> [NEVZTAHUJE_SA_NA] -> Sluzby Cestovneho Ruchu Obstarane Od Inych Osob
  Menovatel Koeficientu -> [NEVZTAHUJE_SA_NA] -> Sluzby Cestovneho Ruchu Obstarane Od Inych Osob

  Sluzby Cestovneho Ruchu Obstarane Od Inych Osob -> [VZTAHUJE_SA_NA] -> Ine Osoby

nodes:
  Paragraf: Paragraf § 65
  Odsek: Paragraf § 65 Odsek 5
  Paragraf: Paragraf § 50

  Organizacia: Cestovna Kancelaria
  Povinnost: Povinnost Postupovat Pri Odpocitani Dane Podla Paragraf § 50
  Pravo: Odpocitanie Dane

  Konanie: Vypocet Koeficientu
  Zaznam: Citatel Koeficientu
  Zaznam: Menovatel Koeficientu

  Sluzba: Sluzby Cestovneho Ruchu Obstarane Od Inych Osob
  Osoba: Ine Osoby


---

chunk: 931
path: ['§ 66', '6']
path_as_text: Paragraf § 66 Odsek 6
text: (6) Ak sa obchodník v prípadoch podľa odseku 5 rozhodne pre osobitnú úpravu uplatňovania dane, je povinný tento postup uplatňovať najmenej dva kalendárne roky.

relations:
  Paragraf § 66 -> [OBSAHUJE] -> Paragraf § 66 Odsek 6
  Paragraf § 66 -> [OBSAHUJE] -> Paragraf § 66 Odsek 5
  Paragraf § 66 Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 66 Odsek 5

  Paragraf § 66 Odsek 6 -> [UPRAVUJE] -> Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky

  Obchodnik -> [MA_POVINNOST] -> Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky

  Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky -> [MA_PODMIENKU] -> Rozhodnutie Obchodnika Pre Osobitnu Upravu Uplatnovania Dane V Pripadoch Podla Paragraf § 66 Odsek 5
  Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky -> [VZTAHUJE_SA_NA] -> Osobitna Uprava Uplatnovania Dane
  Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky -> [MA_OBDOBIE] -> Najmenej Dva Kalendarne Roky

  Rozhodnutie Obchodnika Pre Osobitnu Upravu Uplatnovania Dane V Pripadoch Podla Paragraf § 66 Odsek 5 -> [VZTAHUJE_SA_NA] -> Obchodnik
  Rozhodnutie Obchodnika Pre Osobitnu Upravu Uplatnovania Dane V Pripadoch Podla Paragraf § 66 Odsek 5 -> [VZTAHUJE_SA_NA] -> Osobitna Uprava Uplatnovania Dane
  Rozhodnutie Obchodnika Pre Osobitnu Upravu Uplatnovania Dane V Pripadoch Podla Paragraf § 66 Odsek 5 -> [JE_PODLA] -> Paragraf § 66 Odsek 5

  Osobitna Uprava Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Dan

nodes:
  Paragraf: Paragraf § 66
  Odsek: Paragraf § 66 Odsek 6
  Odsek: Paragraf § 66 Odsek 5

  Subjekt: Obchodnik
  Konanie: Osobitna Uprava Uplatnovania Dane
  Dan: Dan

  Povinnost: Povinnost Uplatnovat Osobitnu Upravu Uplatnovania Dane Najmenej Dva Kalendarne Roky
  Podmienka: Rozhodnutie Obchodnika Pre Osobitnu Upravu Uplatnovania Dane V Pripadoch Podla Paragraf § 66 Odsek 5
  Obdobie: Najmenej Dva Kalendarne Roky


---

chunk: 935
path: ['§ 66', '9']
path_as_text: Paragraf § 66 Odsek 9
text: (9) Obchodník, ktorý uplatňuje osobitnú úpravu, je povinný na účely určenia základu dane podľa odseku 3 viesť osobitne záznamy o predajných cenách a kúpnych cenách tovarov.

relations:
  Paragraf § 66 -> [OBSAHUJE] -> Paragraf § 66 Odsek 9
  Paragraf § 66 -> [OBSAHUJE] -> Paragraf § 66 Odsek 3
  Paragraf § 66 Odsek 9 -> [ODKAZUJE_NA] -> Paragraf § 66 Odsek 3

  Paragraf § 66 Odsek 9 -> [UPRAVUJE] -> Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov

  Obchodnik -> [MA_POVINNOST] -> Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov

  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [MA_PODMIENKU] -> Uplatnovanie Osobitnej Upravy Obchodnikom
  Uplatnovanie Osobitnej Upravy Obchodnikom -> [VZTAHUJE_SA_NA] -> Obchodnik
  Uplatnovanie Osobitnej Upravy Obchodnikom -> [VZTAHUJE_SA_NA] -> Osobitna Uprava

  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [VZTAHUJE_SA_NA] -> Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [VZTAHUJE_SA_NA] -> Urcenie Zakladu Dane Podla Paragraf § 66 Odsek 3

  Urcenie Zakladu Dane Podla Paragraf § 66 Odsek 3 -> [URCUJE] -> Zaklad Dane
  Urcenie Zakladu Dane Podla Paragraf § 66 Odsek 3 -> [JE_PODLA] -> Paragraf § 66 Odsek 3

  Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [OBSAHUJE] -> Predajne Ceny Tovarov
  Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [OBSAHUJE] -> Kupne Ceny Tovarov

  Predajne Ceny Tovarov -> [VZTAHUJE_SA_NA] -> Tovary
  Kupne Ceny Tovarov -> [VZTAHUJE_SA_NA] -> Tovary

nodes:
  Paragraf: Paragraf § 66
  Odsek: Paragraf § 66 Odsek 9
  Odsek: Paragraf § 66 Odsek 3

  Subjekt: Obchodnik
  Konanie: Uplatnovanie Osobitnej Upravy Obchodnikom
  Konanie: Osobitna Uprava

  Povinnost: Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Zaznam: Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov

  Suma: Predajne Ceny Tovarov
  Suma: Kupne Ceny Tovarov
  Tovar: Tovary

  Konanie: Urcenie Zakladu Dane Podla Paragraf § 66 Odsek 3
  Dan: Zaklad Dane


---

chunk: 954
path: ['§ 67', '6']
path_as_text: Paragraf § 67 Odsek 6
text: (6) Platiteľ, ktorý vyrába investičné zlato alebo pretvára zlato na investičné zlato, môže odpočítať daň z tovarov a služieb prijatých na túto činnosť. Osobitné úpravy uplatňovania dane na služby dodávané osobe inej ako zdaniteľnej osobe, na predaj tovaru na diaľku a určité domáce dodania tovaru

relations:
  Paragraf § 67 -> [OBSAHUJE] -> Paragraf § 67 Odsek 6
  Paragraf § 67 Odsek 6 -> [UPRAVUJE] -> Pravo Na Odpocitanie Dane Pri Vyrobe Investicneho Zlata Alebo Pretvarani Zlata Na Investicne Zlato

  Platitel -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane Pri Vyrobe Investicneho Zlata Alebo Pretvarani Zlata Na Investicne Zlato

  Pravo Na Odpocitanie Dane Pri Vyrobe Investicneho Zlata Alebo Pretvarani Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Dan Z Tovarov A Sluzieb Prijatych Na Tuto Cinnost
  Pravo Na Odpocitanie Dane Pri Vyrobe Investicneho Zlata Alebo Pretvarani Zlata Na Investicne Zlato -> [MA_PODMIENKU] -> Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato

  Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Vyroba Investicneho Zlata
  Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Pretvaranie Zlata Na Investicne Zlato

  Vyroba Investicneho Zlata -> [VZTAHUJE_SA_NA] -> Investicne Zlato
  Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Zlato
  Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Investicne Zlato

  Dan Z Tovarov A Sluzieb Prijatych Na Tuto Cinnost -> [VZTAHUJE_SA_NA] -> Tovary Prijate Na Tuto Cinnost
  Dan Z Tovarov A Sluzieb Prijatych Na Tuto Cinnost -> [VZTAHUJE_SA_NA] -> Sluzby Prijate Na Tuto Cinnost

  Tovary Prijate Na Tuto Cinnost -> [VZTAHUJE_SA_NA] -> Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato
  Sluzby Prijate Na Tuto Cinnost -> [VZTAHUJE_SA_NA] -> Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato

nodes:
  Paragraf: Paragraf § 67
  Odsek: Paragraf § 67 Odsek 6

  Subjekt: Platitel

  Pravo: Pravo Na Odpocitanie Dane Pri Vyrobe Investicneho Zlata Alebo Pretvarani Zlata Na Investicne Zlato
  Dan: Dan Z Tovarov A Sluzieb Prijatych Na Tuto Cinnost

  Podmienka: Vyroba Investicneho Zlata Alebo Pretvaranie Zlata Na Investicne Zlato

  Konanie: Vyroba Investicneho Zlata
  Konanie: Pretvaranie Zlata Na Investicne Zlato

  Tovar: Investicne Zlato
  Tovar: Zlato
  Tovar: Tovary Prijate Na Tuto Cinnost
  Sluzba: Sluzby Prijate Na Tuto Cinnost


---

chunk: 976
path: ['§ 68a', '10', 'b)']
path_as_text: Paragraf § 68a Odsek 10 Pismeno b)
text: (10) Zdaniteľná osoba neusadená na území Európskej únie je povinná v daňovom priznaní uviesť b) celkovú hodnotu služieb podľa § 68 ods. 1 písm. a) bez dane dodaných v zdaňovacom období, výšku dane pre každú sadzbu dane, sadzbu dane a celkovú výšku splatnej dane, a to v členení  podľa členských štátov spotreby, v ktorých vznikla daňová povinnosť.

relations:
  Paragraf § 68a -> [OBSAHUJE] -> Paragraf § 68a Odsek 10
  Paragraf § 68a Odsek 10 -> [OBSAHUJE] -> Paragraf § 68a Odsek 10 Pismeno b)
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno a)

  Paragraf § 68a Odsek 10 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno a)
  Paragraf § 68a Odsek 10 Pismeno b) -> [UPRAVUJE] -> Povinnost Uviest Udaje V Danovom Priznani

  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_POVINNOST] -> Povinnost Uviest Udaje V Danovom Priznani
  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_PODMIENKU] -> Neusadenie Na Uzemi Europskej Unie

  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Celkova Hodnota Sluzieb Podla Paragraf § 68 Odsek 1 Pismeno a) Bez Dane Dodanych V Zdanovacom Obdobi
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Celkova Vyska Splatnej Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [MA_PODMIENKU] -> Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost

  Celkova Hodnota Sluzieb Podla Paragraf § 68 Odsek 1 Pismeno a) Bez Dane Dodanych V Zdanovacom Obdobi -> [VZTAHUJE_SA_NA] -> Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  Celkova Hodnota Sluzieb Podla Paragraf § 68 Odsek 1 Pismeno a) Bez Dane Dodanych V Zdanovacom Obdobi -> [MA_OBDOBIE] -> Zdanovacie Obdobie

  Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a) -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno a)

  Vyska Dane Pre Kazdu Sadzbu Dane -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Celkova Vyska Splatnej Dane -> [VZTAHUJE_SA_NA] -> Dan

  Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost -> [VZTAHUJE_SA_NA] -> Clensky Stat Spotreby
  Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost -> [VZTAHUJE_SA_NA] -> Danova Povinnost
  Danova Povinnost -> [VZNIKA] -> Clensky Stat Spotreby

nodes:
  Paragraf: Paragraf § 68a
  Odsek: Paragraf § 68a Odsek 10
  Pismeno: Paragraf § 68a Odsek 10 Pismeno b)
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno a)

  Subjekt: Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
  Lokacia: Uzemie Europskej Unie
  Podmienka: Neusadenie Na Uzemi Europskej Unie

  DanovePriznanie: Danove Priznanie
  Povinnost: Povinnost Uviest Udaje V Danovom Priznani

  Suma: Celkova Hodnota Sluzieb Podla Paragraf § 68 Odsek 1 Pismeno a) Bez Dane Dodanych V Zdanovacom Obdobi
  Sluzba: Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  ZdanovacieObdobie: Zdanovacie Obdobie

  Dan: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Dan: Celkova Vyska Splatnej Dane
  Dan: Dan

  Stat: Clensky Stat Spotreby
  Povinnost: Danova Povinnost
  Podmienka: Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost


---

chunk: 977
path: ['§ 68a', '11']
path_as_text: Paragraf § 68a Odsek 11
text: (11) Sumy v daňovom priznaní sa uvádzajú v eurách. Ak sa úhrada za dodané služby podľa § 68 ods. 1 písm. a) uskutoční v inej mene ako v eurách, použije sa na prepočet tejto úhrady na eurá referenčný výmenný kurz určený a vyhlásený Európskou centrálnou bankou alebo Národnou bankou Slovenska5a) platný posledný deň príslušného zdaňovacieho obdobia alebo nasledujúci deň, ak nebol v posledný deň zdaňovacieho obdobia tento kurz určený a vyhlásený.

relations:
  Paragraf § 68a -> [OBSAHUJE] -> Paragraf § 68a Odsek 11
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno a)

  Paragraf § 68a Odsek 11 -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno a)
  Paragraf § 68a Odsek 11 -> [UPRAVUJE] -> Uvadzanie Sum V Danovom Priznani V Eurach
  Paragraf § 68a Odsek 11 -> [UPRAVUJE] -> Prepocet Uhrady Za Dodane Sluzby Na Eura

  Danove Priznanie -> [OBSAHUJE] -> Sumy V Danovom Priznani
  Sumy V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Eura

  Prepocet Uhrady Za Dodane Sluzby Na Eura -> [VZTAHUJE_SA_NA] -> Uhrada Za Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  Prepocet Uhrady Za Dodane Sluzby Na Eura -> [VZTAHUJE_SA_NA] -> Eura
  Prepocet Uhrady Za Dodane Sluzby Na Eura -> [MA_PODMIENKU] -> Uhrada Za Dodane Sluzby Uskutocnena V Inej Mene Ako V Eurach
  Prepocet Uhrady Za Dodane Sluzby Na Eura -> [MA_HODNOTU] -> Referencny Vymenny Kurz

  Uhrada Za Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a) -> [VZTAHUJE_SA_NA] -> Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a) -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno a)
  Uhrada Za Dodane Sluzby Uskutocnena V Inej Mene Ako V Eurach -> [VZTAHUJE_SA_NA] -> Ina Mena Ako Eura

  Europska Centralna Banka -> [URCUJE] -> Referencny Vymenny Kurz
  Narodna Banka Slovenska -> [URCUJE] -> Referencny Vymenny Kurz

  Referencny Vymenny Kurz -> [MA_DATUM] -> Posledny Den Prislusneho Zdanovacieho Obdobia
  Posledny Den Prislusneho Zdanovacieho Obdobia -> [PATRI_DO] -> Prislusne Zdanovacie Obdobie

  Referencny Vymenny Kurz -> [MA_DATUM] -> Nasledujuci Den
  Nasledujuci Den -> [MA_PODMIENKU] -> Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia
  Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia -> [MA_DATUM] -> Posledny Den Prislusneho Zdanovacieho Obdobia

nodes:
  Paragraf: Paragraf § 68a
  Odsek: Paragraf § 68a Odsek 11
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno a)

  DanovePriznanie: Danove Priznanie
  Suma: Sumy V Danovom Priznani
  Mena: Eura
  Mena: Ina Mena Ako Eura

  Platba: Uhrada Za Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  Sluzba: Dodane Sluzby Podla Paragraf § 68 Odsek 1 Pismeno a)
  Konanie: Prepocet Uhrady Za Dodane Sluzby Na Eura

  Podmienka: Uhrada Za Dodane Sluzby Uskutocnena V Inej Mene Ako V Eurach
  Podmienka: Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia

  Kurz: Referencny Vymenny Kurz
  Banka: Europska Centralna Banka
  Banka: Narodna Banka Slovenska

  ZdanovacieObdobie: Prislusne Zdanovacie Obdobie
  Datum: Posledny Den Prislusneho Zdanovacieho Obdobia
  Datum: Nasledujuci Den


---

chunk: 1000
path: ['§ 68b', '14', 'a)']
path_as_text: Paragraf § 68b Odsek 14 Pismeno a)
text: (14) Ak sa tovar odosiela alebo prepravuje z iných členských štátov, zdaniteľná osoba uvedená v odseku 2 je povinná v daňovom priznaní uviesť aj celkovú hodnotu dodaných tovarov podľa § 68 ods. 1 písm. b) bez dane, výšku dane pre každú sadzbu dane, sadzbu dane a celkovú výšku splatnej dane, a to v členení podľa členských štátov, z ktorých sa tovar odosiela alebo prepravuje, ak sa uskutočňuje a) predaj tovaru na diaľku na území Európskej únie podľa § 14 ods. 1 písm. a),

relations:
  Paragraf § 68b -> [OBSAHUJE] -> Paragraf § 68b Odsek 14
  Paragraf § 68b Odsek 14 -> [OBSAHUJE] -> Paragraf § 68b Odsek 14 Pismeno a)
  Paragraf § 68b -> [OBSAHUJE] -> Paragraf § 68b Odsek 2
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno b)
  Paragraf § 14 -> [OBSAHUJE] -> Paragraf § 14 Odsek 1
  Paragraf § 14 Odsek 1 -> [OBSAHUJE] -> Paragraf § 14 Odsek 1 Pismeno a)

  Paragraf § 68b Odsek 14 -> [ODKAZUJE_NA] -> Paragraf § 68b Odsek 2
  Paragraf § 68b Odsek 14 -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno b)
  Paragraf § 68b Odsek 14 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 14 Odsek 1 Pismeno a)

  Paragraf § 68b Odsek 14 Pismeno a) -> [UPRAVUJE] -> Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie

  Zdanitelna Osoba Uvedena V Paragraf § 68b Odsek 2 -> [JE_PODLA] -> Paragraf § 68b Odsek 2
  Zdanitelna Osoba Uvedena V Paragraf § 68b Odsek 2 -> [MA_POVINNOST] -> Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie

  Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [MA_PODMIENKU] -> Tovar Sa Odosiela Alebo Prepravuje Z Inych Clenskych Statov
  Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [MA_PODMIENKU] -> Uskutocnenie Predaja Tovaru Na Dialku Na Uzemi Europskej Unie Podla Paragraf § 14 Odsek 1 Pismeno a)

  Danove Priznanie -> [OBSAHUJE] -> Celkova Hodnota Dodanych Tovarov Podla Paragraf § 68 Odsek 1 Pismeno b) Bez Dane
  Danove Priznanie -> [OBSAHUJE] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Danove Priznanie -> [OBSAHUJE] -> Sadzba Dane
  Danove Priznanie -> [OBSAHUJE] -> Celkova Vyska Splatnej Dane
  Danove Priznanie -> [MA_PODMIENKU] -> Clenenie Podla Clenskych Statov Z Ktorych Sa Tovar Odosiela Alebo Prepravuje

  Celkova Hodnota Dodanych Tovarov Podla Paragraf § 68 Odsek 1 Pismeno b) Bez Dane -> [VZTAHUJE_SA_NA] -> Dodane Tovary Podla Paragraf § 68 Odsek 1 Pismeno b)
  Dodane Tovary Podla Paragraf § 68 Odsek 1 Pismeno b) -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno b)

  Vyska Dane Pre Kazdu Sadzbu Dane -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Celkova Vyska Splatnej Dane -> [VZTAHUJE_SA_NA] -> Splatna Dan

  Tovar Sa Odosiela Alebo Prepravuje Z Inych Clenskych Statov -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Sa Odosiela Alebo Prepravuje Z Inych Clenskych Statov -> [VZTAHUJE_SA_NA] -> Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje

  Clenenie Podla Clenskych Statov Z Ktorych Sa Tovar Odosiela Alebo Prepravuje -> [VZTAHUJE_SA_NA] -> Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje

  Uskutocnenie Predaja Tovaru Na Dialku Na Uzemi Europskej Unie Podla Paragraf § 14 Odsek 1 Pismeno a) -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Tovar
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [NACHADZA_SA_V] -> Uzemie Europskej Unie
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [JE_PODLA] -> Paragraf § 14 Odsek 1 Pismeno a)

nodes:
  Paragraf: Paragraf § 68b
  Odsek: Paragraf § 68b Odsek 14
  Pismeno: Paragraf § 68b Odsek 14 Pismeno a)
  Odsek: Paragraf § 68b Odsek 2
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno b)
  Paragraf: Paragraf § 14
  Odsek: Paragraf § 14 Odsek 1
  Pismeno: Paragraf § 14 Odsek 1 Pismeno a)

  Subjekt: Zdanitelna Osoba Uvedena V Paragraf § 68b Odsek 2
  Povinnost: Povinnost Uviest Udaje V Danovom Priznani Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie
  DanovePriznanie: Danove Priznanie

  Suma: Celkova Hodnota Dodanych Tovarov Podla Paragraf § 68 Odsek 1 Pismeno b) Bez Dane
  Tovar: Dodane Tovary Podla Paragraf § 68 Odsek 1 Pismeno b)
  Tovar: Tovar

  Suma: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Suma: Celkova Vyska Splatnej Dane
  Dan: Splatna Dan

  Stat: Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje
  Lokacia: Uzemie Europskej Unie
  Konanie: Predaj Tovaru Na Dialku Na Uzemi Europskej Unie

  Podmienka: Tovar Sa Odosiela Alebo Prepravuje Z Inych Clenskych Statov
  Podmienka: Uskutocnenie Predaja Tovaru Na Dialku Na Uzemi Europskej Unie Podla Paragraf § 14 Odsek 1 Pismeno a)
  Podmienka: Clenenie Podla Clenskych Statov Z Ktorych Sa Tovar Odosiela Alebo Prepravuje


---

chunk: 1017
path: ['§ 68c', '4']
path_as_text: Paragraf § 68c Odsek 4
text: (4) Ak sa zdaniteľná osoba, ktorá uskutočňuje predaj tovaru na diaľku podľa § 68 ods. 1 písm. c) a ktorá nie je zastúpená sprostredkovateľom, rozhodne pre uplatňovanie osobitnej úpravy a členským štátom identifikácie je tuzemsko alebo si tuzemsko zvolí ako členský štát identifikácie, je povinná predtým, ako začne uplatňovať osobitnú úpravu, oznámiť toto rozhodnutie daňovému úradu. Toto oznámenie musí obsahovať obchodné meno, adresu, elektronickú adresu vrátane webových sídiel, identifikačné číslo pre daň alebo národné daňové číslo a ďalšie údaje uvedené v osobitnom predpise.28aa) Ak zdaniteľná osoba spĺňa podmienky na uplatňovanie osobitnej úpravy, daňový úrad jej oznámi, že jej povoľuje uplatňovanie osobitnej úpravy; proti tomuto rozhodnutiu nemožno podať odvolanie.

relations:
  Paragraf § 68c -> [OBSAHUJE] -> Paragraf § 68c Odsek 4
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno c)

  Paragraf § 68c Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno c)
  Paragraf § 68c Odsek 4 -> [ODKAZUJE_NA] -> Osobitny Predpis
  Paragraf § 68c Odsek 4 -> [UPRAVUJE] -> Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku
  Predaj Tovaru Na Dialku -> [VZTAHUJE_SA_NA] -> Tovar
  Predaj Tovaru Na Dialku -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno c)

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [MA_PODMIENKU] -> Nie Je Zastupena Sprostredkovatelom
  Nie Je Zastupena Sprostredkovatelom -> [VZTAHUJE_SA_NA] -> Sprostredkovatel

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [MA_PODMIENKU] -> Rozhodnutie Pre Uplatnovanie Osobitnej Upravy
  Rozhodnutie Pre Uplatnovanie Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [MA_PODMIENKU] -> Tuzemsko Je Clenskym Statom Identifikacie Alebo Je Zvolene Ako Clensky Stat Identifikacie
  Tuzemsko Je Clenskym Statom Identifikacie Alebo Je Zvolene Ako Clensky Stat Identifikacie -> [VZTAHUJE_SA_NA] -> Tuzemsko
  Tuzemsko Je Clenskym Statom Identifikacie Alebo Je Zvolene Ako Clensky Stat Identifikacie -> [VZTAHUJE_SA_NA] -> Clensky Stat Identifikacie

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [MA_POVINNOST] -> Povinnost Oznamit Rozhodnutie O Uplatnovani Osobitnej Upravy Danovemu Uradu
  Povinnost Oznamit Rozhodnutie O Uplatnovani Osobitnej Upravy Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy
  Povinnost Oznamit Rozhodnutie O Uplatnovani Osobitnej Upravy Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Danovy Urad
  Povinnost Oznamit Rozhodnutie O Uplatnovani Osobitnej Upravy Danovemu Uradu -> [MA_LEHOTU] -> Pred Zacatim Uplatnovania Osobitnej Upravy

  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Obchodne Meno
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Adresa
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Elektronicka Adresa Vratane Webovych Sidel
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Identifikacne Cislo Pre Dan
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Narodne Danove Cislo
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Dalsie Udaje Uvedene V Osobitnom Predpise
  Dalsie Udaje Uvedene V Osobitnom Predpise -> [JE_PODLA] -> Osobitny Predpis

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [SPLNA_PODMIENKY] -> Podmienky Na Uplatnovanie Osobitnej Upravy
  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Povolení Uplatnovania Osobitnej Upravy
  Rozhodnutie O Povolení Uplatnovania Osobitnej Upravy -> [ROZHODUJE_O] -> Uplatnovanie Osobitnej Upravy
  Rozhodnutie O Povolení Uplatnovania Osobitnej Upravy -> [MA_PODMIENKU] -> Podmienky Na Uplatnovanie Osobitnej Upravy

  Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [NEMA_NAROK_NA] -> Odvolanie Proti Rozhodnutiu O Povolení Uplatnovania Osobitnej Upravy
  Odvolanie Proti Rozhodnutiu O Povolení Uplatnovania Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Rozhodnutie O Povolení Uplatnovania Osobitnej Upravy

nodes:
  Paragraf: Paragraf § 68c
  Odsek: Paragraf § 68c Odsek 4
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)

  Osoba: Zdanitelna Osoba Uskutocnujuca Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c)
  Osoba: Sprostredkovatel
  Organizacia: Danovy Urad

  Konanie: Predaj Tovaru Na Dialku
  Tovar: Tovar
  Konanie: Uplatnovanie Osobitnej Upravy

  Stat: Clensky Stat Identifikacie
  Stat: Tuzemsko

  Podmienka: Nie Je Zastupena Sprostredkovatelom
  Podmienka: Rozhodnutie Pre Uplatnovanie Osobitnej Upravy
  Podmienka: Tuzemsko Je Clenskym Statom Identifikacie Alebo Je Zvolene Ako Clensky Stat Identifikacie
  Podmienka: Podmienky Na Uplatnovanie Osobitnej Upravy

  Povinnost: Povinnost Oznamit Rozhodnutie O Uplatnovani Osobitnej Upravy Danovemu Uradu
  Oznamenie: Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy
  Lehota: Pred Zacatim Uplatnovania Osobitnej Upravy

  Zaznam: Obchodne Meno
  Adresa: Adresa
  Adresa: Elektronicka Adresa Vratane Webovych Sidel
  Zaznam: Identifikacne Cislo Pre Dan
  Zaznam: Narodne Danove Cislo
  Zaznam: Dalsie Udaje Uvedene V Osobitnom Predpise
  PravnyPredpis: Osobitny Predpis

  Rozhodnutie: Rozhodnutie O Povolení Uplatnovania Osobitnej Upravy
  Dokument: Odvolanie Proti Rozhodnutiu O Povolení Uplatnovania Osobitnej Upravy


---

chunk: 1023
path: ['§ 68c', '8']
path_as_text: Paragraf § 68c Odsek 8
text: (8) Identifikačné číslo pre daň pridelené podľa odseku 7 písm. a) a c) a evidenčné identifikačné číslo pridelené podľa odseku 7 písm. b) sa môže použiť len na účely uplatňovania osobitnej úpravy.

relations:
  Paragraf § 68c -> [OBSAHUJE] -> Paragraf § 68c Odsek 8
  Paragraf § 68c -> [OBSAHUJE] -> Paragraf § 68c Odsek 7
  Paragraf § 68c Odsek 7 -> [OBSAHUJE] -> Paragraf § 68c Odsek 7 Pismeno a)
  Paragraf § 68c Odsek 7 -> [OBSAHUJE] -> Paragraf § 68c Odsek 7 Pismeno b)
  Paragraf § 68c Odsek 7 -> [OBSAHUJE] -> Paragraf § 68c Odsek 7 Pismeno c)

  Paragraf § 68c Odsek 8 -> [ODKAZUJE_NA] -> Paragraf § 68c Odsek 7 Pismeno a)
  Paragraf § 68c Odsek 8 -> [ODKAZUJE_NA] -> Paragraf § 68c Odsek 7 Pismeno b)
  Paragraf § 68c Odsek 8 -> [ODKAZUJE_NA] -> Paragraf § 68c Odsek 7 Pismeno c)

  Paragraf § 68c Odsek 8 -> [UPRAVUJE] -> Obmedzenie Pouzitia Identifikacneho Cisla Pre Dan A Evidencneho Identifikacneho Cisla

  Identifikacne Cislo Pre Dan Pridelene Podla Paragraf § 68c Odsek 7 Pismeno a) A Pismeno c) -> [JE_PODLA] -> Paragraf § 68c Odsek 7 Pismeno a)
  Identifikacne Cislo Pre Dan Pridelene Podla Paragraf § 68c Odsek 7 Pismeno a) A Pismeno c) -> [JE_PODLA] -> Paragraf § 68c Odsek 7 Pismeno c)

  Evidencne Identifikacne Cislo Pridelene Podla Paragraf § 68c Odsek 7 Pismeno b) -> [JE_PODLA] -> Paragraf § 68c Odsek 7 Pismeno b)

  Pouzitie Identifikacneho Cisla Pre Dan -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Pridelene Podla Paragraf § 68c Odsek 7 Pismeno a) A Pismeno c)
  Pouzitie Identifikacneho Cisla Pre Dan -> [MA_PODMIENKU] -> Len Na Ucely Uplatnovania Osobitnej Upravy
  Pouzitie Identifikacneho Cisla Pre Dan -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy

  Pouzitie Evidencneho Identifikacneho Cisla -> [VZTAHUJE_SA_NA] -> Evidencne Identifikacne Cislo Pridelene Podla Paragraf § 68c Odsek 7 Pismeno b)
  Pouzitie Evidencneho Identifikacneho Cisla -> [MA_PODMIENKU] -> Len Na Ucely Uplatnovania Osobitnej Upravy
  Pouzitie Evidencneho Identifikacneho Cisla -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy

nodes:
  Paragraf: Paragraf § 68c
  Odsek: Paragraf § 68c Odsek 8
  Odsek: Paragraf § 68c Odsek 7
  Pismeno: Paragraf § 68c Odsek 7 Pismeno a)
  Pismeno: Paragraf § 68c Odsek 7 Pismeno b)
  Pismeno: Paragraf § 68c Odsek 7 Pismeno c)

  Zaznam: Identifikacne Cislo Pre Dan Pridelene Podla Paragraf § 68c Odsek 7 Pismeno a) A Pismeno c)
  Zaznam: Evidencne Identifikacne Cislo Pridelene Podla Paragraf § 68c Odsek 7 Pismeno b)

  Konanie: Uplatnovanie Osobitnej Upravy
  Pravo: Pouzitie Identifikacneho Cisla Pre Dan
  Pravo: Pouzitie Evidencneho Identifikacneho Cisla

  Podmienka: Len Na Ucely Uplatnovania Osobitnej Upravy
  Podmienka: Obmedzenie Pouzitia Identifikacneho Cisla Pre Dan A Evidencneho Identifikacneho Cisla


---

chunk: 1046
path: ['§ 68c', '21', 'b)']
path_as_text: Paragraf § 68c Odsek 21 Pismeno b)
text: (21) Daňové priznanie musí obsahovať tieto údaje: b) celkovú hodnotu predaja tovaru na diaľku podľa § 68 ods. 1 písm. c) bez dane, pri ktorom vznikla daňová povinnosť v zdaňovacom období, výšku dane pre každú sadzbu dane, sadzbu dane a celkovú výšku splatnej dane, a to v členení podľa členských štátov spotreby, v ktorých vznikla daňová povinnosť.

relations:
  Paragraf § 68c -> [OBSAHUJE] -> Paragraf § 68c Odsek 21
  Paragraf § 68c Odsek 21 -> [OBSAHUJE] -> Paragraf § 68c Odsek 21 Pismeno b)
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno c)

  Paragraf § 68c Odsek 21 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno c)
  Paragraf § 68c Odsek 21 Pismeno b) -> [URCUJE] -> Obsah Danoveho Priznania

  Danove Priznanie -> [OBSAHUJE] -> Celkova Hodnota Predaja Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) Bez Dane
  Danove Priznanie -> [OBSAHUJE] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Danove Priznanie -> [OBSAHUJE] -> Sadzba Dane
  Danove Priznanie -> [OBSAHUJE] -> Celkova Vyska Splatnej Dane
  Danove Priznanie -> [MA_PODMIENKU] -> Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost

  Celkova Hodnota Predaja Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) Bez Dane -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c)
  Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [VZTAHUJE_SA_NA] -> Tovar
  Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno c)

  Danova Povinnost -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c)
  Danova Povinnost -> [MA_OBDOBIE] -> Zdanovacie Obdobie

  Vyska Dane Pre Kazdu Sadzbu Dane -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Celkova Vyska Splatnej Dane -> [VZTAHUJE_SA_NA] -> Splatna Dan

  Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost -> [VZTAHUJE_SA_NA] -> Clensky Stat Spotreby
  Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost -> [VZTAHUJE_SA_NA] -> Danova Povinnost
  Danova Povinnost -> [VZNIKA] -> Clensky Stat Spotreby

nodes:
  Paragraf: Paragraf § 68c
  Odsek: Paragraf § 68c Odsek 21
  Pismeno: Paragraf § 68c Odsek 21 Pismeno b)
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)

  DanovePriznanie: Danove Priznanie
  Zaznam: Obsah Danoveho Priznania

  Suma: Celkova Hodnota Predaja Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c) Bez Dane
  Suma: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Suma: Celkova Vyska Splatnej Dane
  Dan: Splatna Dan

  Konanie: Predaj Tovaru Na Dialku Podla Paragraf § 68 Odsek 1 Pismeno c)
  Tovar: Tovar

  Povinnost: Danova Povinnost
  ZdanovacieObdobie: Zdanovacie Obdobie
  Stat: Clensky Stat Spotreby

  Podmienka: Clenenie Podla Clenskych Statov Spotreby V Ktorych Vznikla Danova Povinnost


---

chunk: 1058
path: ['§ 68ca', '6', 'c)']
path_as_text: Paragraf § 68ca Odsek 6 Pismeno c)
text: (6) Ak je členským štátom spotreby Slovenská republika, osoba, ktorá uplatňuje alebo uplatňovala osobitnú úpravu podľa § 68a až 68c alebo podľa ustanovení zákona platného v inom členskom štáte zodpovedajúcich § 68a až 68c, c) je povinná podať daňovému úradu elektronickými prostriedkami osobitné tlačivo do 30 dní odo dňa zistenia, že neuviedla daň alebo daň má byť vyššia, ako bola uvedená v podanom konečnom daňovom priznaní28ae)alebo predchádzajúcich daňových priznaniach po podaní konečného daňového priznania alebo

relations:
  Paragraf § 68ca -> [OBSAHUJE] -> Paragraf § 68ca Odsek 6
  Paragraf § 68ca Odsek 6 -> [OBSAHUJE] -> Paragraf § 68ca Odsek 6 Pismeno c)

  Paragraf § 68ca Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 68a
  Paragraf § 68ca Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 68b
  Paragraf § 68ca Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 68c
  Paragraf § 68ca Odsek 6 Pismeno c) -> [UPRAVUJE] -> Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami

  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [MA_POVINNOST] -> Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami
  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [VZTAHUJE_SA_NA] -> Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State

  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [MA_PODMIENKU] -> Clenskym Statom Spotreby Je Slovenska Republika
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [MA_PODMIENKU] -> Osoba Uplatnuje Alebo Uplatnovala Osobitnu Upravu
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [VZTAHUJE_SA_NA] -> Osobitne Tlacivo
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [VZTAHUJE_SA_NA] -> Danovy Urad
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [MA_PODMIENKU] -> Elektronicke Prostriedky
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [MA_LEHOTU] -> Lehota 30 Dni Odo Dna Zistenia
  Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami -> [VYPLYVA_Z] -> Zistenie Neuvedenej Dane Alebo Vyssej Dane

  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [PODAVA] -> Osobitne Tlacivo
  Osobitne Tlacivo -> [VZTAHUJE_SA_NA] -> Danovy Urad

  Clenskym Statom Spotreby Je Slovenska Republika -> [VZTAHUJE_SA_NA] -> Clensky Stat Spotreby
  Clenskym Statom Spotreby Je Slovenska Republika -> [VZTAHUJE_SA_NA] -> Slovenska Republika

  Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State -> [JE_PODLA] -> Paragraf § 68a
  Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State -> [JE_PODLA] -> Paragraf § 68b
  Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State -> [JE_PODLA] -> Paragraf § 68c
  Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c

  Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c -> [VZTAHUJE_SA_NA] -> Zakon Platny V Inom Clenskom State
  Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c -> [ODKAZUJE_NA] -> Paragraf § 68a
  Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c -> [ODKAZUJE_NA] -> Paragraf § 68b
  Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c -> [ODKAZUJE_NA] -> Paragraf § 68c

  Lehota 30 Dni Odo Dna Zistenia -> [VYPLYVA_Z] -> Den Zistenia

  Zistenie Neuvedenej Dane Alebo Vyssej Dane -> [VZTAHUJE_SA_NA] -> Neuvedena Dan
  Zistenie Neuvedenej Dane Alebo Vyssej Dane -> [VZTAHUJE_SA_NA] -> Vyssia Dan Ako Uvedena Dan
  Vyssia Dan Ako Uvedena Dan -> [VZTAHUJE_SA_NA] -> Podane Konecne Danove Priznanie
  Vyssia Dan Ako Uvedena Dan -> [VZTAHUJE_SA_NA] -> Predchadzajuce Danove Priznania Po Podani Konecneho Danoveho Priznania

nodes:
  Paragraf: Paragraf § 68ca
  Odsek: Paragraf § 68ca Odsek 6
  Pismeno: Paragraf § 68ca Odsek 6 Pismeno c)

  Paragraf: Paragraf § 68a
  Paragraf: Paragraf § 68b
  Paragraf: Paragraf § 68c

  Stat: Slovenska Republika
  Stat: Clensky Stat Spotreby

  Osoba: Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu
  Konanie: Osobitna Uprava Podla Paragraf § 68a Az § 68c Alebo Zodpovedajucich Ustanoveni Zakona Platneho V Inom Clenskom State

  PravnyPredpis: Zakon Platny V Inom Clenskom State
  Zaznam: Ustanovenia Zakona Platneho V Inom Clenskom State Zodpovedajuce Paragrafu § 68a Az § 68c

  Organizacia: Danovy Urad
  Dokument: Osobitne Tlacivo
  Dokument: Elektronicke Prostriedky

  Povinnost: Povinnost Podat Osobitne Tlacivo Danovemu Uradu Elektronickymi Prostriedkami

  Lehota: Lehota 30 Dni Odo Dna Zistenia
  Datum: Den Zistenia

  Dovod: Zistenie Neuvedenej Dane Alebo Vyssej Dane
  Dan: Neuvedena Dan
  Dan: Vyssia Dan Ako Uvedena Dan

  DanovePriznanie: Podane Konecne Danove Priznanie
  DanovePriznanie: Predchadzajuce Danove Priznania Po Podani Konecneho Danoveho Priznania

  Podmienka: Clenskym Statom Spotreby Je Slovenska Republika
  Podmienka: Osoba Uplatnuje Alebo Uplatnovala Osobitnu Upravu


---

chunk: 1069
path: ['§ 68cb', '4']
path_as_text: Paragraf § 68cb Odsek 4
text: (4) Osoba, ktorá má povolenie podľa odseku 3, vyberie daň od osoby, pre ktorú je dovezený tovar určený, a túto vybranú daň je povinná zaplatiť colnému úradu.

relations:
  Paragraf § 68cb -> [OBSAHUJE] -> Paragraf § 68cb Odsek 4
  Paragraf § 68cb -> [OBSAHUJE] -> Paragraf § 68cb Odsek 3
  Paragraf § 68cb Odsek 4 -> [ODKAZUJE_NA] -> Paragraf § 68cb Odsek 3

  Osoba S Povolenim Podla Paragraf § 68cb Odsek 3 -> [MA] -> Povolenie Podla Paragraf § 68cb Odsek 3
  Povolenie Podla Paragraf § 68cb Odsek 3 -> [JE_PODLA] -> Paragraf § 68cb Odsek 3

  Osoba S Povolenim Podla Paragraf § 68cb Odsek 3 -> [PRIJIMA] -> Vybrana Dan
  Vybrana Dan -> [JE_TYPOM] -> Dan
  Vybrana Dan -> [VZTAHUJE_SA_NA] -> Osoba Pre Ktoru Je Dovezeny Tovar Urceny

  Osoba Pre Ktoru Je Dovezeny Tovar Urceny -> [PLATI] -> Vybrana Dan
  Dovezeny Tovar -> [VZTAHUJE_SA_NA] -> Osoba Pre Ktoru Je Dovezeny Tovar Urceny

  Osoba S Povolenim Podla Paragraf § 68cb Odsek 3 -> [MA_POVINNOST] -> Povinnost Zaplatit Vybranu Dan Colnemu Uradu
  Povinnost Zaplatit Vybranu Dan Colnemu Uradu -> [VZTAHUJE_SA_NA] -> Vybrana Dan
  Povinnost Zaplatit Vybranu Dan Colnemu Uradu -> [VZTAHUJE_SA_NA] -> Colny Urad
  Osoba S Povolenim Podla Paragraf § 68cb Odsek 3 -> [PLATI] -> Vybrana Dan

nodes:
  Paragraf: Paragraf § 68cb
  Odsek: Paragraf § 68cb Odsek 4
  Odsek: Paragraf § 68cb Odsek 3

  Osoba: Osoba S Povolenim Podla Paragraf § 68cb Odsek 3
  Rozhodnutie: Povolenie Podla Paragraf § 68cb Odsek 3

  Osoba: Osoba Pre Ktoru Je Dovezeny Tovar Urceny
  Tovar: Dovezeny Tovar

  Dan: Dan
  Dan: Vybrana Dan

  Povinnost: Povinnost Zaplatit Vybranu Dan Colnemu Uradu
  Organizacia: Colny Urad


---

chunk: 1092
path: ['§ 68d', '11', 'b)']
path_as_text: Paragraf § 68d Odsek 11 Pismeno b)
text: (11) Platiteľ je povinný skončiť uplatňovanie osobitnej úpravy, ak b) sa stane členom skupiny, a to dňom, ktorý predchádza dňu, keď sa stal členom skupiny,

relations:
  Paragraf § 68d -> [OBSAHUJE] -> Paragraf § 68d Odsek 11
  Paragraf § 68d Odsek 11 -> [OBSAHUJE] -> Paragraf § 68d Odsek 11 Pismeno b)

  Paragraf § 68d Odsek 11 Pismeno b) -> [UPRAVUJE] -> Skoncenie Uplatnovania Osobitnej Upravy

  Platitel -> [MA_POVINNOST] -> Skoncenie Uplatnovania Osobitnej Upravy
  Skoncenie Uplatnovania Osobitnej Upravy -> [MA_PODMIENKU] -> Platitel Sa Stane Clenom Skupiny
  Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Dnu Ked Sa Platitel Stal Clenom Skupiny

  Platitel Sa Stane Clenom Skupiny -> [VZTAHUJE_SA_NA] -> Platitel
  Platitel Sa Stane Clenom Skupiny -> [VZTAHUJE_SA_NA] -> Skupina
  Platitel Sa Stane Clenom Skupiny -> [MA_DATUM] -> Den Ked Sa Platitel Stal Clenom Skupiny

  Platitel -> [MA_STATUS] -> Clen Skupiny
  Clen Skupiny -> [VZTAHUJE_SA_NA] -> Skupina

nodes:
  Paragraf: Paragraf § 68d
  Odsek: Paragraf § 68d Odsek 11
  Pismeno: Paragraf § 68d Odsek 11 Pismeno b)

  Subjekt: Platitel
  Organizacia: Skupina
  Status: Clen Skupiny

  Povinnost: Skoncenie Uplatnovania Osobitnej Upravy
  Podmienka: Platitel Sa Stane Clenom Skupiny

  Datum: Den Predchadzajuci Dnu Ked Sa Platitel Stal Clenom Skupiny
  Datum: Den Ked Sa Platitel Stal Clenom Skupiny


---

chunk: 1099
path: ['§ 68d', '14', 'a)']
path_as_text: Paragraf § 68d Odsek 14 Pismeno a)
text: (14) Daňový úrad uloží pokutu do výšky 10 000 eur, ak a) platiteľ uplatňuje osobitnú úpravu a nesplnil podmienky podľa odseku 1,

relations:
  Paragraf § 68d -> [OBSAHUJE] -> Paragraf § 68d Odsek 14
  Paragraf § 68d Odsek 14 -> [OBSAHUJE] -> Paragraf § 68d Odsek 14 Pismeno a)
  Paragraf § 68d -> [OBSAHUJE] -> Paragraf § 68d Odsek 1
  Paragraf § 68d Odsek 14 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 68d Odsek 1

  Paragraf § 68d Odsek 14 Pismeno a) -> [UPRAVUJE] -> Pokuta Do Vysky 10 000 Eur

  Danovy Urad -> [VYDAVA] -> Pokuta Do Vysky 10 000 Eur
  Pokuta Do Vysky 10 000 Eur -> [MA_SUMU] -> Suma Do Vysky 10 000 Eur
  Pokuta Do Vysky 10 000 Eur -> [MA_PODMIENKU] -> Platitel Uplatnuje Osobitnu Upravu A Nesplnil Podmienky Podla Paragraf § 68d Odsek 1

  Platitel Uplatnuje Osobitnu Upravu A Nesplnil Podmienky Podla Paragraf § 68d Odsek 1 -> [VZTAHUJE_SA_NA] -> Platitel
  Platitel Uplatnuje Osobitnu Upravu A Nesplnil Podmienky Podla Paragraf § 68d Odsek 1 -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Platitel Uplatnuje Osobitnu Upravu A Nesplnil Podmienky Podla Paragraf § 68d Odsek 1 -> [VZTAHUJE_SA_NA] -> Nesplnenie Podmienok Podla Paragraf § 68d Odsek 1

  Platitel -> [MA] -> Uplatnovanie Osobitnej Upravy
  Platitel -> [NESPLNA_PODMIENKY] -> Podmienky Podla Paragraf § 68d Odsek 1
  Podmienky Podla Paragraf § 68d Odsek 1 -> [JE_PODLA] -> Paragraf § 68d Odsek 1
  Nesplnenie Podmienok Podla Paragraf § 68d Odsek 1 -> [VZTAHUJE_SA_NA] -> Podmienky Podla Paragraf § 68d Odsek 1

nodes:
  Paragraf: Paragraf § 68d
  Odsek: Paragraf § 68d Odsek 14
  Pismeno: Paragraf § 68d Odsek 14 Pismeno a)
  Odsek: Paragraf § 68d Odsek 1

  Organizacia: Danovy Urad
  Subjekt: Platitel

  Sankcia: Pokuta Do Vysky 10 000 Eur
  Suma: Suma Do Vysky 10 000 Eur

  Konanie: Uplatnovanie Osobitnej Upravy

  Podmienka: Podmienky Podla Paragraf § 68d Odsek 1
  Podmienka: Platitel Uplatnuje Osobitnu Upravu A Nesplnil Podmienky Podla Paragraf § 68d Odsek 1

  Dovod: Nesplnenie Podmienok Podla Paragraf § 68d Odsek 1


---

chunk: 1115
path: ['§ 68f', '6', 'c)']
path_as_text: Paragraf § 68f Odsek 6 Pismeno c)
text: (6) Zdaniteľná osoba, ktorá má na území iného členského štátu sídlo, miesto podnikania, bydlisko alebo sa na území iného členského štátu obvykle zdržiava, nemôže uplatňovať oslobodenie od dane podľa odseku 2, počnúc c) dodaním tovaru alebo služby, ktorým hodnota bez dane dodaných tovarov a služieb touto zdaniteľnou osobou, ktoré sa zahŕňajú do ročného obratu v Únii, presiahla v prebiehajúcom kalendárnom roku 100 000 eur.

relations:
  Paragraf § 68f -> [OBSAHUJE] -> Paragraf § 68f Odsek 6
  Paragraf § 68f Odsek 6 -> [OBSAHUJE] -> Paragraf § 68f Odsek 6 Pismeno c)
  Paragraf § 68f -> [OBSAHUJE] -> Paragraf § 68f Odsek 2
  Paragraf § 68f Odsek 6 Pismeno c) -> [ODKAZUJE_NA] -> Paragraf § 68f Odsek 2
  Zdanitelna Osoba -> [MA_ADRESU] -> Sidlo
  Zdanitelna Osoba -> [MA_ADRESU] -> Miesto Podnikania
  Zdanitelna Osoba -> [MA_ADRESU] -> Bydlisko
  Sidlo -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Miesto Podnikania -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Bydlisko -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Oslobodenie Od Dane Podla Paragrafu § 68f Odsek 2 -> [OSLOBODZUJE_OD] -> Dan
  Oslobodenie Od Dane Podla Paragrafu § 68f Odsek 2 -> [VYPLYVA_Z] -> Paragraf § 68f Odsek 2
  Zdanitelna Osoba -> [NEMA_NAROK_NA] -> Oslobodenie Od Dane Podla Paragrafu § 68f Odsek 2
  Zdanitelna Osoba -> [DODAVA] -> Dodanie Tovaru Alebo Sluzby
  Dodanie Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Sluzba
  Dodanie Tovaru Alebo Sluzby -> [MA_HODNOTU] -> Hodnota Bez Dane Dodanych Tovarov A Sluzieb
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb -> [PATRI_DO] -> Rocny Obrat V Unii
  Rocny Obrat V Unii -> [VZTAHUJE_SA_NA] -> Unia
  Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [VZTAHUJE_SA_NA] -> Hodnota Bez Dane Dodanych Tovarov A Sluzieb
  Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [MA_SUMU] -> 100 000 Eur
  100 000 Eur -> [MA] -> Eur
  Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [MA_OBDOBIE] -> Prebiehajuci Kalendarny Rok
  Zdanitelna Osoba -> [NESPLNA_PODMIENKY] -> Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku
  Paragraf § 68f Odsek 6 Pismeno c) -> [URCUJE] -> Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku

nodes:
  Paragraf: Paragraf § 68f
  Odsek: Paragraf § 68f Odsek 6
  Pismeno: Paragraf § 68f Odsek 6 Pismeno c)
  Odsek: Paragraf § 68f Odsek 2
  Subjekt: Zdanitelna Osoba
  Stat: Iny Clensky Stat
  Adresa: Sidlo
  Adresa: Miesto Podnikania
  Adresa: Bydlisko
  Pravo: Oslobodenie Od Dane Podla Paragrafu § 68f Odsek 2
  Dan: Dan
  Konanie: Dodanie Tovaru Alebo Sluzby
  Tovar: Tovar
  Sluzba: Sluzba
  Suma: Hodnota Bez Dane Dodanych Tovarov A Sluzieb
  Obrat: Rocny Obrat V Unii
  Organizacia: Unia
  Obdobie: Prebiehajuci Kalendarny Rok
  Suma: 100 000 Eur
  Mena: Eur
  Podmienka: Presiahnutie Hodnoty Bez Dane 100 000 Eur V Prebiehajucom Kalendarnom Roku

---

chunk: 1138
path: ['§ 68g', '15', 'a)']
path_as_text: Paragraf § 68g Odsek 15 Pismeno a)
text: (15) Daňový úrad bezodkladne a) rozhodne o odňatí individuálneho identifikačného čísla s príponou EX, ak  1. ročný obrat v Únii malého podniku tuzemskej osoby v prebiehajúcom kalendárnom roku, uvedený vo výkaze podľa odseku 8 alebo podľa odseku 14, presiahol 100 000 eur, 2. malý podnik tuzemskej osoby prestal vykonávať podnikanie podľa § 3, 3. malý podnik tuzemskej osoby už nemá v tuzemsku sídlo, miesto podnikania, bydlisko alebo sa v tuzemsku obvykle nezdržiava, 4. po doručení oznámenia podľa písmena b) malý podnik tuzemskej osoby neuplatňuje osobitnú úpravu na území žiadneho iného členského štátu, 5. sa malý podnik tuzemskej osoby stal členom skupiny podľa § 4b alebo § 4c,

relations:
  Paragraf § 68G -> [OBSAHUJE] -> Paragraf § 68G Odsek 15
  Paragraf § 68G Odsek 15 -> [OBSAHUJE] -> Paragraf § 68G Odsek 15 Pismeno a)
  Paragraf § 68G -> [OBSAHUJE] -> Paragraf § 68G Odsek 8
  Paragraf § 68G -> [OBSAHUJE] -> Paragraf § 68G Odsek 14
  Paragraf § 68G Odsek 15 Pismeno a) -> [VZTAHUJE_SA_NA] -> Danovy Urad
  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Odnati Individualneho Identifikacneho Cisla S Priponou Ex
  Rozhodnutie O Odnati Individualneho Identifikacneho Cisla S Priponou Ex -> [ROZHODUJE_O] -> Individualne Identifikacne Cislo S Priponou Ex
  Maly Podnik Tuzemskej Osoby -> [PATRI_DO] -> Tuzemska Osoba
  Maly Podnik Tuzemskej Osoby -> [MA] -> Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby
  Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby -> [MA_OBDOBIE] -> Prebiehajuci Kalendarny Rok
  Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby -> [MA_HODNOTU] -> Suma 100000 Eur
  Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby -> [VYPLYVA_Z] -> Vykaz Podla Paragrafu § 68G Odsek 8
  Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby -> [VYPLYVA_Z] -> Vykaz Podla Paragrafu § 68G Odsek 14
  Maly Podnik Tuzemskej Osoby -> [ZANIKA] -> Podnikanie Podla Paragrafu § 3
  Maly Podnik Tuzemskej Osoby -> [MA_ADRESU] -> SIdlo V Tuzemsku
  Maly Podnik Tuzemskej Osoby -> [MA_ADRESU] -> Miesto Podnikania V Tuzemsku
  Maly Podnik Tuzemskej Osoby -> [MA_ADRESU] -> Bydlisko V Tuzemsku
  Maly Podnik Tuzemskej Osoby -> [MA_STATUS] -> Obvykle Zdrziavanie Sa V Tuzemsku
  SIdlo V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Podnikania V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Bydlisko V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Obvykle Zdrziavanie Sa V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Maly Podnik Tuzemskej Osoby -> [PRIJIMA] -> Oznamenie Podla Paragrafu § 68G Odsek 15 Pismeno b)
  Maly Podnik Tuzemskej Osoby -> [NEVZTAHUJE_SA_NA] -> Osobitna Uprava
  Osobitna Uprava -> [VZTAHUJE_SA_NA] -> Uzemi Ziadneho Ineho Clenskeho Statu
  Maly Podnik Tuzemskej Osoby -> [PATRI_DO] -> Skupina Podla Paragrafu § 4B
  Maly Podnik Tuzemskej Osoby -> [PATRI_DO] -> Skupina Podla Paragrafu § 4C

nodes:
  Organizacia: Danovy Urad
  Paragraf: Paragraf § 68G
  Odsek: Paragraf § 68G Odsek 15
  Pismeno: Paragraf § 68G Odsek 15 Pismeno a)
  Pismeno: Paragraf § 68G Odsek 15 Pismeno b)
  Odsek: Paragraf § 68G Odsek 8
  Odsek: Paragraf § 68G Odsek 14
  Paragraf: Paragraf § 3
  Paragraf: Paragraf § 4B
  Paragraf: Paragraf § 4C
  Rozhodnutie: Rozhodnutie O Odnati Individualneho Identifikacneho Cisla S Priponou Ex
  Zaznam: Individualne Identifikacne Cislo S Priponou Ex
  Subjekt: Maly Podnik Tuzemskej Osoby
  Osoba: Tuzemska Osoba
  Obrat: Rocny Obrat V Unii Maleho Podniku Tuzemskej Osoby
  Obdobie: Prebiehajuci Kalendarny Rok
  Suma: Suma 100000 Eur
  Dokument: Vykaz Podla Paragrafu § 68G Odsek 8
  Dokument: Vykaz Podla Paragrafu § 68G Odsek 14
  Konanie: Podnikanie Podla Paragrafu § 3
  Adresa: SIdlo V Tuzemsku
  Adresa: Miesto Podnikania V Tuzemsku
  Adresa: Bydlisko V Tuzemsku
  Status: Obvykle Zdrziavanie Sa V Tuzemsku
  Stat: Tuzemsko
  Oznamenie: Oznamenie Podla Paragrafu § 68G Odsek 15 Pismeno b)
  Konanie: Osobitna Uprava
  Lokacia: Uzemi Ziadneho Ineho Clenskeho Statu
  Organizacia: Skupina Podla Paragrafu § 4B
  Organizacia: Skupina Podla Paragrafu § 4C

---

chunk: 1140
path: ['§ 68g', '16']
path_as_text: Paragraf § 68g Odsek 16
text: (16) Na účely uplatňovania tejto osobitnej úpravy sa písomnosti doručujú elektronickými prostriedkami spôsobom podľa osobitného predpisu.28ag)

relations:
  Paragraf § 68g -> [OBSAHUJE] -> Paragraf § 68g Odsek 16
  Paragraf § 68g Odsek 16 -> [UPRAVUJE] -> Uplatnovanie Osobitnej Upravy
  Dorucovanie Pisomnosti Elektronickymi Prostriedkami -> [VZTAHUJE_SA_NA] -> Pisomnosti
  Dorucovanie Pisomnosti Elektronickymi Prostriedkami -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy

nodes:
  Paragraf: Paragraf § 68g
  Odsek: Paragraf § 68g Odsek 16
  Konanie: Uplatnovanie Osobitnej Upravy
  Dokument: Pisomnosti
  Povinnost: Dorucovanie Pisomnosti Elektronickymi Prostriedkami
  PravnyPredpis: Osobitny Predpis 28ag)

---

chunk: 1161
path: ['§ 69', '12', 'h)']
path_as_text: Paragraf § 69 Odsek 12 Pismeno h)
text: (12) Platiteľ, ktorý má pridelené identifikačné číslo pre daň podľa § 4, § 4b, § 4c alebo § 5 a ktorý je príjemcom plnenia od iného platiteľa, ktorý má pridelené identifikačné číslo pre daň podľa § 4, § 4b, § 4c alebo § 5, je povinný platiť daň vzťahujúcu sa na h) dodanie mobilných telefónov, ktoré sú vyrobené alebo prispôsobené na použitie v spojení s licencovanou sieťou a fungujú na stanovených frekvenciách bez ohľadu na to, či majú alebo nemajú iné využitie, ak základ dane vo faktúre za dodanie mobilných telefónov je 5 000 eur a viac,

relations:
  Paragraf § 69 -> [OBSAHUJE] -> Paragraf § 69 Odsek 12
  Paragraf § 69 Odsek 12 -> [OBSAHUJE] -> Paragraf § 69 Odsek 12 Pismeno h)
  Platitel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Iny Platitel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 4
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 4B
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 4C
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 5
  Platitel -> [JE_TYPOM] -> Prijemca Plnenia
  Prijemca Plnenia -> [PRIJIMA] -> Iny Platitel
  Platitel -> [MA_POVINNOST] -> Povinnost Platit Dan
  Povinnost Platit Dan -> [VZTAHUJE_SA_NA] -> Dan
  Platitel -> [PLATI] -> Dan
  Dan -> [VZTAHUJE_SA_NA] -> Dodanie Mobilnych Telefonov
  Paragraf § 69 Odsek 12 Pismeno h) -> [VZTAHUJE_SA_NA] -> Dodanie Mobilnych Telefonov
  Dodanie Mobilnych Telefonov -> [VZTAHUJE_SA_NA] -> Mobilne Telefony
  Mobilne Telefony -> [VZTAHUJE_SA_NA] -> Licencovana Siet
  Mobilne Telefony -> [VZTAHUJE_SA_NA] -> Stanovene Frekvencie
  Faktura Za Dodanie Mobilnych Telefonov -> [VZTAHUJE_SA_NA] -> Dodanie Mobilnych Telefonov
  Faktura Za Dodanie Mobilnych Telefonov -> [MA_SUMU] -> Zaklad Dane Vo Fakture Za Dodanie Mobilnych Telefonov
  Povinnost Platit Dan -> [MA_PODMIENKU] -> Zaklad Dane Vo Fakture Za Dodanie Mobilnych Telefonov

nodes:
  Paragraf: Paragraf § 69
  Odsek: Paragraf § 69 Odsek 12
  Pismeno: Paragraf § 69 Odsek 12 Pismeno h)
  Paragraf: Paragraf § 4
  Paragraf: Paragraf § 4B
  Paragraf: Paragraf § 4C
  Paragraf: Paragraf § 5
  Subjekt: Platitel
  Subjekt: Iny Platitel
  Subjekt: Prijemca Plnenia
  Zaznam: Identifikacne Cislo Pre Dan
  Povinnost: Povinnost Platit Dan
  Dan: Dan
  Tovar: Dodanie Mobilnych Telefonov
  Tovar: Mobilne Telefony
  Sluzba: Licencovana Siet
  Zaznam: Stanovene Frekvencie
  Dokument: Faktura Za Dodanie Mobilnych Telefonov
  Suma: Zaklad Dane Vo Fakture Za Dodanie Mobilnych Telefonov

---

chunk: 1184
path: ['§ 69a', '9']
path_as_text: Paragraf § 69a Odsek 9
text: (9) Proti rozhodnutiu, ktorým Daňový úrad Bratislava zrušil daňovému zástupcovi osobitné identifikačné číslo pre daň podľa odseku 8 písm. b), nemožno podať odvolanie.

relations:
  Paragraf § 69a -> [OBSAHUJE] -> Paragraf § 69a Odsek 9
  Paragraf § 69a -> [OBSAHUJE] -> Paragraf § 69a Odsek 8
  Paragraf § 69a Odsek 8 -> [OBSAHUJE] -> Paragraf § 69a Odsek 8 Pismeno b)
  Rozhodnutie O Zruseni Osobitneho Identifikacneho Cisla Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 69a Odsek 8 Pismeno b)
  Danovy Urad Bratislava -> [VYDAVA] -> Rozhodnutie O Zruseni Osobitneho Identifikacneho Cisla Pre Dan
  Rozhodnutie O Zruseni Osobitneho Identifikacneho Cisla Pre Dan -> [RUSI] -> Osobitne Identifikacne Cislo Pre Dan
  Danovy Zastupca -> [MA] -> Osobitne Identifikacne Cislo Pre Dan
  Rozhodnutie O Zruseni Osobitneho Identifikacneho Cisla Pre Dan -> [NEMA_NAROK_NA] -> Podanie Odvolania

nodes:
  Paragraf: Paragraf § 69a
  Odsek: Paragraf § 69a Odsek 9
  Odsek: Paragraf § 69a Odsek 8
  Pismeno: Paragraf § 69a Odsek 8 Pismeno b)
  Rozhodnutie: Rozhodnutie O Zruseni Osobitneho Identifikacneho Cisla Pre Dan
  Organizacia: Danovy Urad Bratislava
  Subjekt: Danovy Zastupca
  Registracia: Osobitne Identifikacne Cislo Pre Dan
  Pravo: Podanie Odvolania

---

chunk: 1207
path: ['§ 70', '2', 'g)']
path_as_text: Paragraf § 70 Odsek 2 Pismeno g)
text: (2) Platiteľ vedie podrobné záznamy podľa jednotlivých zdaňovacích období o g) premiestnení tovaru, vrátení tovaru alebo nahradení zdaniteľnej osoby podľa § 8a ods. 1 písm.

relations:
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 2
  Paragraf § 70 Odsek 2 -> [OBSAHUJE] -> Paragraf § 70 Odsek 2 Pismeno g)
  Paragraf § 8a -> [OBSAHUJE] -> Paragraf § 8a Odsek 1
  Platitel -> [UCHOVAVA] -> Podrobne Zaznamy
  Podrobne Zaznamy -> [MA_OBDOBIE] -> Jednotlive Zdanovacie Obdobia
  Podrobne Zaznamy -> [VZTAHUJE_SA_NA] -> Premiestnenie Tovaru
  Podrobne Zaznamy -> [VZTAHUJE_SA_NA] -> Vratenie Tovaru
  Podrobne Zaznamy -> [VZTAHUJE_SA_NA] -> Nahradenie Zdanitelnej Osoby
  Premiestnenie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Vratenie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Nahradenie Zdanitelnej Osoby -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Nahradenie Zdanitelnej Osoby -> [VYPLYVA_Z] -> Paragraf § 8a Odsek 1
  Podrobne Zaznamy -> [VYPLYVA_Z] -> Paragraf § 70 Odsek 2 Pismeno g)

nodes:
  Subjekt: Platitel
  Zaznam: Podrobne Zaznamy
  ZdanovacieObdobie: Jednotlive Zdanovacie Obdobia
  Konanie: Premiestnenie Tovaru
  Konanie: Vratenie Tovaru
  Konanie: Nahradenie Zdanitelnej Osoby
  Tovar: Tovar
  Subjekt: Zdanitelna Osoba
  Paragraf: Paragraf § 70
  Odsek: Paragraf § 70 Odsek 2
  Pismeno: Paragraf § 70 Odsek 2 Pismeno g)
  Paragraf: Paragraf § 8a
  Odsek: Paragraf § 8a Odsek 1

---

chunk: 1230
path: ['§ 70', '11', 'a)']
path_as_text: Paragraf § 70 Odsek 11 Pismeno a)
text: (11) Záznamy podľa a) odsekov 1 až 6 a 10 sa uchovávajú do konca kalendárneho roka, v ktorom uplynie desať rokov od skončenia roka, ktorého sa týkajú,

relations:
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 11
  Paragraf § 70 Odsek 11 -> [OBSAHUJE] -> Paragraf § 70 Odsek 11 Pismeno a)
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 1
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 2
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 3
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 4
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 5
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 6
  Paragraf § 70 -> [OBSAHUJE] -> Paragraf § 70 Odsek 10
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 1
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 2
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 3
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 4
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 5
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 6
  Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10 -> [ODKAZUJE_NA] -> Paragraf § 70 Odsek 10
  Paragraf § 70 Odsek 11 Pismeno a) -> [URCUJE] -> Uchovavanie Zaznamov
  Uchovavanie Zaznamov -> [VZTAHUJE_SA_NA] -> Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10
  Uchovavanie Zaznamov -> [MA_LEHOTU] -> Do Konca Kalendarneho Roka V Ktorom Uplynie Desat Rokov Od Skoncenia Roka Ktoreho Sa Tykaju

nodes:
  Paragraf: Paragraf § 70
  Odsek: Paragraf § 70 Odsek 11
  Pismeno: Paragraf § 70 Odsek 11 Pismeno a)
  Odsek: Paragraf § 70 Odsek 1
  Odsek: Paragraf § 70 Odsek 2
  Odsek: Paragraf § 70 Odsek 3
  Odsek: Paragraf § 70 Odsek 4
  Odsek: Paragraf § 70 Odsek 5
  Odsek: Paragraf § 70 Odsek 6
  Odsek: Paragraf § 70 Odsek 10
  Zaznam: Zaznamy Podla Paragrafu § 70 Odsek 1 Az 6 A 10
  Povinnost: Uchovavanie Zaznamov
  Lehota: Do Konca Kalendarneho Roka V Ktorom Uplynie Desat Rokov Od Skoncenia Roka Ktoreho Sa Tykaju

---

chunk: 1253
path: ['§ 70a', '8', 'b)']
path_as_text: Paragraf § 70a Odsek 8 Pismeno b)
text: (8) Záznamy podľa odseku 2, ktoré je povinný viesť tuzemský poskytovateľ platobných služieb podľa odseku 3, musia obsahovať b) meno a priezvisko príjemcu platby alebo obchodné meno alebo názov príjemcu platby, uvedené v záznamoch poskytovateľa platobných služieb,

relations:
  Paragraf § 70a -> [OBSAHUJE] -> Paragraf § 70a Odsek 8
  Paragraf § 70a Odsek 8 -> [OBSAHUJE] -> Paragraf § 70a Odsek 8 Pismeno b)
  Tuzemsky Poskytovatel Platobnych Sluzieb -> [MA_POVINNOST] -> Povinnost Viest Zaznamy
  Povinnost Viest Zaznamy -> [VZTAHUJE_SA_NA] -> Zaznamy Podla Paragrafu § 70a Odsek 2
  Zaznamy Podla Paragrafu § 70a Odsek 2 -> [OBSAHUJE] -> Prijemca Platby

nodes:
  Paragraf: Paragraf § 70a
  Odsek: Paragraf § 70a Odsek 8
  Pismeno: Paragraf § 70a Odsek 8 Pismeno b)
  Odsek: Paragraf § 70a Odsek 2
  Odsek: Paragraf § 70a Odsek 3
  Zaznam: Zaznamy Podla Paragrafu § 70a Odsek 2
  Organizacia: Tuzemsky Poskytovatel Platobnych Sluzieb
  Subjekt: Prijemca Platby
  Povinnost: Povinnost Viest Zaznamy

---

chunk: 1276
path: ['§ 71', '2']
path_as_text: Paragraf § 71 Odsek 2
text: (2) Za faktúru sa považuje aj každý doklad alebo oznámenie, ktoré mení pôvodnú faktúru a osobitne a jednoznačne sa na ňu vzťahuje. Faktúrou podľa prvej vety nie je opravný doklad podľa § 25a.

relations:
  Paragraf § 71 -> [OBSAHUJE] -> Paragraf § 71 Odsek 2
  Paragraf § 71 Odsek 2 -> [DEFINUJE] -> Faktura
  Doklad Meniaci Povodnu Fakturu -> [JE_TYPOM] -> Faktura
  Oznamenie Meniace Povodnu Fakturu -> [JE_TYPOM] -> Faktura
  Doklad Meniaci Povodnu Fakturu -> [MENI] -> Povodna Faktura
  Oznamenie Meniace Povodnu Fakturu -> [MENI] -> Povodna Faktura
  Doklad Meniaci Povodnu Fakturu -> [VZTAHUJE_SA_NA] -> Povodna Faktura
  Oznamenie Meniace Povodnu Fakturu -> [VZTAHUJE_SA_NA] -> Povodna Faktura
  Opravny Doklad -> [VYPLYVA_Z] -> Paragraf § 25A
  Opravny Doklad -> [NEVZTAHUJE_SA_NA] -> Faktura

nodes:
  Paragraf: Paragraf § 71
  Odsek: Paragraf § 71 Odsek 2
  Paragraf: Paragraf § 25A
  Dokument: Faktura
  Dokument: Povodna Faktura
  Dokument: Doklad Meniaci Povodnu Fakturu
  Oznamenie: Oznamenie Meniace Povodnu Fakturu
  Dokument: Opravny Doklad

---

chunk: 1299
path: ['§ 73', '1', 'd)']
path_as_text: Paragraf § 73 Odsek 1 Pismeno d)
text: (1) Faktúra podľa § 72 musí byť vyhotovená do 15 dní d) od konca kalendárneho mesiaca, v ktorom bola dodaná služba alebo prijatá platba pred dodaním služby s miestom dodania podľa § 15 ods. 1 v inom členskom štáte,

relations:
  Paragraf § 73 -> [OBSAHUJE] -> Paragraf § 73 Odsek 1
  Paragraf § 73 Odsek 1 -> [OBSAHUJE] -> Paragraf § 73 Odsek 1 Pismeno d)
  Paragraf § 15 -> [OBSAHUJE] -> Paragraf § 15 Odsek 1
  Paragraf § 73 Odsek 1 Pismeno d) -> [UPRAVUJE] -> Vyhotovenie Faktury
  Vyhotovenie Faktury -> [VZTAHUJE_SA_NA] -> Faktura
  Faktura -> [ODKAZUJE_NA] -> Paragraf § 72
  Vyhotovenie Faktury -> [MA_LEHOTU] -> Lehota 15 Dni
  Lehota 15 Dni -> [VYPLYVA_Z] -> Koniec Kalendarneho Mesiaca
  Koniec Kalendarneho Mesiaca -> [MA_OBDOBIE] -> Kalendarneho Mesiaca
  Kalendarneho Mesiaca -> [VZTAHUJE_SA_NA] -> Dodanie Sluzby
  Dodanie Sluzby -> [DODAVA] -> Sluzba
  Kalendarneho Mesiaca -> [VZTAHUJE_SA_NA] -> Prijatie Platby Pred Dodanim Sluzby
  Prijatie Platby Pred Dodanim Sluzby -> [PRIJIMA] -> Platba
  Prijatie Platby Pred Dodanim Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Sluzby
  Sluzba -> [MA] -> Miesto Dodania Sluzby
  Miesto Dodania Sluzby -> [ODKAZUJE_NA] -> Paragraf § 15 Odsek 1
  Miesto Dodania Sluzby -> [NACHADZA_SA_V] -> Iny Clensky Stat

nodes:
  Dokument: Faktura
  Paragraf: Paragraf § 72
  Paragraf: Paragraf § 73
  Odsek: Paragraf § 73 Odsek 1
  Pismeno: Paragraf § 73 Odsek 1 Pismeno d)
  Paragraf: Paragraf § 15
  Odsek: Paragraf § 15 Odsek 1
  Povinnost: Vyhotovenie Faktury
  Lehota: Lehota 15 Dni
  Datum: Koniec Kalendarneho Mesiaca
  Obdobie: Kalendarneho Mesiaca
  Konanie: Dodanie Sluzby
  Sluzba: Sluzba
  Konanie: Prijatie Platby Pred Dodanim Sluzby
  Platba: Platba
  Lokacia: Miesto Dodania Sluzby
  Stat: Iny Clensky Stat

---

chunk: 1322
path: ['§ 74', '5']
path_as_text: Paragraf § 74 Odsek 5
text: (5) Za identifikačné číslo pre daň sa na účely vyhotovenia zjednodušenej faktúry podľa odseku 3 písm. b) považuje aj daňové identifikačné číslo, ktoré bolo platiteľovi pridelené podľa osobitného predpisu,29aaa) alebo identifikačné číslo, ktoré bolo platiteľovi pridelené podľa osobitného predpisu,29aab) ak ku dňu vyhotovenia tejto zjednodušenej faktúry platiteľ, ktorý splnil registračnú povinnosť, nemá pridelené identifikačné číslo pre daň podľa § 4 alebo § 5.

relations:
  Paragraf § 74 -> [OBSAHUJE] -> Paragraf § 74 Odsek 5
  Paragraf § 74 -> [OBSAHUJE] -> Paragraf § 74 Odsek 3
  Paragraf § 74 Odsek 3 -> [OBSAHUJE] -> Paragraf § 74 Odsek 3 Pismeno b)
  Zjednodusena Faktura -> [ODKAZUJE_NA] -> Paragraf § 74 Odsek 3 Pismeno b)
  Vyhotovenie Zjednodusenej Faktury -> [VZTAHUJE_SA_NA] -> Zjednodusena Faktura
  Danove Identifikacne Cislo -> [JE_TYPOM] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo -> [JE_TYPOM] -> Identifikacne Cislo Pre Dan
  Danove Identifikacne Cislo -> [VZTAHUJE_SA_NA] -> Platitel
  Identifikacne Cislo -> [VZTAHUJE_SA_NA] -> Platitel
  Danove Identifikacne Cislo -> [VYPLYVA_Z] -> Osobitny Predpis 29aaa)
  Identifikacne Cislo -> [VYPLYVA_Z] -> Osobitny Predpis 29aab)
  Platitel -> [MA_POVINNOST] -> Registracna Povinnost
  Platitel -> [SPLNA_PODMIENKY] -> Splnenie Registracnej Povinnosti
  Nepridelenie Identifikacneho Cisla Pre Dan Ku Dnu Vyhotovenia Zjednodusenej Faktury -> [MA_DATUM] -> Den Vyhotovenia Zjednodusenej Faktury
  Nepridelenie Identifikacneho Cisla Pre Dan Ku Dnu Vyhotovenia Zjednodusenej Faktury -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 4
  Identifikacne Cislo Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 5
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Vyhotovenie Zjednodusenej Faktury

nodes:
  Paragraf: Paragraf § 74
  Odsek: Paragraf § 74 Odsek 5
  Odsek: Paragraf § 74 Odsek 3
  Pismeno: Paragraf § 74 Odsek 3 Pismeno b)
  Paragraf: Paragraf § 4
  Paragraf: Paragraf § 5
  Dokument: Zjednodusena Faktura
  Konanie: Vyhotovenie Zjednodusenej Faktury
  Zaznam: Identifikacne Cislo Pre Dan
  Zaznam: Danove Identifikacne Cislo
  Zaznam: Identifikacne Cislo
  Subjekt: Platitel
  Povinnost: Registracna Povinnost
  Podmienka: Splnenie Registracnej Povinnosti
  Podmienka: Nepridelenie Identifikacneho Cisla Pre Dan Ku Dnu Vyhotovenia Zjednodusenej Faktury
  Datum: Den Vyhotovenia Zjednodusenej Faktury
  PravnyPredpis: Osobitny Predpis 29aaa)
  PravnyPredpis: Osobitny Predpis 29aab)

---

chunk: 1345
path: ['§ 76a', '5']
path_as_text: Paragraf § 76a Odsek 5
text: (5) Finančné riaditeľstvo vedie evidenciu európskych doručovacích štandardov a certifikovaných poskytovateľov doručovacej služby pôsobiacich na území Slovenskej republiky a zverejňuje ju prostredníctvom svojho webového sídla.

relations:
  Paragraf § 76A -> [OBSAHUJE] -> Paragraf § 76A Odsek 5
  Financne Riaditelstvo -> [UCHOVAVA] -> Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby
  Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby -> [OBSAHUJE] -> Europske Dorucovacie Standardy
  Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby -> [OBSAHUJE] -> Certifikovani Poskytovatelia Dorucovacej Sluzby
  Certifikovani Poskytovatelia Dorucovacej Sluzby -> [POSKYTUJE] -> Dorucovacia Sluzba
  Certifikovani Poskytovatelia Dorucovacej Sluzby -> [NACHADZA_SA_V] -> Slovenska Republika
  Financne Riaditelstvo -> [VYDAVA] -> Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby
  Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby -> [NACHADZA_SA_V] -> Webove Sidlo Financneho Riaditelstva

nodes:
  Paragraf: Paragraf § 76A
  Odsek: Paragraf § 76A Odsek 5
  Organizacia: Financne Riaditelstvo
  Zaznam: Evidencia Europskych Dorucovacich Standardov A Certifikovanych Poskytovatelov Dorucovacej Sluzby
  Dokument: Europske Dorucovacie Standardy
  Organizacia: Certifikovani Poskytovatelia Dorucovacej Sluzby
  Sluzba: Dorucovacia Sluzba
  Stat: Slovenska Republika
  Dokument: Webove Sidlo Financneho Riaditelstva

---

chunk: 1368
path: ['§ 78', '2', 'a)']
path_as_text: Paragraf § 78 Odsek 2 Pismeno a)
text: (2) Platiteľ, ktorý a) má sídlo, miesto podnikania alebo prevádzkareň v tuzemsku, a ak nemá takéto miesto, ale má bydlisko v tuzemsku alebo sa v tuzemsku obvykle zdržiava, je povinný podať daňové priznanie do 25 dní po skončení každého zdaňovacieho obdobia a v tej istej lehote je povinný zaplatiť vlastnú daňovú povinnosť, okrem platiteľa, ktorý  1. splnil registračnú povinnosť podľa § 4, a ktorému nebolo do uplynutia lehoty na podanie daňového priznania za prvé zdaňovacie obdobie pridelené identifikačné číslo pre daň podľa § 4, ktorý je povinný podať daňové priznanie a zaplatiť vlastnú daňovú povinnosť do piatich  pracovných dní odo dňa doručenia rozhodnutia o registrácii pre daň podľa § 4 ods. 4, 2. uskutočňuje v tuzemsku výlučne plnenia oslobodené od dane podľa § 28 až 42, na ktorého sa povinnosť podávať daňové priznanie vzťahuje počnúc zdaňovacím obdobím, v ktorom mu vznikne povinnosť platiť daň zo zdaniteľného obchodu, ktorý nie je oslobodený od dane podľa § 28 až 42, alebo počnúc prvým zdaňovacím obdobím, ak sa stal platiteľom 1. januára kalendárneho roka nasledujúceho po kalendárnom roku, za ktorý presiahol obrat podľa § 4 ods. 1 písm. a) výlučne z dodaní tovarov a služieb, ktoré sú oslobodené od dane podľa § 37 až 39, a od tohto presiahnutia obratu do 31. decembra prebiehajúceho kalendárneho roka dodal v tuzemsku tovar alebo službu, ktorá nie je oslobodená od dane podľa § 28 až 42,

relations:


nodes:


---

chunk: 1391
path: ['§ 78a', '4', 'g)']
path_as_text: Paragraf § 78a Odsek 4 Pismeno g)
text: (4) Z faktúr uvedených v odseku 2 alebo z dokladu podľa odseku 3 sa v kontrolnom výkaze uvádzajú tieto údaje: g) druh a množstvo tovaru, ak je faktúra vyhotovená o dodaní tovaru, z ktorého je povinný platiť daň príjemca plnenia podľa § 69 ods. 12 písm. h) a i).

relations:
  Paragraf § 78a -> [OBSAHUJE] -> Paragraf § 78a Odsek 2
  Paragraf § 78a -> [OBSAHUJE] -> Paragraf § 78a Odsek 3
  Paragraf § 78a -> [OBSAHUJE] -> Paragraf § 78a Odsek 4
  Paragraf § 78a Odsek 4 -> [OBSAHUJE] -> Paragraf § 78a Odsek 4 Pismeno g)
  Paragraf § 69 -> [OBSAHUJE] -> Paragraf § 69 Odsek 12
  Paragraf § 69 Odsek 12 -> [OBSAHUJE] -> Paragraf § 69 Odsek 12 Pismeno h)
  Paragraf § 69 Odsek 12 -> [OBSAHUJE] -> Paragraf § 69 Odsek 12 Pismeno i)
  Kontrolny Vykaz -> [OBSAHUJE] -> Druh Tovaru
  Kontrolny Vykaz -> [OBSAHUJE] -> Mnozstvo Tovaru
  Faktura -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Druh Tovaru
  Dodanie Tovaru -> [MA_HODNOTU] -> Mnozstvo Tovaru
  Prijemca Plnenia -> [MA_POVINNOST] -> Povinnost Platit Dan
  Povinnost Platit Dan -> [VZTAHUJE_SA_NA] -> Dan

nodes:
  Paragraf: Paragraf § 78a
  Odsek: Paragraf § 78a Odsek 2
  Odsek: Paragraf § 78a Odsek 3
  Odsek: Paragraf § 78a Odsek 4
  Pismeno: Paragraf § 78a Odsek 4 Pismeno g)
  Paragraf: Paragraf § 69
  Odsek: Paragraf § 69 Odsek 12
  Pismeno: Paragraf § 69 Odsek 12 Pismeno h)
  Pismeno: Paragraf § 69 Odsek 12 Pismeno i)
  Dokument: Faktura
  Dokument: Doklad
  Dokument: Kontrolny Vykaz
  Tovar: Druh Tovaru
  Mnozstvo: Mnozstvo Tovaru
  Konanie: Dodanie Tovaru
  Subjekt: Prijemca Plnenia
  Dan: Dan
  Povinnost: Povinnost Platit Dan

---

chunk: 1414
path: ['§ 79', '3']
path_as_text: Paragraf § 79 Odsek 3
text: (3) Platiteľ, ktorý spĺňa podmienky podľa odseku 2, vyznačí túto skutočnosť v daňovom priznaní za zdaňovacie obdobie, v ktorom nadmerný odpočet vznikol.

relations:
  Paragraf § 79 -> [OBSAHUJE] -> Paragraf § 79 Odsek 3
  Paragraf § 79 -> [OBSAHUJE] -> Paragraf § 79 Odsek 2
  Platitel -> [SPLNA_PODMIENKY] -> Podmienky Podla Paragraf § 79 Odsek 2
  Podmienky Podla Paragraf § 79 Odsek 2 -> [VYPLYVA_Z] -> Paragraf § 79 Odsek 2
  Platitel -> [MA_POVINNOST] -> Vyznačenie Splnenia Podmienok V Danovom Priznani
  Vyznačenie Splnenia Podmienok V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie Vzniku Nadmerneho Odpocetu
  Nadmerny Odpocet -> [MA_OBDOBIE] -> Zdanovacie Obdobie Vzniku Nadmerneho Odpocetu

nodes:
  Subjekt: Platitel
  Paragraf: Paragraf § 79
  Odsek: Paragraf § 79 Odsek 3
  Odsek: Paragraf § 79 Odsek 2
  Podmienka: Podmienky Podla Paragraf § 79 Odsek 2
  Povinnost: Vyznačenie Splnenia Podmienok V Danovom Priznani
  DanovePriznanie: Danove Priznanie
  ZdanovacieObdobie: Zdanovacie Obdobie Vzniku Nadmerneho Odpocetu
  NadmernyOdpocet: Nadmerny Odpocet

---

chunk: 1437
path: ['§ 80', '2']
path_as_text: Paragraf § 80 Odsek 2
text: (2) Platiteľ môže podať súhrnný výkaz za kalendárny štvrťrok, ak hodnota tovarov podľa odseku 1 písm. a) až c) nepresiahne v príslušnom kalendárnom štvrťroku a súčasne v predchádzajúcich štyroch kalendárnych štvrťrokoch hodnotu 50 000 eur; možnosť podať súhrnný výkaz za kalendárny štvrťrok prestáva platiť od skončenia kalendárneho mesiaca, v ktorom hodnota tovarov podľa odseku 1 písm. a) až c) presiahne v príslušnom kalendárnom štvrťroku hodnotu 50 000 eur, a platiteľ je povinný podať súhrnný výkaz osobitne za každý kalendárny mesiac príslušného kalendárneho štvrťroka.

relations:
  Paragraf § 80 -> [OBSAHUJE] -> Paragraf § 80 Odsek 2
  Paragraf § 80 -> [OBSAHUJE] -> Paragraf § 80 Odsek 1
  Paragraf § 80 Odsek 1 -> [OBSAHUJE] -> Paragraf § 80 Odsek 1 Pismeno a)
  Paragraf § 80 Odsek 1 -> [OBSAHUJE] -> Paragraf § 80 Odsek 1 Pismeno b)
  Paragraf § 80 Odsek 1 -> [OBSAHUJE] -> Paragraf § 80 Odsek 1 Pismeno c)
  Platitel -> [MA_PRAVO] -> Podanie Suhrnneho Vykazu Za Kalendarny Stvrtrok
  Podanie Suhrnneho Vykazu Za Kalendarny Stvrtrok -> [VZTAHUJE_SA_NA] -> Suhrnny Vykaz
  Podanie Suhrnneho Vykazu Za Kalendarny Stvrtrok -> [MA_OBDOBIE] -> Kalendarny Stvrtrok
  Podanie Suhrnneho Vykazu Za Kalendarny Stvrtrok -> [MA_PODMIENKU] -> Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur -> [VZTAHUJE_SA_NA] -> Tovary Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c)
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur -> [MA_SUMU] -> 50000 Eur
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur -> [MA_OBDOBIE] -> Prislusny Kalendarny Stvrtrok
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur -> [MA_OBDOBIE] -> Predchadzajuce Styri Kalenderne Stvrtroky
  Moznost Podat Suhrnny Vykaz Za Kalendarny Stvrtrok -> [ZANIKA] -> Skoncenie Kalenderneho Mesiaca
  Moznost Podat Suhrnny Vykaz Za Kalendarny Stvrtrok -> [MA_PODMIENKU] -> Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Presiahne 50000 Eur
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Presiahne 50000 Eur -> [VZTAHUJE_SA_NA] -> Tovary Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c)
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Presiahne 50000 Eur -> [MA_SUMU] -> 50000 Eur
  Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Presiahne 50000 Eur -> [MA_OBDOBIE] -> Prislusny Kalendarny Stvrtrok
  Platitel -> [MA_POVINNOST] -> Povinnost Podat Suhrnny Vykaz Za Kazdy Kalendarny Mesiac
  Povinnost Podat Suhrnny Vykaz Za Kazdy Kalendarny Mesiac -> [VZTAHUJE_SA_NA] -> Suhrnny Vykaz
  Povinnost Podat Suhrnny Vykaz Za Kazdy Kalendarny Mesiac -> [MA_OBDOBIE] -> Kalendarny Mesiac
  Povinnost Podat Suhrnny Vykaz Za Kazdy Kalendarny Mesiac -> [MA_OBDOBIE] -> Prislusny Kalendarny Stvrtrok

nodes:
  Paragraf: Paragraf § 80
  Odsek: Paragraf § 80 Odsek 2
  Odsek: Paragraf § 80 Odsek 1
  Pismeno: Paragraf § 80 Odsek 1 Pismeno a)
  Pismeno: Paragraf § 80 Odsek 1 Pismeno b)
  Pismeno: Paragraf § 80 Odsek 1 Pismeno c)
  Subjekt: Platitel
  Dokument: Suhrnny Vykaz
  Pravo: Podanie Suhrnneho Vykazu Za Kalendarny Stvrtrok
  Povinnost: Povinnost Podat Suhrnny Vykaz Za Kazdy Kalendarny Mesiac
  Podmienka: Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Nepresiahne 50000 Eur
  Podmienka: Hodnota Tovarov Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c) Presiahne 50000 Eur
  Tovar: Tovary Podla Paragrafu § 80 Odsek 1 Pismeno a) Az c)
  Obdobie: Kalendarny Stvrtrok
  Obdobie: Prislusny Kalendarny Stvrtrok
  Obdobie: Predchadzajuce Styri Kalenderne Stvrtroky
  Obdobie: Kalendarny Mesiac
  Datum: Skoncenie Kalenderneho Mesiaca
  Suma: 50000 Eur
  Pravo: Moznost Podat Suhrnny Vykaz Za Kalendarny Stvrtrok

---

chunk: 1460
path: ['§ 81', '7']
path_as_text: Paragraf § 81 Odsek 7
text: (7) Daňová povinnosť podľa odseku 5 nevzniká pri zániku platiteľa bez likvidácie, keď právny nástupca je platiteľom alebo sa stáva platiteľom podľa § 4 ods. 1 písm. c) alebo podľa § 5 ods. 4 písm. a).

relations:
  Paragraf § 81 -> [OBSAHUJE] -> Paragraf § 81 Odsek 7
  Paragraf § 81 -> [OBSAHUJE] -> Paragraf § 81 Odsek 5
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1
  Paragraf § 4 Odsek 1 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1 Pismeno c)
  Paragraf § 5 -> [OBSAHUJE] -> Paragraf § 5 Odsek 4
  Paragraf § 5 Odsek 4 -> [OBSAHUJE] -> Paragraf § 5 Odsek 4 Pismeno a)
  Danova Povinnost Podla Paragrafu § 81 Odsek 5 -> [VYPLYVA_Z] -> Paragraf § 81 Odsek 5
  Danova Povinnost Podla Paragrafu § 81 Odsek 5 -> [NEVZTAHUJE_SA_NA] -> Zanik Platitela Bez Likvidacie
  Zanik Platitela Bez Likvidacie -> [VZTAHUJE_SA_NA] -> Platitel
  Pravny Nastupca -> [JE_TYPOM] -> Platitel
  Pravny Nastupca -> [REGISTRUJE] -> Statie Sa Platitelom Podla Paragrafu § 4 Odsek 1 Pismeno c)
  Statie Sa Platitelom Podla Paragrafu § 4 Odsek 1 Pismeno c) -> [VYPLYVA_Z] -> Paragraf § 4 Odsek 1 Pismeno c)
  Pravny Nastupca -> [REGISTRUJE] -> Statie Sa Platitelom Podla Paragrafu § 5 Odsek 4 Pismeno a)
  Statie Sa Platitelom Podla Paragrafu § 5 Odsek 4 Pismeno a) -> [VYPLYVA_Z] -> Paragraf § 5 Odsek 4 Pismeno a)

nodes:
  Paragraf: Paragraf § 81
  Odsek: Paragraf § 81 Odsek 7
  Odsek: Paragraf § 81 Odsek 5
  Paragraf: Paragraf § 4
  Odsek: Paragraf § 4 Odsek 1
  Pismeno: Paragraf § 4 Odsek 1 Pismeno c)
  Paragraf: Paragraf § 5
  Odsek: Paragraf § 5 Odsek 4
  Pismeno: Paragraf § 5 Odsek 4 Pismeno a)
  Povinnost: Danova Povinnost Podla Paragrafu § 81 Odsek 5
  Dovod: Zanik Platitela Bez Likvidacie
  Subjekt: Platitel
  Subjekt: Pravny Nastupca
  Registracia: Statie Sa Platitelom Podla Paragrafu § 4 Odsek 1 Pismeno c)
  Registracia: Statie Sa Platitelom Podla Paragrafu § 5 Odsek 4 Pismeno a)

---

chunk: 1483
path: ['§ 85', '1']
path_as_text: Paragraf § 85 Odsek 1
text: (1) Podľa doterajších predpisov sa až do uplynutia posudzujú všetky lehoty, ktoré začali plynúť pred účinnosťou tohto zákona.

relations:
  Paragraf § 85 -> [OBSAHUJE] -> Paragraf § 85 Odsek 1
  Paragraf § 85 Odsek 1 -> [ODKAZUJE_NA] -> Doterajsie Predpisy
  Paragraf § 85 Odsek 1 -> [UPRAVUJE] -> Lehoty Zacinajuce Plynut Pred Ucinnostou Tohto Zakona
  Lehoty Zacinajuce Plynut Pred Ucinnostou Tohto Zakona -> [VYPLYVA_Z] -> Doterajsie Predpisy
  Lehoty Zacinajuce Plynut Pred Ucinnostou Tohto Zakona -> [MA_DATUM] -> Ucinnost Tohto Zakona
  Ucinnost Tohto Zakona -> [VZTAHUJE_SA_NA] -> Tento Zakon
  Lehoty Zacinajuce Plynut Pred Ucinnostou Tohto Zakona -> [MA_DATUM] -> Uplynutie Lehot

nodes:
  Paragraf: Paragraf § 85
  Odsek: Paragraf § 85 Odsek 1
  PravnyPredpis: Doterajsie Predpisy
  PravnyPredpis: Tento Zakon
  Lehota: Lehoty Zacinajuce Plynut Pred Ucinnostou Tohto Zakona
  Datum: Ucinnost Tohto Zakona
  Datum: Uplynutie Lehot

---

chunk: 1506
path: ['§ 85', '19', 'c)']
path_as_text: Paragraf § 85 Odsek 19 Pismeno c)
text: (19) Tovar podľa odsekov 17 a 18 nie je predmetom dane, ak c) tovarom podľa odseku 17 písm. a) je dopravný prostriedok, ktorý bol nadobudnutý alebo dovezený do 30. apríla 2004 vrátane v súlade s daňovými podmienkami platnými na domácom trhu štátu, ktorý je členským štátom k 30. aprílu 2004 alebo sa stane členským štátom 1. mája 2004, a nebol pri vývoze oslobodený od dane ani daň viažuca sa na dopravný prostriedok nebola vrátená; táto podmienka sa považuje za splnenú, ak dopravný prostriedok bol prvýkrát použitý pred 1. májom 1996 alebo ak je výška dane pri jeho dovoze zanedbateľná.

relations:
  Paragraf § 85 -> [OBSAHUJE] -> Paragraf § 85 Odsek 17
  Paragraf § 85 -> [OBSAHUJE] -> Paragraf § 85 Odsek 18
  Paragraf § 85 -> [OBSAHUJE] -> Paragraf § 85 Odsek 19
  Paragraf § 85 Odsek 17 -> [OBSAHUJE] -> Paragraf § 85 Odsek 17 Pismeno a)
  Paragraf § 85 Odsek 19 -> [OBSAHUJE] -> Paragraf § 85 Odsek 19 Pismeno c)
  Tovar Podla Paragrafu § 85 Odsek 17 Pismeno a) -> [JE_TYPOM] -> Dopravny Prostriedok
  Tovar Podla Paragrafu § 85 Odsek 17 -> [NIE_JE_PREDMETOM_DANE] -> Dan
  Tovar Podla Paragrafu § 85 Odsek 18 -> [NIE_JE_PREDMETOM_DANE] -> Dan
  Tovar Podla Paragrafu § 85 Odsek 17 -> [MA_PODMIENKU] -> Nadobudnutie Alebo Dovezenie Dopravneho Prostriedku Do 30. Aprila 2004 Vratane
  Tovar Podla Paragrafu § 85 Odsek 18 -> [MA_PODMIENKU] -> Nadobudnutie Alebo Dovezenie Dopravneho Prostriedku Do 30. Aprila 2004 Vratane
  Nadobudnutie Alebo Dovezenie Dopravneho Prostriedku Do 30. Aprila 2004 Vratane -> [MA_DATUM] -> 30. April 2004
  Nadobudnutie Alebo Dovezenie Dopravneho Prostriedku Do 30. Aprila 2004 Vratane -> [MA_PODMIENKU] -> Danove Podmienky Platne Na Domacom Trhu Clenskeho Statu
  Danove Podmienky Platne Na Domacom Trhu Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Clensky Stat K 30. Aprilu 2004
  Danove Podmienky Platne Na Domacom Trhu Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Stat Ktory Sa Stane Clenskym Statom 1. Maja 2004
  Clensky Stat K 30. Aprilu 2004 -> [MA_DATUM] -> 30. April 2004
  Stat Ktory Sa Stane Clenskym Statom 1. Maja 2004 -> [MA_DATUM] -> 1. Maj 2004
  Tovar Podla Paragrafu § 85 Odsek 17 -> [MA_PODMIENKU] -> Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze
  Tovar Podla Paragrafu § 85 Odsek 18 -> [MA_PODMIENKU] -> Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze
  Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze -> [NEVZTAHUJE_SA_NA] -> Dan
  Tovar Podla Paragrafu § 85 Odsek 17 -> [MA_PODMIENKU] -> Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok
  Tovar Podla Paragrafu § 85 Odsek 18 -> [MA_PODMIENKU] -> Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok
  Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok -> [VZTAHUJE_SA_NA] -> Dan Viazuca Sa Na Dopravny Prostriedok
  Dan Viazuca Sa Na Dopravny Prostriedok -> [VZTAHUJE_SA_NA] -> Dopravny Prostriedok
  Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze -> [SPLNA_PODMIENKY] -> Pouzitie Dopravneho Prostriedku Prvykrat Pred 1. Majom 1996
  Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok -> [SPLNA_PODMIENKY] -> Pouzitie Dopravneho Prostriedku Prvykrat Pred 1. Majom 1996
  Pouzitie Dopravneho Prostriedku Prvykrat Pred 1. Majom 1996 -> [MA_DATUM] -> 1. Maj 1996
  Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze -> [SPLNA_PODMIENKY] -> Zanedbatelna Vyska Dane Pri Dovoze Dopravneho Prostriedku
  Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok -> [SPLNA_PODMIENKY] -> Zanedbatelna Vyska Dane Pri Dovoze Dopravneho Prostriedku
  Zanedbatelna Vyska Dane Pri Dovoze Dopravneho Prostriedku -> [VZTAHUJE_SA_NA] -> Dan

nodes:
  Paragraf: Paragraf § 85
  Odsek: Paragraf § 85 Odsek 17
  Odsek: Paragraf § 85 Odsek 18
  Odsek: Paragraf § 85 Odsek 19
  Pismeno: Paragraf § 85 Odsek 17 Pismeno a)
  Pismeno: Paragraf § 85 Odsek 19 Pismeno c)
  Tovar: Tovar Podla Paragrafu § 85 Odsek 17
  Tovar: Tovar Podla Paragrafu § 85 Odsek 18
  Tovar: Tovar Podla Paragrafu § 85 Odsek 17 Pismeno a)
  Vozidlo: Dopravny Prostriedok
  Dan: Dan
  Dan: Dan Viazuca Sa Na Dopravny Prostriedok
  Podmienka: Danove Podmienky Platne Na Domacom Trhu Clenskeho Statu
  Podmienka: Nadobudnutie Alebo Dovezenie Dopravneho Prostriedku Do 30. Aprila 2004 Vratane
  Podmienka: Neoslobodenie Dopravneho Prostriedku Od Dane Pri Vyvoze
  Podmienka: Nevratenie Dane Viazucej Sa Na Dopravny Prostriedok
  Podmienka: Pouzitie Dopravneho Prostriedku Prvykrat Pred 1. Majom 1996
  Podmienka: Zanedbatelna Vyska Dane Pri Dovoze Dopravneho Prostriedku
  Datum: 30. April 2004
  Datum: 1. Maj 2004
  Datum: 1. Maj 1996
  Stat: Clensky Stat K 30. Aprilu 2004
  Stat: Stat Ktory Sa Stane Clenskym Statom 1. Maja 2004

---

chunk: 1529
path: ['§ 85e', '8']
path_as_text: Paragraf § 85e Odsek 8
text: (8) Vo faktúre, ktorá bola vyhotovená po 1. januári 2009 a vzťahuje sa k zdaniteľnému obchodu, pri ktorom vznikla daňová povinnosť pred 1. januárom 2009 a platba bola dohodnutá v slovenských korunách, sa údaje o základe dane a výške dane uvedú v eurách aj v slovenských korunách.

relations:
  Paragraf § 85e -> [OBSAHUJE] -> Paragraf § 85e Odsek 8
  Paragraf § 85e Odsek 8 -> [UPRAVUJE] -> Uvedenie Udajov O Zaklade Dane A Vyske Dane V Eurach A Slovenskych Korunach
  Faktura -> [MA_PODMIENKU] -> Vyhotovenie Faktury Po 1. Januari 2009
  Vyhotovenie Faktury Po 1. Januari 2009 -> [MA_DATUM] -> Datum 1. Januar 2009
  Faktura -> [VZTAHUJE_SA_NA] -> Zdanitelny Obchod
  Zdanitelny Obchod -> [MA_POVINNOST] -> Danova Povinnost
  Danova Povinnost -> [VZNIKA] -> Vznik Danovej Povinnosti Pred 1. Januari 2009
  Vznik Danovej Povinnosti Pred 1. Januari 2009 -> [MA_DATUM] -> Datum 1. Januar 2009
  Platba -> [MA_PODMIENKU] -> Dohodnutie Platby V Slovenskych Korunach
  Dohodnutie Platby V Slovenskych Korunach -> [MA_HODNOTU] -> Slovenska Koruna
  Uvedenie Udajov O Zaklade Dane A Vyske Dane V Eurach A Slovenskych Korunach -> [VZTAHUJE_SA_NA] -> Faktura
  Uvedenie Udajov O Zaklade Dane A Vyske Dane V Eurach A Slovenskych Korunach -> [MA] -> Udaje O Zaklade Dane
  Udaje O Zaklade Dane -> [VZTAHUJE_SA_NA] -> Zaklad Dane
  Uvedenie Udajov O Zaklade Dane A Vyske Dane V Eurach A Slovenskych Korunach -> [MA] -> Udaje O Vyske Dane
  Udaje O Vyske Dane -> [VZTAHUJE_SA_NA] -> Vyska Dane
  Udaje O Zaklade Dane -> [MA_HODNOTU] -> Euro
  Udaje O Zaklade Dane -> [MA_HODNOTU] -> Slovenska Koruna
  Udaje O Vyske Dane -> [MA_HODNOTU] -> Euro
  Udaje O Vyske Dane -> [MA_HODNOTU] -> Slovenska Koruna

nodes:
  Paragraf: Paragraf § 85e
  Odsek: Paragraf § 85e Odsek 8
  Dokument: Faktura
  Podmienka: Vyhotovenie Faktury Po 1. Januari 2009
  Datum: Datum 1. Januar 2009
  Konanie: Zdanitelny Obchod
  Povinnost: Danova Povinnost
  Podmienka: Vznik Danovej Povinnosti Pred 1. Januari 2009
  Platba: Platba
  Podmienka: Dohodnutie Platby V Slovenskych Korunach
  Mena: Slovenska Koruna
  Mena: Euro
  Zaznam: Udaje O Zaklade Dane
  Dan: Zaklad Dane
  Zaznam: Udaje O Vyske Dane
  Suma: Vyska Dane
  Povinnost: Uvedenie Udajov O Zaklade Dane A Vyske Dane V Eurach A Slovenskych Korunach

---

chunk: 1552
path: ['§ 85kb', '2']
path_as_text: Paragraf § 85kb Odsek 2
text: (2) Poverená osoba podľa § 4 ods. 3 v znení účinnom do 31. decembra 2013 uvedie vo svojom daňovom priznaní, dodatočnom daňovom priznaní, súhrnom výkaze a dodatočnom súhrnnom výkaze údaje týkajúce sa spoločného podnikania združenia za obdobia do konca kalendárneho roka 2013. Za daň vzťahujúcu sa na spoločné podnikanie do konca kalendárneho roka 2013 zodpovedajú všetci účastníci združenia spoločne a nerozdielne.

relations:
  Paragraf § 85Kb -> [OBSAHUJE] -> Paragraf § 85Kb Odsek 2
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 3
  Poverena Osoba -> [ODKAZUJE_NA] -> Paragraf § 4 Odsek 3
  Poverena Osoba -> [PODAVA] -> Danove Priznanie
  Poverena Osoba -> [PODAVA] -> Dodatocne Danove Priznanie
  Poverena Osoba -> [PREDKLADA] -> Suhrnny Vykaz
  Poverena Osoba -> [PREDKLADA] -> Dodatocny Suhrnny Vykaz
  Danove Priznanie -> [OBSAHUJE] -> Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia
  Dodatocne Danove Priznanie -> [OBSAHUJE] -> Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia
  Suhrnny Vykaz -> [OBSAHUJE] -> Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia
  Dodatocny Suhrnny Vykaz -> [OBSAHUJE] -> Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia
  Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia -> [VZTAHUJE_SA_NA] -> Spolocne Podnikanie Zdruzenia
  Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia -> [MA_OBDOBIE] -> Obdobia Do Konca Kalendarneho Roka 2013
  Dan Vztahujuca Sa Na Spolocne Podnikanie Do Konca Kalendarneho Roka 2013 -> [VZTAHUJE_SA_NA] -> Spolocne Podnikanie Zdruzenia
  Dan Vztahujuca Sa Na Spolocne Podnikanie Do Konca Kalendarneho Roka 2013 -> [MA_OBDOBIE] -> Obdobia Do Konca Kalendarneho Roka 2013
  Ucastnici Zdruzenia -> [ZODPOVEDA_ZA] -> Dan Vztahujuca Sa Na Spolocne Podnikanie Do Konca Kalendarneho Roka 2013
  Ucastnici Zdruzenia -> [MA_POVINNOST] -> Spolocna A Nerozdielna Zodpovednost

nodes:
  Paragraf: Paragraf § 85Kb
  Odsek: Paragraf § 85Kb Odsek 2
  Paragraf: Paragraf § 4
  Odsek: Paragraf § 4 Odsek 3
  Osoba: Poverena Osoba
  DanovePriznanie: Danove Priznanie
  DanovePriznanie: Dodatocne Danove Priznanie
  Dokument: Suhrnny Vykaz
  Dokument: Dodatocny Suhrnny Vykaz
  Zaznam: Udaje TykaJuce Sa Spolocneho Podnikania Zdruzenia
  Konanie: Spolocne Podnikanie Zdruzenia
  Obdobie: Obdobia Do Konca Kalendarneho Roka 2013
  Dan: Dan Vztahujuca Sa Na Spolocne Podnikanie Do Konca Kalendarneho Roka 2013
  Subjekt: Ucastnici Zdruzenia
  Povinnost: Spolocna A Nerozdielna Zodpovednost
  Datum: Datum 31. Decembra 2013

---

chunk: 1575
path: ['§ 85kj', '2']
path_as_text: Paragraf § 85kj Odsek 2
text: (2) Ak chce zdaniteľná osoba okrem zdaniteľnej osoby, ktorá uplatňuje osobitnú úpravu podľa § 68b v znení účinnom do 30. júna 2021, uplatňovať od 1. júla 2021 osobitnú úpravu podľa § 68b v znení účinnom od 1. júla 2021, oznámi elektronickými prostriedkami začatie činnosti daňovému úradu najskôr 1. apríla 2021 a najneskôr 10. júna 2021. Ak zdaniteľná osoba, ktorá chce uplatňovať osobitnú úpravu podľa § 68b v znení účinnom od 1. júla 2021, nemá v tuzemsku pridelené identifikačné číslo pre daň, musí v oznámení o začatí činnosti uviesť obchodné meno, adresu, elektronickú adresu vrátane webových sídiel a ďalšie údaje uvedené v osobitnom predpise.28aa)

relations:
  Paragraf § 85kj -> [OBSAHUJE] -> Paragraf § 85kj Odsek 2
  Paragraf § 85kj Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 68b
  Zdanitelna Osoba -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy Podla Paragrafu § 68b Od 1. Jula 2021
  Zdanitelna Osoba Uplatnujuca Osobitnu Upravu Podla Paragrafu § 68b Do 30. Juna 2021 -> [NEVZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy Podla Paragrafu § 68b Od 1. Jula 2021
  Uplatnovanie Osobitnej Upravy Podla Paragrafu § 68b Od 1. Jula 2021 -> [ODKAZUJE_NA] -> Paragraf § 68b
  Uplatnovanie Osobitnej Upravy Podla Paragrafu § 68b Od 1. Jula 2021 -> [MA_DATUM] -> Datum 1. Jula 2021
  Zdanitelna Osoba Uplatnujuca Osobitnu Upravu Podla Paragrafu § 68b Do 30. Juna 2021 -> [MA_DATUM] -> Datum 30. Juna 2021
  Zdanitelna Osoba -> [OZNAMUJE] -> Oznamenie O Zacati Cinnosti
  Oznamenie O Zacati Cinnosti -> [VZTAHUJE_SA_NA] -> Zacatie Cinnosti
  Zdanitelna Osoba -> [DORUCUJE] -> Danovy Urad
  Oznamenie O Zacati Cinnosti -> [DORUCUJE] -> Danovy Urad
  Oznamenie O Zacati Cinnosti -> [MA] -> Elektronicke Prostriedky
  Oznamenie O Zacati Cinnosti -> [MA_LEHOTU] -> Lehota Na Oznamenie Zacatia Cinnosti
  Lehota Na Oznamenie Zacatia Cinnosti -> [MA_DATUM] -> Datum 1. Aprila 2021
  Lehota Na Oznamenie Zacatia Cinnosti -> [MA_DATUM] -> Datum 10. Juna 2021
  Zdanitelna Osoba -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Oznamenie O Zacati Cinnosti -> [OBSAHUJE] -> Obchodne Meno
  Oznamenie O Zacati Cinnosti -> [OBSAHUJE] -> Adresa
  Oznamenie O Zacati Cinnosti -> [OBSAHUJE] -> Elektronicka Adresa
  Oznamenie O Zacati Cinnosti -> [OBSAHUJE] -> Webove Sidla
  Oznamenie O Zacati Cinnosti -> [ODKAZUJE_NA] -> Osobitny Predpis 28aa

nodes:
  Paragraf: Paragraf § 85kj
  Odsek: Paragraf § 85kj Odsek 2
  Paragraf: Paragraf § 68b
  Osoba: Zdanitelna Osoba
  Osoba: Zdanitelna Osoba Uplatnujuca Osobitnu Upravu Podla Paragrafu § 68b Do 30. Juna 2021
  Konanie: Uplatnovanie Osobitnej Upravy Podla Paragrafu § 68b Od 1. Jula 2021
  Oznamenie: Oznamenie O Zacati Cinnosti
  Konanie: Zacatie Cinnosti
  Organizacia: Danovy Urad
  Dokument: Elektronicke Prostriedky
  Lehota: Lehota Na Oznamenie Zacatia Cinnosti
  Datum: Datum 1. Aprila 2021
  Datum: Datum 10. Juna 2021
  Datum: Datum 1. Jula 2021
  Datum: Datum 30. Juna 2021
  Zaznam: Identifikacne Cislo Pre Dan
  Zaznam: Obchodne Meno
  Adresa: Adresa
  Adresa: Elektronicka Adresa
  Adresa: Webove Sidla
  PravnyPredpis: Osobitny Predpis 28aa

---

chunk: 1598
path: ['§ 85km', '5']
path_as_text: Paragraf § 85km Odsek 5
text: (5) Ak platiteľ dostane opravný doklad podľa § 25a ods. 7 písm. a) po 31. decembri 2022 z dôvodu, že dodávateľ opravil základ dane podľa § 25a ods. 3, pretože sa pohľadávka stala nevymožiteľnou podľa § 25a ods. 2 písm. a) alebo písm. f) v znení účinnom do 31. decembra 2022, je povinný opraviť odpočítanú daň v tom zdaňovacom období, v ktorom dostane tento doklad. Na opravu odpočítanej dane sa uplatní § 53b v znení účinnom do 31. decembra 2022 a na povinnosť vykázať opravu odpočítanej dane v kontrolnom výkaze sa uplatní § 78a v znení účinnom do 31. decembra 2022.

relations:
  Paragraf § 85km -> [OBSAHUJE] -> Paragraf § 85km Odsek 5
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 7
  Paragraf § 25a Odsek 7 -> [OBSAHUJE] -> Paragraf § 25a Odsek 7 Pismeno a)
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 3
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 2
  Paragraf § 25a Odsek 2 -> [OBSAHUJE] -> Paragraf § 25a Odsek 2 Pismeno a)
  Paragraf § 25a Odsek 2 -> [OBSAHUJE] -> Paragraf § 25a Odsek 2 Pismeno f)
  Opravny Doklad -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 7 Pismeno a)
  Platitel -> [PRIJIMA] -> Opravny Doklad
  Dodavatel -> [MA] -> Oprava Zakladu Dane
  Oprava Zakladu Dane -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 3
  Pohladavka -> [MA_STATUS] -> Nevymozitelna
  Pohladavka -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 2 Pismeno a)
  Pohladavka -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 2 Pismeno f)
  Platitel -> [MA_POVINNOST] -> Povinnost Opravit Odpocitanu Dan
  Povinnost Opravit Odpocitanu Dan -> [VZTAHUJE_SA_NA] -> Oprava Odpocitanej Dane
  Oprava Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Povinnost Opravit Odpocitanu Dan -> [MA_OBDOBIE] -> Zdanovacie Obdobie Dostania Dokladu
  Zdanovacie Obdobie Dostania Dokladu -> [VZTAHUJE_SA_NA] -> Opravny Doklad
  Oprava Odpocitanej Dane -> [VYPLYVA_Z] -> Paragraf § 53b
  Paragraf § 53b -> [MA_OBDOBIE] -> Znenie Ucinne Do 31. Decembra 2022
  Znenie Ucinne Do 31. Decembra 2022 -> [MA_DATUM] -> Datum 31. December 2022
  Povinnost Vykazat Opravu Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Oprava Odpocitanej Dane
  Povinnost Vykazat Opravu Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Kontrolny Vykaz
  Povinnost Vykazat Opravu Odpocitanej Dane -> [VYPLYVA_Z] -> Paragraf § 78a
  Paragraf § 78a -> [MA_OBDOBIE] -> Znenie Ucinne Do 31. Decembra 2022

nodes:
  Paragraf: Paragraf § 85km
  Odsek: Paragraf § 85km Odsek 5
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 7
  Pismeno: Paragraf § 25a Odsek 7 Pismeno a)
  Odsek: Paragraf § 25a Odsek 3
  Odsek: Paragraf § 25a Odsek 2
  Pismeno: Paragraf § 25a Odsek 2 Pismeno a)
  Pismeno: Paragraf § 25a Odsek 2 Pismeno f)
  Paragraf: Paragraf § 53b
  Paragraf: Paragraf § 78a
  Subjekt: Platitel
  Subjekt: Dodavatel
  Dokument: Opravny Doklad
  Konanie: Oprava Zakladu Dane
  Konanie: Oprava Odpocitanej Dane
  Povinnost: Povinnost Opravit Odpocitanu Dan
  Povinnost: Povinnost Vykazat Opravu Odpocitanej Dane
  Dan: Odpocitana Dan
  ZdanovacieObdobie: Zdanovacie Obdobie Dostania Dokladu
  Dokument: Kontrolny Vykaz
  Pohladavka: Pohladavka
  Status: Nevymozitelna
  Datum: Datum 31. December 2022
  Obdobie: Znenie Ucinne Do 31. Decembra 2022

---

chunk: 1600
path: ['§ 85km', '7']
path_as_text: Paragraf § 85km Odsek 7
text: (7) Platiteľ má právo opraviť opravenú odpočítanú daň, ak po 31. decembri 2022 dostane opravný doklad podľa § 25a ods. 7 písm. b) z dôvodu, že dodávateľ opravil základ dane podľa § 25a ods. 6, pretože prijal akúkoľvek platbu v súvislosti s nevymožiteľnou pohľadávkou podľa § 25a ods. 2 písm. a) alebo písm. f) v znení účinnom do 31. decembra 2022. Na opravu opravenej odpočítanej dane sa uplatní § 53b v znení účinnom do 31. decembra 2022 a na povinnosť vykázať opravu opravenej odpočítanej dane v kontrolnom výkaze sa uplatní § 78a v znení účinnom do 31. decembra 2022.

relations:
  Paragraf § 85km -> [OBSAHUJE] -> Paragraf § 85km Odsek 7
  Paragraf § 85km Odsek 7 -> [UPRAVUJE] -> Platitel
  Platitel -> [MA_PRAVO] -> Pravo Opravit Opravenu Odpocitanu Dan
  Pravo Opravit Opravenu Odpocitanu Dan -> [VZTAHUJE_SA_NA] -> Opravena Odpocitana Dan
  Pravo Opravit Opravenu Odpocitanu Dan -> [MA_PODMIENKU] -> Opravny Doklad
  Opravny Doklad -> [MA_DATUM] -> Datum 31. Decembra 2022
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 7
  Paragraf § 25a Odsek 7 -> [OBSAHUJE] -> Paragraf § 25a Odsek 7 Pismeno b)
  Dodavatel -> [MENI] -> Oprava Zakladu Dane
  Oprava Zakladu Dane -> [VZTAHUJE_SA_NA] -> Zaklad Dane
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 6
  Dodavatel -> [PRIJIMA] -> Platba
  Platba -> [SUVISI_S] -> Nevymozitelna Pohladavka
  Paragraf § 25a -> [OBSAHUJE] -> Paragraf § 25a Odsek 2
  Paragraf § 25a Odsek 2 -> [OBSAHUJE] -> Paragraf § 25a Odsek 2 Pismeno a)
  Paragraf § 25a Odsek 2 -> [OBSAHUJE] -> Paragraf § 25a Odsek 2 Pismeno f)
  Oprava Opravenej Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Opravena Odpocitana Dan
  Povinnost Vykazat Opravu Opravenej Odpocitanej Dane V Kontrolnom Vykaze -> [VZTAHUJE_SA_NA] -> Oprava Opravenej Odpocitanej Dane
  Povinnost Vykazat Opravu Opravenej Odpocitanej Dane V Kontrolnom Vykaze -> [VZTAHUJE_SA_NA] -> Kontrolny Vykaz

nodes:
  Paragraf: Paragraf § 85km
  Odsek: Paragraf § 85km Odsek 7
  Subjekt: Platitel
  Pravo: Pravo Opravit Opravenu Odpocitanu Dan
  Dan: Opravena Odpocitana Dan
  Dokument: Opravny Doklad
  Datum: Datum 31. Decembra 2022
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 7
  Pismeno: Paragraf § 25a Odsek 7 Pismeno b)
  Subjekt: Dodavatel
  Konanie: Oprava Zakladu Dane
  Dan: Zaklad Dane
  Odsek: Paragraf § 25a Odsek 6
  Platba: Platba
  Pohladavka: Nevymozitelna Pohladavka
  Odsek: Paragraf § 25a Odsek 2
  Pismeno: Paragraf § 25a Odsek 2 Pismeno a)
  Pismeno: Paragraf § 25a Odsek 2 Pismeno f)
  Konanie: Oprava Opravenej Odpocitanej Dane
  Paragraf: Paragraf § 53b
  Povinnost: Povinnost Vykazat Opravu Opravenej Odpocitanej Dane V Kontrolnom Vykaze
  Dokument: Kontrolny Vykaz
  Paragraf: Paragraf § 78a

---

