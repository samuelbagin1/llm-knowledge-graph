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
  Platitel -> [PODAVA] -> Ziadost O Registraciu Pre Dan
  Platitel -> [MA_POVINNOST] -> Ziadost O Registraciu Pre Dan
  Ziadost O Registraciu Pre Dan -> [VZTAHUJE_SA_NA] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Danovy Urad -> [REGISTRUJE] -> Platitel
  Platitel -> [MA] -> Registracia Pre Dan
  Danovy Urad -> [VYDAVA] -> Identifikacne Cislo Pre Dan
  Platitel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Registracii Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [ROZHODUJE_O] -> Registracia Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [MA_LEHOTU] -> Lehota Desat Dni Od Dorucenia Ziadosti O Registraciu Pre Dan
  Danovy Urad -> [PRIJIMA] -> Ziadost O Registraciu Pre Dan
  Zdanitelna Osoba -> [NADOBUDA] -> Platitel
  Identifikacne Cislo Pre Dan -> [MA_STATUS] -> Platnost Identifikacneho Cisla Pre Dan

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
  Ziadost: Ziadost O Registraciu Pre Dan
  Registracia: Registracia Pre Dan
  Zaznam: Identifikacne Cislo Pre Dan
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Dan: Dan
  Lehota: Lehota Desat Dni Od Dorucenia Ziadosti O Registraciu Pre Dan
  Dokument: Doklady Podla Paragraf § 4 Odsek 3
  Status: Platnost Identifikacneho Cisla Pre Dan

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
  Bydlisko -> [JE_TYPOM] -> Adresa Trvaleho Pobytu
  Adresa Trvaleho Pobytu -> [VZTAHUJE_SA_NA] -> Fyzicka Osoba
  Adresa Trvaleho Pobytu -> [NACHADZA_SA_V] -> Tuzemsko
  Bydlisko -> [JE_TYPOM] -> Trvale Miesto Pobytu
  Trvale Miesto Pobytu -> [VZTAHUJE_SA_NA] -> Fyzicka Osoba Bez Trvaleho Pobytu V Tuzemsku
  Trvale Miesto Pobytu -> [NACHADZA_SA_V] -> Zahranicie

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
  Lokacia: Tuzemsko
  Lokacia: Zahranicie

---

chunk: 62
path: ['§ 4b', '6']
path_as_text: Paragraf § 4b Odsek 6
text: (6) Ak sa člen skupiny rozhodne vystúpiť zo skupiny alebo musí vystúpiť zo skupiny z dôvodu neplnenia podmienok podľa § 4a, zástupca skupiny je povinný bezodkladne podať žiadosť o zmenu registrácie skupiny; ak je vystupujúcim členom skupiny zástupca skupiny, žiadosť musí obsahovať aj označenie člena skupiny, ktorý bol určený členmi skupiny ako nový zástupca skupiny. Daňový úrad vydá bezodkladne rozhodnutie o zmene registrácie skupiny, proti ktorému nemožno podať odvolanie; účinky zmeny registrácie skupiny nastávajú v deň uvedený v rozhodnutí, ktorý nesmie byť neskorší ako 30. deň odo dňa podania žiadosti o zmenu registrácie skupiny. Daňový úrad, ktorý je miestne príslušný pre vystupujúceho člena skupiny, zaregistruje vystupujúceho člena skupiny za samostatného platiteľa ku dňu, keď nastali účinky zmeny registrácie skupiny a pridelí mu identifikačné číslo pre daň; proti tomuto rozhodnutiu nemožno podať odvolanie. Práva a povinnosti skupiny vyplývajúce z tohto zákona prechádzajú na zdaniteľnú osobu, ktorá vystúpila zo skupiny, dňom, keď nastali účinky zmeny registrácie skupiny, a to v rozsahu, v akom sa vzťahujú na plnenia uskutočnené a prijaté touto zdaniteľnou osobou.

relations:
  Paragraf § 4b -> [OBSAHUJE] -> Paragraf § 4b Odsek 6
  Neplnenie Podmienok Podla Paragrafu § 4a -> [VYPLYVA_Z] -> Podmienky Podla Paragrafu § 4a
  Podmienky Podla Paragrafu § 4a -> [ODKAZUJE_NA] -> Paragraf § 4a
  Clen Skupiny -> [PATRI_DO] -> Skupina
  Vystupujuci Clen Skupiny -> [JE_TYPOM] -> Clen Skupiny
  Vystupujuci Clen Skupiny -> [PATRI_DO] -> Skupina
  Zastupca Skupiny -> [PODAVA] -> Ziadost O Zmenu Registracie Skupiny
  Zastupca Skupiny -> [MA_POVINNOST] -> Ziadost O Zmenu Registracie Skupiny
  Ziadost O Zmenu Registracie Skupiny -> [VZTAHUJE_SA_NA] -> Zmena Registracie Skupiny
  Ziadost O Zmenu Registracie Skupiny -> [OBSAHUJE] -> Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny
  Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny -> [VZTAHUJE_SA_NA] -> Novy Zastupca Skupiny
  Clenovia Skupiny -> [URCUJE] -> Novy Zastupca Skupiny
  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Zmene Registracie Skupiny
  Rozhodnutie O Zmene Registracie Skupiny -> [ROZHODUJE_O] -> Zmena Registracie Skupiny
  Odvolanie Proti Rozhodnutiu O Zmene Registracie Skupiny -> [NEVZTAHUJE_SA_NA] -> Rozhodnutie O Zmene Registracie Skupiny
  Zmena Registracie Skupiny -> [MA_STATUS] -> Ucinok Zmeny Registracie Skupiny
  Ucinok Zmeny Registracie Skupiny -> [MA_DATUM] -> Den Uvedeny V Rozhodnuti
  Den Uvedeny V Rozhodnuti -> [VYPLYVA_Z] -> Rozhodnutie O Zmene Registracie Skupiny
  Den Uvedeny V Rozhodnuti -> [MA_LEHOTU] -> Lehota 30 Dni Od Podania Ziadosti O Zmenu Registracie Skupiny
  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [VZTAHUJE_SA_NA] -> Vystupujuci Clen Skupiny
  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [REGISTRUJE] -> Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela
  Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela -> [VZTAHUJE_SA_NA] -> Vystupujuci Clen Skupiny
  Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela -> [MA_STATUS] -> Samostatny Platitel
  Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela -> [MA_DATUM] -> Ucinok Zmeny Registracie Skupiny
  Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Odvolanie Proti Rozhodnutiu O Registracii Za Samostatneho Platitela -> [NEVZTAHUJE_SA_NA] -> Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela
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
  Dovod: Neplnenie Podmienok Podla Paragrafu § 4a
  Podmienka: Podmienky Podla Paragrafu § 4a
  Ziadost: Ziadost O Zmenu Registracie Skupiny
  Zaznam: Oznacenie Clena Skupiny Ako Noveho Zastupcu Skupiny
  Organizacia: Danovy Urad
  Rozhodnutie: Rozhodnutie O Zmene Registracie Skupiny
  Registracia: Zmena Registracie Skupiny
  Konanie: Odvolanie Proti Rozhodnutiu O Zmene Registracie Skupiny
  Status: Ucinok Zmeny Registracie Skupiny
  Datum: Den Uvedeny V Rozhodnuti
  Lehota: Lehota 30 Dni Od Podania Ziadosti O Zmenu Registracie Skupiny
  Organizacia: Miestne Prislusny Danovy Urad Pre Vystupujuceho Clena Skupiny
  Registracia: Registracia Vystupujuceho Clena Skupiny Za Samostatneho Platitela
  Status: Samostatny Platitel
  Zaznam: Identifikacne Cislo Pre Dan
  Dan: Dan
  Konanie: Odvolanie Proti Rozhodnutiu O Registracii Za Samostatneho Platitela
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
  Zahranicna Osoba -> [JE_TYPOM] -> Zdanitelna Osoba
  Zahranicna Osoba -> [NEVZTAHUJE_SA_NA] -> Tuzemsko
  Zahranicna Osoba -> [MA_STATUS] -> Platitel
  Zahranicna Osoba -> [NADOBUDA] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [VZTAHUJE_SA_NA] -> Tovar
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [NACHADZA_SA_V] -> Tuzemsko
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [VYPLYVA_Z] -> Iny Clensky Stat
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [JE_PREDMETOM_DANE] -> Dan
  Maly Podnik Zahranicnej Osoby -> [JE_TYPOM] -> Zahranicna Osoba
  Maly Podnik Zahranicnej Osoby -> [MA_PRAVO] -> Oslobodenie Od Dane
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_STATUS] -> Nadobudnutie Tovaru Povazovane Za Zdanene
  Paragraf § 5 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu

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
  Status: Platitel
  Stat: Tuzemsko
  Stat: Iny Clensky Stat
  Konanie: Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Tovar: Tovar
  Dan: Dan
  Subjekt: Maly Podnik Zahranicnej Osoby
  Pravo: Oslobodenie Od Dane
  Status: Nadobudnutie Tovaru Povazovane Za Zdanene

---

chunk: 93
path: ['§ 5a']
path_as_text: Paragraf § 5a
text: Na účely tohto zákona sa za platiteľa, ktorý má pridelené identifikačné číslo pre daň podľa § 4, § 4b, § 4c alebo § 5, považuje platiteľ, ktorý a) sa stal platiteľom po doručení rozhodnutia o registrácii pre daň podľa § 4, a to počnúc dňom, keď sa stal platiteľom; ak platiteľ nesplnil oznamovaciu povinnosť podľa § 4 ods. 5, tak počnúc 1. januárom kalendárneho roka nasledujúceho po kalendárnom roku, v ktorom presiahol obrat podľa § 4 ods. 1 písm. a),  b) sa stal platiteľom pred doručením rozhodnutia o registrácii pre daň podľa § 4 alebo § 5, a to počnúc dňom doručenia tohto rozhodnutia alebo c) je skupinou, počnúc dňom, ku ktorému daňový úrad vykoná registráciu skupiny.

relations:
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 5a
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 4
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 4b
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 4c
  Tento Zakon -> [OBSAHUJE] -> Paragraf § 5
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 5
  Paragraf § 4 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1
  Paragraf § 4 Odsek 1 -> [OBSAHUJE] -> Paragraf § 4 Odsek 1 Pismeno a)
  Paragraf § 5a -> [DEFINUJE] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel S Identifikacnym Cislom Pre Dan -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [PRIJIMA] -> Rozhodnutie O Registracii Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [ROZHODUJE_O] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [MA_DATUM] -> Den Ked Sa Stal Platitelom
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [MA_POVINNOST] -> Oznamovacia Povinnost
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [NESPLNA_PODMIENKY] -> Oznamovacia Povinnost
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [MA_DATUM] -> 1. Januar Kalendarneho Roka Nasledujuceho Po Roku Presiahnutia Obratu
  Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan -> [MA_HODNOTU] -> Obrat
  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan -> [PRIJIMA] -> Rozhodnutie O Registracii Pre Dan
  Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan -> [MA_DATUM] -> Den Dorucenia Rozhodnutia
  Skupina -> [JE_TYPOM] -> Platitel S Identifikacnym Cislom Pre Dan
  Danovy Urad -> [REGISTRUJE] -> Skupina
  Danovy Urad -> [REGISTRUJE] -> Registracia Skupiny
  Registracia Skupiny -> [VZTAHUJE_SA_NA] -> Skupina
  Registracia Skupiny -> [MA_DATUM] -> Den Registracie Skupiny
  Skupina -> [MA_DATUM] -> Den Registracie Skupiny

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
  Subjekt: Platitel Po Doruceni Rozhodnutia O Registracii Pre Dan
  Subjekt: Platitel Pred Dorucenim Rozhodnutia O Registracii Pre Dan
  Organizacia: Skupina
  Organizacia: Danovy Urad
  Dan: Dan
  Zaznam: Identifikacne Cislo Pre Dan
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Registracia: Registracia Pre Dan
  Registracia: Registracia Skupiny
  Povinnost: Oznamovacia Povinnost
  Obrat: Obrat
  Datum: Den Ked Sa Stal Platitelom
  Datum: Den Dorucenia Rozhodnutia
  Datum: 1. Januar Kalendarneho Roka Nasledujuceho Po Roku Presiahnutia Obratu
  Datum: Den Registracie Skupiny

---

chunk: 103
path: ['§ 6a', '2']
path_as_text: Paragraf § 6a Odsek 2
text: (2) Ak zdaniteľná osoba spĺňa podmienky na registráciu podľa § 5 a je registrovaná podľa § 4, považuje sa za platiteľa registrovaného podľa § 5 odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava; túto skutočnosť je povinná oznámiť daňovému úradu do desiatich dní odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava.

relations:
  Paragraf § 6a -> [OBSAHUJE] -> Paragraf § 6a Odsek 2
  Zdanitelna Osoba -> [SPLNA_PODMIENKY] -> Podmienky Na Registraciu Podla Paragrafu § 5
  Zdanitelna Osoba -> [MA] -> Registracia Podla Paragrafu § 4
  Zdanitelna Osoba -> [JE_TYPOM] -> Platitel Registrovany Podla Paragrafu § 5
  Platitel Registrovany Podla Paragrafu § 5 -> [MA_DATUM] -> Den Prestania Mat Miesto V Tuzemsku
  Sidlo V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Podnikania V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Prevadzkaren V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Bydlisko V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Kde Sa Obvykle Zdrziava V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Skutocnost Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Sidlo V Tuzemsku
  Skutocnost Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Miesto Podnikania V Tuzemsku
  Skutocnost Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Prevadzkaren V Tuzemsku
  Skutocnost Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Bydlisko V Tuzemsku
  Skutocnost Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Miesto Kde Sa Obvykle Zdrziava V Tuzemsku
  Zdanitelna Osoba -> [MA_POVINNOST] -> Povinnost Oznamenia Skutocnosti Danovemu Uradu
  Povinnost Oznamenia Skutocnosti Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Oznamenie Skutocnosti Prestania Mat Miesto V Tuzemsku
  Povinnost Oznamenia Skutocnosti Danovemu Uradu -> [MA_LEHOTU] -> Lehota Do Desiatich Dni
  Lehota Do Desiatich Dni -> [VYPLYVA_Z] -> Den Prestania Mat Miesto V Tuzemsku
  Zdanitelna Osoba -> [OZNAMUJE] -> Danovy Urad
  Oznamenie Skutocnosti Prestania Mat Miesto V Tuzemsku -> [VZTAHUJE_SA_NA] -> Skutocnost Prestania Mat Miesto V Tuzemsku

nodes:
  Paragraf: Paragraf § 6a
  Odsek: Paragraf § 6a Odsek 2
  Paragraf: Paragraf § 5
  Paragraf: Paragraf § 4
  Subjekt: Zdanitelna Osoba
  Podmienka: Podmienky Na Registraciu Podla Paragrafu § 5
  Registracia: Registracia Podla Paragrafu § 5
  Registracia: Registracia Podla Paragrafu § 4
  Subjekt: Platitel Registrovany Podla Paragrafu § 5
  Stat: Tuzemsko
  Adresa: Sidlo V Tuzemsku
  Adresa: Miesto Podnikania V Tuzemsku
  Lokacia: Prevadzkaren V Tuzemsku
  Adresa: Bydlisko V Tuzemsku
  Lokacia: Miesto Kde Sa Obvykle Zdrziava V Tuzemsku
  Organizacia: Danovy Urad
  Povinnost: Povinnost Oznamenia Skutocnosti Danovemu Uradu
  Oznamenie: Oznamenie Skutocnosti Prestania Mat Miesto V Tuzemsku
  Lehota: Lehota Do Desiatich Dni
  Datum: Den Prestania Mat Miesto V Tuzemsku
  Dovod: Skutocnost Prestania Mat Miesto V Tuzemsku

---

chunk: 124
path: ['§ 8', '4', 'h)']
path_as_text: Paragraf § 8 Odsek 4 Pismeno h)
text: (4) Za dodanie tovaru sa považuje aj premiestnenie tovaru, ktorý je vo vlastníctve zdaniteľnej osoby, z tuzemska do iného členského štátu, ak je tento tovar odoslaný alebo prepravený ňou alebo na jej účet do iného členského štátu na účely jej podnikania. Takéto premiestnenie sa považuje za dodanie tovaru za protihodnotu okrem premiestnenia tovaru, ktoré spĺňa podmienky režimu call-off stock podľa § 8a, a okrem premiestnenia tovaru h) na dočasné použitie na obdobie nepresahujúce 24 mesiacov na území iného členského štátu, v ktorom by sa dovoz toho istého tovaru z územia tretieho štátu považoval za prepustený do režimu dočasné použitie s úplným oslobodením od dovozného cla,

relations:
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4
  Paragraf § 8 Odsek 4 -> [OBSAHUJE] -> Paragraf § 8 Odsek 4 Pismeno h)
  Paragraf § 8 Odsek 4 Pismeno h) -> [ODKAZUJE_NA] -> Paragraf § 8a
  Premiestnenie Tovaru -> [JE_TYPOM] -> Dodanie Tovaru
  Premiestnenie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar -> [PATRI_DO] -> Zdanitelna Osoba
  Premiestnenie Tovaru -> [VYPLYVA_Z] -> Tuzemsko
  Premiestnenie Tovaru -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat
  Premiestnenie Tovaru -> [VZTAHUJE_SA_NA] -> Podnikanie Zdanitelnej Osoby
  Premiestnenie Tovaru -> [JE_TYPOM] -> Dodanie Tovaru Za Protihodnotu
  Premiestnenie Tovaru -> [MA_PODMIENKU] -> Podmienky Rezimu Call-Off Stock
  Docasne Pouzitie Tovaru -> [MA_OBDOBIE] -> Obdobie Nepresahujuce 24 Mesiacov
  Docasne Pouzitie Tovaru -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Dovoz Tovaru -> [VYPLYVA_Z] -> Tretí Stat
  Dovoz Tovaru -> [MA_STATUS] -> Rezim Docasne Pouzitie
  Rezim Docasne Pouzitie -> [MA_PRAVO] -> Uplne Oslobodenie Od Dovozneho Cla
  Uplne Oslobodenie Od Dovozneho Cla -> [OSLOBODZUJE_OD] -> Dovozne Clo

nodes:
  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 4
  Pismeno: Paragraf § 8 Odsek 4 Pismeno h)
  Paragraf: Paragraf § 8a
  Konanie: Dodanie Tovaru
  Konanie: Premiestnenie Tovaru
  Tovar: Tovar
  Osoba: Zdanitelna Osoba
  Stat: Tuzemsko
  Stat: Iny Clensky Stat
  Stat: Tretí Stat
  Konanie: Podnikanie Zdanitelnej Osoby
  Konanie: Dodanie Tovaru Za Protihodnotu
  Podmienka: Podmienky Rezimu Call-Off Stock
  Konanie: Docasne Pouzitie Tovaru
  Obdobie: Obdobie Nepresahujuce 24 Mesiacov
  Konanie: Dovoz Tovaru
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
  Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i) -> [VYPLYVA_Z] -> Paragraf § 8 Odsek 4
  Prestanie Plnenia Podmienky -> [VZTAHUJE_SA_NA] -> Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)
  Premiestnenie Tovaru -> [JE_TYPOM] -> Dodanie Tovaru Za Protihodnotu
  Premiestnenie Tovaru -> [VYPLYVA_Z] -> Prestanie Plnenia Podmienky
  Premiestnenie Tovaru -> [VYPLYVA_Z] -> Paragraf § 8 Odsek 5

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
  Konanie: Dodanie Tovaru Za Protihodnotu
  Podmienka: Podmienky Podla Paragraf § 8 Odsek 4 Pismeno a) Az i)
  Dovod: Prestanie Plnenia Podmienky

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
  Jednoucelovy Poukaz -> [MA] -> Miesto Dodania Tovaru
  Jednoucelovy Poukaz -> [MA] -> Miesto Dodania Sluzby
  Miesto Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Dodania Sluzby -> [VZTAHUJE_SA_NA] -> Sluzba
  Jednoucelovy Poukaz -> [MA] -> Dan Splatna Z Tovaru Alebo Sluzby
  Dan Splatna Z Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Tovar
  Dan Splatna Z Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Sluzba

nodes:
  PravnyPredpis: Tento Zakon
  Paragraf: Paragraf § 9a
  Odsek: Paragraf § 9a Odsek 1
  Pismeno: Paragraf § 9a Odsek 1 Pismeno b)
  Dokument: Jednoucelovy Poukaz
  Dokument: Poukaz
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
  Paragraf § 9A -> [OBSAHUJE] -> Paragraf § 9A Odsek 6
  Paragraf § 9A -> [OBSAHUJE] -> Paragraf § 9A Odsek 5
  Prevod Viacuceloveho Poukazu -> [VZTAHUJE_SA_NA] -> Viacucelovy Poukaz
  Ina Zdanitelna Osoba -> [VZTAHUJE_SA_NA] -> Prevod Viacuceloveho Poukazu
  Ina Zdanitelna Osoba -> [NEVZTAHUJE_SA_NA] -> Dodavatel Tovaru
  Ina Zdanitelna Osoba -> [NEVZTAHUJE_SA_NA] -> Dodavatel Sluzby
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu -> [SUVISI_S] -> Prevod Viacuceloveho Poukazu
  Ina Zdanitelna Osoba -> [DODAVA] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu
  Distribucna Sluzba -> [JE_TYPOM] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu
  Propagacna Sluzba -> [JE_TYPOM] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu
  Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu -> [JE_PREDMETOM_DANE] -> Dan
  Paragraf § 9A Odsek 6 -> [UPRAVUJE] -> Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu

nodes:
  Paragraf: Paragraf § 9A
  Odsek: Paragraf § 9A Odsek 6
  Odsek: Paragraf § 9A Odsek 5
  Subjekt: Ina Zdanitelna Osoba
  Subjekt: Dodavatel Tovaru
  Subjekt: Dodavatel Sluzby
  Konanie: Prevod Viacuceloveho Poukazu
  Dokument: Viacucelovy Poukaz
  Sluzba: Sluzba Dodana V Suvislosti S Prevodom Viacuceloveho Poukazu
  Sluzba: Distribucna Sluzba
  Sluzba: Propagacna Sluzba
  Dan: Dan

---

chunk: 172
path: ['§ 11', '7']
path_as_text: Paragraf § 11 Odsek 7
text: (7) Nadobúdateľ podľa odseku 4 písm. b) sa môže rozhodnúť, že bude zdaňovať nadobudnutie tovaru pred dosiahnutím hodnoty 14 000 eur a toto svoje rozhodnutie oznámi písomne daňovému úradu pri podaní žiadosti o registráciu pre daň (§ 7). Zdaňovanie nadobudnutia tovaru je nadobúdateľ povinný uplatňovať najmenej po dobu dvoch kalendárnych rokov.

relations:
  Paragraf § 11 -> [OBSAHUJE] -> Paragraf § 11 Odsek 7
  Paragraf § 11 -> [OBSAHUJE] -> Paragraf § 11 Odsek 4
  Paragraf § 11 Odsek 4 -> [OBSAHUJE] -> Paragraf § 11 Odsek 4 Pismeno b)
  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 11 Odsek 4 Pismeno b)
  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [MA_PRAVO] -> Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Rozhodnutie O Zdanovani Nadobudnutia Tovaru -> [VZTAHUJE_SA_NA] -> Zdanovanie Nadobudnutia Tovaru
  Zdanovanie Nadobudnutia Tovaru -> [VZTAHUJE_SA_NA] -> Nadobudnutie Tovaru
  Nadobudnutie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Rozhodnutie O Zdanovani Nadobudnutia Tovaru -> [MA_PODMIENKU] -> Hodnota 14 000 Eur
  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [OZNAMUJE] -> Pisomne Oznamenie Rozhodnutia Danovemu Uradu
  Pisomne Oznamenie Rozhodnutia Danovemu Uradu -> [VZTAHUJE_SA_NA] -> Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Pisomne Oznamenie Rozhodnutia Danovemu Uradu -> [DORUCUJE] -> Danovy Urad
  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [PODAVA] -> Ziadost O Registraciu Pre Dan
  Ziadost O Registraciu Pre Dan -> [VZTAHUJE_SA_NA] -> Registracia Pre Dan
  Registracia Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Registracia Pre Dan -> [ODKAZUJE_NA] -> Paragraf § 7
  Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b) -> [MA_POVINNOST] -> Zdanovanie Nadobudnutia Tovaru
  Zdanovanie Nadobudnutia Tovaru -> [MA_LEHOTU] -> Dva Kalendarne Roky

nodes:
  Paragraf: Paragraf § 11
  Odsek: Paragraf § 11 Odsek 7
  Odsek: Paragraf § 11 Odsek 4
  Pismeno: Paragraf § 11 Odsek 4 Pismeno b)
  Paragraf: Paragraf § 7
  Subjekt: Nadobudatel Podla Paragraf § 11 Odsek 4 Pismeno b)
  Konanie: Nadobudnutie Tovaru
  Tovar: Tovar
  Rozhodnutie: Rozhodnutie O Zdanovani Nadobudnutia Tovaru
  Povinnost: Zdanovanie Nadobudnutia Tovaru
  Suma: Hodnota 14 000 Eur
  Oznamenie: Pisomne Oznamenie Rozhodnutia Danovemu Uradu
  Organizacia: Danovy Urad
  Ziadost: Ziadost O Registraciu Pre Dan
  Registracia: Registracia Pre Dan
  Dan: Dan
  Lehota: Dva Kalendarne Roky

---

chunk: 186
path: ['§ 13', '1', 'a)']
path_as_text: Paragraf § 13 Odsek 1 Pismeno a)
text: (1) Miestom dodania tovaru, a) ak je dodanie tovaru spojené s odoslaním alebo prepravou tovaru, je miesto, kde sa tovar nachádza v čase, keď sa odoslanie alebo preprava tovaru osobe, ktorej má byť tovar dodaný, začína uskutočňovať, s výnimkou podľa písmena b), odseku 2 a § 14,

relations:
  Paragraf § 13 -> [OBSAHUJE] -> Paragraf § 13 Odsek 1
  Paragraf § 13 Odsek 1 -> [OBSAHUJE] -> Paragraf § 13 Odsek 1 Pismeno a)
  Paragraf § 13 -> [OBSAHUJE] -> Paragraf § 13 Odsek 2
  Paragraf § 13 Odsek 1 Pismeno a) -> [UPRAVUJE] -> Dodanie Tovaru
  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [MA_PODMIENKU] -> Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru
  Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Osoba Ktorej Ma Byt Tovar Dodany
  Miesto Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Miesto Dodania Tovaru -> [JE_TYPOM] -> Miesto Kde Sa Tovar Nachadza Pri Zacati Odoslania Alebo Prepravy
  Tovar -> [NACHADZA_SA_V] -> Miesto Kde Sa Tovar Nachadza Pri Zacati Odoslania Alebo Prepravy
  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 13 Odsek 1 Pismeno b)
  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 13 Odsek 2
  Paragraf § 13 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 14

nodes:
  Paragraf: Paragraf § 13
  Odsek: Paragraf § 13 Odsek 1
  Pismeno: Paragraf § 13 Odsek 1 Pismeno a)
  Pismeno: Paragraf § 13 Odsek 1 Pismeno b)
  Odsek: Paragraf § 13 Odsek 2
  Paragraf: Paragraf § 14
  Konanie: Dodanie Tovaru
  Tovar: Tovar
  Konanie: Odoslanie Tovaru
  Konanie: Preprava Tovaru
  Osoba: Osoba Ktorej Ma Byt Tovar Dodany
  Lokacia: Miesto Dodania Tovaru
  Lokacia: Miesto Kde Sa Tovar Nachadza Pri Zacati Odoslania Alebo Prepravy
  Podmienka: Dodanie Tovaru Spojene S Odoslanim Alebo Prepravou Tovaru

---

chunk: 195
path: ['§ 13a', '2']
path_as_text: Paragraf § 13a Odsek 2
text: (2) Na účely odseku 1 je prostrednou osobou dodávateľ, ktorý v reťazci dodaní nie je prvým dodávateľom a ktorý odosiela alebo prepravuje tovar alebo na účet ktorého je tovar odoslaný alebo prepravený treťou osobou.

relations:
  Paragraf § 13A -> [OBSAHUJE] -> Paragraf § 13A Odsek 1
  Paragraf § 13A -> [OBSAHUJE] -> Paragraf § 13A Odsek 2
  Paragraf § 13A Odsek 2 -> [ODKAZUJE_NA] -> Paragraf § 13A Odsek 1
  Paragraf § 13A Odsek 2 -> [DEFINUJE] -> Prostredna Osoba
  Prostredna Osoba -> [JE_TYPOM] -> Dodavatel
  Prostredna Osoba -> [NEVZTAHUJE_SA_NA] -> Prvy Dodavatel
  Prostredna Osoba -> [PATRI_DO] -> Retazec Dodani
  Prostredna Osoba -> [DODAVA] -> Tovar
  Prostredna Osoba -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Prostredna Osoba -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Odoslanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Tretia Osoba -> [VZTAHUJE_SA_NA] -> Odoslanie Tovaru
  Tretia Osoba -> [VZTAHUJE_SA_NA] -> Preprava Tovaru

nodes:
  Paragraf: Paragraf § 13A
  Odsek: Paragraf § 13A Odsek 1
  Odsek: Paragraf § 13A Odsek 2
  Osoba: Prostredna Osoba
  Subjekt: Dodavatel
  Subjekt: Prvy Dodavatel
  Konanie: Retazec Dodani
  Tovar: Tovar
  Osoba: Tretia Osoba
  Konanie: Odoslanie Tovaru
  Konanie: Preprava Tovaru

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
  Miesto Skoncenia Prepravy Tovaru -> [JE_TYPOM] -> Miesto Skutocneho Skoncenia Prepravy Tovaru
  Miesto Skoncenia Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

nodes:
  Paragraf: Paragraf § 16
  Odsek: Paragraf § 16 Odsek 5
  Odsek: Paragraf § 16 Odsek 6
  Odsek: Paragraf § 16 Odsek 7
  Pismeno: Paragraf § 16 Odsek 7 Pismeno c)
  Lokacia: Miesto Skoncenia Prepravy Tovaru
  Lokacia: Miesto Skutocneho Skoncenia Prepravy Tovaru
  Sluzba: Preprava Tovaru
  Tovar: Tovar

---

chunk: 218
path: ['§ 16', '8']
path_as_text: Paragraf § 16 Odsek 8
text: (8) Miestom dodania doplnkových služieb pri preprave, napríklad nakladanie, vykladanie, manipulácia a podobné služby, ak sú tieto služby dodané osobe inej ako zdaniteľnej osobe, je  miesto, kde sa tieto služby fyzicky vykonajú.

relations:
  Paragraf § 16 -> [OBSAHUJE] -> Paragraf § 16 Odsek 8
  Paragraf § 16 Odsek 8 -> [UPRAVUJE] -> Doplnkove Sluzby Pri Preprave
  Nakladanie -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Vykladanie -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Manipulacia -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Podobne Sluzby -> [JE_TYPOM] -> Doplnkove Sluzby Pri Preprave
  Doplnkove Sluzby Pri Preprave -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba
  Paragraf § 16 Odsek 8 -> [URCUJE] -> Miesto Fyzickeho Vykonania Sluzieb
  Miesto Fyzickeho Vykonania Sluzieb -> [VZTAHUJE_SA_NA] -> Doplnkove Sluzby Pri Preprave

nodes:
  Paragraf: Paragraf § 16
  Odsek: Paragraf § 16 Odsek 8
  Sluzba: Doplnkove Sluzby Pri Preprave
  Sluzba: Nakladanie
  Sluzba: Vykladanie
  Sluzba: Manipulacia
  Sluzba: Podobne Sluzby
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Lokacia: Miesto Fyzickeho Vykonania Sluzieb

---

chunk: 241
path: ['§ 16a', '1', 'b)']
path_as_text: Paragraf § 16a Odsek 1 Pismeno b)
text: (1) Miestom dodania tovaru pri predaji tovaru na diaľku na území Európskej únie je miesto, kde sa odoslanie alebo preprava tovaru začína, a miestom dodania pri dodaní telekomunikačných služieb, služieb rozhlasového vysielania a televízneho vysielania a elektronických služieb, ktoré sú dodané osobe inej ako zdaniteľnej osobe, je miesto, kde má dodávateľ služby sídlo, miesto podnikania alebo prevádzkareň, a ak nemá sídlo, miesto podnikania alebo prevádzkareň, miestom dodania služby je jeho bydlisko alebo miesto, kde sa obvykle zdržiava, ak b) tovar sa odosiela alebo prepravuje do iného členského štátu, ako je členský štát podľa písmena a), alebo služba sa dodáva osobe, ktorá má sídlo, bydlisko alebo miesto, kde sa obvykle zdržiava, v inom členskom štáte, ako je členský štát podľa písmena a) a

relations:
  Paragraf § 16a -> [OBSAHUJE] -> Paragraf § 16a Odsek 1
  Paragraf § 16a Odsek 1 -> [OBSAHUJE] -> Paragraf § 16a Odsek 1 Pismeno b)
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie -> [JE_TYPOM] -> Miesto Zacatia Odoslania Alebo Prepravy Tovaru
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Miesto Zacatia Odoslania Alebo Prepravy Tovaru -> [VZTAHUJE_SA_NA] -> Odoslanie Alebo Preprava Tovaru
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Telekomunikacne Sluzby
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzby Rozhlasoveho Vysielania A Televizneho Vysielania
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Elektronicke Sluzby
  Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba
  Miesto Dodania Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb
  Dodavatel Sluzby -> [MA_ADRESU] -> Sidlo Dodavatela Sluzby
  Dodavatel Sluzby -> [MA_ADRESU] -> Miesto Podnikania Dodavatela Sluzby
  Dodavatel Sluzby -> [MA_ADRESU] -> Prevadzkaren Dodavatela Sluzby
  Dodavatel Sluzby -> [MA_ADRESU] -> Bydlisko Dodavatela Sluzby
  Dodavatel Sluzby -> [NACHADZA_SA_V] -> Miesto Kde Sa Dodavatel Sluzby Obvykle Zdrziava
  Miesto Dodania Sluzby -> [JE_TYPOM] -> Sidlo Dodavatela Sluzby
  Miesto Dodania Sluzby -> [JE_TYPOM] -> Miesto Podnikania Dodavatela Sluzby
  Miesto Dodania Sluzby -> [JE_TYPOM] -> Prevadzkaren Dodavatela Sluzby
  Miesto Dodania Sluzby -> [JE_TYPOM] -> Bydlisko Dodavatela Sluzby
  Miesto Dodania Sluzby -> [JE_TYPOM] -> Miesto Kde Sa Dodavatel Sluzby Obvykle Zdrziava
  Tovar -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat Ako Clensky Stat Podla Pismena a)
  Iny Clensky Stat Ako Clensky Stat Podla Pismena a) -> [SUVISI_S] -> Clensky Stat Podla Pismena a)
  Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a) -> [NACHADZA_SA_V] -> Iny Clensky Stat Ako Clensky Stat Podla Pismena a)
  Paragraf § 16a Odsek 1 Pismeno b) -> [VZTAHUJE_SA_NA] -> Tovar
  Paragraf § 16a Odsek 1 Pismeno b) -> [VZTAHUJE_SA_NA] -> Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)

nodes:
  Paragraf: Paragraf § 16a
  Odsek: Paragraf § 16a Odsek 1
  Pismeno: Paragraf § 16a Odsek 1 Pismeno b)
  Lokacia: Miesto Dodania Tovaru Pri Predaji Tovaru Na Dialku Na Uzemi Europskej Unie
  Lokacia: Miesto Zacatia Odoslania Alebo Prepravy Tovaru
  Konanie: Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Tovar: Tovar
  Konanie: Odoslanie Alebo Preprava Tovaru
  Konanie: Dodanie Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania A Televizneho Vysielania A Elektronickych Sluzieb
  Sluzba: Telekomunikacne Sluzby
  Sluzba: Sluzby Rozhlasoveho Vysielania A Televizneho Vysielania
  Sluzba: Elektronicke Sluzby
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Lokacia: Miesto Dodania Sluzby
  Adresa: Sidlo Dodavatela Sluzby
  Adresa: Miesto Podnikania Dodavatela Sluzby
  Adresa: Prevadzkaren Dodavatela Sluzby
  Adresa: Bydlisko Dodavatela Sluzby
  Lokacia: Miesto Kde Sa Dodavatel Sluzby Obvykle Zdrziava
  Osoba: Dodavatel Sluzby
  Stat: Iny Clensky Stat Ako Clensky Stat Podla Pismena a)
  Stat: Clensky Stat Podla Pismena a)
  Osoba: Osoba So Sidlom Bydliskom Alebo Miestom Obvykleho Zdrziavania V Inom Clenskom State Ako Clensky Stat Podla Pismena a)

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
  Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode -> [URCUJE] -> Miesto Podla Paragraf § 17 Odsek 1
  Miesto Nadobudnutia Tovaru Z Ineho Clenskeho Statu Pri Trojstrannom Obchode -> [VZTAHUJE_SA_NA] -> Trojstranny Obchod
  Prvy Odberatel -> [MA] -> Tovar
  Prvy Odberatel -> [DODAVA] -> Nasledne Dodanie Tovaru
  Nasledne Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Nasledne Dodanie Tovaru -> [NACHADZA_SA_V] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [NACHADZA_SA_V] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru
  Druhy Odberatel -> [JE_TYPOM] -> Osoba Identifikovana Pre Dan
  Osoba Identifikovana Pre Dan -> [VZTAHUJE_SA_NA] -> Dan
  Druhy Odberatel -> [NACHADZA_SA_V] -> Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru
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
  Subjekt: Prvy Odberatel
  Subjekt: Druhy Odberatel
  Tovar: Tovar
  Konanie: Nasledne Dodanie Tovaru
  Stat: Clensky Stat Skoncenia Odoslania Alebo Prepravy Tovaru
  Konanie: Odoslanie Alebo Preprava Tovaru
  Dan: Dan
  Osoba: Osoba Identifikovana Pre Dan
  Povinnost: Povinnost Platit Dan

---

chunk: 264
path: ['§ 19', '7']
path_as_text: Paragraf § 19 Odsek 7
text: (7) Pri dodaní tovaru prostredníctvom predajných automatov, prípadne iných obdobných prístrojov uvádzaných do chodu mincami, bankovkami, známkami alebo inými platobnými prostriedkami nahrádzajúcimi peniaze vzniká daňová povinnosť dňom, keď sa vyberú peniaze alebo známky z prístroja alebo iným spôsobom sa zistí výška obratu.

relations:
  Paragraf § 19 -> [OBSAHUJE] -> Paragraf § 19 Odsek 7
  Paragraf § 19 Odsek 7 -> [UPRAVUJE] -> Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov
  Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru Prostrednictvom Predajnych Automatov
  Dodanie Tovaru Prostrednictvom Predajnych Automatov -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru Prostrednictvom Predajnych Automatov -> [SUVISI_S] -> Predajne Automaty Alebo Obdobne Pristroje
  Predajne Automaty Alebo Obdobne Pristroje -> [MA] -> Mince Bankovky Znamky Alebo Ine Platobne Prostriedky Nahradzajuce Peniaze
  Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov -> [VZNIKA] -> Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu
  Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu -> [SUVISI_S] -> Vyska Obratu

nodes:
  Paragraf: Paragraf § 19
  Odsek: Paragraf § 19 Odsek 7
  Konanie: Dodanie Tovaru Prostrednictvom Predajnych Automatov
  Tovar: Tovar
  Majetok: Predajne Automaty Alebo Obdobne Pristroje
  Platba: Mince Bankovky Znamky Alebo Ine Platobne Prostriedky Nahradzajuce Peniaze
  Povinnost: Danova Povinnost Pri Dodani Tovaru Prostrednictvom Predajnych Automatov
  Datum: Den Vybratia Penazi Alebo Znamok Z Pristroja Alebo Zistenia Vysky Obratu
  Obrat: Vyska Obratu

---

chunk: 279
path: ['§ 21', '4']
path_as_text: Paragraf § 21 Odsek 4
text: (4) Ak daňová povinnosť pri dovoze tovaru vznikne podľa odseku 1 písm. c), daň sa zníži o sumu dane zaplatenej pri prepustení tovaru do voľného obehu vrátane konečného použitia alebo pri prepustení do colného režimu dočasné použitie s čiastočným oslobodením od dovozného cla alebo o sumu dane priznanej podľa § 84a ods. 3.

relations:
  Paragraf § 21 -> [OBSAHUJE] -> Paragraf § 21 Odsek 4
  Paragraf § 21 -> [OBSAHUJE] -> Paragraf § 21 Odsek 1
  Paragraf § 21 Odsek 1 -> [OBSAHUJE] -> Paragraf § 21 Odsek 1 Pismeno c)
  Paragraf § 84A -> [OBSAHUJE] -> Paragraf § 84A Odsek 3
  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dan
  Znizenie Dane -> [MENI] -> Dan
  Znizenie Dane -> [MA_SUMU] -> Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [VYPLYVA_Z] -> Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [VZTAHUJE_SA_NA] -> Tovar
  Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia -> [MA_STATUS] -> Volny Obeh Vratane Konecneho Pouzitia
  Znizenie Dane -> [MA_SUMU] -> Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [VYPLYVA_Z] -> Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [MA_STATUS] -> Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla -> [OSLOBODZUJE_OD] -> Dovozne Clo
  Znizenie Dane -> [MA_SUMU] -> Suma Dane Priznanej Podla Paragrafu § 84A Odsek 3

nodes:
  Paragraf: Paragraf § 21
  Odsek: Paragraf § 21 Odsek 4
  Odsek: Paragraf § 21 Odsek 1
  Pismeno: Paragraf § 21 Odsek 1 Pismeno c)
  Paragraf: Paragraf § 84A
  Odsek: Paragraf § 84A Odsek 3
  Povinnost: Danova Povinnost Pri Dovoze Tovaru
  Konanie: Dovoz Tovaru
  Tovar: Tovar
  Dan: Dan
  Konanie: Znizenie Dane
  Suma: Suma Dane Zaplatenej Pri Prepusteni Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Konanie: Prepustenie Tovaru Do Volneho Obehu Vratane Konecneho Pouzitia
  Status: Volny Obeh Vratane Konecneho Pouzitia
  Suma: Suma Dane Zaplatenej Pri Prepusteni Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Konanie: Prepustenie Do Colneho Rezimu Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Status: Colny Rezim Docasne Pouzitie S Ciastocnym Oslobodenim Od Dovozneho Cla
  Dan: Dovozne Clo
  Suma: Suma Dane Priznanej Podla Paragrafu § 84A Odsek 3

---

chunk: 287
path: ['§ 22', '3']
path_as_text: Paragraf § 22 Odsek 3
text: (3) Do základu dane podľa odseku 1 sa nezahŕňajú výdavky platené v mene a na účet kupujúceho alebo zákazníka, ktoré dodávateľ požaduje od kupujúceho alebo zákazníka (ďalej len „prechodné položky“). Pri dodaní tovaru v zálohovaných obaloch sa do základu dane podľa odseku 1 nezahŕňa záloha na zálohované obaly, ktoré sú dodané spolu s tovarom. Pri dodaní nápoja v zálohovanom jednorazovom obale na nápoje6abd) sa do základu dane podľa odseku 1 nezahŕňa záloh na tento obal.

relations:
  Paragraf § 22 -> [OBSAHUJE] -> Paragraf § 22 Odsek 3
  Paragraf § 22 -> [OBSAHUJE] -> Paragraf § 22 Odsek 1
  Paragraf § 22 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 22 Odsek 1
  Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika -> [NEVZTAHUJE_SA_NA] -> Zaklad Dane
  Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika -> [JE_TYPOM] -> Prechodne Polozky
  Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika -> [VZTAHUJE_SA_NA] -> Kupujuci
  Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika -> [VZTAHUJE_SA_NA] -> Zakaznik
  Dodavatel -> [MA_NAROK_NA] -> Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
  Dodanie Tovaru V Zalohovanych Obaloch -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru V Zalohovanych Obaloch -> [VZTAHUJE_SA_NA] -> Zalohovane Obaly
  Zaloha Na Zalohovane Obaly -> [NEVZTAHUJE_SA_NA] -> Zaklad Dane
  Zaloha Na Zalohovane Obaly -> [VZTAHUJE_SA_NA] -> Zalohovane Obaly
  Dodanie Napoja V Zalohovanom Jednorazovom Obale Na Napoje -> [VZTAHUJE_SA_NA] -> Napoj
  Dodanie Napoja V Zalohovanom Jednorazovom Obale Na Napoje -> [VZTAHUJE_SA_NA] -> Zalohovany Jednorazovy Obal Na Napoje
  Zaloha Na Zalohovany Jednorazovy Obal Na Napoje -> [NEVZTAHUJE_SA_NA] -> Zaklad Dane
  Zaloha Na Zalohovany Jednorazovy Obal Na Napoje -> [VZTAHUJE_SA_NA] -> Zalohovany Jednorazovy Obal Na Napoje

nodes:
  Paragraf: Paragraf § 22
  Odsek: Paragraf § 22 Odsek 3
  Odsek: Paragraf § 22 Odsek 1
  Dan: Zaklad Dane
  Suma: Vydavky Platene V Mene A Na Ucet Kupujuceho Alebo Zakaznika
  Suma: Prechodne Polozky
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
  Prve Miesto Urcenia V Tuzemsku -> [VYPLYVA_Z] -> Nakladny List
  Prve Miesto Urcenia V Tuzemsku -> [VYPLYVA_Z] -> Iny Sprievodny Dokument
  Nakladny List -> [SUVISI_S] -> Dovazany Tovar
  Iny Sprievodny Dokument -> [SUVISI_S] -> Dovazany Tovar
  Dovazany Tovar -> [NACHADZA_SA_V] -> Tuzemsko
  Prve Miesto Urcenia V Tuzemsku -> [MA_PODMIENKU] -> Miesto Prvej Prekladky Tovaru V Tuzemsku
  Miesto Prvej Prekladky Tovaru V Tuzemsku -> [MA_PODMIENKU] -> Neuvedenie Prveho Miesta Urcenia V Tuzemsku
  Miesto Prvej Prekladky Tovaru V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Miesto Prvej Prekladky Tovaru V Tuzemsku -> [SUVISI_S] -> Dovazany Tovar

nodes:
  Paragraf: Paragraf § 24
  Odsek: Paragraf § 24 Odsek 3
  Odsek: Paragraf § 24 Odsek 2
  Pismeno: Paragraf § 24 Odsek 2 Pismeno b)
  Lokacia: Prve Miesto Urcenia V Tuzemsku
  Stat: Tuzemsko
  Dokument: Nakladny List
  Dokument: Iny Sprievodny Dokument
  Tovar: Dovazany Tovar
  Lokacia: Miesto Prvej Prekladky Tovaru V Tuzemsku
  Podmienka: Neuvedenie Prveho Miesta Urcenia V Tuzemsku

---

chunk: 320
path: ['§ 25', '5', 'a)']
path_as_text: Paragraf § 25 Odsek 5 Pismeno a)
text: (5) Ak pri dovoze tovaru vznikne daňová povinnosť v tuzemsku právnickej osobe z iného členského štátu, ktorá nie je zdaniteľnou osobou, colný orgán vráti tejto osobe daň zaplatenú pri dovoze, ak a) ide o tovar odoslaný alebo prepravený z územia tretieho štátu a miestom určenia tovaru je iný členský štát ako tuzemsko a

relations:
  Paragraf § 25 -> [OBSAHUJE] -> Paragraf § 25 Odsek 5
  Paragraf § 25 Odsek 5 -> [OBSAHUJE] -> Paragraf § 25 Odsek 5 Pismeno a)
  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Danova Povinnost Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Pravnicka Osoba Z Ineho Clenskeho Statu
  Danova Povinnost Pri Dovoze Tovaru -> [NACHADZA_SA_V] -> Tuzemsko
  Pravnicka Osoba Z Ineho Clenskeho Statu -> [PATRI_DO] -> Iny Clensky Stat
  Pravnicka Osoba Z Ineho Clenskeho Statu -> [MA_STATUS] -> Nie Je Zdanitelnou Osobou
  Colny Organ -> [PLATI] -> Dan Zaplatena Pri Dovoze
  Pravnicka Osoba Z Ineho Clenskeho Statu -> [MA_NAROK_NA] -> Vratenie Dane Zaplatenej Pri Dovoze
  Vratenie Dane Zaplatenej Pri Dovoze -> [VZTAHUJE_SA_NA] -> Dan Zaplatena Pri Dovoze
  Vratenie Dane Zaplatenej Pri Dovoze -> [MA_PODMIENKU] -> Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu
  Vratenie Dane Zaplatenej Pri Dovoze -> [MA_PODMIENKU] -> Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Tovar Odoslany Alebo Prepraveny Z Uzemia Tretieho Statu -> [VZTAHUJE_SA_NA] -> Uzemie Tretieho Statu
  Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko -> [VZTAHUJE_SA_NA] -> Iny Clensky Stat
  Miesto Urcenia Tovaru Je Iny Clensky Stat Ako Tuzemsko -> [NEVZTAHUJE_SA_NA] -> Tuzemsko

nodes:
  Paragraf: Paragraf § 25
  Odsek: Paragraf § 25 Odsek 5
  Pismeno: Paragraf § 25 Odsek 5 Pismeno a)
  Osoba: Pravnicka Osoba Z Ineho Clenskeho Statu
  Status: Nie Je Zdanitelnou Osobou
  Organizacia: Colny Organ
  Dan: Dan Zaplatena Pri Dovoze
  Povinnost: Danova Povinnost Pri Dovoze Tovaru
  Konanie: Dovoz Tovaru
  Tovar: Tovar
  Stat: Tuzemsko
  Stat: Iny Clensky Stat
  Lokacia: Uzemie Tretieho Statu
  Pravo: Vratenie Dane Zaplatenej Pri Dovoze
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
  Platitel -> [NEMA_NAROK_NA] -> Oprava Zakladu Dane
  Oprava Zakladu Dane -> [VZTAHUJE_SA_NA] -> Nevymozitelna Pohladavka
  Tovar -> [DODAVA] -> Odberatel Dlznik
  Sluzba -> [POSKYTUJE] -> Odberatel Dlznik
  Konkurz Na Majetok Odberatela Dlznika -> [VZTAHUJE_SA_NA] -> Majetok Odberatela Dlznika
  Majetok Odberatela Dlznika -> [PATRI_DO] -> Odberatel Dlznik

nodes:
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 4
  Pismeno: Paragraf § 25a Odsek 4 Pismeno b)
  Odsek: Paragraf § 25a Odsek 2
  Subjekt: Platitel
  Konanie: Oprava Zakladu Dane
  Pohladavka: Nevymozitelna Pohladavka
  Tovar: Tovar
  Sluzba: Sluzba
  Subjekt: Odberatel Dlznik
  Konanie: Konkurz Na Majetok Odberatela Dlznika
  Majetok: Majetok Odberatela Dlznika

---

chunk: 356
path: ['§ 25a', '10', 'e)']
path_as_text: Paragraf § 25a Odsek 10 Pismeno e)
text: (10) Opravný doklad podľa odseku 7 písm. b) musí obsahovať e) sumu, ktorú platiteľ prijal v súvislosti s nevymožiteľnou pohľadávkou podľa odseku 2 alebo jej časťou, a z toho sumu prislúchajúcej dane,

relations:
  Opravny Doklad -> [ODKAZUJE_NA] -> Paragraf § 25A Odsek 7 Pismeno b)
  Paragraf § 25A Odsek 10 Pismeno e) -> [URCUJE] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Opravny Doklad -> [OBSAHUJE] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Platitel -> [PRIJIMA] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou -> [SUVISI_S] -> Nevymozitelna Pohladavka
  Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou -> [SUVISI_S] -> Cast Nevymozitelnej Pohladavky
  Nevymozitelna Pohladavka -> [ODKAZUJE_NA] -> Paragraf § 25A Odsek 2
  Cast Nevymozitelnej Pohladavky -> [JE_SUCASTOU] -> Nevymozitelna Pohladavka
  Opravny Doklad -> [OBSAHUJE] -> Suma Prisluchajucej Dane
  Suma Prisluchajucej Dane -> [MA_HODNOTU] -> Prisluchajuca Dan

nodes:
  Pismeno: Paragraf § 25A Odsek 10 Pismeno e)
  Pismeno: Paragraf § 25A Odsek 7 Pismeno b)
  Odsek: Paragraf § 25A Odsek 2
  Dokument: Opravny Doklad
  Subjekt: Platitel
  Pohladavka: Nevymozitelna Pohladavka
  Pohladavka: Cast Nevymozitelnej Pohladavky
  Suma: Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Suma: Suma Prisluchajucej Dane
  Dan: Prisluchajuca Dan

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
  Platitel -> [KONA_V_MENE] -> Oprava Zakladu Dane
  Oprava Zakladu Dane -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 3
  Platitel -> [MA_POVINNOST] -> Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a)
  Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a) -> [VYPLYVA_Z] -> Paragraf § 25a Odsek 11 Pismeno a)
  Oprava Zakladu Dane -> [JE_SUCASTOU] -> Danove Priznanie
  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Oprava Zakladu Dane -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Oprava Zakladu Dane -> [MA_STATUS] -> Neuznana Oprava Zakladu Dane

nodes:
  Paragraf: Paragraf § 25a
  Odsek: Paragraf § 25a Odsek 13
  Odsek: Paragraf § 25a Odsek 3
  Odsek: Paragraf § 25a Odsek 11
  Pismeno: Paragraf § 25a Odsek 11 Pismeno a)
  Subjekt: Platitel
  Konanie: Oprava Zakladu Dane
  Povinnost: Povinnost Podla Paragraf § 25a Odsek 11 Pismeno a)
  DanovePriznanie: Danove Priznanie
  ZdanovacieObdobie: Zdanovacie Obdobie
  Status: Neuznana Oprava Zakladu Dane

---

chunk: 379
path: ['§ 27', '4']
path_as_text: Paragraf § 27 Odsek 4
text: (4) Na účely správneho zatriedenia tovaru do číselného kódu podľa prílohy č. 7 sa použije záväzná informácia o nomenklatúrnom zatriedení tovaru vydaná colným orgánom podľa osobitného predpisu.6b)

relations:
  Paragraf § 27 -> [OBSAHUJE] -> Paragraf § 27 Odsek 4
  Ciselny Kod Podla Prilohy C. 7 -> [VYPLYVA_Z] -> Priloha C. 7
  Tovar -> [MA_IDENTIFIKATOR] -> Ciselny Kod Podla Prilohy C. 7
  Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru -> [VZTAHUJE_SA_NA] -> Nomenklaturne Zatriedenie Tovaru
  Nomenklaturne Zatriedenie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Colny Organ -> [VYDAVA] -> Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru
  Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru -> [ODKAZUJE_NA] -> Osobitny Predpis
  Zavazna Informacia O Nomenklaturnom Zatriedeni Tovaru -> [VYPLYVA_Z] -> Paragraf § 27 Odsek 4

nodes:
  Paragraf: Paragraf § 27
  Odsek: Paragraf § 27 Odsek 4
  Tovar: Tovar
  Zaznam: Ciselny Kod Podla Prilohy C. 7
  Priloha: Priloha C. 7
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
  Kulturne Sluzby -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Tovary Uzko Suvisiace S Kulturnymi Sluzbami -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Tovary Uzko Suvisiace S Kulturnymi Sluzbami -> [SUVISI_S] -> Kulturne Sluzby
  Kulturne Sluzby -> [MA_PODMIENKU] -> Pravnicka Osoba Zriadena Zakonom
  Kulturne Sluzby -> [MA_PODMIENKU] -> Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky
  Kulturne Sluzby -> [MA_PODMIENKU] -> Pravnicka Osoba Zriadena Vyssim Uzemnym Celkom
  Kulturne Sluzby -> [MA_PODMIENKU] -> Pravnicka Osoba Zriadena Obcou
  Kulturne Sluzby -> [MA_PODMIENKU] -> Pravnicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2
  Kulturne Sluzby -> [MA_PODMIENKU] -> Fyzicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2
  Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky -> [VYPLYVA_Z] -> Ministerstvo Kultury Slovenskej Republiky
  Pravnicka Osoba Zriadena Vyssim Uzemnym Celkom -> [VYPLYVA_Z] -> Vyssi Uzemny Celok
  Pravnicka Osoba Zriadena Obcou -> [VYPLYVA_Z] -> Obec
  Pravnicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2 -> [SPLNA_PODMIENKY] -> Podmienka Podla Paragrafu § 30 Odsek 2
  Fyzicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2 -> [SPLNA_PODMIENKY] -> Podmienka Podla Paragrafu § 30 Odsek 2

nodes:
  Paragraf: Paragraf § 34
  Paragraf: Paragraf § 30
  Odsek: Paragraf § 30 Odsek 2
  Dan: Dan
  Sluzba: Kulturne Sluzby
  Tovar: Tovary Uzko Suvisiace S Kulturnymi Sluzbami
  Osoba: Pravnicka Osoba Zriadena Zakonom
  Osoba: Pravnicka Osoba Zriadena Ministerstvom Kultury Slovenskej Republiky
  Osoba: Pravnicka Osoba Zriadena Vyssim Uzemnym Celkom
  Osoba: Pravnicka Osoba Zriadena Obcou
  Osoba: Pravnicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2
  Osoba: Fyzicka Osoba Splnajuca Podmienku Podla Paragrafu § 30 Odsek 2
  Podmienka: Podmienka Podla Paragrafu § 30 Odsek 2
  Organizacia: Ministerstvo Kultury Slovenskej Republiky
  Organizacia: Vyssi Uzemny Celok
  Organizacia: Obec

---

chunk: 425
path: ['§ 39', '1', 'c)']
path_as_text: Paragraf § 39 Odsek 1 Pismeno c)
text: (1) Oslobodené od dane sú: c) činnosti týkajúce sa vkladov a bežných účtov vrátane ich sprostredkovania,

relations:
  Paragraf § 39 -> [OBSAHUJE] -> Paragraf § 39 Odsek 1
  Paragraf § 39 Odsek 1 -> [OBSAHUJE] -> Paragraf § 39 Odsek 1 Pismeno c)
  Cinnosti Tykajuce Sa Vkladov -> [VZTAHUJE_SA_NA] -> Vklady
  Cinnosti Tykajuce Sa Beznych Uctov -> [VZTAHUJE_SA_NA] -> Bezne Ucty
  Sprostredkovanie Cinnosti Tykajucich Sa Vkladov -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Vkladov
  Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov -> [VZTAHUJE_SA_NA] -> Cinnosti Tykajuce Sa Beznych Uctov
  Cinnosti Tykajuce Sa Vkladov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Cinnosti Tykajuce Sa Beznych Uctov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Sprostredkovanie Cinnosti Tykajucich Sa Vkladov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Paragraf § 39 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Cinnosti Tykajuce Sa Vkladov
  Paragraf § 39 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Cinnosti Tykajuce Sa Beznych Uctov
  Paragraf § 39 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Sprostredkovanie Cinnosti Tykajucich Sa Vkladov
  Paragraf § 39 Odsek 1 Pismeno c) -> [UPRAVUJE] -> Sprostredkovanie Cinnosti Tykajucich Sa Beznych Uctov

nodes:
  Paragraf: Paragraf § 39
  Odsek: Paragraf § 39 Odsek 1
  Pismeno: Paragraf § 39 Odsek 1 Pismeno c)
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
  Tovar -> [VYPLYVA_Z] -> Tuzemsko
  Tovar -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Predavajuci -> [DODAVA] -> Tovar
  Nadobudatel -> [NADOBUDA] -> Tovar
  Tretia Osoba -> [KONA_V_MENE] -> Predavajuci
  Tretia Osoba -> [KONA_V_MENE] -> Nadobudatel
  Nadobudatel -> [VYPLYVA_Z] -> Paragraf § 43 Odsek 1 Pismeno a)
  Nadobudatel -> [MA] -> Identifikacia Pre Dan V Inom Clenskom State
  Identifikacia Pre Dan V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacia Pre Dan V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Nadobudatel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State
  Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Dan
  Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State -> [VYPLYVA_Z] -> Iny Clensky Stat
  Nadobudatel -> [OZNAMUJE] -> Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State
  Nadobudatel -> [OZNAMUJE] -> Dodavatel

nodes:
  Paragraf: Paragraf § 43
  Odsek: Paragraf § 43 Odsek 1
  Pismeno: Paragraf § 43 Odsek 1 Pismeno a)
  Pismeno: Paragraf § 43 Odsek 1 Pismeno b)
  Konanie: Dodanie Tovaru
  Tovar: Tovar
  Dan: Dan
  Lokacia: Tuzemsko
  Stat: Iny Clensky Stat
  Osoba: Predavajuci
  Osoba: Nadobudatel
  Osoba: Tretia Osoba
  Osoba: Dodavatel
  Registracia: Identifikacia Pre Dan V Inom Clenskom State
  Zaznam: Identifikacne Cislo Pre Dan Pridelene V Inom Clenskom State

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
  Platitel -> [MA_POVINNOST] -> Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane
  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane -> [VZTAHUJE_SA_NA] -> Podmienky Oslobodenia Od Dane
  Podmienky Oslobodenia Od Dane -> [VZTAHUJE_SA_NA] -> Dan
  Podmienky Oslobodenia Od Dane -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 1
  Podmienky Oslobodenia Od Dane -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 2
  Podmienky Oslobodenia Od Dane -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 3
  Podmienky Oslobodenia Od Dane -> [ODKAZUJE_NA] -> Paragraf § 43 Odsek 4
  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane -> [MA] -> Doklad O Odoslani Tovaru
  Doklad O Odoslani Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodavatel -> [POSKYTUJE] -> Preprava Tovaru
  Odberatel -> [POSKYTUJE] -> Preprava Tovaru
  Postovy Podnik -> [POSKYTUJE] -> Preprava Tovaru
  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane -> [MA] -> Kopia Dokladu O Preprave Tovaru
  Kopia Dokladu O Preprave Tovaru -> [VZTAHUJE_SA_NA] -> Preprava Tovaru
  Kopia Dokladu O Preprave Tovaru -> [OBSAHUJE] -> Prevzatie Tovaru V Inom Clenskom State
  Odberatel -> [PRIJIMA] -> Prevzatie Tovaru V Inom Clenskom State
  Osoba Poverena Odberatelom -> [KONA_V_MENE] -> Odberatel
  Osoba Poverena Odberatelom -> [PRIJIMA] -> Prevzatie Tovaru V Inom Clenskom State
  Prevzatie Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Tovar
  Prevzatie Tovaru V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane -> [MA] -> Iny Doklad
  Iny Doklad -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru V Inom Clenskom State

nodes:
  Paragraf: Paragraf § 43
  Odsek: Paragraf § 43 Odsek 5
  Pismeno: Paragraf § 43 Odsek 5 Pismeno b)
  Odsek: Paragraf § 43 Odsek 1
  Odsek: Paragraf § 43 Odsek 2
  Odsek: Paragraf § 43 Odsek 3
  Odsek: Paragraf § 43 Odsek 4
  Subjekt: Platitel
  Povinnost: Povinnost Preukazat Splnenie Podmienok Oslobodenia Od Dane
  Podmienka: Podmienky Oslobodenia Od Dane
  Dan: Dan
  Dokument: Doklad O Odoslani Tovaru
  Dokument: Kopia Dokladu O Preprave Tovaru
  Dokument: Iny Doklad
  Tovar: Tovar
  Sluzba: Preprava Tovaru
  Subjekt: Dodavatel
  Subjekt: Odberatel
  Organizacia: Postovy Podnik
  Osoba: Osoba Poverena Odberatelom
  Stat: Iny Clensky Stat
  Konanie: Prevzatie Tovaru V Inom Clenskom State

---

chunk: 471
path: ['§ 45', '4', 'b)']
path_as_text: Paragraf § 45 Odsek 4 Pismeno b)
text: (4) Zo záznamov vedených na určenie dane musí byť zrejmé b) u druhého odberateľa, ak použije pri trojstrannom obchode identifikačné číslo pre daň pridelené v tuzemsku, základ dane, suma dane a názov alebo meno a adresa prvého odberateľa.

relations:
  Paragraf § 45 -> [OBSAHUJE] -> Paragraf § 45 Odsek 4
  Paragraf § 45 Odsek 4 -> [OBSAHUJE] -> Paragraf § 45 Odsek 4 Pismeno b)
  Zaznamy Vedene Na Urcenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Druhy Odberatel -> [SUVISI_S] -> Trojstranny Obchod
  Druhy Odberatel -> [MA_IDENTIFIKATOR] -> Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Zaznamy Vedene Na Urcenie Dane -> [OBSAHUJE] -> Zaklad Dane
  Zaznamy Vedene Na Urcenie Dane -> [OBSAHUJE] -> Suma Dane
  Zaznamy Vedene Na Urcenie Dane -> [OBSAHUJE] -> Nazov Alebo Meno Prveho Odberatela
  Zaznamy Vedene Na Urcenie Dane -> [OBSAHUJE] -> Adresa Prveho Odberatela
  Prvy Odberatel -> [MA_ADRESU] -> Adresa Prveho Odberatela
  Prvy Odberatel -> [MA_IDENTIFIKATOR] -> Nazov Alebo Meno Prveho Odberatela

nodes:
  Paragraf: Paragraf § 45
  Odsek: Paragraf § 45 Odsek 4
  Pismeno: Paragraf § 45 Odsek 4 Pismeno b)
  Zaznam: Zaznamy Vedene Na Urcenie Dane
  Dan: Dan
  Subjekt: Druhy Odberatel
  Konanie: Trojstranny Obchod
  Dokument: Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Suma: Zaklad Dane
  Suma: Suma Dane
  Subjekt: Prvy Odberatel
  Dokument: Nazov Alebo Meno Prveho Odberatela
  Adresa: Adresa Prveho Odberatela

---

chunk: 484
path: ['§ 47', '6']
path_as_text: Paragraf § 47 Odsek 6
text: (6) Oslobodené od dane sú služby vrátane prepravných a s nimi súvisiacich doplnkových služieb, iné ako služby oslobodené od dane podľa § 28 až 41, ktoré sú priamo spojené s vývozom tovaru a s tovarom pod colným opatrením podľa § 18 ods. 2.

relations:
  Paragraf § 47 -> [OBSAHUJE] -> Paragraf § 47 Odsek 6
  Sluzby Priamo Spojene S Vyvozom Tovaru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Prepravne Sluzby Priamo Spojene S Vyvozom Tovaru -> [JE_TYPOM] -> Sluzby Priamo Spojene S Vyvozom Tovaru
  Doplnkove Sluzby Suvisiace S Prepravnymi Sluzbami Priamo Spojene S Vyvozom Tovaru -> [JE_TYPOM] -> Sluzby Priamo Spojene S Vyvozom Tovaru
  Sluzby Priamo Spojene S Vyvozom Tovaru -> [SUVISI_S] -> Vyvoz Tovaru
  Vyvoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Sluzby Priamo Spojene S Vyvozom Tovaru -> [SUVISI_S] -> Tovar Pod Colnym Opatrenim
  Paragraf § 18 -> [OBSAHUJE] -> Paragraf § 18 Odsek 2
  Sluzby Priamo Spojene S Vyvozom Tovaru -> [NEVZTAHUJE_SA_NA] -> Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41

nodes:
  Paragraf: Paragraf § 47
  Odsek: Paragraf § 47 Odsek 6
  Dan: Dan
  Sluzba: Sluzby Priamo Spojene S Vyvozom Tovaru
  Sluzba: Prepravne Sluzby Priamo Spojene S Vyvozom Tovaru
  Sluzba: Doplnkove Sluzby Suvisiace S Prepravnymi Sluzbami Priamo Spojene S Vyvozom Tovaru
  Konanie: Vyvoz Tovaru
  Tovar: Tovar
  Tovar: Tovar Pod Colnym Opatrenim
  Sluzba: Sluzby Oslobodene Od Dane Podla Paragrafov § 28 Az § 41
  Paragraf: Paragraf § 18
  Odsek: Paragraf § 18 Odsek 2
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

---

chunk: 494
path: ['§ 47', '13', 'b)']
path_as_text: Paragraf § 47 Odsek 13 Pismeno b)
text: (13) Oslobodené od dane je bezodplatné dodanie tovaru formou daru poskytnutého na základe  písomnej darovacej zmluvy uzatvorenej medzi platiteľom a Ministerstvom vnútra Slovenskej republiky na účel vývozu tovaru mimo územia Európskej únie ako súčasť humanitárnej činnosti a dobročinnej činnosti. Ministerstvo vnútra Slovenskej republiky za každý kalendárny rok do 15. januára nasledujúceho kalendárneho roka predloží finančnému riaditeľstvu b) zoznam evidenčných čísiel colných vyhlásení o vývoze tovaru darovaného platiteľom za príslušný kalendárny rok.

relations:
  Paragraf § 47 -> [OBSAHUJE] -> Paragraf § 47 Odsek 13
  Paragraf § 47 Odsek 13 -> [OBSAHUJE] -> Paragraf § 47 Odsek 13 Pismeno b)
  Bezodplatne Dodanie Tovaru Formou Daru -> [VYPLYVA_Z] -> Paragraf § 47 Odsek 13
  Bezodplatne Dodanie Tovaru Formou Daru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Bezodplatne Dodanie Tovaru Formou Daru -> [VYPLYVA_Z] -> Pisomna Darovacia Zmluva
  Pisomna Darovacia Zmluva -> [VZTAHUJE_SA_NA] -> Platitel
  Pisomna Darovacia Zmluva -> [VZTAHUJE_SA_NA] -> Ministerstvo Vnutra Slovenskej Republiky
  Pisomna Darovacia Zmluva -> [VZTAHUJE_SA_NA] -> Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [NEVZTAHUJE_SA_NA] -> Uzemie Europskej Unie
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [JE_SUCASTOU] -> Humanitarna Cinnost
  Vyvoz Tovaru Mimo Uzemia Europskej Unie -> [JE_SUCASTOU] -> Dobrocinna Cinnost
  Ministerstvo Vnutra Slovenskej Republiky -> [PREDKLADA] -> Zoznam Evidencnych Cisiel Colnych Vyhlaseni
  Ministerstvo Vnutra Slovenskej Republiky -> [PREDKLADA] -> Financne Riaditelstvo
  Ministerstvo Vnutra Slovenskej Republiky -> [MA_LEHOTU] -> Do 15. Januara Nasledujuceho Kalendarskeho Roka
  Do 15. Januara Nasledujuceho Kalendarskeho Roka -> [MA_OBDOBIE] -> Nasledujuci Kalendarny Rok
  Ministerstvo Vnutra Slovenskej Republiky -> [MA_OBDOBIE] -> Kalendarny Rok
  Zoznam Evidencnych Cisiel Colnych Vyhlaseni -> [VZTAHUJE_SA_NA] -> Colne Vyhlasenie O Vyvoze Tovaru
  Colne Vyhlasenie O Vyvoze Tovaru -> [VZTAHUJE_SA_NA] -> Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Colne Vyhlasenie O Vyvoze Tovaru -> [VZTAHUJE_SA_NA] -> Darovany Tovar
  Darovany Tovar -> [VYPLYVA_Z] -> Platitel
  Zoznam Evidencnych Cisiel Colnych Vyhlaseni -> [MA_OBDOBIE] -> Prislusny Kalendarny Rok
  Paragraf § 47 Odsek 13 Pismeno b) -> [UPRAVUJE] -> Zoznam Evidencnych Cisiel Colnych Vyhlaseni

nodes:
  Paragraf: Paragraf § 47
  Odsek: Paragraf § 47 Odsek 13
  Pismeno: Paragraf § 47 Odsek 13 Pismeno b)
  Konanie: Bezodplatne Dodanie Tovaru Formou Daru
  Dan: Dan
  Zmluva: Pisomna Darovacia Zmluva
  Subjekt: Platitel
  Organizacia: Ministerstvo Vnutra Slovenskej Republiky
  Konanie: Vyvoz Tovaru Mimo Uzemia Europskej Unie
  Lokacia: Uzemie Europskej Unie
  Konanie: Humanitarna Cinnost
  Konanie: Dobrocinna Cinnost
  Obdobie: Kalendarny Rok
  Lehota: Do 15. Januara Nasledujuceho Kalendarskeho Roka
  Obdobie: Nasledujuci Kalendarny Rok
  Organizacia: Financne Riaditelstvo
  Dokument: Zoznam Evidencnych Cisiel Colnych Vyhlaseni
  Dokument: Colne Vyhlasenie O Vyvoze Tovaru
  Tovar: Darovany Tovar
  Obdobie: Prislusny Kalendarny Rok

---

chunk: 517
path: ['§ 48', '2', 'w)']
path_as_text: Paragraf § 48 Odsek 2 Pismeno w)
text: (2) Tovar, ktorý je prepustený do colného režimu voľný obeh s oslobodením od cla podľa osobitného predpisu,22) je oslobodený od dane, ak ide o w) rôzne dokumenty a predmety,

relations:
  Paragraf § 48 -> [OBSAHUJE] -> Paragraf § 48 Odsek 2
  Paragraf § 48 Odsek 2 -> [OBSAHUJE] -> Paragraf § 48 Odsek 2 Pismeno w)
  Rozne Dokumenty A Predmety -> [JE_TYPOM] -> Tovar
  Tovar -> [MA_PODMIENKU] -> Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu
  Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu -> [ODKAZUJE_NA] -> Osobitny Predpis
  Rozne Dokumenty A Predmety -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Paragraf § 48 Odsek 2 Pismeno w) -> [UPRAVUJE] -> Rozne Dokumenty A Predmety

nodes:
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 2
  Pismeno: Paragraf § 48 Odsek 2 Pismeno w)
  Tovar: Tovar
  Tovar: Rozne Dokumenty A Predmety
  Dan: Dan
  Podmienka: Prepustenie Do Colneho Rezimu Volny Obeh S Oslobodenim Od Cla Podla Osobitneho Predpisu
  PravnyPredpis: Osobitny Predpis

---

chunk: 525
path: ['§ 48', '5', 'a)']
path_as_text: Paragraf § 48 Odsek 5 Pismeno a)
text: (5) Oslobodený od dane je dovoz tovaru a) osobami, ktoré požívajú výsady a imunity podľa medzinárodného práva,23) ak sa na tento dovoz vzťahuje oslobodenie od cla,

relations:
  Paragraf § 48 -> [OBSAHUJE] -> Paragraf § 48 Odsek 5
  Paragraf § 48 Odsek 5 -> [OBSAHUJE] -> Paragraf § 48 Odsek 5 Pismeno a)
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dovoz Tovaru -> [JE_OSLOBODENE_OD_DANE] -> Dan
  Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava -> [MA_PRAVO] -> Vysady A Imunity Podla Medzinarodneho Prava
  Vysady A Imunity Podla Medzinarodneho Prava -> [VYPLYVA_Z] -> Medzinarodne Pravo
  Dovoz Tovaru -> [VZTAHUJE_SA_NA] -> Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava
  Dovoz Tovaru -> [MA_PODMIENKU] -> Oslobodenie Od Cla
  Oslobodenie Od Cla -> [OSLOBODZUJE_OD] -> Clo

nodes:
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 5
  Pismeno: Paragraf § 48 Odsek 5 Pismeno a)
  Konanie: Dovoz Tovaru
  Tovar: Tovar
  Osoba: Osoby Pozivajuce Vysady A Imunity Podla Medzinarodneho Prava
  Pravo: Vysady A Imunity Podla Medzinarodneho Prava
  PravnyPredpis: Medzinarodne Pravo
  Dan: Dan
  Dan: Clo
  Pravo: Oslobodenie Od Cla

---

chunk: 540
path: ['§ 48a', '1', 'a)']
path_as_text: Paragraf § 48a Odsek 1 Pismeno a)
text: (1) Na účely tohto ustanovenia sa rozumie a) cestujúcim leteckou dopravou osoba cestujúca leteckým dopravným prostriedkom okrem dopravného prostriedku súkromného rekreačného lietania,

relations:
  Paragraf § 48a -> [OBSAHUJE] -> Paragraf § 48a Odsek 1
  Paragraf § 48a Odsek 1 -> [OBSAHUJE] -> Paragraf § 48a Odsek 1 Pismeno a)
  Paragraf § 48a Odsek 1 Pismeno a) -> [DEFINUJE] -> Cestujuci Leteckou Dopravou
  Cestujuci Leteckou Dopravou -> [VZTAHUJE_SA_NA] -> Letecky Dopravny Prostriedok
  Cestujuci Leteckou Dopravou -> [NEVZTAHUJE_SA_NA] -> Dopravny Prostriedok Sukromneho Rekreacneho Lietania

nodes:
  Paragraf: Paragraf § 48a
  Odsek: Paragraf § 48a Odsek 1
  Pismeno: Paragraf § 48a Odsek 1 Pismeno a)
  Osoba: Cestujuci Leteckou Dopravou
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
  Colny Urad -> [VYDAVA] -> Rozhodnutie O Zabezpeceni Dane
  Rozhodnutie O Zabezpeceni Dane -> [ROZHODUJE_O] -> Zabezpecenie Dane
  Rozhodnutie O Zabezpeceni Dane -> [URCUJE] -> Vyska Zabezpecenia Dane
  Rozhodnutie O Zabezpeceni Dane -> [URCUJE] -> Lehota Na Zaplatenie Zabezpecenia Dane
  Zaplatenie Zabezpecenia Dane -> [MA_LEHOTU] -> Lehota Na Zaplatenie Zabezpecenia Dane
  Zaplatenie Zabezpecenia Dane -> [MA_SUMU] -> Vyska Zabezpecenia Dane
  Osoba Podla Paragraf § 48b Odsek 1 -> [ODKAZUJE_NA] -> Paragraf § 48b Odsek 1
  Osoba Podla Paragraf § 48b Odsek 1 -> [PLATI] -> Zaplatenie Zabezpecenia Dane
  Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane -> [VZTAHUJE_SA_NA] -> Rozhodnutie O Zabezpeceni Dane
  Rozhodnutie O Zabezpeceni Dane -> [NEMA_NAROK_NA] -> Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane
  Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3 -> [OSLOBODZUJE_OD] -> Dan
  Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3 -> [ODKAZUJE_NA] -> Paragraf § 48 Odsek 3
  Colny Urad -> [NEVZTAHUJE_SA_NA] -> Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3

nodes:
  Paragraf: Paragraf § 48b
  Odsek: Paragraf § 48b Odsek 2
  Odsek: Paragraf § 48b Odsek 1
  Paragraf: Paragraf § 48
  Odsek: Paragraf § 48 Odsek 3
  Organizacia: Colny Urad
  Rozhodnutie: Rozhodnutie O Zabezpeceni Dane
  Povinnost: Zabezpecenie Dane
  Suma: Vyska Zabezpecenia Dane
  Lehota: Lehota Na Zaplatenie Zabezpecenia Dane
  Platba: Zaplatenie Zabezpecenia Dane
  Dokument: Odvolanie Proti Rozhodnutiu O Zabezpeceni Dane
  Osoba: Osoba Podla Paragraf § 48b Odsek 1
  Pravo: Oslobodenie Od Dane Podla Paragraf § 48 Odsek 3
  Dan: Dan

---

chunk: 566
path: ['§ 48b', '3', 'c)']
path_as_text: Paragraf § 48b Odsek 3 Pismeno c)
text: (3) Colný úrad uvoľní zabezpečenie dane do desiatich dní od predloženia dôkazu o tom, že odoslanie alebo preprava tovaru sa skončila v inom členskom štáte okrem odseku 4. Dôkazom, že odoslanie alebo preprava tovaru sa skončila v inom členskom štáte, je doklad o prevzatí tovaru príjemcom v inom členskom štáte. Doklad o prevzatí tovaru musí obsahovať c) adresu miesta a dátum prevzatia tovaru v inom členskom štáte, ak odoslanie alebo prepravu tovaru vykoná dodávateľ, alebo adresu miesta a dátum skončenia prepravy, ak odoslanie alebo prepravu tovaru vykoná odberateľ,

relations:
  Paragraf § 48B -> [OBSAHUJE] -> Paragraf § 48B Odsek 3
  Paragraf § 48B Odsek 3 -> [OBSAHUJE] -> Paragraf § 48B Odsek 3 Pismeno c)
  Paragraf § 48B -> [OBSAHUJE] -> Paragraf § 48B Odsek 4
  Colny Urad -> [MA_POVINNOST] -> Povinnost Uvolnit Zabezpecenie Dane
  Povinnost Uvolnit Zabezpecenie Dane -> [VZTAHUJE_SA_NA] -> Zabezpecenie Dane
  Povinnost Uvolnit Zabezpecenie Dane -> [MA_LEHOTU] -> Lehota Do Desiatich Dni
  Povinnost Uvolnit Zabezpecenie Dane -> [MA_PODMIENKU] -> Predlozenie Dokazu
  Predlozenie Dokazu -> [PREDKLADA] -> Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Odoslanie Alebo Preprava Tovaru
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Povinnost Uvolnit Zabezpecenie Dane -> [NEVZTAHUJE_SA_NA] -> Paragraf § 48B Odsek 4
  Doklad O Prevzati Tovaru -> [JE_TYPOM] -> Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Doklad O Prevzati Tovaru -> [VZTAHUJE_SA_NA] -> Prevzatie Tovaru Prijemcom V Inom Clenskom State
  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [VZTAHUJE_SA_NA] -> Tovar
  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [PRIJIMA] -> Prijemca
  Prevzatie Tovaru Prijemcom V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Adresa Miesta Prevzatia Tovaru
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Datum Prevzatia Tovaru
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Dodavatel
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Adresa Miesta Skoncenia Prepravy
  Doklad O Prevzati Tovaru -> [OBSAHUJE] -> Datum Skoncenia Prepravy
  Odoslanie Alebo Preprava Tovaru -> [VZTAHUJE_SA_NA] -> Odberatel

nodes:
  Paragraf: Paragraf § 48B
  Odsek: Paragraf § 48B Odsek 3
  Pismeno: Paragraf § 48B Odsek 3 Pismeno c)
  Odsek: Paragraf § 48B Odsek 4
  Organizacia: Colny Urad
  Povinnost: Povinnost Uvolnit Zabezpecenie Dane
  Platba: Zabezpecenie Dane
  Lehota: Lehota Do Desiatich Dni
  Konanie: Predlozenie Dokazu
  Dokument: Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru V Inom Clenskom State
  Konanie: Odoslanie Alebo Preprava Tovaru
  Tovar: Tovar
  Stat: Iny Clensky Stat
  Dokument: Doklad O Prevzati Tovaru
  Konanie: Prevzatie Tovaru Prijemcom V Inom Clenskom State
  Subjekt: Prijemca
  Adresa: Adresa Miesta Prevzatia Tovaru
  Datum: Datum Prevzatia Tovaru
  Subjekt: Dodavatel
  Subjekt: Odberatel
  Adresa: Adresa Miesta Skoncenia Prepravy
  Datum: Datum Skoncenia Prepravy

---

chunk: 586
path: ['§ 48ca', '4', 'a)']
path_as_text: Paragraf § 48ca Odsek 4 Pismeno a)
text: (4) Prevádzkovateľ colného skladu je povinný viesť záznamy v členení podľa kalendárnych mesiacov o a) množstve tovaru v metrických tonách umiestneného do colného skladu, dátume umiestnenia tovaru a osobe, pre ktorú bol tento tovar umiestnený,

relations:
  Paragraf § 48ca -> [OBSAHUJE] -> Paragraf § 48ca Odsek 4
  Paragraf § 48ca Odsek 4 -> [OBSAHUJE] -> Paragraf § 48ca Odsek 4 Pismeno a)
  Paragraf § 48ca Odsek 4 Pismeno a) -> [UPRAVUJE] -> Vedenie Zaznamov
  Prevadzkovatel Colneho Skladu -> [MA_POVINNOST] -> Vedenie Zaznamov
  Vedenie Zaznamov -> [VZTAHUJE_SA_NA] -> Zaznamy
  Zaznamy -> [MA_OBDOBIE] -> Kalendarny Mesiac
  Zaznamy -> [MA_HODNOTU] -> Mnozstvo Tovaru V Metrickych Tonach
  Zaznamy -> [MA_DATUM] -> Datum Umiestnenia Tovaru
  Zaznamy -> [VZTAHUJE_SA_NA] -> Osoba Pre Ktoru Bol Tovar Umiestneny
  Mnozstvo Tovaru V Metrickych Tonach -> [VZTAHUJE_SA_NA] -> Tovar Umiestneny Do Colneho Skladu
  Tovar Umiestneny Do Colneho Skladu -> [NACHADZA_SA_V] -> Colny Sklad

nodes:
  Paragraf: Paragraf § 48ca
  Odsek: Paragraf § 48ca Odsek 4
  Pismeno: Paragraf § 48ca Odsek 4 Pismeno a)
  Subjekt: Prevadzkovatel Colneho Skladu
  Povinnost: Vedenie Zaznamov
  Zaznam: Zaznamy
  Obdobie: Kalendarny Mesiac
  Tovar: Tovar Umiestneny Do Colneho Skladu
  Lokacia: Colny Sklad
  Mnozstvo: Mnozstvo Tovaru V Metrickych Tonach
  Datum: Datum Umiestnenia Tovaru
  Osoba: Osoba Pre Ktoru Bol Tovar Umiestneny

---

chunk: 607
path: ['§ 48d', '11', 'b)']
path_as_text: Paragraf § 48d Odsek 11 Pismeno b)
text: (11) Povolenie na prevádzkovanie osobitného skladu zaniká dňom b) vyhlásenia konkurzu alebo dňom vstupu do likvidácie,

relations:
  Paragraf § 48d -> [OBSAHUJE] -> Paragraf § 48d Odsek 11
  Paragraf § 48d Odsek 11 -> [OBSAHUJE] -> Paragraf § 48d Odsek 11 Pismeno b)
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [VYPLYVA_Z] -> Paragraf § 48d Odsek 11 Pismeno b)
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Den Vyhlasenia Konkurzu
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Den Vstupu Do Likvidacie
  Den Vyhlasenia Konkurzu -> [VZTAHUJE_SA_NA] -> Vyhlasenie Konkurzu
  Den Vstupu Do Likvidacie -> [VZTAHUJE_SA_NA] -> Vstup Do Likvidacie

nodes:
  Paragraf: Paragraf § 48d
  Odsek: Paragraf § 48d Odsek 11
  Pismeno: Paragraf § 48d Odsek 11 Pismeno b)
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
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [VYPLYVA_Z] -> Paragraf § 48d Odsek 11 Pismeno d)
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Osobitneho Skladu
  Prevadzkovatel Osobitneho Skladu -> [MA_STATUS] -> Platitel
  Povolenie Na Prevadzkovanie Osobitneho Skladu -> [ZANIKA] -> Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu
  Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu -> [VYPLYVA_Z] -> Platitel

nodes:
  Paragraf: Paragraf § 48d
  Odsek: Paragraf § 48d Odsek 11
  Pismeno: Paragraf § 48d Odsek 11 Pismeno d)
  Rozhodnutie: Povolenie Na Prevadzkovanie Osobitneho Skladu
  Subjekt: Prevadzkovatel Osobitneho Skladu
  Status: Platitel
  Status: Zanik Povolenia Na Prevadzkovanie Osobitneho Skladu

---

chunk: 632
path: ['§ 48e', '9', 'b)']
path_as_text: Paragraf § 48e Odsek 9 Pismeno b)
text: (9) Osoba, ktorá spôsobí, že sa tovar vyjme z daňového skladu, je povinná predtým, ako nastane táto skutočnosť, oznámiť prevádzkovateľovi daňového skladu identifikačné číslo pre daň pridelené v tuzemsku a doručiť mu b) faktúru, ktorú vyhotovila o dodaní tovaru, ak v súvislosti s dodaním tovaru dochádza k vyňatiu tovaru z daňového skladu, alebo iný doklad, ktorý preukazuje dodanie tovaru, ak faktúra nie je vyhotovená pred vyňatím tovaru z daňového skladu.

relations:
  Paragraf § 48e -> [OBSAHUJE] -> Paragraf § 48e Odsek 9
  Paragraf § 48e Odsek 9 -> [OBSAHUJE] -> Paragraf § 48e Odsek 9 Pismeno b)
  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [ZODPOVEDA_ZA] -> Vynatie Tovaru Z Danoveho Skladu
  Vynatie Tovaru Z Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Tovar
  Vynatie Tovaru Z Danoveho Skladu -> [VZTAHUJE_SA_NA] -> Danovy Sklad
  Prevadzkovatel Danoveho Skladu -> [MA] -> Danovy Sklad
  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [MA_POVINNOST] -> Povinnost Oznamit Identifikacne Cislo Pre Dan
  Osoba Sposobujuca Vynatie Tovaru Z Danoveho Skladu -> [MA_POVINNOST] -> Povinnost Dorucit Fakturu Alebo Iny Doklad
  Povinnost Oznamit Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Povinnost Oznamit Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Danoveho Skladu
  Povinnost Dorucit Fakturu Alebo Iny Doklad -> [VZTAHUJE_SA_NA] -> Faktura O Dodani Tovaru
  Povinnost Dorucit Fakturu Alebo Iny Doklad -> [VZTAHUJE_SA_NA] -> Iny Doklad Preukazujuci Dodanie Tovaru
  Povinnost Dorucit Fakturu Alebo Iny Doklad -> [VZTAHUJE_SA_NA] -> Prevadzkovatel Danoveho Skladu
  Povinnost Oznamit Identifikacne Cislo Pre Dan -> [MA_LEHOTU] -> Lehota Pred Vynatim Tovaru Z Danoveho Skladu
  Povinnost Dorucit Fakturu Alebo Iny Doklad -> [MA_LEHOTU] -> Lehota Pred Vynatim Tovaru Z Danoveho Skladu
  Identifikacne Cislo Pre Dan Pridelene V Tuzemsku -> [VZTAHUJE_SA_NA] -> Tuzemsko
  Faktura O Dodani Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Iny Doklad Preukazujuci Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Dodanie Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Dodanie Tovaru -> [SUVISI_S] -> Vynatie Tovaru Z Danoveho Skladu
  Povinnost Oznamit Identifikacne Cislo Pre Dan -> [VYPLYVA_Z] -> Paragraf § 48e Odsek 9
  Povinnost Dorucit Fakturu Alebo Iny Doklad -> [VYPLYVA_Z] -> Paragraf § 48e Odsek 9 Pismeno b)

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
  Dokument: Identifikacne Cislo Pre Dan Pridelene V Tuzemsku
  Dokument: Faktura O Dodani Tovaru
  Dokument: Iny Doklad Preukazujuci Dodanie Tovaru
  Povinnost: Povinnost Oznamit Identifikacne Cislo Pre Dan
  Povinnost: Povinnost Dorucit Fakturu Alebo Iny Doklad
  Lehota: Lehota Pred Vynatim Tovaru Z Danoveho Skladu
  Stat: Tuzemsko

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
  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane Pri Prechodnych Polozkach
  Odpocitanie Dane Pri Prechodnych Polozkach -> [VZTAHUJE_SA_NA] -> Dan
  Odpocitanie Dane Pri Prechodnych Polozkach -> [MA_PODMIENKU] -> Prechodne Polozky
  Prechodne Polozky -> [VYPLYVA_Z] -> Paragraf § 22 Odsek 3
  Platitel -> [VYPLYVA_Z] -> Paragraf § 49 Odsek 7 Pismeno b)

nodes:
  Paragraf: Paragraf § 49
  Odsek: Paragraf § 49 Odsek 7
  Pismeno: Paragraf § 49 Odsek 7 Pismeno b)
  Paragraf: Paragraf § 22
  Odsek: Paragraf § 22 Odsek 3
  Subjekt: Platitel
  Dan: Dan
  Pravo: Odpocitanie Dane Pri Prechodnych Polozkach
  Podmienka: Prechodne Polozky

---

chunk: 655
path: ['§ 50', '2', 'a)']
path_as_text: Paragraf § 50 Odsek 2 Pismeno a)
text: (2) Koeficient sa vypočíta ako podiel, v ktorého čitateli je hodnota bez dane dodaných tovarov a služieb za kalendárny rok, pri ktorých je daň odpočítateľná, a v ktorého menovateli je hodnota bez dane zo všetkých dodaných tovarov a služieb za kalendárny rok. Pri výpočte koeficientu sa do čitateľa ani do menovateľa koeficientu neuvádza hodnota z a) predaja podniku alebo časti podniku tvoriacej samostatnú organizačnú zložku,

relations:
  Paragraf § 50 -> [OBSAHUJE] -> Paragraf § 50 Odsek 2
  Paragraf § 50 Odsek 2 -> [OBSAHUJE] -> Paragraf § 50 Odsek 2 Pismeno a)
  Paragraf § 50 Odsek 2 -> [UPRAVUJE] -> Vypocet Koeficientu
  Vypocet Koeficientu -> [MA_HODNOTU] -> Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna
  Vypocet Koeficientu -> [MA_HODNOTU] -> Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [VZTAHUJE_SA_NA] -> Dodane Tovary A Sluzby
  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [VZTAHUJE_SA_NA] -> Dodane Tovary A Sluzby
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [MA_OBDOBIE] -> Kalendarny Rok
  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [MA_OBDOBIE] -> Kalendarny Rok
  Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna -> [VZTAHUJE_SA_NA] -> Dan
  Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok -> [VZTAHUJE_SA_NA] -> Dan
  Vypocet Koeficientu -> [NEVZTAHUJE_SA_NA] -> Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku
  Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku -> [VZTAHUJE_SA_NA] -> Podnik Alebo Cast Podniku Tvoriaca Samostatnu Organizacnu Zlozku

nodes:
  Paragraf: Paragraf § 50
  Odsek: Paragraf § 50 Odsek 2
  Pismeno: Paragraf § 50 Odsek 2 Pismeno a)
  Konanie: Vypocet Koeficientu
  Suma: Hodnota Bez Dane Dodanych Tovarov A Sluzieb Za Kalendarny Rok Pri Ktorych Je Dan Odpocitatelna
  Suma: Hodnota Bez Dane Zo Vsetkych Dodanych Tovarov A Sluzieb Za Kalendarny Rok
  Tovar: Dodane Tovary A Sluzby
  Obdobie: Kalendarny Rok
  Dan: Dan
  Konanie: Predaj Podniku Alebo Casti Podniku Tvoriacej Samostatnu Organizacnu Zlozku
  Organizacia: Podnik Alebo Cast Podniku Tvoriaca Samostatnu Organizacnu Zlozku

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
  Platitel -> [MA_PRAVO] -> Odpocitanie Dane
  Platitel -> [MA_PRAVO] -> Pomerne Odpocitanie Dane
  Oprava Odpocitanej Dane Vo Vyssej Vyske -> [VZTAHUJE_SA_NA] -> Odpocitana Dan Vo Vyssej Vyske
  Platitel -> [MA_POVINNOST] -> Oprava Odpocitanej Dane Vo Vyssej Vyske
  Oprava Odpocitanej Dane Vo Vyssej Vyske -> [MA_PODMIENKU] -> Prvotne Pouzitie
  Oprava Odpocitanej Dane Vo Vyssej Vyske -> [VYPLYVA_Z] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 1
  Oprava Odpocitanej Dane Vo Vyssej Vyske -> [NEVZTAHUJE_SA_NA] -> Dodanie Tovaru
  Oprava Odpocitanej Dane Vo Vyssej Vyske -> [NEVZTAHUJE_SA_NA] -> Dodanie Sluzby
  Oprava Odpocitanej Dane V Nizsej Vyske -> [VZTAHUJE_SA_NA] -> Odpocitana Dan V Nizsej Vyske
  Platitel -> [MA_PRAVO] -> Oprava Odpocitanej Dane V Nizsej Vyske
  Oprava Odpocitanej Dane V Nizsej Vyske -> [MA_PODMIENKU] -> Prvotne Pouzitie
  Oprava Odpocitanej Dane V Nizsej Vyske -> [VYPLYVA_Z] -> Paragraf § 53 Odsek 1 Pismeno c) Bod 2
  Odpocitanie Dane -> [MA_OBDOBIE] -> Zdanovacie Obdobie

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
  Pravo: Odpocitanie Dane
  Pravo: Pomerne Odpocitanie Dane
  Dan: Odpocitana Dan Vo Vyssej Vyske
  Dan: Odpocitana Dan V Nizsej Vyske
  Povinnost: Oprava Odpocitanej Dane Vo Vyssej Vyske
  Pravo: Oprava Odpocitanej Dane V Nizsej Vyske
  Podmienka: Prvotne Pouzitie
  Tovar: Dodanie Tovaru
  Sluzba: Dodanie Sluzby

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
  Platitel -> [MA_POVINNOST] -> Oprava Odpocitanej Dane
  Oprava Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Odpocitana Dan -> [VZTAHUJE_SA_NA] -> Sluzba Vykonana Na Investicnom Majetku
  Sluzba Vykonana Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno a)
  Sluzba Vykonana Na Investicnom Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno b)
  Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno a)
  Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 54 Odsek 2 Pismeno b)
  Platitel -> [DODAVA] -> Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno a)
  Platitel -> [DODAVA] -> Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno b)
  Platitel -> [ODKAZUJE_NA] -> Paragraf § 53a Odsek 1
  Oprava Odpocitanej Dane -> [MA_OBDOBIE] -> Obdobie Od Dodania Investicneho Majetku Do 60 Kalendarnych Mesiacov
  Oprava Odpocitanej Dane -> [MA_OBDOBIE] -> Obdobie Od Dodania Investicneho Majetku Do 240 Kalendarnych Mesiacov
  Obdobie Od Dodania Investicneho Majetku Do 60 Kalendarnych Mesiacov -> [MA_DATUM] -> Kalendarny Mesiac Dodania Investicneho Majetku
  Obdobie Od Dodania Investicneho Majetku Do 60 Kalendarnych Mesiacov -> [MA_LEHOTU] -> Uplynutie 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Obdobie Od Dodania Investicneho Majetku Do 240 Kalendarnych Mesiacov -> [MA_DATUM] -> Kalendarny Mesiac Dodania Investicneho Majetku
  Obdobie Od Dodania Investicneho Majetku Do 240 Kalendarnych Mesiacov -> [MA_LEHOTU] -> Uplynutie 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Platitel -> [MA_POVINNOST] -> Pomerne Odpocitanie Dane Zo Sluzby Vykonanej Na Investicnom Majetku
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
  Dan: Odpocitana Dan
  Povinnost: Oprava Odpocitanej Dane
  Sluzba: Sluzba Vykonana Na Investicnom Majetku
  Majetok: Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno a)
  Majetok: Investicny Majetok Podla Paragrafu § 54 Odsek 2 Pismeno b)
  Obdobie: Obdobie Od Dodania Investicneho Majetku Do 60 Kalendarnych Mesiacov
  Obdobie: Obdobie Od Dodania Investicneho Majetku Do 240 Kalendarnych Mesiacov
  Datum: Kalendarny Mesiac Dodania Investicneho Majetku
  Lehota: Uplynutie 60 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Lehota: Uplynutie 240 Kalendarneho Mesiaca Od Uplatnenia Odpocitania Dane
  Dan: Pomerne Odpocitanie Dane Zo Sluzby Vykonanej Na Investicnom Majetku

---

chunk: 701
path: ['§ 54', '1', 'b)']
path_as_text: Paragraf § 54 Odsek 1 Pismeno b)
text: (1) Ak v období nasledujúcom po zdaňovacom období, v ktorom došlo k prvotnému použitiu investičného majetku, platiteľ zmení účel jeho použitia, b) má právo upraviť odpočítanú daň, ak v dôsledku tejto zmeny bola daň pri prvotnom použití tohto investičného majetku odpočítaná v nižšej výške, v akej mohla byť odpočítaná v kalendárnom roku, v ktorom došlo k zmene účelu použitia tohto investičného majetku.

relations:
  Paragraf § 54 -> [OBSAHUJE] -> Paragraf § 54 Odsek 1
  Paragraf § 54 Odsek 1 -> [OBSAHUJE] -> Paragraf § 54 Odsek 1 Pismeno b)
  Paragraf § 54 Odsek 1 Pismeno b) -> [UPRAVUJE] -> Pravo Upravit Odpocitanu Dan
  Platitel -> [MA_PRAVO] -> Pravo Upravit Odpocitanu Dan
  Pravo Upravit Odpocitanu Dan -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Pravo Upravit Odpocitanu Dan -> [VYPLYVA_Z] -> Zmena Ucelu Pouzitia Investicneho Majetku
  Zmena Ucelu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Prvotne Pouzitie Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku -> [SUVISI_S] -> Prvotne Pouzitie Investicneho Majetku
  Obdobie Nasledujuce Po Zdanovacom Obdobi -> [SUVISI_S] -> Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Zmena Ucelu Pouzitia Investicneho Majetku -> [MA_OBDOBIE] -> Obdobie Nasledujuce Po Zdanovacom Obdobi
  Zmena Ucelu Pouzitia Investicneho Majetku -> [MA_OBDOBIE] -> Kalendarny Rok Zmeny Ucelu Pouzitia Investicneho Majetku
  Odpocitana Dan -> [MA_SUMU] -> Nizsia Vyska Odpocitanej Dane
  Odpocitana Dan -> [SUVISI_S] -> Prvotne Pouzitie Investicneho Majetku

nodes:
  Paragraf: Paragraf § 54
  Odsek: Paragraf § 54 Odsek 1
  Pismeno: Paragraf § 54 Odsek 1 Pismeno b)
  Subjekt: Platitel
  Majetok: Investicny Majetok
  Pravo: Pravo Upravit Odpocitanu Dan
  Dan: Odpocitana Dan
  Dovod: Zmena Ucelu Pouzitia Investicneho Majetku
  Konanie: Prvotne Pouzitie Investicneho Majetku
  ZdanovacieObdobie: Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Obdobie: Obdobie Nasledujuce Po Zdanovacom Obdobi
  Obdobie: Kalendarny Rok Zmeny Ucelu Pouzitia Investicneho Majetku
  Suma: Nizsia Vyska Odpocitanej Dane

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
  Prvotne Pouzitie Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Obdobie Nasledujuce Po Zdanovacom Obdobi -> [VYPLYVA_Z] -> Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku
  Platitel -> [MENI] -> Zmena Rozsahu Pouzitia Investicneho Majetku
  Zmena Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Zmena Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Pouzitie Investicneho Majetku Na Ucely Podnikania
  Zmena Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Pouzitie Investicneho Majetku Na Iny Ucel Ako Na Podnikanie
  Platitel -> [MA_PRAVO] -> Uprava Odpocitanej Dane
  Uprava Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Odpocitana Dan
  Uprava Odpocitanej Dane -> [MA_PODMIENKU] -> Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske -> [VZTAHUJE_SA_NA] -> Prvotne Pouzitie Investicneho Majetku
  Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske -> [MA_OBDOBIE] -> Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku
  Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku -> [VZTAHUJE_SA_NA] -> Zmena Rozsahu Pouzitia Investicneho Majetku

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
  Majetok: Investicny Majetok
  Konanie: Prvotne Pouzitie Investicneho Majetku
  Konanie: Zmena Rozsahu Pouzitia Investicneho Majetku
  Konanie: Pouzitie Investicneho Majetku Na Ucely Podnikania
  Konanie: Pouzitie Investicneho Majetku Na Iny Ucel Ako Na Podnikanie
  Pravo: Uprava Odpocitanej Dane
  Dan: Odpocitana Dan
  Dan: Cast Dane
  Dan: Dan Pri Prvotnom Pouziti Investicneho Majetku Odpocitana V Nizsej Vyske
  Obdobie: Kalendarny Rok Zmeny Rozsahu Pouzitia Investicneho Majetku
  Obdobie: Obdobie Nasledujuce Po Zdanovacom Obdobi
  ZdanovacieObdobie: Zdanovacie Obdobie Prvotneho Pouzitia Investicneho Majetku

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
  Platitel -> [MA_POVINNOST] -> Registracna Povinnost
  Platitel -> [SUVISI_S] -> Nesplnenie Registracnej Povinnosti
  Nesplnenie Registracnej Povinnosti -> [VZTAHUJE_SA_NA] -> Registracna Povinnost
  Platitel -> [PODAVA] -> Danove Priznanie
  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Danove Priznanie -> [MA_LEHOTU] -> Lehota Podla Paragrafu § 78 Odsek 2
  Platitel -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane
  Pravo Na Odpocitanie Dane -> [VZTAHUJE_SA_NA] -> Dan Ina Ako V Paragraf § 55 Odsek 1
  Dan Ina Ako V Paragraf § 55 Odsek 1 -> [ODKAZUJE_NA] -> Paragraf § 55 Odsek 1
  Iny Platitel -> [MA] -> Dan Ina Ako V Paragraf § 55 Odsek 1
  Dan Ina Ako V Paragraf § 55 Odsek 1 -> [NACHADZA_SA_V] -> Tuzemsko
  Dan Ina Ako V Paragraf § 55 Odsek 1 -> [VZTAHUJE_SA_NA] -> Tovar
  Dan Ina Ako V Paragraf § 55 Odsek 1 -> [VZTAHUJE_SA_NA] -> Sluzba

nodes:
  Paragraf: Paragraf § 55
  Odsek: Paragraf § 55 Odsek 3
  Pismeno: Paragraf § 55 Odsek 3 Pismeno a)
  Paragraf: Paragraf § 78
  Odsek: Paragraf § 78 Odsek 2
  Paragraf: Paragraf § 49
  Paragraf: Paragraf § 50
  Paragraf: Paragraf § 51
  Odsek: Paragraf § 51 Odsek 1
  Odsek: Paragraf § 51 Odsek 3
  Odsek: Paragraf § 51 Odsek 5
  Odsek: Paragraf § 55 Odsek 4
  Odsek: Paragraf § 55 Odsek 1
  Subjekt: Platitel
  Subjekt: Iny Platitel
  Povinnost: Registracna Povinnost
  Dovod: Nesplnenie Registracnej Povinnosti
  ZdanovacieObdobie: Zdanovacie Obdobie
  DanovePriznanie: Danove Priznanie
  Lehota: Lehota Podla Paragrafu § 78 Odsek 2
  Pravo: Pravo Na Odpocitanie Dane
  Dan: Dan Ina Ako V Paragraf § 55 Odsek 1
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
  Ziadatel -> [MA_NAROK_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Ziadatel -> [MA_PODMIENKU] -> Zdanitelne Obchody S Pravom Na Odpocitanie Dane
  Zdanitelne Obchody S Pravom Na Odpocitanie Dane -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane
  Pravo Na Odpocitanie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Pravo Na Odpocitanie Dane -> [VZNIKA] -> Clensky Stat
  Ziadatel -> [NACHADZA_SA_V] -> Clensky Stat
  Ziadatel -> [MA_ADRESU] -> Sidlo
  Ziadatel -> [MA_ADRESU] -> Miesto Podnikania
  Ziadatel -> [MA] -> Prevadzkaren
  Ziadatel -> [MA_ADRESU] -> Bydlisko
  Ziadatel -> [NACHADZA_SA_V] -> Miesto Obvykleho Zdrziavania
  Sidlo -> [NACHADZA_SA_V] -> Clensky Stat
  Miesto Podnikania -> [NACHADZA_SA_V] -> Clensky Stat
  Prevadzkaren -> [NACHADZA_SA_V] -> Clensky Stat
  Bydlisko -> [NACHADZA_SA_V] -> Clensky Stat
  Miesto Obvykleho Zdrziavania -> [NACHADZA_SA_V] -> Clensky Stat
  Ziadatel -> [MA_NAROK_NA] -> Vratenie Pomernej Vysky Dane
  Vratenie Pomernej Vysky Dane -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Pomernej Vysky Dane -> [MA_PODMIENKU] -> Pravidla Platne V Clenskom State
  Pravidla Platne V Clenskom State -> [VZTAHUJE_SA_NA] -> Clensky Stat
  Ziadatel -> [MA_PODMIENKU] -> Zdanitelne Obchody Bez Prava Na Odpocitanie Dane
  Zdanitelne Obchody Bez Prava Na Odpocitanie Dane -> [NEMA_NAROK_NA] -> Pravo Na Odpocitanie Dane

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
  Stat: Clensky Stat
  Adresa: Sidlo
  Adresa: Miesto Podnikania
  Lokacia: Prevadzkaren
  Adresa: Bydlisko
  Lokacia: Miesto Obvykleho Zdrziavania
  Podmienka: Pravidla Platne V Clenskom State

---

chunk: 770
path: ['§ 55b', '4', 'd)']
path_as_text: Paragraf § 55b Odsek 4 Pismeno d)
text: (4) Druh nadobudnutého tovaru a služieb sa vyjadruje týmito číselnými kódmi: d) poplatky za užívanie ciest a diaľnic číselným kódom 4,

relations:
  Paragraf § 55b -> [OBSAHUJE] -> Paragraf § 55b Odsek 4
  Paragraf § 55b Odsek 4 -> [OBSAHUJE] -> Paragraf § 55b Odsek 4 Pismeno d)
  Paragraf § 55b Odsek 4 Pismeno d) -> [UPRAVUJE] -> Poplatky Za Uzivanie Ciest A Dialnic
  Druh Nadobudnuteho Tovaru A Sluzieb -> [OBSAHUJE] -> Poplatky Za Uzivanie Ciest A Dialnic
  Poplatky Za Uzivanie Ciest A Dialnic -> [MA_IDENTIFIKATOR] -> Ciselny Kod 4

nodes:
  Paragraf: Paragraf § 55b
  Odsek: Paragraf § 55b Odsek 4
  Pismeno: Paragraf § 55b Odsek 4 Pismeno d)
  Tovar: Druh Nadobudnuteho Tovaru A Sluzieb
  Sluzba: Poplatky Za Uzivanie Ciest A Dialnic
  Zaznam: Ciselny Kod 4

---

chunk: 771
path: ['§ 55b', '4', 'e)']
path_as_text: Paragraf § 55b Odsek 4 Pismeno e)
text: (4) Druh nadobudnutého tovaru a služieb sa vyjadruje týmito číselnými kódmi: e) cestovné náklady týkajúce sa osobnej dopravy číselným kódom 5,

relations:
  Paragraf § 55b -> [OBSAHUJE] -> Paragraf § 55b Odsek 4
  Paragraf § 55b Odsek 4 -> [OBSAHUJE] -> Paragraf § 55b Odsek 4 Pismeno e)
  Paragraf § 55b Odsek 4 Pismeno e) -> [URCUJE] -> Cestovne Naklady Tykajuce Sa Osobnej Dopravy
  Cestovne Naklady Tykajuce Sa Osobnej Dopravy -> [MA_HODNOTU] -> Ciselny Kod 5

nodes:
  Paragraf: Paragraf § 55b
  Odsek: Paragraf § 55b Odsek 4
  Pismeno: Paragraf § 55b Odsek 4 Pismeno e)
  Sluzba: Cestovne Naklady Tykajuce Sa Osobnej Dopravy
  Mnozstvo: Ciselny Kod 5

---

chunk: 793
path: ['§ 55d', '8']
path_as_text: Paragraf § 55d Odsek 8
text: (8) Daňový úrad Bratislava vráti daň na účet vedený v banke v tuzemsku alebo na základe žiadosti žiadateľa na účet vedený v zahraničnej banke v inom členskom štáte, ak ju nemožno použiť podľa osobitného predpisu.27bd) Pri vrátení dane na účet vedený v zahraničnej banke v inom členskom štáte sa od sumy dane odpočítajú bankové poplatky za prevod peňažných prostriedkov.

relations:
  Paragraf § 55D -> [OBSAHUJE] -> Paragraf § 55D Odsek 8
  Danovy Urad Bratislava -> [PLATI] -> Dan
  Danovy Urad Bratislava -> [PLATI] -> Bankovy Ucet Vedeny V Banke V Tuzemsku
  Bankovy Ucet Vedeny V Banke V Tuzemsku -> [MA] -> Banka V Tuzemsku
  Banka V Tuzemsku -> [NACHADZA_SA_V] -> Tuzemsko
  Ziadatel -> [PODAVA] -> Ziadost Ziadatela
  Danovy Urad Bratislava -> [PLATI] -> Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State
  Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State -> [MA] -> Zahranicna Banka V Inom Clenskom State
  Zahranicna Banka V Inom Clenskom State -> [NACHADZA_SA_V] -> Iny Clensky Stat
  Dan -> [ODKAZUJE_NA] -> Osobitny Predpis 27Bd
  Bankove Poplatky Za Prevod Penaznych Prostriedkov -> [VZTAHUJE_SA_NA] -> Prevod Penaznych Prostriedkov
  Suma Dane -> [MA_HODNOTU] -> Bankove Poplatky Za Prevod Penaznych Prostriedkov

nodes:
  Paragraf: Paragraf § 55D
  Odsek: Paragraf § 55D Odsek 8
  Organizacia: Danovy Urad Bratislava
  Dan: Dan
  BankovyUcet: Bankovy Ucet Vedeny V Banke V Tuzemsku
  Banka: Banka V Tuzemsku
  Stat: Tuzemsko
  Ziadost: Ziadost Ziadatela
  Osoba: Ziadatel
  BankovyUcet: Bankovy Ucet Vedeny V Zahranicnej Banke V Inom Clenskom State
  Banka: Zahranicna Banka V Inom Clenskom State
  Stat: Iny Clensky Stat
  PravnyPredpis: Osobitny Predpis 27Bd
  Suma: Suma Dane
  Suma: Bankove Poplatky Za Prevod Penaznych Prostriedkov
  Platba: Prevod Penaznych Prostriedkov

---

chunk: 812
path: ['§ 57', '2']
path_as_text: Paragraf § 57 Odsek 2
text: (2) Žiadosť o vrátenie dane môže podať zahraničná osoba z tretieho štátu aj za obdobie kalendárneho polroka, ak suma dane, ktorej vrátenie žiada, je najmenej 1 000 eur, a ak taká žiadosť bola podaná za prvý kalendárny polrok, suma dane, ktorej vrátenie žiada za druhý kalendárny polrok, je najmenej 50 eur. Žiadosť o vrátenie dane za kalendárny polrok sa podáva najneskôr v lehote podľa odseku 1.

relations:
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 1
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 2
  Paragraf § 57 Odsek 2 -> [UPRAVUJE] -> Ziadost O Vratenie Dane
  Zahranicna Osoba Z Tretieho Statu -> [VZTAHUJE_SA_NA] -> Treti Stat
  Zahranicna Osoba Z Tretieho Statu -> [PODAVA] -> Ziadost O Vratenie Dane
  Zahranicna Osoba Z Tretieho Statu -> [MA_PRAVO] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Ziadost O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Kalendarny Polrok
  Prvy Kalendarny Polrok -> [JE_TYPOM] -> Kalendarny Polrok
  Druhy Kalendarny Polrok -> [JE_TYPOM] -> Kalendarny Polrok
  Ziadost O Vratenie Dane -> [MA_SUMU] -> Suma Dane Najmenej 1000 Eur
  Suma Dane Najmenej 1000 Eur -> [VZTAHUJE_SA_NA] -> Dan
  Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Prvy Kalendarny Polrok
  Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Druhy Kalendarny Polrok
  Ziadost O Vratenie Dane -> [MA_SUMU] -> Suma Dane Najmenej 50 Eur
  Suma Dane Najmenej 50 Eur -> [VZTAHUJE_SA_NA] -> Dan
  Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok -> [VZTAHUJE_SA_NA] -> Ziadost O Vratenie Dane
  Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok -> [MA_LEHOTU] -> Lehota Podla Paragraf § 57 Odsek 1
  Lehota Podla Paragraf § 57 Odsek 1 -> [ODKAZUJE_NA] -> Paragraf § 57 Odsek 1

nodes:
  Paragraf: Paragraf § 57
  Odsek: Paragraf § 57 Odsek 1
  Odsek: Paragraf § 57 Odsek 2
  Ziadost: Ziadost O Vratenie Dane
  Osoba: Zahranicna Osoba Z Tretieho Statu
  Stat: Treti Stat
  Pravo: Vratenie Dane
  Dan: Dan
  Obdobie: Kalendarny Polrok
  Obdobie: Prvy Kalendarny Polrok
  Obdobie: Druhy Kalendarny Polrok
  Suma: Suma Dane Najmenej 1000 Eur
  Suma: Suma Dane Najmenej 50 Eur
  Lehota: Lehota Podla Paragraf § 57 Odsek 1
  Konanie: Podanie Ziadosti O Vratenie Dane Za Kalendarny Polrok

---

chunk: 816
path: ['§ 57', '5', 'a)']
path_as_text: Paragraf § 57 Odsek 5 Pismeno a)
text: (5) Zahraničná osoba z tretieho štátu musí v žiadosti o vrátenie dane vyhlásiť, že a) spĺňa podmienky podľa § 56 ods. 2,

relations:
  Paragraf § 57 -> [OBSAHUJE] -> Paragraf § 57 Odsek 5
  Paragraf § 57 Odsek 5 -> [OBSAHUJE] -> Paragraf § 57 Odsek 5 Pismeno a)
  Paragraf § 56 -> [OBSAHUJE] -> Paragraf § 56 Odsek 2
  Zahranicna Osoba Z Tretieho Statu -> [VZTAHUJE_SA_NA] -> Treti Stat
  Ziadost O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Zahranicna Osoba Z Tretieho Statu -> [SPLNA_PODMIENKY] -> Podmienky Podla Paragraf § 56 Odsek 2
  Zahranicna Osoba Z Tretieho Statu -> [MA_POVINNOST] -> Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane
  Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane -> [VZTAHUJE_SA_NA] -> Ziadost O Vratenie Dane

nodes:
  Paragraf: Paragraf § 57
  Odsek: Paragraf § 57 Odsek 5
  Pismeno: Paragraf § 57 Odsek 5 Pismeno a)
  Paragraf: Paragraf § 56
  Odsek: Paragraf § 56 Odsek 2
  Osoba: Zahranicna Osoba Z Tretieho Statu
  Stat: Treti Stat
  Ziadost: Ziadost O Vratenie Dane
  Dan: Dan
  Podmienka: Podmienky Podla Paragraf § 56 Odsek 2
  Povinnost: Povinnost Vyhlasit Splnenie Podmienok V Ziadosti O Vratenie Dane

---

chunk: 839
path: ['§ 59', '6']
path_as_text: Paragraf § 59 Odsek 6
text: (6) Nárok na vrátenie dane zaniká, ak sa platiteľovi alebo poverenej osobe nepredložia doklady uvedené v odseku 3 do šiestich mesiacov od konca mesiaca, v ktorom bol tovar predaný.

relations:
  Paragraf § 59 -> [OBSAHUJE] -> Paragraf § 59 Odsek 6
  Paragraf § 59 -> [OBSAHUJE] -> Paragraf § 59 Odsek 3
  Paragraf § 59 Odsek 6 -> [ODKAZUJE_NA] -> Paragraf § 59 Odsek 3
  Paragraf § 59 Odsek 6 -> [UPRAVUJE] -> Narok Na Vratenie Dane
  Narok Na Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Narok Na Vratenie Dane -> [ZANIKA] -> Nepredlozenie Dokladov
  Nepredlozenie Dokladov -> [VZTAHUJE_SA_NA] -> Doklady Uvedene V Paragraf § 59 Odsek 3
  Doklady Uvedene V Paragraf § 59 Odsek 3 -> [VYPLYVA_Z] -> Paragraf § 59 Odsek 3
  Nepredlozenie Dokladov -> [VZTAHUJE_SA_NA] -> Platitel
  Nepredlozenie Dokladov -> [VZTAHUJE_SA_NA] -> Poverena Osoba
  Nepredlozenie Dokladov -> [MA_LEHOTU] -> Sest Mesiacov Od Konca Mesiaca Predaja Tovaru
  Sest Mesiacov Od Konca Mesiaca Predaja Tovaru -> [VZTAHUJE_SA_NA] -> Tovar

nodes:
  Paragraf: Paragraf § 59
  Odsek: Paragraf § 59 Odsek 6
  Odsek: Paragraf § 59 Odsek 3
  Pravo: Narok Na Vratenie Dane
  Dan: Dan
  Subjekt: Platitel
  Osoba: Poverena Osoba
  Dokument: Doklady Uvedene V Paragraf § 59 Odsek 3
  Podmienka: Nepredlozenie Dokladov
  Lehota: Sest Mesiacov Od Konca Mesiaca Predaja Tovaru
  Tovar: Tovar

---

chunk: 853
path: ['§ 61', '1']
path_as_text: Paragraf § 61 Odsek 1
text: (1) Osoby iných štátov, ktoré požívajú výsady a imunity podľa medzinárodného práva,23) a medzinárodné organizácie24) a ich pracovníci (ďalej len „zahraničný zástupca“) majú nárok na vrátenie dane zaplatenej v cenách tovarov a služieb určených na ich spotrebu.

relations:
  Paragraf § 61 -> [OBSAHUJE] -> Paragraf § 61 Odsek 1
  Osoby Inych Statov -> [JE_TYPOM] -> Zahranicny Zastupca
  Medzinarodne Organizacie -> [JE_TYPOM] -> Zahranicny Zastupca
  Pracovnici Medzinarodnych Organizacii -> [JE_TYPOM] -> Zahranicny Zastupca
  Pracovnici Medzinarodnych Organizacii -> [PATRI_DO] -> Medzinarodne Organizacie
  Osoby Inych Statov -> [MA_PRAVO] -> Vysady A Imunity
  Vysady A Imunity -> [VYPLYVA_Z] -> Medzinarodne Pravo
  Zahranicny Zastupca -> [MA_NAROK_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan Zaplatena V Cenach Tovarov A Sluzieb
  Dan Zaplatena V Cenach Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Tovary Urcene Na Spotrebu Zahranicneho Zastupcu
  Dan Zaplatena V Cenach Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzby Urcene Na Spotrebu Zahranicneho Zastupcu

nodes:
  Paragraf: Paragraf § 61
  Odsek: Paragraf § 61 Odsek 1
  Osoba: Osoby Inych Statov
  Organizacia: Medzinarodne Organizacie
  Osoba: Pracovnici Medzinarodnych Organizacii
  Subjekt: Zahranicny Zastupca
  Pravo: Vysady A Imunity
  PravnyPredpis: Medzinarodne Pravo
  Platba: Vratenie Dane
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
  Vratenie Dane -> [VYPLYVA_Z] -> Paragraf § 61 Odsek 3
  Obdobne Zvyhodnenie -> [VYPLYVA_Z] -> Paragraf § 61 Odsek 3
  Zahranicni Zastupcovia -> [MA_NAROK_NA] -> Vratenie Dane
  Zahranicni Zastupcovia -> [VZTAHUJE_SA_NA] -> Staty Zahranicnych Zastupcov
  Vratenie Dane -> [MA_PODMIENKU] -> Vzajomnost Podla Paragrafu § 61 Odsek 3
  Staty Zahranicnych Zastupcov -> [POSKYTUJE] -> Vratenie Dane
  Staty Zahranicnych Zastupcov -> [POSKYTUJE] -> Obdobne Zvyhodnenie
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Osoby Slovenskej Republiky
  Obdobne Zvyhodnenie -> [VZTAHUJE_SA_NA] -> Osoby Slovenskej Republiky
  Iny Stat -> [POSKYTUJE] -> Vratenie Dane
  Iny Stat -> [POSKYTUJE] -> Obdobne Zvyhodnenie
  Slovenska Republika -> [POSKYTUJE] -> Vratenie Dane
  Vratenie Dane -> [MA_HODNOTU] -> Rozsah Vratenia Dane
  Vzajomnost Podla Paragrafu § 61 Odsek 3 -> [NEVZTAHUJE_SA_NA] -> Medzinarodne Organizacie
  Vzajomnost Podla Paragrafu § 61 Odsek 3 -> [NEVZTAHUJE_SA_NA] -> Pracovnici Medzinarodnych Organizacii

nodes:
  Paragraf: Paragraf § 61
  Odsek: Paragraf § 61 Odsek 3
  Pravo: Vratenie Dane
  Pravo: Obdobne Zvyhodnenie
  Subjekt: Zahranicni Zastupcovia
  Stat: Staty Zahranicnych Zastupcov
  Stat: Iny Stat
  Stat: Slovenska Republika
  Osoba: Osoby Slovenskej Republiky
  Organizacia: Medzinarodne Organizacie
  Osoba: Pracovnici Medzinarodnych Organizacii
  Podmienka: Vzajomnost Podla Paragrafu § 61 Odsek 3
  Mnozstvo: Rozsah Vratenia Dane

---

chunk: 885
path: ['§ 62', '3']
path_as_text: Paragraf § 62 Odsek 3
text: (3) Vrátenie dane môže zahraničný zástupca žiadať len v prípade, ak celková cena vrátane dane na jednom doklade o kúpe tovarov alebo služieb s výnimkou dokladu o kúpe pohonných látok je najmenej 33,19 eura. Ak iný štát viaže vrátenie dane osobám Slovenskej republiky na doklad o kúpe tovarov alebo služieb, na ktorom je celková cena vyššia ako 33,19 eura, môže zahraničný zástupca tohto štátu žiadať vrátenie dane z takého dokladu, na ktorom je celková cena najmenej vo výške určenej týmto štátom.

relations:
  Paragraf § 62 -> [OBSAHUJE] -> Paragraf § 62 Odsek 3
  Zahranicny Zastupca -> [MA_PRAVO] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Vratenie Dane -> [MA_PODMIENKU] -> Podmienka Celkova Cena Najmenej 33,19 Eura
  Podmienka Celkova Cena Najmenej 33,19 Eura -> [VZTAHUJE_SA_NA] -> Celkova Cena Vratane Dane
  Podmienka Celkova Cena Najmenej 33,19 Eura -> [MA_SUMU] -> Suma 33,19 Eura
  Celkova Cena Vratane Dane -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Doklad O Kupe Tovarov Alebo Sluzieb -> [VZTAHUJE_SA_NA] -> Tovar
  Doklad O Kupe Tovarov Alebo Sluzieb -> [VZTAHUJE_SA_NA] -> Sluzba
  Doklad O Kupe Pohonnych Latok -> [VZTAHUJE_SA_NA] -> Pohonne Latky
  Podmienka Celkova Cena Najmenej 33,19 Eura -> [NEVZTAHUJE_SA_NA] -> Doklad O Kupe Pohonnych Latok
  Iny Stat -> [URCUJE] -> Vratenie Dane
  Osoba Slovenskej Republiky -> [PATRI_DO] -> Slovenska Republika
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Osoba Slovenskej Republiky
  Vratenie Dane -> [MA_PODMIENKU] -> Podmienka Celkova Cena Vyssia Ako 33,19 Eura
  Podmienka Celkova Cena Vyssia Ako 33,19 Eura -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Podmienka Celkova Cena Vyssia Ako 33,19 Eura -> [MA_SUMU] -> Suma 33,19 Eura
  Zahranicny Zastupca -> [KONA_V_MENE] -> Iny Stat
  Zahranicny Zastupca -> [MA_NAROK_NA] -> Vratenie Dane
  Vratenie Dane -> [VYPLYVA_Z] -> Doklad O Kupe Tovarov Alebo Sluzieb
  Vratenie Dane -> [MA_PODMIENKU] -> Podmienka Celkova Cena Najmenej Vo Vyske Urcenej Inym Statom
  Podmienka Celkova Cena Najmenej Vo Vyske Urcenej Inym Statom -> [MA_SUMU] -> Suma Urcena Inym Statom
  Suma Urcena Inym Statom -> [VYPLYVA_Z] -> Iny Stat
  Podmienka Celkova Cena Najmenej Vo Vyske Urcenej Inym Statom -> [VZTAHUJE_SA_NA] -> Doklad O Kupe Tovarov Alebo Sluzieb

nodes:
  Paragraf: Paragraf § 62
  Odsek: Paragraf § 62 Odsek 3
  Subjekt: Zahranicny Zastupca
  Pravo: Vratenie Dane
  Dan: Dan
  Dokument: Doklad O Kupe Tovarov Alebo Sluzieb
  Dokument: Doklad O Kupe Pohonnych Latok
  Tovar: Tovar
  Sluzba: Sluzba
  Tovar: Pohonne Latky
  Suma: Celkova Cena Vratane Dane
  Suma: Suma 33,19 Eura
  Podmienka: Podmienka Celkova Cena Najmenej 33,19 Eura
  Stat: Iny Stat
  Osoba: Osoba Slovenskej Republiky
  Stat: Slovenska Republika
  Podmienka: Podmienka Celkova Cena Vyssia Ako 33,19 Eura
  Suma: Suma Urcena Inym Statom
  Podmienka: Podmienka Celkova Cena Najmenej Vo Vyske Urcenej Inym Statom

---

chunk: 894
path: ['§ 62aa', '5']
path_as_text: Paragraf § 62aa Odsek 5
text: (5) Ak sa prestali plniť podmienky na vrátenie dane podľa odseku 1 a rozhodnutie o vrátení dane už bolo vydané, Daňový úrad Bratislava toto rozhodnutie zruší. Ak sa prestali plniť podmienky na vrátenie dane podľa odseku 1 len čiastočne, Daňový úrad Bratislava novým rozhodnutím zruší rozhodnutie o vrátení dane a určí sumu dane, na vrátenie ktorej má Európska komisia, agentúra alebo orgán zriadený podľa práva Európskej únie nárok.

relations:
  Paragraf § 62Aa -> [OBSAHUJE] -> Paragraf § 62Aa Odsek 5
  Paragraf § 62Aa -> [OBSAHUJE] -> Paragraf § 62Aa Odsek 1
  Paragraf § 62Aa Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 62Aa Odsek 1
  Podmienky Na Vratenie Dane -> [VZTAHUJE_SA_NA] -> Vratenie Dane
  Vratenie Dane -> [VZTAHUJE_SA_NA] -> Dan
  Podmienky Na Vratenie Dane -> [VYPLYVA_Z] -> Paragraf § 62Aa Odsek 1
  Danovy Urad Bratislava -> [RUSI] -> Rozhodnutie O Vrateni Dane
  Danovy Urad Bratislava -> [VYDAVA] -> Nove Rozhodnutie
  Nove Rozhodnutie -> [RUSI] -> Rozhodnutie O Vrateni Dane
  Nove Rozhodnutie -> [URCUJE] -> Suma Dane Na Vratenie
  Suma Dane Na Vratenie -> [VZTAHUJE_SA_NA] -> Dan
  Europska Komisia -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Agentura -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Organ Zriadeny Podla Prava Europskej Unie -> [MA_NAROK_NA] -> Suma Dane Na Vratenie
  Organ Zriadeny Podla Prava Europskej Unie -> [VYPLYVA_Z] -> Pravo Europskej Unie

nodes:
  Paragraf: Paragraf § 62Aa
  Odsek: Paragraf § 62Aa Odsek 5
  Odsek: Paragraf § 62Aa Odsek 1
  Podmienka: Podmienky Na Vratenie Dane
  Pravo: Vratenie Dane
  Dan: Dan
  Rozhodnutie: Rozhodnutie O Vrateni Dane
  Rozhodnutie: Nove Rozhodnutie
  Organizacia: Danovy Urad Bratislava
  Suma: Suma Dane Na Vratenie
  Organizacia: Europska Komisia
  Organizacia: Agentura
  Organizacia: Organ Zriadeny Podla Prava Europskej Unie
  PravnyPredpis: Pravo Europskej Unie

---

chunk: 908
path: ['§ 65', '5']
path_as_text: Paragraf § 65 Odsek 5
text: (5) Ak je cestovná kancelária povinná postupovať pri odpočítaní dane podľa § 50, pri výpočte koeficientu neuvádza do čitateľa ani menovateľa služby cestovného ruchu obstarané od iných osôb.

relations:
  Paragraf § 65 -> [OBSAHUJE] -> Paragraf § 65 Odsek 5
  Paragraf § 65 Odsek 5 -> [ODKAZUJE_NA] -> Paragraf § 50
  Cestovna Kancelaria -> [MA_POVINNOST] -> Povinnost Postupovat Pri Odpocitani Dane
  Povinnost Postupovat Pri Odpocitani Dane -> [VZTAHUJE_SA_NA] -> Odpocitanie Dane
  Povinnost Postupovat Pri Odpocitani Dane -> [ODKAZUJE_NA] -> Paragraf § 50
  Sluzby Cestovneho Ruchu Obstarane Od Inych Osob -> [VYPLYVA_Z] -> Ine Osoby

nodes:
  Paragraf: Paragraf § 65
  Odsek: Paragraf § 65 Odsek 5
  Paragraf: Paragraf § 50
  Organizacia: Cestovna Kancelaria
  Povinnost: Povinnost Postupovat Pri Odpocitani Dane
  Pravo: Odpocitanie Dane
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
  Paragraf § 66 Odsek 6 -> [VZTAHUJE_SA_NA] -> Obchodnik
  Rozhodnutie Pre Osobitnu Upravu Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Obchodnik
  Rozhodnutie Pre Osobitnu Upravu Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Osobitna Uprava Uplatnovania Dane
  Obchodnik -> [MA_POVINNOST] -> Uplatnovanie Osobitnej Upravy Uplatnovania Dane
  Uplatnovanie Osobitnej Upravy Uplatnovania Dane -> [MA_PODMIENKU] -> Rozhodnutie Pre Osobitnu Upravu Uplatnovania Dane
  Uplatnovanie Osobitnej Upravy Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Osobitna Uprava Uplatnovania Dane
  Uplatnovanie Osobitnej Upravy Uplatnovania Dane -> [MA_LEHOTU] -> Najmenej Dva Kalendarne Roky
  Paragraf § 66 Odsek 6 -> [URCUJE] -> Uplatnovanie Osobitnej Upravy Uplatnovania Dane

nodes:
  Paragraf: Paragraf § 66
  Odsek: Paragraf § 66 Odsek 6
  Odsek: Paragraf § 66 Odsek 5
  Subjekt: Obchodnik
  Konanie: Osobitna Uprava Uplatnovania Dane
  Podmienka: Rozhodnutie Pre Osobitnu Upravu Uplatnovania Dane
  Povinnost: Uplatnovanie Osobitnej Upravy Uplatnovania Dane
  Lehota: Najmenej Dva Kalendarne Roky

---

chunk: 935
path: ['§ 66', '9']
path_as_text: Paragraf § 66 Odsek 9
text: (9) Obchodník, ktorý uplatňuje osobitnú úpravu, je povinný na účely určenia základu dane podľa odseku 3 viesť osobitne záznamy o predajných cenách a kúpnych cenách tovarov.

relations:
  Paragraf § 66 Odsek 9 -> [JE_SUCASTOU] -> Paragraf § 66
  Paragraf § 66 Odsek 3 -> [JE_SUCASTOU] -> Paragraf § 66
  Obchodnik -> [MA] -> Uplatnovanie Osobitnej Upravy
  Obchodnik -> [MA_POVINNOST] -> Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [VZTAHUJE_SA_NA] -> Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [OBSAHUJE] -> Predajne Ceny Tovarov
  Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [OBSAHUJE] -> Kupne Ceny Tovarov
  Predajne Ceny Tovarov -> [VZTAHUJE_SA_NA] -> Tovary
  Kupne Ceny Tovarov -> [VZTAHUJE_SA_NA] -> Tovary
  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [VZTAHUJE_SA_NA] -> Urcenie Zakladu Dane
  Urcenie Zakladu Dane -> [URCUJE] -> Zaklad Dane
  Urcenie Zakladu Dane -> [ODKAZUJE_NA] -> Paragraf § 66 Odsek 3
  Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov -> [VYPLYVA_Z] -> Paragraf § 66 Odsek 9

nodes:
  Paragraf: Paragraf § 66
  Odsek: Paragraf § 66 Odsek 9
  Odsek: Paragraf § 66 Odsek 3
  Subjekt: Obchodnik
  Konanie: Uplatnovanie Osobitnej Upravy
  Povinnost: Povinnost Viest Osobitne Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Zaznam: Zaznamy O Predajnych Cenach A Kupnych Cenach Tovarov
  Suma: Predajne Ceny Tovarov
  Suma: Kupne Ceny Tovarov
  Tovar: Tovary
  Konanie: Urcenie Zakladu Dane
  Dan: Zaklad Dane

---

chunk: 954
path: ['§ 67', '6']
path_as_text: Paragraf § 67 Odsek 6
text: (6) Platiteľ, ktorý vyrába investičné zlato alebo pretvára zlato na investičné zlato, môže odpočítať daň z tovarov a služieb prijatých na túto činnosť. Osobitné úpravy uplatňovania dane na služby dodávané osobe inej ako zdaniteľnej osobe, na predaj tovaru na diaľku a určité domáce dodania tovaru

relations:
  Paragraf § 67 -> [OBSAHUJE] -> Paragraf § 67 Odsek 6
  Platitel -> [MA_PRAVO] -> Odpocitanie Dane
  Odpocitanie Dane -> [VZTAHUJE_SA_NA] -> Dan Z Tovarov A Sluzieb Prijatych Na Cinnost
  Dan Z Tovarov A Sluzieb Prijatych Na Cinnost -> [VZTAHUJE_SA_NA] -> Tovary Prijate Na Cinnost
  Dan Z Tovarov A Sluzieb Prijatych Na Cinnost -> [VZTAHUJE_SA_NA] -> Sluzby Prijate Na Cinnost
  Platitel -> [MA] -> Vyroba Investicneho Zlata
  Vyroba Investicneho Zlata -> [VZTAHUJE_SA_NA] -> Investicne Zlato
  Platitel -> [MA] -> Pretvaranie Zlata Na Investicne Zlato
  Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Zlato
  Pretvaranie Zlata Na Investicne Zlato -> [VZTAHUJE_SA_NA] -> Investicne Zlato
  Osobitne Upravy Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Sluzby Dodavane Osobe Inej Ako Zdanitelnej Osobe
  Sluzby Dodavane Osobe Inej Ako Zdanitelnej Osobe -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba
  Osobitne Upravy Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku
  Predaj Tovaru Na Dialku -> [VZTAHUJE_SA_NA] -> Tovar Predavany Na Dialku
  Osobitne Upravy Uplatnovania Dane -> [VZTAHUJE_SA_NA] -> Urcite Domace Dodania Tovaru
  Urcite Domace Dodania Tovaru -> [VZTAHUJE_SA_NA] -> Domaci Tovar

nodes:
  Paragraf: Paragraf § 67
  Odsek: Paragraf § 67 Odsek 6
  Subjekt: Platitel
  Tovar: Investicne Zlato
  Tovar: Zlato
  Konanie: Vyroba Investicneho Zlata
  Konanie: Pretvaranie Zlata Na Investicne Zlato
  Pravo: Odpocitanie Dane
  Dan: Dan Z Tovarov A Sluzieb Prijatych Na Cinnost
  Tovar: Tovary Prijate Na Cinnost
  Sluzba: Sluzby Prijate Na Cinnost
  Dan: Osobitne Upravy Uplatnovania Dane
  Sluzba: Sluzby Dodavane Osobe Inej Ako Zdanitelnej Osobe
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Konanie: Predaj Tovaru Na Dialku
  Konanie: Urcite Domace Dodania Tovaru
  Tovar: Tovar Predavany Na Dialku
  Tovar: Domaci Tovar

---

chunk: 976
path: ['§ 68a', '10', 'b)']
path_as_text: Paragraf § 68a Odsek 10 Pismeno b)
text: (10) Zdaniteľná osoba neusadená na území Európskej únie je povinná v daňovom priznaní uviesť b) celkovú hodnotu služieb podľa § 68 ods. 1 písm. a) bez dane dodaných v zdaňovacom období, výšku dane pre každú sadzbu dane, sadzbu dane a celkovú výšku splatnej dane, a to v členení  podľa členských štátov spotreby, v ktorých vznikla daňová povinnosť.

relations:
  Paragraf § 68A -> [OBSAHUJE] -> Paragraf § 68A Odsek 10
  Paragraf § 68A Odsek 10 -> [OBSAHUJE] -> Paragraf § 68A Odsek 10 Pismeno b)
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno a)
  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_POVINNOST] -> Povinnost Uviest Udaje V Danovom Priznani
  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [NEVZTAHUJE_SA_NA] -> Uzemie Europskej Unie
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Celkova Hodnota Sluzieb Bez Dane
  Celkova Hodnota Sluzieb Bez Dane -> [VZTAHUJE_SA_NA] -> Sluzby Podla Paragrafu § 68 Odsek 1 Pismeno a)
  Sluzby Podla Paragrafu § 68 Odsek 1 Pismeno a) -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno a)
  Sluzby Podla Paragrafu § 68 Odsek 1 Pismeno a) -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Vyska Dane Pre Kazdu Sadzbu Dane -> [MA] -> Sadzba Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Celkova Vyska Splatnej Dane
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Clensky Stat Spotreby
  Danova Povinnost -> [VZNIKA] -> Clensky Stat Spotreby

nodes:
  Paragraf: Paragraf § 68A
  Odsek: Paragraf § 68A Odsek 10
  Pismeno: Paragraf § 68A Odsek 10 Pismeno b)
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno a)
  Subjekt: Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
  Lokacia: Uzemie Europskej Unie
  DanovePriznanie: Danove Priznanie
  Povinnost: Povinnost Uviest Udaje V Danovom Priznani
  Suma: Celkova Hodnota Sluzieb Bez Dane
  Sluzba: Sluzby Podla Paragrafu § 68 Odsek 1 Pismeno a)
  ZdanovacieObdobie: Zdanovacie Obdobie
  Dan: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Dan: Celkova Vyska Splatnej Dane
  Stat: Clensky Stat Spotreby
  Povinnost: Danova Povinnost

---

chunk: 977
path: ['§ 68a', '11']
path_as_text: Paragraf § 68a Odsek 11
text: (11) Sumy v daňovom priznaní sa uvádzajú v eurách. Ak sa úhrada za dodané služby podľa § 68 ods. 1 písm. a) uskutoční v inej mene ako v eurách, použije sa na prepočet tejto úhrady na eurá referenčný výmenný kurz určený a vyhlásený Európskou centrálnou bankou alebo Národnou bankou Slovenska5a) platný posledný deň príslušného zdaňovacieho obdobia alebo nasledujúci deň, ak nebol v posledný deň zdaňovacieho obdobia tento kurz určený a vyhlásený.

relations:
  Paragraf § 68a -> [OBSAHUJE] -> Paragraf § 68a Odsek 11
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno a)
  Danove Priznanie -> [MA_SUMU] -> Sumy V Danovom Priznani
  Sumy V Danovom Priznani -> [MA] -> Eura
  Uhrada Za Dodane Sluzby -> [VZTAHUJE_SA_NA] -> Dodane Sluzby
  Dodane Sluzby -> [VYPLYVA_Z] -> Paragraf § 68 Odsek 1 Pismeno a)
  Uhrada Za Dodane Sluzby -> [MA] -> Ina Mena Ako Eura
  Prepocet Uhrady Na Eura -> [VZTAHUJE_SA_NA] -> Uhrada Za Dodane Sluzby
  Prepocet Uhrady Na Eura -> [VZTAHUJE_SA_NA] -> Eura
  Prepocet Uhrady Na Eura -> [MA_HODNOTU] -> Referencny Vymenny Kurz
  Europska Centralna Banka -> [URCUJE] -> Referencny Vymenny Kurz
  Narodna Banka Slovenska -> [URCUJE] -> Referencny Vymenny Kurz
  Referencny Vymenny Kurz -> [MA_DATUM] -> Posledny Den Prislusneho Zdanovacieho Obdobia
  Posledny Den Prislusneho Zdanovacieho Obdobia -> [PATRI_DO] -> Prislusne Zdanovacie Obdobie
  Referencny Vymenny Kurz -> [MA_DATUM] -> Nasledujuci Den
  Nasledujuci Den -> [MA_PODMIENKU] -> Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia
  Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia -> [MA_DATUM] -> Posledny Den Prislusneho Zdanovacieho Obdobia
  Paragraf § 68a Odsek 11 -> [UPRAVUJE] -> Danove Priznanie
  Paragraf § 68a Odsek 11 -> [UPRAVUJE] -> Prepocet Uhrady Na Eura

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
  Platba: Uhrada Za Dodane Sluzby
  Sluzba: Dodane Sluzby
  Konanie: Prepocet Uhrady Na Eura
  Kurz: Referencny Vymenny Kurz
  Banka: Europska Centralna Banka
  Banka: Narodna Banka Slovenska
  ZdanovacieObdobie: Prislusne Zdanovacie Obdobie
  Datum: Posledny Den Prislusneho Zdanovacieho Obdobia
  Datum: Nasledujuci Den
  Podmienka: Kurz Neurceny A Nevyhlaseny V Posledny Den Zdanovacieho Obdobia

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
  Zdanitelna Osoba Uvedena V Paragraf § 68b Odsek 2 -> [MA_POVINNOST] -> Povinnost Uviest Udaje V Danovom Priznani
  Povinnost Uviest Udaje V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Danove Priznanie -> [MA_HODNOTU] -> Celkova Hodnota Dodanych Tovarov Bez Dane
  Celkova Hodnota Dodanych Tovarov Bez Dane -> [VZTAHUJE_SA_NA] -> Dodane Tovary Podla Paragraf § 68 Odsek 1 Pismeno b)
  Danove Priznanie -> [MA_SUMU] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Vyska Dane Pre Kazdu Sadzbu Dane -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Danove Priznanie -> [MA] -> Sadzba Dane
  Danove Priznanie -> [MA_SUMU] -> Celkova Vyska Splatnej Dane
  Celkova Vyska Splatnej Dane -> [VZTAHUJE_SA_NA] -> Splatna Dan
  Danove Priznanie -> [VZTAHUJE_SA_NA] -> Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje
  Tovar Odosielany Alebo Prepravovany Z Inych Clenskych Statov -> [VZTAHUJE_SA_NA] -> Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje
  Povinnost Uviest Udaje V Danovom Priznani -> [MA_PODMIENKU] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [NACHADZA_SA_V] -> Uzemi Europskej Unie
  Paragraf § 68b Odsek 14 Pismeno a) -> [UPRAVUJE] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie

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
  Povinnost: Povinnost Uviest Udaje V Danovom Priznani
  DanovePriznanie: Danove Priznanie
  Suma: Celkova Hodnota Dodanych Tovarov Bez Dane
  Tovar: Dodane Tovary Podla Paragraf § 68 Odsek 1 Pismeno b)
  Suma: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Suma: Celkova Vyska Splatnej Dane
  Dan: Splatna Dan
  Stat: Clenske Staty Z Ktorych Sa Tovar Odosiela Alebo Prepravuje
  Tovar: Tovar Odosielany Alebo Prepravovany Z Inych Clenskych Statov
  Konanie: Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Lokacia: Uzemi Europskej Unie

---

chunk: 1017
path: ['§ 68c', '4']
path_as_text: Paragraf § 68c Odsek 4
text: (4) Ak sa zdaniteľná osoba, ktorá uskutočňuje predaj tovaru na diaľku podľa § 68 ods. 1 písm. c) a ktorá nie je zastúpená sprostredkovateľom, rozhodne pre uplatňovanie osobitnej úpravy a členským štátom identifikácie je tuzemsko alebo si tuzemsko zvolí ako členský štát identifikácie, je povinná predtým, ako začne uplatňovať osobitnú úpravu, oznámiť toto rozhodnutie daňovému úradu. Toto oznámenie musí obsahovať obchodné meno, adresu, elektronickú adresu vrátane webových sídiel, identifikačné číslo pre daň alebo národné daňové číslo a ďalšie údaje uvedené v osobitnom predpise.28aa) Ak zdaniteľná osoba spĺňa podmienky na uplatňovanie osobitnej úpravy, daňový úrad jej oznámi, že jej povoľuje uplatňovanie osobitnej úpravy; proti tomuto rozhodnutiu nemožno podať odvolanie.

relations:
  Paragraf § 68C -> [OBSAHUJE] -> Paragraf § 68C Odsek 4
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno c)
  Zdanitelna Osoba -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku
  Predaj Tovaru Na Dialku -> [VZTAHUJE_SA_NA] -> Tovar
  Predaj Tovaru Na Dialku -> [VYPLYVA_Z] -> Paragraf § 68 Odsek 1 Pismeno c)
  Zdanitelna Osoba -> [JE_ZASTUPENA] -> Sprostredkovatel
  Zdanitelna Osoba -> [MA_PRAVO] -> Uplatnovanie Osobitnej Upravy
  Clensky Stat Identifikacie -> [MA] -> Tuzemsko
  Rozhodnutie Pre Uplatnovanie Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Zdanitelna Osoba -> [OZNAMUJE] -> Rozhodnutie Pre Uplatnovanie Osobitnej Upravy
  Zdanitelna Osoba -> [OZNAMUJE] -> Danovy Urad
  Zdanitelna Osoba -> [MA_POVINNOST] -> Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Obchodne Meno
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Adresa
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Elektronicka Adresa
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Webove Sidlo
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Identifikacne Cislo Pre Dan
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Narodne Danove Cislo
  Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy -> [OBSAHUJE] -> Dalsie Udaje Uvedene V Osobitnom Predpise
  Zdanitelna Osoba -> [SPLNA_PODMIENKY] -> Podmienky Na Uplatnovanie Osobitnej Upravy
  Podmienky Na Uplatnovanie Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Danovy Urad -> [VYDAVA] -> Povolenie Uplatnovania Osobitnej Upravy
  Povolenie Uplatnovania Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Zdanitelna Osoba -> [NEMA_NAROK_NA] -> Odvolanie

nodes:
  Paragraf: Paragraf § 68C
  Odsek: Paragraf § 68C Odsek 4
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)
  Osoba: Zdanitelna Osoba
  Konanie: Predaj Tovaru Na Dialku
  Tovar: Tovar
  Osoba: Sprostredkovatel
  Konanie: Uplatnovanie Osobitnej Upravy
  Stat: Clensky Stat Identifikacie
  Stat: Tuzemsko
  Rozhodnutie: Rozhodnutie Pre Uplatnovanie Osobitnej Upravy
  Oznamenie: Oznamenie Rozhodnutia O Uplatnovani Osobitnej Upravy
  Organizacia: Danovy Urad
  Dokument: Obchodne Meno
  Adresa: Adresa
  Adresa: Elektronicka Adresa
  Adresa: Webove Sidlo
  Dokument: Identifikacne Cislo Pre Dan
  Dokument: Narodne Danove Cislo
  Zaznam: Dalsie Udaje Uvedene V Osobitnom Predpise
  Podmienka: Podmienky Na Uplatnovanie Osobitnej Upravy
  Rozhodnutie: Povolenie Uplatnovania Osobitnej Upravy
  Konanie: Odvolanie

---

chunk: 1023
path: ['§ 68c', '8']
path_as_text: Paragraf § 68c Odsek 8
text: (8) Identifikačné číslo pre daň pridelené podľa odseku 7 písm. a) a c) a evidenčné identifikačné číslo pridelené podľa odseku 7 písm. b) sa môže použiť len na účely uplatňovania osobitnej úpravy.

relations:
  Paragraf § 68c Odsek 7 Pismeno a) -> [JE_SUCASTOU] -> Paragraf § 68c Odsek 7
  Paragraf § 68c Odsek 7 Pismeno b) -> [JE_SUCASTOU] -> Paragraf § 68c Odsek 7
  Paragraf § 68c Odsek 7 Pismeno c) -> [JE_SUCASTOU] -> Paragraf § 68c Odsek 7
  Paragraf § 68c Odsek 8 -> [UPRAVUJE] -> Identifikacne Cislo Pre Dan
  Paragraf § 68c Odsek 8 -> [UPRAVUJE] -> Evidencne Identifikacne Cislo
  Identifikacne Cislo Pre Dan -> [VYPLYVA_Z] -> Paragraf § 68c Odsek 7 Pismeno a)
  Identifikacne Cislo Pre Dan -> [VYPLYVA_Z] -> Paragraf § 68c Odsek 7 Pismeno c)
  Evidencne Identifikacne Cislo -> [VYPLYVA_Z] -> Paragraf § 68c Odsek 7 Pismeno b)
  Pouzitie Identifikacneho Cisla Pre Dan -> [VZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan
  Pouzitie Identifikacneho Cisla Pre Dan -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Pouzitie Evidencneho Identifikacneho Cisla -> [VZTAHUJE_SA_NA] -> Evidencne Identifikacne Cislo
  Pouzitie Evidencneho Identifikacneho Cisla -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy

nodes:
  Odsek: Paragraf § 68c Odsek 8
  Odsek: Paragraf § 68c Odsek 7
  Pismeno: Paragraf § 68c Odsek 7 Pismeno a)
  Pismeno: Paragraf § 68c Odsek 7 Pismeno b)
  Pismeno: Paragraf § 68c Odsek 7 Pismeno c)
  Zaznam: Identifikacne Cislo Pre Dan
  Zaznam: Evidencne Identifikacne Cislo
  Konanie: Uplatnovanie Osobitnej Upravy
  Pravo: Pouzitie Identifikacneho Cisla Pre Dan
  Pravo: Pouzitie Evidencneho Identifikacneho Cisla

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
  Paragraf § 68c Odsek 21 Pismeno b) -> [UPRAVUJE] -> Danove Priznanie
  Paragraf § 68c Odsek 21 Pismeno b) -> [ODKAZUJE_NA] -> Paragraf § 68 Odsek 1 Pismeno c)
  Danove Priznanie -> [MA_HODNOTU] -> Celkova Hodnota Predaja Tovaru Na Dialku Bez Dane
  Celkova Hodnota Predaja Tovaru Na Dialku Bez Dane -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku
  Predaj Tovaru Na Dialku -> [VZTAHUJE_SA_NA] -> Tovar
  Danova Povinnost -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku
  Danova Povinnost -> [VZNIKA] -> Zdanovacie Obdobie
  Danove Priznanie -> [MA_SUMU] -> Vyska Dane Pre Kazdu Sadzbu Dane
  Vyska Dane Pre Kazdu Sadzbu Dane -> [VZTAHUJE_SA_NA] -> Sadzba Dane
  Danove Priznanie -> [MA] -> Sadzba Dane
  Danove Priznanie -> [MA_SUMU] -> Celkova Vyska Splatnej Dane
  Celkova Vyska Splatnej Dane -> [VZTAHUJE_SA_NA] -> Splatna Dan
  Danove Priznanie -> [VZTAHUJE_SA_NA] -> Clensky Stat Spotreby
  Danova Povinnost -> [VZNIKA] -> Clensky Stat Spotreby

nodes:
  Paragraf: Paragraf § 68c
  Odsek: Paragraf § 68c Odsek 21
  Pismeno: Paragraf § 68c Odsek 21 Pismeno b)
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)
  DanovePriznanie: Danove Priznanie
  Suma: Celkova Hodnota Predaja Tovaru Na Dialku Bez Dane
  Konanie: Predaj Tovaru Na Dialku
  Tovar: Tovar
  Povinnost: Danova Povinnost
  ZdanovacieObdobie: Zdanovacie Obdobie
  Suma: Vyska Dane Pre Kazdu Sadzbu Dane
  SadzbaDane: Sadzba Dane
  Suma: Celkova Vyska Splatnej Dane
  Dan: Splatna Dan
  Stat: Clensky Stat Spotreby

---

chunk: 1058
path: ['§ 68ca', '6', 'c)']
path_as_text: Paragraf § 68ca Odsek 6 Pismeno c)
text: (6) Ak je členským štátom spotreby Slovenská republika, osoba, ktorá uplatňuje alebo uplatňovala osobitnú úpravu podľa § 68a až 68c alebo podľa ustanovení zákona platného v inom členskom štáte zodpovedajúcich § 68a až 68c, c) je povinná podať daňovému úradu elektronickými prostriedkami osobitné tlačivo do 30 dní odo dňa zistenia, že neuviedla daň alebo daň má byť vyššia, ako bola uvedená v podanom konečnom daňovom priznaní28ae)alebo predchádzajúcich daňových priznaniach po podaní konečného daňového priznania alebo

relations:
  Paragraf § 68ca -> [OBSAHUJE] -> Paragraf § 68ca Odsek 6
  Paragraf § 68ca Odsek 6 -> [OBSAHUJE] -> Paragraf § 68ca Odsek 6 Pismeno c)
  Paragraf § 68ca Odsek 6 Pismeno c) -> [UPRAVUJE] -> Podanie Osobitneho Tlaciva
  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [MA_POVINNOST] -> Podanie Osobitneho Tlaciva
  Podanie Osobitneho Tlaciva -> [VZTAHUJE_SA_NA] -> Osobitne Tlacivo
  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [PODAVA] -> Osobitne Tlacivo
  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [PODAVA] -> Danovy Urad
  Podanie Osobitneho Tlaciva -> [MA_PODMIENKU] -> Elektronicke Prostriedky
  Podanie Osobitneho Tlaciva -> [MA_LEHOTU] -> Lehota 30 Dni Odo Dna Zistenia
  Lehota 30 Dni Odo Dna Zistenia -> [VYPLYVA_Z] -> Den Zistenia
  Podanie Osobitneho Tlaciva -> [VYPLYVA_Z] -> Zistenie Neuvedenej Alebo Vyssej Dane
  Zistenie Neuvedenej Alebo Vyssej Dane -> [VZTAHUJE_SA_NA] -> Neuvedena Dan
  Zistenie Neuvedenej Alebo Vyssej Dane -> [VZTAHUJE_SA_NA] -> Vyssia Dan Ako Uvedena Dan
  Vyssia Dan Ako Uvedena Dan -> [VZTAHUJE_SA_NA] -> Podane Konecne Danove Priznanie
  Vyssia Dan Ako Uvedena Dan -> [VZTAHUJE_SA_NA] -> Predchadzajuce Danove Priznania Po Podani Konecneho Danoveho Priznania
  Osoba Uplatnujuca Alebo Uplatnovala Osobitnu Upravu -> [VZTAHUJE_SA_NA] -> Osobitna Uprava
  Zakon Platny V Inom Clenskom State -> [OBSAHUJE] -> Ustanovenia Zodpovedajuce Paragrafu § 68a Az 68c
  Ustanovenia Zodpovedajuce Paragrafu § 68a Az 68c -> [ODKAZUJE_NA] -> Paragraf § 68a
  Ustanovenia Zodpovedajuce Paragrafu § 68a Az 68c -> [ODKAZUJE_NA] -> Paragraf § 68b
  Ustanovenia Zodpovedajuce Paragrafu § 68a Az 68c -> [ODKAZUJE_NA] -> Paragraf § 68c
  Slovenska Republika -> [JE_TYPOM] -> Clensky Stat Spotreby
  Podanie Osobitneho Tlaciva -> [MA_PODMIENKU] -> Slovenska Republika

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
  Konanie: Osobitna Uprava
  PravnyPredpis: Zakon Platny V Inom Clenskom State
  Paragraf: Ustanovenia Zodpovedajuce Paragrafu § 68a Az 68c
  Organizacia: Danovy Urad
  Dokument: Elektronicke Prostriedky
  Dokument: Osobitne Tlacivo
  Povinnost: Podanie Osobitneho Tlaciva
  Lehota: Lehota 30 Dni Odo Dna Zistenia
  Datum: Den Zistenia
  Dan: Neuvedena Dan
  Dan: Vyssia Dan Ako Uvedena Dan
  DanovePriznanie: Podane Konecne Danove Priznanie
  DanovePriznanie: Predchadzajuce Danove Priznania Po Podani Konecneho Danoveho Priznania
  Dovod: Zistenie Neuvedenej Alebo Vyssej Dane

---

chunk: 1069
path: ['§ 68cb', '4']
path_as_text: Paragraf § 68cb Odsek 4
text: (4) Osoba, ktorá má povolenie podľa odseku 3, vyberie daň od osoby, pre ktorú je dovezený tovar určený, a túto vybranú daň je povinná zaplatiť colnému úradu.

relations:
  Paragraf § 68Cb -> [OBSAHUJE] -> Paragraf § 68Cb Odsek 4
  Paragraf § 68Cb -> [OBSAHUJE] -> Paragraf § 68Cb Odsek 3
  Osoba S Povolenim -> [MA] -> Povolenie Podla Paragrafu § 68Cb Odsek 3
  Povolenie Podla Paragrafu § 68Cb Odsek 3 -> [VYPLYVA_Z] -> Paragraf § 68Cb Odsek 3
  Osoba S Povolenim -> [PRIJIMA] -> Dan
  Osoba Pre Ktoru Je Dovezeny Tovar Urceny -> [PLATI] -> Dan
  Dovezeny Tovar -> [VZTAHUJE_SA_NA] -> Osoba Pre Ktoru Je Dovezeny Tovar Urceny
  Vybrana Dan -> [JE_TYPOM] -> Dan
  Osoba S Povolenim -> [MA_POVINNOST] -> Povinnost Zaplatit Vybranu Dan Colnemu Uradu
  Povinnost Zaplatit Vybranu Dan Colnemu Uradu -> [VZTAHUJE_SA_NA] -> Vybrana Dan
  Osoba S Povolenim -> [PLATI] -> Colny Urad

nodes:
  Paragraf: Paragraf § 68Cb
  Odsek: Paragraf § 68Cb Odsek 4
  Odsek: Paragraf § 68Cb Odsek 3
  Osoba: Osoba S Povolenim
  Rozhodnutie: Povolenie Podla Paragrafu § 68Cb Odsek 3
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
  Platitel -> [MA_POVINNOST] -> Skoncenie Uplatnovania Osobitnej Upravy
  Skoncenie Uplatnovania Osobitnej Upravy -> [MA_PODMIENKU] -> Stat Sa Clenom Skupiny
  Platitel -> [MA_STATUS] -> Clen Skupiny
  Clen Skupiny -> [VZTAHUJE_SA_NA] -> Skupina
  Stat Sa Clenom Skupiny -> [VZTAHUJE_SA_NA] -> Skupina
  Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Dnu Statia Sa Clenom Skupiny
  Skoncenie Uplatnovania Osobitnej Upravy -> [VYPLYVA_Z] -> Paragraf § 68d Odsek 11 Pismeno b)

nodes:
  Paragraf: Paragraf § 68d
  Odsek: Paragraf § 68d Odsek 11
  Pismeno: Paragraf § 68d Odsek 11 Pismeno b)
  Subjekt: Platitel
  Organizacia: Skupina
  Status: Clen Skupiny
  Podmienka: Stat Sa Clenom Skupiny
  Povinnost: Skoncenie Uplatnovania Osobitnej Upravy
  Datum: Den Predchadzajuci Dnu Statia Sa Clenom Skupiny

---

chunk: 1099
path: ['§ 68d', '14', 'a)']
path_as_text: Paragraf § 68d Odsek 14 Pismeno a)
text: (14) Daňový úrad uloží pokutu do výšky 10 000 eur, ak a) platiteľ uplatňuje osobitnú úpravu a nesplnil podmienky podľa odseku 1,

relations:
  Paragraf § 68d -> [OBSAHUJE] -> Paragraf § 68d Odsek 14
  Paragraf § 68d Odsek 14 -> [OBSAHUJE] -> Paragraf § 68d Odsek 14 Pismeno a)
  Paragraf § 68d -> [OBSAHUJE] -> Paragraf § 68d Odsek 1
  Danovy Urad -> [VYDAVA] -> Pokuta Do Vysky 10 000 Eur
  Pokuta Do Vysky 10 000 Eur -> [MA_SUMU] -> Suma 10 000 Eur
  Platitel -> [MA] -> Uplatnovanie Osobitnej Upravy
  Platitel -> [NESPLNA_PODMIENKY] -> Podmienky Podla Odseku 1
  Podmienky Podla Odseku 1 -> [ODKAZUJE_NA] -> Paragraf § 68d Odsek 1
  Nesplnenie Podmienok Podla Odseku 1 -> [VYPLYVA_Z] -> Podmienky Podla Odseku 1
  Pokuta Do Vysky 10 000 Eur -> [VYPLYVA_Z] -> Nesplnenie Podmienok Podla Odseku 1
  Pokuta Do Vysky 10 000 Eur -> [VYPLYVA_Z] -> Paragraf § 68d Odsek 14 Pismeno a)

nodes:
  Organizacia: Danovy Urad
  Subjekt: Platitel
  Sankcia: Pokuta Do Vysky 10 000 Eur
  Suma: Suma 10 000 Eur
  Konanie: Uplatnovanie Osobitnej Upravy
  Dovod: Nesplnenie Podmienok Podla Odseku 1
  Podmienka: Podmienky Podla Odseku 1
  Paragraf: Paragraf § 68d
  Odsek: Paragraf § 68d Odsek 14
  Pismeno: Paragraf § 68d Odsek 14 Pismeno a)
  Odsek: Paragraf § 68d Odsek 1

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

