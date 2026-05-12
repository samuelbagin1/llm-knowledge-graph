chunk: 124
page: 30
text: v ktorom sa vyhotovil doklad o oprave základu dane, a pri oprave základu dane pri nadobudnutí tovaru v tuzemsku z iného členského štátu alebo pri dodaní tovaru alebo služby, keď je povinný platiť daň nadobúdateľ tovaru alebo príjemca služby, rozdiel sa uvedie v daňovom priznaní za zdaňovacie obdobie, v ktorom nadobúdateľ tovaru alebo príjemca služby obdržal doklad o oprave základu dane. Ak sa pri oprave základu dane nevyhotovuje doklad o oprave základu dane, uvedie sa rozdiel medzi pôvodným základom dane a opraveným základom dane a rozdiel medzi pôvodnou daňou a opravenou daňou v daňovom priznaní za zdaňovacie obdobie, v ktorom nastala skutočnosť, ktorá má za následok opravu základu dane. (4) Colný orgán vráti alebo odpustí daň pri dovoze tovaru na základe žiadosti v prípadoch podľa osobitného predpisu6acaa) s výnimkou prípadu, ak platiteľ môže odpočítať daň z dovezeného tovaru v plnom rozsahu; táto výnimka sa nevzťahuje na prípad podľa osobitného predpisu.6acab) Ak pri
relationships:
  Oprava Zakladu Dane -> [MA_DOKLAD] -> Doklad O Oprave Zakladu Dane
  Oprava Zakladu Dane -> [TYKA_SA] -> Zaklad Dane
  Oprava Zakladu Dane -> [TYKA_SA] -> Nadobudnutie Tovaru
  Nadobudnutie Tovaru -> [TYKA_SA] -> Tovar
  Nadobudnutie Tovaru -> [MA_MIESTO] -> Tuzemsko
  Nadobudnutie Tovaru -> [VYCHADZA_Z] -> Iny Clensky Stat
  Oprava Zakladu Dane -> [TYKA_SA] -> Dodanie Tovaru
  Oprava Zakladu Dane -> [TYKA_SA] -> Dodanie Sluzby
  Dodanie Tovaru -> [TYKA_SA] -> Tovar
  Dodanie Sluzby -> [TYKA_SA] -> Sluzba
  Nadobudatel Tovaru -> [JE_POVINNY_PLATIT] -> Dan
  Prijemca Sluzby -> [JE_POVINNY_PLATIT] -> Dan
  Danove Priznanie -> [MA_OBDOBIE] -> Zdanovacie Obdobie
  Danove Priznanie -> [UVADZA] -> Rozdiel
  Nadobudatel Tovaru -> [PRIJIMA] -> Doklad O Oprave Zakladu Dane
  Prijemca Sluzby -> [PRIJIMA] -> Doklad O Oprave Zakladu Dane
  Skutocnost Sposobujuca Opravu Zakladu Dane -> [VZNIKA_PRI] -> Oprava Zakladu Dane
  Dovoz Tovaru -> [TYKA_SA] -> Tovar
  Dan Z Dovezeneho Tovaru -> [VZTAHUJE_SA_NA] -> Tovar
  Colny Organ -> [ROZHODUJE_O] -> Dan Z Dovezeneho Tovaru
  Rozhodnutie O Vrateni Alebo Odpusteni Dane -> [VYCHADZA_Z] -> Ziadost
  Platitel -> [MA_PRAVO] -> Odpocitanie Dane
  Odpocitanie Dane -> [TYKA_SA] -> Dan Z Dovezeneho Tovaru
  Odpocitanie Dane -> [TYKA_SA] -> Dovezeny Tovar

nodes:
  Doklad: Doklad O Oprave Zakladu Dane
  Oprava: Oprava Zakladu Dane
  Hodnota: Zaklad Dane
  Tovar: Tovar
  Tovar: Dovezeny Tovar
  Tuzemsko: Tuzemsko
  ClenskyStat: Iny Clensky Stat
  Sluzba: Sluzba
  Cinnost: Nadobudnutie Tovaru
  Cinnost: Dodanie Tovaru
  Cinnost: Dodanie Sluzby
  Subjekt: Nadobudatel Tovaru
  Subjekt: Prijemca Sluzby
  DanovePriznanie: Danove Priznanie
  ZdanovacieObdobie: Zdanovacie Obdobie
  Hodnota: Rozdiel
  Dan: Dan
  Dan: Dan Z Dovezeneho Tovaru
  StatnyOrgan: Colny Organ
  Ziadost: Ziadost
  Cinnost: Dovoz Tovaru
  ZdanitelnaOsoba: Platitel
  Pravo: Odpocitanie Dane
  Dovod: Skutocnost Sposobujuca Opravu Zakladu Dane
  Rozhodnutie: Rozhodnutie O Vrateni Alebo Odpusteni Dane

chunk: 146
page: 36
text: b) právnickou osobou alebo fyzickou osobou, ktorá spĺňa jednu podmienku alebo viac podmienok podľa § 30 ods. 2, c) ako odborný výcvik a rekvalifikácia poskytované podľa osobitného predpisu.13) (2) Oslobodené od dane je aj dodanie tovaru a dodanie služieb úzko súvisiacich s výchovnými službami a vzdelávacími službami podľa odseku 1 osobami, ktoré poskytujú výchovné služby a vzdelávacie služby podľa odseku 1. § 32 Služby dodávané členom (1) Oslobodené od dane sú služby dodávané ako protihodnota členského príspevku pre vlastných členov politických strán a hnutí, cirkví a náboženských spoločností, občianskych združení vrátane odborových organizácií a profesijných komôr, ak toto oslobodenie od dane nenarušuje hospodársku súťaž; oslobodený od dane je aj tovar dodaný týmito osobami v úzkej súvislosti s dodanou službou. (2) Oslobodené od dane sú služby, ktoré svojim členom dodáva právnická osoba, ak
relationships:
  Pravnicka Osoba -> [SPLNA_PODMIENKY] -> Podmienky Podla Paragrafu 30 Odsek 2
  Fyzicka Osoba -> [SPLNA_PODMIENKY] -> Podmienky Podla Paragrafu 30 Odsek 2
  Podmienky Podla Paragrafu 30 Odsek 2 -> [JE_PODLA] -> Paragraf § 30 Odsek 2

  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Odborny Vycvik A Rekvalifikacia
  Odborny Vycvik A Rekvalifikacia -> [JE_PODLA] -> Osobitny Predpis

  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru
  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Dodanie Sluzieb
  Dodanie Tovaru -> [SUVISI_S] -> Vychovne Sluzby
  Dodanie Tovaru -> [SUVISI_S] -> Vzdelavacie Sluzby
  Dodanie Sluzieb -> [SUVISI_S] -> Vychovne Sluzby
  Dodanie Sluzieb -> [SUVISI_S] -> Vzdelavacie Sluzby

  Sluzby Dodavane Clenom -> [JE_PODLA] -> Paragraf § 32
  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Sluzby Dodavane Clenom
  Sluzby Dodavane Clenom -> [SUVISI_S] -> Clensky Prispevok
  Sluzby Dodavane Clenom -> [TYKA_SA] -> Clenovia
  Sluzby Dodavane Clenom -> [MA_PODMIENKU] -> Nenarusenie Hospodarskej Sutaze

  Clenovia -> [JE_CLENOM] -> Politicke Strany
  Clenovia -> [JE_CLENOM] -> Hnutia
  Clenovia -> [JE_CLENOM] -> Cirkvi
  Clenovia -> [JE_CLENOM] -> Nabozenske Spolocnosti
  Clenovia -> [JE_CLENOM] -> Obcianske Zdruzenia
  Clenovia -> [JE_CLENOM] -> Odborove Organizacie
  Clenovia -> [JE_CLENOM] -> Profesijne Komory

  Pravnicka Osoba -> [DODAVA] -> Sluzby Dodavane Clenom
  Pravnicka Osoba -> [DODAVA] -> Tovar Dodany V Uzkej Suvislosti So Sluzbou
  Tovar Dodany V Uzkej Suvislosti So Sluzbou -> [SUVISI_S] -> Sluzby Dodavane Clenom
  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Tovar Dodany V Uzkej Suvislosti So Sluzbou

nodes:
  Odsek: Paragraf § 30 Odsek 2
  Paragraf: Paragraf § 32
  PravnickaOsoba: Pravnicka Osoba
  FyzickaOsoba: Fyzicka Osoba
  Podmienka: Podmienky Podla Paragrafu 30 Odsek 2
  PravnyPredpis: Osobitny Predpis
  OslobodenieOdDane: Oslobodenie Od Dane
  Cinnost: Odborny Vycvik A Rekvalifikacia
  Cinnost: Dodanie Tovaru
  Cinnost: Dodanie Sluzieb
  Sluzba: Vychovne Sluzby
  Sluzba: Vzdelavacie Sluzby
  Sluzba: Sluzby Dodavane Clenom
  Povinnost: Clensky Prispevok
  ClenSkupiny: Clenovia
  Organizacia: Politicke Strany
  Organizacia: Hnutia
  Organizacia: Cirkvi
  Organizacia: Nabozenske Spolocnosti
  Organizacia: Obcianske Zdruzenia
  Organizacia: Odborove Organizacie
  Organizacia: Profesijne Komory
  Podmienka: Nenarusenie Hospodarskej Sutaze
  Tovar: Tovar Dodany V Uzkej Suvislosti So Sluzbou


chunk: 192
page: 49
text: Zabezpečenie dane pri dovoze tovaru (1) Colný úrad môže žiadať zabezpečenie dane pri dovoze tovaru, pri ktorom sa uplatňuje oslobodenie od dane podľa § 48 ods. 3, pred prepustením tovaru do colného režimu voľný obeh. Zabezpečenie dane je povinná zložiť osoba, ktorá by bola povinná platiť daň, keby sa neuplatnilo oslobodenie od dane podľa § 48 ods. 3, vo výške dane, ktorú by bola povinná platiť, keby sa neuplatnilo oslobodenie od dane podľa § 48 ods. 3. Zabezpečenie dane sa poskytne formou podľa osobitného predpisu.24aa) (2) Colný úrad rozhodnutím určí výšku zabezpečenia dane a lehotu na jej zaplatenie. Proti rozhodnutiu o zabezpečení dane nie je možné podať odvolanie. Ak osoba podľa odseku 1 zabezpečenie dane nezaplatí v lehote a vo výške určenej v rozhodnutí, colný úrad oslobodenie od dane podľa § 48 ods. 3 neuplatní. (3) Colný úrad uvoľní zabezpečenie dane do desiatich dní od predloženia dôkazu o tom, že odoslanie alebo preprava tovaru sa skončila v inom členskom štáte okrem odseku 4. Dôkazom, že
relationships:
  Colny Urad -> [MA_PRAVO] -> Zabezpeka Na Dan Pri Dovoze Tovaru
  Zabezpeka Na Dan Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Dovoz Tovaru -> [TYKA_SA] -> Tovar
  Zabezpeka Na Dan Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Dan
  Zabezpeka Na Dan Pri Dovoze Tovaru -> [VZTAHUJE_SA_NA] -> Oslobodenie Od Dane Podla Paragrafu 48 Odsek 3
  Oslobodenie Od Dane Podla Paragrafu 48 Odsek 3 -> [JE_PODLA] -> Odsek 3
  Paragraf § 48 -> [MA_ODSEK] -> Odsek 3

  Osoba Povinna Platit Dan -> [MA_POVINNOST] -> Zlozenie Zabezpeky Na Dan
  Zlozenie Zabezpeky Na Dan -> [TYKA_SA] -> Zabezpeka Na Dan Pri Dovoze Tovaru
  Osoba Povinna Platit Dan -> [JE_POVINNY_PLATIT] -> Dan
  Osoba Povinna Platit Dan -> [MA_PODMIENKU] -> Neuplatnenie Oslobodenia Od Dane
  Zabezpeka Na Dan Pri Dovoze Tovaru -> [MA_SUMU] -> Suma Dane
  Zabezpeka Na Dan Pri Dovoze Tovaru -> [JE_PODLA] -> Osobitny Predpis

  Colny Urad -> [VYDAVA] -> Rozhodnutie O Zabezpeceni Dane
  Rozhodnutie O Zabezpeceni Dane -> [URCUJE] -> Vyska Zabezpecenia Dane
  Rozhodnutie O Zabezpeceni Dane -> [MA_SUMU] -> Suma Dane
  Rozhodnutie O Zabezpeceni Dane -> [MA_LEHOTU] -> Lehota Na Zaplatenie Zabezpecenia Dane
  Rozhodnutie O Zabezpeceni Dane -> [MA_ODKLADNY_UCINOK] -> Nemoznost Podat Odvolanie

  Nezaplatenie Zabezpeky Na Dan -> [NASTAVA_PRI] -> Neuhradenie V Lehote A Vo Vyske Urcenej V Rozhodnuti
  Neuhradenie V Lehote A Vo Vyske Urcenej V Rozhodnuti -> [NEVZTAHUJE_SA_NA] -> Oslobodenie Od Dane Podla Paragrafu 48 Odsek 3

  Colny Urad -> [VYKONAVA] -> Uvolnenie Zabezpeky Na Dan
  Uvolnenie Zabezpeky Na Dan -> [TYKA_SA] -> Zabezpeka Na Dan Pri Dovoze Tovaru
  Uvolnenie Zabezpeky Na Dan -> [MA_LEHOTU] -> Lehota Do Desiatich Dni
  Uvolnenie Zabezpeky Na Dan -> [VYCHADZA_Z] -> Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru
  Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru -> [PREUKAZUJE] -> Skoncenie Odoslania Alebo Prepravy Tovaru
  Skoncenie Odoslania Alebo Prepravy Tovaru -> [TYKA_SA] -> Tovar
  Skoncenie Odoslania Alebo Prepravy Tovaru -> [MA_MIESTO] -> Iny Clensky Stat

nodes:
  Urad: Colny Urad
  ZabezpekaNaDan: Zabezpeka Na Dan Pri Dovoze Tovaru
  Cinnost: Dovoz Tovaru
  Tovar: Tovar
  Dan: Dan
  OslobodenieOdDane: Oslobodenie Od Dane Podla Paragrafu 48 Odsek 3
  Paragraf: Paragraf § 48
  Odsek: Odsek 3
  Osoba: Osoba Povinna Platit Dan
  Povinnost: Zlozenie Zabezpeky Na Dan
  Podmienka: Neuplatnenie Oslobodenia Od Dane
  Suma: Suma Dane
  PravnyPredpis: Osobitny Predpis
  Rozhodnutie: Rozhodnutie O Zabezpeceni Dane
  Hodnota: Vyska Zabezpecenia Dane
  Lehota: Lehota Na Zaplatenie Zabezpecenia Dane
  Status: Nemoznost Podat Odvolanie
  Cinnost: Nezaplatenie Zabezpeky Na Dan
  Podmienka: Neuhradenie V Lehote A Vo Vyske Urcenej V Rozhodnuti
  Cinnost: Uvolnenie Zabezpeky Na Dan
  Lehota: Lehota Do Desiatich Dni
  Doklad: Dokaz O Skonceni Odoslania Alebo Prepravy Tovaru
  Cinnost: Skoncenie Odoslania Alebo Prepravy Tovaru
  ClenskyStat: Iny Clensky Stat


chunk: 221
page: 56
text: d) ním uplatnená pri dovoze tovaru alebo zaplatená colnému orgánu v tuzemsku pri dovoze tovaru. (3) Platiteľ nemôže odpočítať daň z tovarov a služieb podľa odseku 2, ktoré použije na dodávky tovarov a služieb, ktoré sú oslobodené od dane podľa § 28 až 42, s výnimkou poisťovacích služieb podľa § 37 a finančných služieb podľa § 39, ak sú poskytnuté zákazníkovi, ktorý nemá sídlo, miesto podnikania, prevádzkareň ani bydlisko na území Európskej únie, alebo ak sú tieto služby priamo spojené s vývozom tovaru mimo územia Európskej únie. Platiteľ, ktorý dodáva investičné zlato oslobodené od dane podľa § 67 ods. 3, a platiteľ, ktorý sprostredkováva dodanie investičného zlata oslobodené od dane podľa § 67 ods. 3, nemôže odpočítať daň z tovarov a služieb podľa odseku 2, ktoré použije na túto činnosť, s výnimkou dane z tovarov a služieb podľa § 67 ods. 5 a 6. (4) Ak platiteľ použije tovary a služby pre dodávky tovarov a služieb, pri ktorých môže odpočítať
relationships:
  Dan -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru
  Dan -> [ZAPLATI] -> Colny Organ
  Dovoz Tovaru -> [TYKA_SA] -> Tovar
  Dovoz Tovaru -> [MA_MIESTO] -> Tuzemsko

  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane
  Odpocitanie Dane -> [TYKA_SA] -> Dan
  Dan -> [VZTAHUJE_SA_NA] -> Tovary A Sluzby Podla Odseku 2
  Tovary A Sluzby Podla Odseku 2 -> [JE_PODLA] -> Odsek 2
  Tovary A Sluzby Podla Odseku 2 -> [VZTAHUJE_SA_NA] -> Dodavky Tovarov A Sluzieb
  Dodavky Tovarov A Sluzieb -> [JE_OSLOBODENE_OD] -> Dan
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 28
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 29
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 30
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 31
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 32
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 33
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 34
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 35
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 36
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 37
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 38
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 39
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 40
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 41
  Dodavky Tovarov A Sluzieb -> [JE_PODLA] -> Paragraf § 42
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 28
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 29
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 30
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 31
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 32
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 33
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 34
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 35
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 36
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 37
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 38
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 39
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 40
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 41
  Oslobodenie Od Dane Podla Paragrafov 28 Az 42 -> [JE_PODLA] -> Paragraf § 42

  Vynimka Z Neodpoctu Dane -> [VZTAHUJE_SA_NA] -> Poistovacie Sluzby Podla Paragrafu 37
  Vynimka Z Neodpoctu Dane -> [VZTAHUJE_SA_NA] -> Financne Sluzby Podla Paragrafu 39
  Poistovacie Sluzby Podla Paragrafu 37 -> [JE_PODLA] -> Paragraf § 37
  Financne Sluzby Podla Paragrafu 39 -> [JE_PODLA] -> Paragraf § 39
  Vynimka Z Neodpoctu Dane -> [MA_PODMIENKU] -> Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU
  Vynimka Z Neodpoctu Dane -> [MA_PODMIENKU] -> Priame Spojenie Sluzieb S Vyvozom Tovaru Mimo EU

  Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU -> [NEMA_NAROK_NA] -> Sidlo V EU
  Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU -> [NEMA_NAROK_NA] -> Miesto Podnikania V EU
  Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU -> [NEMA_NAROK_NA] -> Prevadzkarne V EU
  Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU -> [NEMA_NAROK_NA] -> Bydlisko V EU
  Priame Spojenie Sluzieb S Vyvozom Tovaru Mimo EU -> [TYKA_SA] -> Vyvoz Tovaru
  Vyvoz Tovaru -> [TYKA_SA] -> Tovar
  Vyvoz Tovaru -> [MA_MIESTO] -> Mimo Uzemia Europskej Unie

  Platitel -> [DODAVA] -> Investicne Zlato Oslobodene Od Dane Podla Paragrafu 67 Odsek 3
  Investicne Zlato Oslobodene Od Dane Podla Paragrafu 67 Odsek 3 -> [JE_OSLOBODENE_OD] -> Dan
  Investicne Zlato Oslobodene Od Dane Podla Paragrafu 67 Odsek 3 -> [JE_PODLA] -> Paragraf § 67 Odsek 3
  Platitel -> [VYKONAVA] -> Sprostredkovanie Dodania Investicneho Zlata Oslobodeneho Od Dane Podla Paragrafu 67 Odsek 3
  Sprostredkovanie Dodania Investicneho Zlata Oslobodeneho Od Dane Podla Paragrafu 67 Odsek 3 -> [TYKA_SA] -> Investicne Zlato Oslobodene Od Dane Podla Paragrafu 67 Odsek 3
  Sprostredkovanie Dodania Investicneho Zlata Oslobodeneho Od Dane Podla Paragrafu 67 Odsek 3 -> [JE_PODLA] -> Paragraf § 67 Odsek 3
  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane Pri Investicnom Zlate
  Odpocitanie Dane Pri Investicnom Zlate -> [TYKA_SA] -> Tovary A Sluzby Podla Odseku 2
  Odpocitanie Dane Pri Investicnom Zlate -> [MA_VYNIMKU] -> Vynimka Podla Paragrafu 67 Odsek 5 A 6
  Vynimka Podla Paragrafu 67 Odsek 5 A 6 -> [JE_PODLA] -> Paragraf § 67 Odsek 5
  Vynimka Podla Paragrafu 67 Odsek 5 A 6 -> [JE_PODLA] -> Paragraf § 67 Odsek 6

nodes:
  Dan: Dan
  Tovar: Tovar
  Sluzba: Sluzba
  Cinnost: Dovoz Tovaru
  Tuzemsko: Tuzemsko
  Urad: Colny Organ
  ZdanitelnaOsoba: Platitel
  Pravo: Odpocitanie Dane
  Dan: Tovary A Sluzby Podla Odseku 2
  Odsek: Odsek 2
  Cinnost: Dodavky Tovarov A Sluzieb
  OslobodenieOdDane: Oslobodenie Od Dane Podla Paragrafov 28 Az 42
  Paragraf: Paragraf § 28
  Paragraf: Paragraf § 29
  Paragraf: Paragraf § 30
  Paragraf: Paragraf § 31
  Paragraf: Paragraf § 32
  Paragraf: Paragraf § 33
  Paragraf: Paragraf § 34
  Paragraf: Paragraf § 35
  Paragraf: Paragraf § 36
  Paragraf: Paragraf § 38
  Paragraf: Paragraf § 40
  Paragraf: Paragraf § 41
  Paragraf: Paragraf § 42
  Sluzba: Poistovacie Sluzby Podla Paragrafu 37
  Sluzba: Financne Sluzby Podla Paragrafu 39
  Paragraf: Paragraf § 37
  Paragraf: Paragraf § 39
  OslobodenieOdDane: Vynimka Z Neodpoctu Dane
  Subjekt: Zakaznik Bez Sidla Miesta Podnikania Prevadzkarne Ani Bydliska V EU
  Sidlo: Sidlo V EU
  Lokacia: Miesto Podnikania V EU
  Prevazdkaren: Prevadzkarne V EU
  Adresa: Bydlisko V EU
  Podmienka: Priame Spojenie Sluzieb S Vyvozom Tovaru Mimo EU
  Cinnost: Vyvoz Tovaru
  Uzemie: Mimo Uzemia Europskej Unie
  InvesticnyMajetok: Investicne Zlato Oslobodene Od Dane Podla Paragrafu 67 Odsek 3
  Cinnost: Sprostredkovanie Dodania Investicneho Zlata Oslobodeneho Od Dane Podla Paragrafu 67 Odsek 3
  Odsek: Paragraf § 67 Odsek 3
  Pravo: Odpocitanie Dane Pri Investicnom Zlate
  OslobodenieOdDane: Vynimka Podla Paragrafu 67 Odsek 5 A 6
  Odsek: Paragraf § 67 Odsek 5
  Odsek: Paragraf § 67 Odsek 6


chunk: 270
page: 67
text: f) identifikačné číslo pre daň žiadateľa alebo jeho daňové registračné číslo pridelené v inom členskom štáte, g) údaje o bankovom účte žiadateľa vrátane medzinárodného bankového čísla účtu (IBAN) a medzinárodného kódu banky (BIC). (3) Žiadosť o vrátenie dane musí obsahovať okrem údajov podľa odseku 2 údaje z každej faktúry o dodaní tovaru alebo služby a z každého dovozného dokladu, z ktorých žiadateľ žiada vrátenie dane, a to: a) priezvisko a meno alebo názov dodávateľa a adresu jeho sídla, miesta podnikania, prevádzkarne, bydliska alebo miesta, kde sa obvykle zdržiava,
relationships:
  Ziadatel -> [PODAVA] -> Ziadost O Vratenie Dane
  Ziadost O Vratenie Dane -> [TYKA_SA] -> Vratenie Dane
  Vratenie Dane -> [TYKA_SA] -> Dan

  Ziadost O Vratenie Dane -> [MA_OBSAH] -> Identifikacne Cislo Pre Dan
  Ziadost O Vratenie Dane -> [MA_OBSAH] -> Danove Registracne Cislo
  Ziadatel -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan
  Ziadatel -> [MA_IDENTIFIKACNE_CISLO] -> Danove Registracne Cislo
  Danove Registracne Cislo -> [MA_MIESTO] -> Iny Clensky Stat

  Ziadost O Vratenie Dane -> [MA_OBSAH] -> Bankovy Ucet Ziadatela
  Ziadatel -> [MA_DOKLAD] -> Bankovy Ucet Ziadatela
  Bankovy Ucet Ziadatela -> [MA_IDENTIFIKACNE_CISLO] -> Medzinarodne Bankove Cislo Uctu IBAN
  Bankovy Ucet Ziadatela -> [MA_IDENTIFIKACNE_CISLO] -> Medzinarodny Kod Banky BIC

  Ziadost O Vratenie Dane -> [MA_OBSAH] -> Udaje Z Faktury
  Ziadost O Vratenie Dane -> [MA_OBSAH] -> Udaje Z Dovozneho Dokladu
  Udaje Z Faktury -> [VYCHADZA_Z] -> Faktura
  Udaje Z Dovozneho Dokladu -> [VYCHADZA_Z] -> Dovozny Doklad
  Faktura -> [JE_PREDMETOM] -> Dodanie Tovaru Alebo Sluzby
  Dovozny Doklad -> [TYKA_SA] -> Dovoz Tovaru

  Udaje Z Faktury -> [MA_OBSAH] -> Dodavatel
  Udaje Z Faktury -> [MA_OBSAH] -> Meno Alebo Nazov Dodavatela
  Dodavatel -> [MA_NAZOV] -> Meno Alebo Nazov Dodavatela
  Dodavatel -> [MA_SIDLO] -> Sidlo Dodavatela
  Dodavatel -> [MA_MIESTO_PODNIKANIA] -> Miesto Podnikania Dodavatela
  Dodavatel -> [MA_PREVADZKAREN] -> Prevadzkaren Dodavatela
  Dodavatel -> [MA_BYDLISKO] -> Bydlisko Dodavatela
  Dodavatel -> [MA_MIESTO] -> Miesto Kde Sa Obvykle Zdrziava Dodavatel

nodes:
  Ziadost: Ziadost O Vratenie Dane
  Subjekt: Ziadatel
  Dan: Dan
  Cinnost: Vratenie Dane
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  IdentifikacneCislo: Danove Registracne Cislo
  ClenskyStat: Iny Clensky Stat
  BankovyUcet: Bankovy Ucet Ziadatela
  IdentifikacneCislo: Medzinarodne Bankove Cislo Uctu IBAN
  IdentifikacneCislo: Medzinarodny Kod Banky BIC
  Zaznam: Udaje Z Faktury
  Zaznam: Udaje Z Dovozneho Dokladu
  Doklad: Faktura
  Doklad: Dovozny Doklad
  Cinnost: Dodanie Tovaru Alebo Sluzby
  Cinnost: Dovoz Tovaru
  Tovar: Tovar
  Subjekt: Dodavatel
  Hodnota: Meno Alebo Nazov Dodavatela
  Sidlo: Sidlo Dodavatela
  Lokacia: Miesto Podnikania Dodavatela
  Prevazdkaren: Prevadzkaren Dodavatela
  Adresa: Bydlisko Dodavatela
  Lokacia: Miesto Kde Sa Obvykle Zdrziava Dodavatel


chunk: 28
page: 7
text: Strana 8 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. tomuto rozhodnutiu nemožno podať odvolanie. Začatie konania o registrácii skupiny z úradnej moci bráni registrácii členov skupiny podľa § 4b. (3) Daňový úrad Banská Bystrica vyzve spoločného zástupcu podľa odseku 2, aby sa vyjadril k dôvodom na registráciu skupiny z úradnej moci v určenej lehote, ktorá nesmie byť kratšia ako 15 dní odo dňa doručenia výzvy. (4) Ak na základe vyjadrenia podľa odseku 3 nedôjde k vyvráteniu dôvodov na registráciu skupiny z úradnej moci alebo ak spoločný zástupca výzve podľa odseku 3 nevyhovie, Daňový úrad Banská Bystrica rozhodne z úradnej moci o registrácii tých členov skupiny, pri ktorých sú splnené dôvody na registráciu skupiny z úradnej moci, pridelí skupine identifikačné číslo pre daň a určí zástupcu skupiny. Proti tomuto rozhodnutiu môže spoločný zástupca podľa odseku 2 podať do ôsmich dní odo dňa jeho doručenia odvolanie, ktoré má odkladný účinok.
relationships:
  Zakon 222/2004 Z. z. -> [UPRAVUJE] -> Konanie O Registracii Skupiny Z Uradnej Moci

  Zacatie Konania O Registracii Skupiny Z Uradnej Moci -> [TYKA_SA] -> Registracia Skupiny Z Uradnej Moci
  Registracia Clenov Skupiny Podla Paragrafu 4b -> [JE_PODLA] -> Paragraf § 4b
  Zacatie Konania O Registracii Skupiny Z Uradnej Moci -> [NEVZTAHUJE_SA_NA] -> Registracia Clenov Skupiny Podla Paragrafu 4b

  Danovy Urad Banska Bystrica -> [DORUCUJE] -> Vyzva
  Vyzva -> [VZTAHUJE_SA_NA] -> Spolocny Zastupca
  Vyzva -> [MA_OBSAH] -> Vyjadrenie K Dovodom Na Registraciu Skupiny Z Uradnej Moci
  Vyjadrenie K Dovodom Na Registraciu Skupiny Z Uradnej Moci -> [TYKA_SA] -> Dovody Na Registraciu Skupiny Z Uradnej Moci
  Vyzva -> [MA_LEHOTU] -> Lehota Najmenej 15 Dni
  Lehota Najmenej 15 Dni -> [VYCHADZA_Z] -> Den Dorucenia Vyzvy

  Danovy Urad Banska Bystrica -> [ROZHODUJE_O] -> Registracia Clenov Skupiny Z Uradnej Moci
  Registracia Clenov Skupiny Z Uradnej Moci -> [TYKA_SA] -> Clenovia Skupiny
  Registracia Clenov Skupiny Z Uradnej Moci -> [MA_DOVOD] -> Dovody Na Registraciu Skupiny Z Uradnej Moci
  Clenovia Skupiny -> [SPLNA_PODMIENKY] -> Dovody Na Registraciu Skupiny Z Uradnej Moci
  Rozhodnutie O Registracii Skupiny Z Uradnej Moci -> [VYCHADZA_Z] -> Nevyvratenie Dovodov Na Registraciu
  Rozhodnutie O Registracii Skupiny Z Uradnej Moci -> [VYCHADZA_Z] -> Nevyhovenie Vyzve

  Danovy Urad Banska Bystrica -> [PRIDELUJE] -> Identifikacne Cislo Pre Dan
  Skupina -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan
  Danovy Urad Banska Bystrica -> [URCUJE] -> Zastupca Skupiny
  Skupina -> [MA_ZASTUPCU] -> Zastupca Skupiny

  Spolocny Zastupca -> [PODAVA] -> Odvolanie
  Odvolanie -> [TYKA_SA] -> Rozhodnutie O Registracii Skupiny Z Uradnej Moci
  Odvolanie -> [MA_LEHOTU] -> Lehota Do Osmich Dni Od Dorucenia Rozhodnutia
  Odvolanie -> [MA_UCINOK] -> Odkladny Ucinok

nodes:
  PravnyPredpis: Zakon 222/2004 Z. z.
  Konanie: Konanie O Registracii Skupiny Z Uradnej Moci
  Konanie: Zacatie Konania O Registracii Skupiny Z Uradnej Moci
  Registracia: Registracia Skupiny Z Uradnej Moci
  Registracia: Registracia Clenov Skupiny Podla Paragrafu 4b
  Registracia: Registracia Clenov Skupiny Z Uradnej Moci
  Paragraf: Paragraf § 4b
  Urad: Danovy Urad Banska Bystrica
  Vyzva: Vyzva
  Zastupca: Spolocny Zastupca
  Zastupca: Zastupca Skupiny
  Zaznam: Vyjadrenie K Dovodom Na Registraciu Skupiny Z Uradnej Moci
  Dovod: Dovody Na Registraciu Skupiny Z Uradnej Moci
  Lehota: Lehota Najmenej 15 Dni
  Datum: Den Dorucenia Vyzvy
  ClenSkupiny: Clenovia Skupiny
  Rozhodnutie: Rozhodnutie O Registracii Skupiny Z Uradnej Moci
  Dovod: Nevyvratenie Dovodov Na Registraciu
  Dovod: Nevyhovenie Vyzve
  Skupina: Skupina
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  Doklad: Odvolanie
  Lehota: Lehota Do Osmich Dni Od Dorucenia Rozhodnutia
  Status: Odkladny Ucinok


chunk: 335
page: 83
text: (5) Platiteľ, ktorý dodáva investičné zlato oslobodené od dane, môže odpočítať daň a) ním uplatnenú pri dodaní investičného zlata iným platiteľom, ktorý využil možnosť zdanenia podľa odseku 4, b) voči nemu uplatnenú pri dodaní iného ako investičného zlata iným platiteľom, ktoré je ním alebo na jeho účet následne pretvorené na investičné zlato, c) ním uplatnenú pri nadobudnutí iného ako investičného zlata v tuzemsku z iného členského štátu, ktoré je ním alebo na jeho účet následne pretvorené na investičné zlato, d) zaplatenú colnému úradu alebo ním uplatnenú pri dovoze iného ako investičného zlata, ktoré je ním alebo na jeho účet následne pretvorené na investičné zlato, e) voči nemu uplatnenú pri dodaní služieb iným platiteľom, ktoré spočívali v zmene podoby, hmotnosti alebo rýdzosti zlata vrátane investičného zlata.
relationships:
  Platitel -> [DODAVA] -> Investicne Zlato
  Investicne Zlato -> [JE_OSLOBODENE_OD] -> Dan
  Platitel -> [MA_PRAVO] -> Odpocitanie Dane
  Odpocitanie Dane -> [TYKA_SA] -> Dan

  Odpocitanie Dane -> [PLATI_PRE] -> Dan Uplatnena Pri Dodani Investicneho Zlata
  Dan Uplatnena Pri Dodani Investicneho Zlata -> [VZNIKA_PRI] -> Dodanie Investicneho Zlata Inym Platitelom
  Dodanie Investicneho Zlata Inym Platitelom -> [JE_PREDMETOM] -> Investicne Zlato
  Iny Platitel -> [VYKONAVA] -> Vyuzenie Moznosti Zdanenia Podla Odseku 4
  Vyuzenie Moznosti Zdanenia Podla Odseku 4 -> [JE_PODLA] -> Odsek 4

  Odpocitanie Dane -> [PLATI_PRE] -> Dan Uplatnena Pri Dodani Ineho Ako Investicneho Zlata
  Dan Uplatnena Pri Dodani Ineho Ako Investicneho Zlata -> [VZNIKA_PRI] -> Dodanie Ineho Ako Investicneho Zlata Inym Platitelom
  Dodanie Ineho Ako Investicneho Zlata Inym Platitelom -> [JE_PREDMETOM] -> Ine Ako Investicne Zlato
  Pretvorenie Na Investicne Zlato -> [TYKA_SA] -> Ine Ako Investicne Zlato
  Pretvorenie Na Investicne Zlato -> [VYMEDZUJE] -> Investicne Zlato

  Odpocitanie Dane -> [PLATI_PRE] -> Dan Uplatnena Pri Nadobudnuti Ineho Ako Investicneho Zlata
  Dan Uplatnena Pri Nadobudnuti Ineho Ako Investicneho Zlata -> [VZNIKA_PRI] -> Nadobudnutie Ineho Ako Investicneho Zlata V Tuzemsku Z Ineho Clenskeho Statu
  Nadobudnutie Ineho Ako Investicneho Zlata V Tuzemsku Z Ineho Clenskeho Statu -> [JE_PREDMETOM] -> Ine Ako Investicne Zlato
  Nadobudnutie Ineho Ako Investicneho Zlata V Tuzemsku Z Ineho Clenskeho Statu -> [MA_MIESTO] -> Tuzemsko
  Nadobudnutie Ineho Ako Investicneho Zlata V Tuzemsku Z Ineho Clenskeho Statu -> [VYCHADZA_Z] -> Iny Clensky Stat

  Odpocitanie Dane -> [PLATI_PRE] -> Dan Zaplatena Alebo Uplatnena Pri Dovoze Ineho Ako Investicneho Zlata
  Dan Zaplatena Alebo Uplatnena Pri Dovoze Ineho Ako Investicneho Zlata -> [VZNIKA_PRI] -> Dovoz Ineho Ako Investicneho Zlata
  Dovoz Ineho Ako Investicneho Zlata -> [JE_PREDMETOM] -> Ine Ako Investicne Zlato
  Platitel -> [ZAPLATI] -> Dan Zaplatena Alebo Uplatnena Pri Dovoze Ineho Ako Investicneho Zlata
  Dan Zaplatena Alebo Uplatnena Pri Dovoze Ineho Ako Investicneho Zlata -> [TYKA_SA] -> Colny Urad

  Odpocitanie Dane -> [PLATI_PRE] -> Dan Uplatnena Pri Dodani Sluzieb
  Dan Uplatnena Pri Dodani Sluzieb -> [VZNIKA_PRI] -> Dodanie Sluzieb Inym Platitelom
  Dodanie Sluzieb Inym Platitelom -> [JE_PREDMETOM] -> Sluzby Zmeny Podoby Hmotnosti Alebo Rydzosti Zlata
  Sluzby Zmeny Podoby Hmotnosti Alebo Rydzosti Zlata -> [TYKA_SA] -> Zmena Podoby Hmotnosti Alebo Rydzosti Zlata
  Zmena Podoby Hmotnosti Alebo Rydzosti Zlata -> [TYKA_SA] -> Zlato
  Zmena Podoby Hmotnosti Alebo Rydzosti Zlata -> [TYKA_SA] -> Investicne Zlato

nodes:
  ZdanitelnaOsoba: Platitel
  ZdanitelnaOsoba: Iny Platitel
  Dan: Dan
  Pravo: Odpocitanie Dane
  InvesticnyMajetok: Investicne Zlato
  Majetok: Ine Ako Investicne Zlato
  Majetok: Zlato
  Cinnost: Dodanie Investicneho Zlata Inym Platitelom
  Cinnost: Vyuzenie Moznosti Zdanenia Podla Odseku 4
  Odsek: Odsek 4
  Dan: Dan Uplatnena Pri Dodani Investicneho Zlata
  Cinnost: Dodanie Ineho Ako Investicneho Zlata Inym Platitelom
  Dan: Dan Uplatnena Pri Dodani Ineho Ako Investicneho Zlata
  Cinnost: Pretvorenie Na Investicne Zlato
  Cinnost: Nadobudnutie Ineho Ako Investicneho Zlata V Tuzemsku Z Ineho Clenskeho Statu
  Dan: Dan Uplatnena Pri Nadobudnuti Ineho Ako Investicneho Zlata
  Tuzemsko: Tuzemsko
  ClenskyStat: Iny Clensky Stat
  Cinnost: Dovoz Ineho Ako Investicneho Zlata
  Dan: Dan Zaplatena Alebo Uplatnena Pri Dovoze Ineho Ako Investicneho Zlata
  Urad: Colny Urad
  Cinnost: Dodanie Sluzieb Inym Platitelom
  Sluzba: Sluzby Zmeny Podoby Hmotnosti Alebo Rydzosti Zlata
  Cinnost: Zmena Podoby Hmotnosti Alebo Rydzosti Zlata

chunk: 337
page: 84
text: 3. určité domáce dodanie tovaru, ktorým sa rozumie dodanie tovaru na území Európskej únie zdaniteľnou osobou neusadenou na území Európskej únie osobe inej ako zdaniteľnej osobe prostredníctvom zdaniteľnej osoby, ktorá uľahčuje dodanie tovaru podľa § 8 ods. 7 prvej vety, ak sa odoslanie alebo preprava tovaru začína a skončí v tom istom členskom štáte, c) § 68c sa vzťahuje na predaj tovaru na diaľku dovážaného z územia tretích štátov v zásielke, ak vlastná hodnota zásielky nepresahuje 150 eur a tovar nie je predmetom spotrebnej dane. (2) Na účely uplatňovania osobitných úprav podľa odseku 1 sa daňovým priznaním rozumie podanie, ktoré obsahuje údaje podľa osobitného predpisu28aa) potrebné na určenie výšky dane, ktorá sa stala splatnou v každom členskom štáte. (3) Zdaniteľná osoba, ktorá sa rozhodne pre uplatňovanie osobitnej úpravy podľa § 68a, § 68b alebo § 68c, je povinná doručovať písomnosti týkajúce sa osobitnej úpravy daňovému úradu
relationships:
  Urcite Domace Dodanie Tovaru -> [ROZUMIE_SA] -> Dodanie Tovaru Na Uzemi Europskej Unie
  Dodanie Tovaru Na Uzemi Europskej Unie -> [MA_MIESTO] -> Uzemie Europskej Unie
  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [DODAVA] -> Tovar
  Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [NACHADZA_SA_V] -> Uzemie Europskej Unie
  Dodanie Tovaru Na Uzemi Europskej Unie -> [TYKA_SA] -> Osoba Ina Ako Zdanitelna Osoba
  Dodanie Tovaru Na Uzemi Europskej Unie -> [VYCHADZA_Z] -> Ulahcenie Dodania Tovaru
  Zdanitelna Osoba Ulahcujuca Dodanie Tovaru -> [VYKONAVA] -> Ulahcenie Dodania Tovaru
  Ulahcenie Dodania Tovaru -> [JE_PODLA] -> Paragraf § 8 Odsek 7
  Odoslanie Alebo Preprava Tovaru -> [TYKA_SA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [MA_PODMIENKU] -> Zacatie A Skoncenie V Tom Istom Clenskom State
  Zacatie A Skoncenie V Tom Istom Clenskom State -> [VZTAHUJE_SA_NA] -> Ten Isty Clensky Stat

  Paragraf § 68c -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov V Zasielke
  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov V Zasielke -> [TYKA_SA] -> Tovar
  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov V Zasielke -> [VYCHADZA_Z] -> Uzemie Tretich Statov
  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov V Zasielke -> [TYKA_SA] -> Zasielka
  Zasielka -> [MA_HODNOTU] -> Vlastna Hodnota Zasielky
  Vlastna Hodnota Zasielky -> [MA_PODMIENKU] -> Limit 150 Eur
  Limit 150 Eur -> [MA_SUMU] -> 150 Eur
  Tovar -> [NEVZTAHUJE_SA_NA] -> Spotrebna Dan

  Danove Priznanie -> [ROZUMIE_SA] -> Podanie
  Podanie -> [OBSAHUJE] -> Udaje Podla Osobitneho Predpisu
  Udaje Podla Osobitneho Predpisu -> [JE_PODLA] -> Osobitny Predpis 28aa
  Udaje Podla Osobitneho Predpisu -> [MA_UCEL] -> Urcenie Vysky Dane
  Urcenie Vysky Dane -> [TYKA_SA] -> Dan
  Dan -> [MA_STATUS] -> Splatna Dan
  Splatna Dan -> [NACHADZA_SA_V] -> Clensky Stat

  Osobitne Upravy -> [JE_PODLA] -> Odsek 1
  Osobitna Uprava Podla Paragrafu 68a -> [JE_PODLA] -> Paragraf § 68a
  Osobitna Uprava Podla Paragrafu 68b -> [JE_PODLA] -> Paragraf § 68b
  Osobitna Uprava Podla Paragrafu 68c -> [JE_PODLA] -> Paragraf § 68c
  Zdanitelna Osoba -> [MA_POVINNOST] -> Dorucovanie Pisomnosti Tykajucich Sa Osobitnej Upravy
  Dorucovanie Pisomnosti Tykajucich Sa Osobitnej Upravy -> [TYKA_SA] -> Pisomnosti Tykajuce Sa Osobitnej Upravy
  Dorucovanie Pisomnosti Tykajucich Sa Osobitnej Upravy -> [TYKA_SA] -> Danovy Urad

nodes:
  Cinnost: Urcite Domace Dodanie Tovaru
  Cinnost: Dodanie Tovaru Na Uzemi Europskej Unie
  Uzemie: Uzemie Europskej Unie
  ZdanitelnaOsoba: Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
  ZdanitelnaOsoba: Zdanitelna Osoba Ulahcujuca Dodanie Tovaru
  ZdanitelnaOsoba: Zdanitelna Osoba
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Tovar: Tovar
  Cinnost: Ulahcenie Dodania Tovaru
  Odsek: Paragraf § 8 Odsek 7
  Cinnost: Odoslanie Alebo Preprava Tovaru
  Podmienka: Zacatie A Skoncenie V Tom Istom Clenskom State
  ClenskyStat: Ten Isty Clensky Stat
  Paragraf: Paragraf § 68a
  Paragraf: Paragraf § 68b
  Paragraf: Paragraf § 68c
  Cinnost: Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov V Zasielke
  TretiStat: Uzemie Tretich Statov
  Zasielka: Zasielka
  Hodnota: Vlastna Hodnota Zasielky
  Limit: Limit 150 Eur
  Euro: 150 Eur
  SpotrebnaDan: Spotrebna Dan
  DanovePriznanie: Danove Priznanie
  Doklad: Podanie
  Zaznam: Udaje Podla Osobitneho Predpisu
  PravnyPredpis: Osobitny Predpis 28aa
  Vypocet: Urcenie Vysky Dane
  Dan: Dan
  Status: Splatna Dan
  ClenskyStat: Clensky Stat
  Cinnost: Osobitne Upravy
  Odsek: Odsek 1
  Cinnost: Osobitna Uprava Podla Paragrafu 68a
  Cinnost: Osobitna Uprava Podla Paragrafu 68b
  Cinnost: Osobitna Uprava Podla Paragrafu 68c
  Povinnost: Dorucovanie Pisomnosti Tykajucich Sa Osobitnej Upravy
  Doklad: Pisomnosti Tykajuce Sa Osobitnej Upravy
  Urad: Danovy Urad


chunk: 351
page: 87
text: písm. b), b) možno predpokladať, že táto zdaniteľná osoba prestala vykonávať činnosť podľa § 68 ods. 1 písm. b), c) táto zdaniteľná osoba prestala spĺňať podmienky na uplatňovanie osobitnej úpravy alebo d) táto zdaniteľná osoba opakovane porušuje povinnosti týkajúce sa uplatňovania osobitnej úpravy. (11) Daňový úrad o zrušení povolenia podľa odseku 10 vydá rozhodnutie; proti tomuto rozhodnutiu môže zdaniteľná osoba uvedená v odseku 2 podať odvolanie, ktoré nemá odkladný
relationships:
  Paragraf § 68 -> [OBSAHUJE] -> Odsek 1
  Paragraf § 68 -> [OBSAHUJE] -> Odsek 10
  Paragraf § 68 -> [OBSAHUJE] -> Odsek 11
  Odsek 1 -> [MA_PISMENO] -> Pismeno b)

  Zrusenie Povolenia -> [JE_PODLA] -> Odsek 10
  Zrusenie Povolenia -> [MA_DOVOD] -> Predpoklad Ukoncenia Cinnosti Podla Paragrafu 68 Odsek 1 Pismeno B
  Zrusenie Povolenia -> [MA_DOVOD] -> Nesplnenie Podmienok Na Uplatnovanie Osobitnej Upravy
  Zrusenie Povolenia -> [MA_DOVOD] -> Opakovane Porusovanie Povinnosti Tykajucich Sa Osobitnej Upravy

  Predpoklad Ukoncenia Cinnosti Podla Paragrafu 68 Odsek 1 Pismeno B -> [TYKA_SA] -> Cinnost Podla Paragrafu 68 Odsek 1 Pismeno B
  Cinnost Podla Paragrafu 68 Odsek 1 Pismeno B -> [JE_PODLA] -> Pismeno b)
  Zdanitelna Osoba -> [NEVZTAHUJE_SA_NA] -> Cinnost Podla Paragrafu 68 Odsek 1 Pismeno B

  Zdanitelna Osoba -> [NESPLNA_PODMIENKY] -> Podmienky Na Uplatnovanie Osobitnej Upravy
  Podmienky Na Uplatnovanie Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Osobitna Uprava

  Opakovane Porusovanie Povinnosti Tykajucich Sa Osobitnej Upravy -> [TYKA_SA] -> Povinnosti Tykajuce Sa Uplatnovania Osobitnej Upravy
  Povinnosti Tykajuce Sa Uplatnovania Osobitnej Upravy -> [TYKA_SA] -> Osobitna Uprava

  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Zruseni Povolenia
  Rozhodnutie O Zruseni Povolenia -> [TYKA_SA] -> Zrusenie Povolenia
  Rozhodnutie O Zruseni Povolenia -> [JE_PODLA] -> Odsek 11

  Zdanitelna Osoba -> [PODAVA] -> Odvolanie
  Odvolanie -> [TYKA_SA] -> Rozhodnutie O Zruseni Povolenia
  Odvolanie -> [MA_ODKLADNY_UCINOK] -> Bez Odkladneho Ucinku
  Zdanitelna Osoba -> [JE_PODLA] -> Odsek 2

nodes:
  Urad: Danovy Urad
  ZdanitelnaOsoba: Zdanitelna Osoba
  Rozhodnutie: Rozhodnutie O Zruseni Povolenia
  Rozhodnutie: Zrusenie Povolenia
  Zaznam: Odvolanie
  Status: Bez Odkladneho Ucinku
  Paragraf: Paragraf § 68
  Odsek: Odsek 1
  Odsek: Odsek 2
  Odsek: Odsek 10
  Odsek: Odsek 11
  Pismeno: Pismeno b)
  Dovod: Predpoklad Ukoncenia Cinnosti Podla Paragrafu 68 Odsek 1 Pismeno B
  Dovod: Nesplnenie Podmienok Na Uplatnovanie Osobitnej Upravy
  Dovod: Opakovane Porusovanie Povinnosti Tykajucich Sa Osobitnej Upravy
  Cinnost: Cinnost Podla Paragrafu 68 Odsek 1 Pismeno B
  Podmienka: Podmienky Na Uplatnovanie Osobitnej Upravy
  Povinnost: Povinnosti Tykajuce Sa Uplatnovania Osobitnej Upravy
  Status: Osobitna Uprava


chunk: 394
page: 97
text: osoby sídlo, miesto podnikania, bydlisko alebo v ktorom sa obvykle zdržiava. (2) Dodanie tovaru a služby uskutočnené v tuzemsku malým podnikom zahraničnej osoby je oslobodené od dane, ak je malý podnik zahraničnej osoby v členskom štáte usadenia identifikovaný pre túto osobitnú úpravu, a to odo dňa, kedy mu bolo príslušným orgánom
relationships:
  Osoba -> [MA_SIDLO] -> Sidlo
  Osoba -> [MA_MIESTO_PODNIKANIA] -> Miesto Podnikania
  Osoba -> [MA_BYDLISKO] -> Bydlisko
  Osoba -> [MA_MIESTO] -> Miesto Kde Sa Osoba Obvykle Zdrziava

  Maly Podnik Zahranicnej Osoby -> [JE_SUCASTOU] -> Zahranicna Osoba
  Maly Podnik Zahranicnej Osoby -> [NACHADZA_SA_V] -> Clensky Stat Usadenia
  Maly Podnik Zahranicnej Osoby -> [MA_STATUS] -> Identifikacia Pre Osobitnu Upravu
  Identifikacia Pre Osobitnu Upravu -> [VZTAHUJE_SA_NA] -> Osobitna Uprava
  Identifikacia Pre Osobitnu Upravu -> [MA_MIESTO] -> Clensky Stat Usadenia

  Maly Podnik Zahranicnej Osoby -> [DODAVA] -> Tovar
  Maly Podnik Zahranicnej Osoby -> [POSKYTUJE] -> Sluzba
  Dodanie Tovaru A Sluzby -> [USKUTOCNUJE] -> Maly Podnik Zahranicnej Osoby
  Dodanie Tovaru A Sluzby -> [TYKA_SA] -> Tovar
  Dodanie Tovaru A Sluzby -> [TYKA_SA] -> Sluzba
  Dodanie Tovaru A Sluzby -> [MA_MIESTO] -> Tuzemsko
  Dodanie Tovaru A Sluzby -> [JE_OSLOBODENE_OD] -> Dan
  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru A Sluzby
  Oslobodenie Od Dane -> [MA_PODMIENKU] -> Identifikacia Pre Osobitnu Upravu

nodes:
  Osoba: Osoba
  Osoba: Zahranicna Osoba
  Sidlo: Sidlo
  Lokacia: Miesto Podnikania
  Adresa: Bydlisko
  Lokacia: Miesto Kde Sa Osoba Obvykle Zdrziava
  Podnik: Maly Podnik Zahranicnej Osoby
  ClenskyStat: Clensky Stat Usadenia
  Status: Identifikacia Pre Osobitnu Upravu
  Status: Osobitna Uprava
  Cinnost: Dodanie Tovaru A Sluzby
  Tovar: Tovar
  Sluzba: Sluzba
  Tuzemsko: Tuzemsko
  OslobodenieOdDane: Oslobodenie Od Dane
  Dan: Dan


chunk: 413
page: 101
text: v ktorom sa malý podnik tuzemskej osoby rozhodol neuplatňovať osobitnú úpravu, alebo vyznačenie skutočnosti, že sa malý podnik tuzemskej osoby rozhodol skončiť uplatňovanie osobitnej úpravy. Malý podnik tuzemskej osoby nesmie uplatňovať osobitnú úpravu v členskom štáte označenom v oznámení podľa prvej vety alebo v žiadnom členskom štáte, ak v tomto oznámení vyznačil skutočnosť, že sa rozhodol skončiť uplatňovanie osobitnej úpravy, od prvého dňa nasledujúceho kalendárneho štvrťroka po doručení oznámenia alebo od prvého dňa druhého kalendárneho mesiaca nasledujúceho kalendárneho štvrťroka, ak bolo oznámenie doručené počas
relationships:
  Oznamenie -> [MA_OBSAH] -> Clensky Stat Oznaceny V Oznameni
  Oznamenie -> [MA_OBSAH] -> Skutocnost Skoncenia Uplatnovania Osobitnej Upravy

  Maly Podnik Tuzemskej Osoby -> [VYKONAVA] -> Rozhodnutie Neuplatnovat Osobitnu Upravu
  Rozhodnutie Neuplatnovat Osobitnu Upravu -> [TYKA_SA] -> Osobitna Uprava
  Rozhodnutie Neuplatnovat Osobitnu Upravu -> [MA_MIESTO] -> Clensky Stat Oznaceny V Oznameni

  Maly Podnik Tuzemskej Osoby -> [VYKONAVA] -> Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy
  Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy -> [TYKA_SA] -> Osobitna Uprava
  Skutocnost Skoncenia Uplatnovania Osobitnej Upravy -> [TYKA_SA] -> Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy

  Maly Podnik Tuzemskej Osoby -> [NEMA_NAROK_NA] -> Uplatnovanie Osobitnej Upravy
  Uplatnovanie Osobitnej Upravy -> [TYKA_SA] -> Osobitna Uprava
  Uplatnovanie Osobitnej Upravy -> [MA_MIESTO] -> Clensky Stat Oznaceny V Oznameni
  Uplatnovanie Osobitnej Upravy -> [MA_VYNIMKU] -> Ziadny Clensky Stat

  Zakaz Uplatnovania Osobitnej Upravy -> [VZTAHUJE_SA_NA] -> Uplatnovanie Osobitnej Upravy
  Zakaz Uplatnovania Osobitnej Upravy -> [PLATI_OD] -> Prvy Den Nasledujuceho Kalendarneho Stvrtroka Po Doruceni Oznamenia
  Zakaz Uplatnovania Osobitnej Upravy -> [PLATI_OD] -> Prvy Den Druheho Kalendarneho Mesiaca Nasledujuceho Kalendarneho Stvrtroka
  Zakaz Uplatnovania Osobitnej Upravy -> [VYCHADZA_Z] -> Dorucenie Oznamenia

nodes:
  Podnik: Maly Podnik Tuzemskej Osoby
  Oznamenie: Oznamenie
  ClenskyStat: Clensky Stat Oznaceny V Oznameni
  ClenskyStat: Ziadny Clensky Stat
  Cinnost: Osobitna Uprava
  Cinnost: Uplatnovanie Osobitnej Upravy
  Cinnost: Rozhodnutie Neuplatnovat Osobitnu Upravu
  Cinnost: Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy
  Status: Skutocnost Skoncenia Uplatnovania Osobitnej Upravy
  Status: Zakaz Uplatnovania Osobitnej Upravy
  Datum: Prvy Den Nasledujuceho Kalendarneho Stvrtroka Po Doruceni Oznamenia
  Datum: Prvy Den Druheho Kalendarneho Mesiaca Nasledujuceho Kalendarneho Stvrtroka
  Cinnost: Dorucenie Oznamenia


chunk: 439
page: 108
text: c) v režime call-off stock, v rozsahu podľa osobitného predpisu,28dc) h) tovare, ktorý bol pre neho odoslaný alebo prepravený do tuzemska z iného členského štátu v režime call-off stock, v rozsahu podľa osobitného predpisu, i) oprave základu dane podľa § 25 a 25a, j) oprave odpočítanej dane podľa § 53, § 53a, § 53b ods. 1 a 2 a oprave opravenej odpočítanej dane podľa § 53b ods. 4. (3) Platiteľ vedie osobitne záznamy potrebné na identifikáciu hnuteľného hmotného majetku z iného členského štátu, na ktorom vykoná práce alebo ho oceňuje, ak tieto práce vykonáva pre osobu identifikovanú pre daň v inom členskom štáte. (4) Osoba registrovaná pre daň podľa § 7 alebo 7a je povinná viesť záznamy o  a) nadobudnutých tovaroch z iného členského štátu, b) dodaných službách s miestom dodania v inom členskom štáte podľa § 15 ods. 1, c) prijatých službách, pri ktorých je povinná platiť daň podľa § 69 ods. 3, d) tovare, ktorý bol pre ňu odoslaný alebo prepravený do tuzemska z iného členského štátu
relationships:
  Platitel -> [VIES_ZAZNAMY_O] -> Tovar V Rezime Call-Off Stock
  Tovar V Rezime Call-Off Stock -> [JE_PODLA] -> Osobitny Predpis
  Tovar V Rezime Call-Off Stock -> [MA_STATUS] -> Rezim Call-Off Stock

  Platitel -> [VIES_ZAZNAMY_O] -> Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock
  Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock -> [TYKA_SA] -> Tovar
  Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock -> [MA_MIESTO] -> Tuzemsko
  Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock -> [VYCHADZA_Z] -> Iny Clensky Stat
  Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock -> [MA_STATUS] -> Rezim Call-Off Stock
  Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock -> [JE_PODLA] -> Osobitny Predpis

  Platitel -> [VIES_ZAZNAMY_O] -> Oprava Zakladu Dane
  Oprava Zakladu Dane -> [JE_PODLA] -> Paragraf § 25
  Oprava Zakladu Dane -> [JE_PODLA] -> Paragraf § 25a

  Platitel -> [VIES_ZAZNAMY_O] -> Oprava Odpocitanej Dane
  Oprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 53
  Oprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 53a
  Oprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 53b Odsek 1
  Oprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 53b Odsek 2

  Platitel -> [VIES_ZAZNAMY_O] -> Oprava Opravenej Odpocitanej Dane
  Oprava Opravenej Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 53b Odsek 4

  Platitel -> [VIES_ZAZNAMY_O] -> Hnutelny Hmotny Majetok Z Ineho Clenskeho Statu
  Hnutelny Hmotny Majetok Z Ineho Clenskeho Statu -> [VYCHADZA_Z] -> Iny Clensky Stat
  Zaznamy O Hnutelnom Hmotnom Majetku -> [MA_UCEL] -> Identifikacia Hnutelneho Hmotneho Majetku
  Platitel -> [VYKONAVA] -> Prace Na Hnutelnom Hmotnom Majetku
  Platitel -> [VYKONAVA] -> Ocenovanie Hnutelneho Hmotneho Majetku
  Prace Na Hnutelnom Hmotnom Majetku -> [TYKA_SA] -> Hnutelny Hmotny Majetok Z Ineho Clenskeho Statu
  Ocenovanie Hnutelneho Hmotneho Majetku -> [TYKA_SA] -> Hnutelny Hmotny Majetok Z Ineho Clenskeho Statu
  Prace Na Hnutelnom Hmotnom Majetku -> [PLATI_PRE] -> Osoba Identifikovana Pre Dan V Inom Clenskom State
  Ocenovanie Hnutelneho Hmotneho Majetku -> [PLATI_PRE] -> Osoba Identifikovana Pre Dan V Inom Clenskom State

  Osoba Registrovana Pre Dan Podla Paragrafu 7 Alebo 7a -> [JE_PODLA] -> Paragraf § 7
  Osoba Registrovana Pre Dan Podla Paragrafu 7 Alebo 7a -> [JE_PODLA] -> Paragraf § 7a
  Osoba Registrovana Pre Dan Podla Paragrafu 7 Alebo 7a -> [MA_POVINNOST] -> Vedenie Zaznamov
  Vedenie Zaznamov -> [TYKA_SA] -> Nadobudnute Tovary Z Ineho Clenskeho Statu
  Vedenie Zaznamov -> [TYKA_SA] -> Dodane Sluzby S Miestom Dodania V Inom Clenskom State
  Dodane Sluzby S Miestom Dodania V Inom Clenskom State -> [JE_PODLA] -> Paragraf § 15 Odsek 1
  Vedenie Zaznamov -> [TYKA_SA] -> Prijate Sluzby
  Prijate Sluzby -> [JE_PODLA] -> Paragraf § 69 Odsek 3
  Prijate Sluzby -> [JE_POVINNY_PLATIT] -> Dan
  Vedenie Zaznamov -> [TYKA_SA] -> Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu

nodes:
  ZdanitelnaOsoba: Platitel
  ZdanitelnaOsoba: Osoba Registrovana Pre Dan Podla Paragrafu 7 Alebo 7a
  Osoba: Osoba Identifikovana Pre Dan V Inom Clenskom State
  Status: Rezim Call-Off Stock
  PravnyPredpis: Osobitny Predpis
  Tovar: Tovar
  Tovar: Tovar V Rezime Call-Off Stock
  Tovar: Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu V Rezime Call-Off Stock
  Tovar: Nadobudnute Tovary Z Ineho Clenskeho Statu
  Tovar: Tovar Odoslany Alebo Prepraveny Do Tuzemska Z Ineho Clenskeho Statu
  Tuzemsko: Tuzemsko
  ClenskyStat: Iny Clensky Stat
  Oprava: Oprava Zakladu Dane
  Oprava: Oprava Odpocitanej Dane
  Oprava: Oprava Opravenej Odpocitanej Dane
  Majetok: Hnutelny Hmotny Majetok Z Ineho Clenskeho Statu
  Zaznam: Zaznamy O Hnutelnom Hmotnom Majetku
  Cinnost: Identifikacia Hnutelneho Hmotneho Majetku
  Cinnost: Prace Na Hnutelnom Hmotnom Majetku
  Cinnost: Ocenovanie Hnutelneho Hmotneho Majetku
  Povinnost: Vedenie Zaznamov
  Sluzba: Dodane Sluzby S Miestom Dodania V Inom Clenskom State
  Sluzba: Prijate Sluzby
  Dan: Dan
  Paragraf: Paragraf § 25
  Paragraf: Paragraf § 25a
  Paragraf: Paragraf § 53
  Paragraf: Paragraf § 53a
  Odsek: Paragraf § 53b Odsek 1
  Odsek: Paragraf § 53b Odsek 2
  Odsek: Paragraf § 53b Odsek 4
  Paragraf: Paragraf § 7
  Paragraf: Paragraf § 7a
  Odsek: Paragraf § 15 Odsek 1
  Odsek: Paragraf § 69 Odsek 3


chunk: 483
page: 119
text: dopravného prostriedku a na požiadanie daňového úradu poskytnúť ďalšie informácie potrebné na správne určenie dane. (6) Osoba, ktorá nie je platiteľom a príležitostne dodá nový dopravný prostriedok do iného členského štátu, uplatňuje odpočítanie dane v daňovom priznaní. Daňový úrad vráti uplatnenú odpočítanú daň do 30 dní od podania daňového priznania. (7) Každá osoba, ktorá je povinná podať daňové priznanie, uvedie v daňovom priznaní všetky údaje potrebné na výpočet celkovej splatnej dane a na výpočet celkovej odpočítateľnej dane vrátane celkovej hodnoty tovarov a služieb vzťahujúcej sa na celkovú daň a na odpočítavanú daň a celkovej hodnoty tovarov a služieb oslobodených od dane. Hodnota tovarov a služieb oslobodených od dane sa uvádza v daňovom priznaní za zdaňovacie obdobie, v ktorom vznikla daňová povinnosť. Údaje v daňovom priznaní sa zaokrúhľujú na najbližší eurocent do 0,005 eura nadol a od 0,005 eura vrátane nahor.
relationships:
  Osoba Ktora Nie Je Platitelom -> [USKUTOCNUJE] -> Prilezitostne Dodanie Noveho Dopravneho Prostriedku
  Prilezitostne Dodanie Noveho Dopravneho Prostriedku -> [TYKA_SA] -> Novy Dopravny Prostriedok
  Prilezitostne Dodanie Noveho Dopravneho Prostriedku -> [MA_MIESTO] -> Iny Clensky Stat

  Osoba Ktora Nie Je Platitelom -> [MA_NAROK_NA] -> Odpocitanie Dane
  Odpocitanie Dane -> [TYKA_SA] -> Dan
  Odpocitanie Dane -> [UVADZA] -> Danove Priznanie
  Osoba Ktora Nie Je Platitelom -> [PODAVA] -> Danove Priznanie

  Danovy Urad -> [VYKONAVA] -> Vratenie Uplatnenej Odpocitanej Dane
  Vratenie Uplatnenej Odpocitanej Dane -> [TYKA_SA] -> Odpocitana Dan
  Vratenie Uplatnenej Odpocitanej Dane -> [MA_LEHOTU] -> Lehota 30 Dni Od Podania Danoveho Priznania
  Lehota 30 Dni Od Podania Danoveho Priznania -> [VYCHADZA_Z] -> Podanie Danoveho Priznania

  Osoba Povinna Podat Danove Priznanie -> [MA_POVINNOST] -> Podanie Danoveho Priznania
  Podanie Danoveho Priznania -> [TYKA_SA] -> Danove Priznanie
  Danove Priznanie -> [MA_OBSAH] -> Udaje Potrebne Na Vypocet Celkovej Splatnej Dane
  Danove Priznanie -> [MA_OBSAH] -> Udaje Potrebne Na Vypocet Celkovej Odpocitatelnej Dane
  Udaje Potrebne Na Vypocet Celkovej Splatnej Dane -> [TYKA_SA] -> Celkova Splatna Dan
  Udaje Potrebne Na Vypocet Celkovej Odpocitatelnej Dane -> [TYKA_SA] -> Celkova Odpocitatelna Dan
  Danove Priznanie -> [MA_OBSAH] -> Celkova Hodnota Tovarov A Sluzieb
  Celkova Hodnota Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Celkova Splatna Dan
  Celkova Hodnota Tovarov A Sluzieb -> [VZTAHUJE_SA_NA] -> Odpocitavana Dan
  Danove Priznanie -> [MA_OBSAH] -> Hodnota Tovarov A Sluzieb Oslobodenych Od Dane

  Hodnota Tovarov A Sluzieb Oslobodenych Od Dane -> [MA_OBDOBIE] -> Zdanovacie Obdobie Vzniku Danovej Povinnosti
  Danova Povinnost -> [VZNIKA_PRI] -> Zdanovacie Obdobie Vzniku Danovej Povinnosti

  Udaje V Danovom Priznani -> [MA_VLASTNOST] -> Zaokruhlovanie Na Najblizsi Eurocent
  Zaokruhlovanie Na Najblizsi Eurocent -> [MA_PODMIENKU] -> Zaokruhlenie Do 0,005 Eura Nadol
  Zaokruhlovanie Na Najblizsi Eurocent -> [MA_PODMIENKU] -> Zaokruhlenie Od 0,005 Eura Vratane Nahor

nodes:
  SpravcaDane: Danovy Urad
  Osoba: Osoba Ktora Nie Je Platitelom
  Osoba: Osoba Povinna Podat Danove Priznanie
  DanovePriznanie: Danove Priznanie
  Cinnost: Podanie Danoveho Priznania
  Cinnost: Prilezitostne Dodanie Noveho Dopravneho Prostriedku
  Vozidlo: Novy Dopravny Prostriedok
  ClenskyStat: Iny Clensky Stat
  Pravo: Odpocitanie Dane
  Cinnost: Vratenie Uplatnenej Odpocitanej Dane
  Dan: Dan
  Dan: Odpocitana Dan
  Dan: Celkova Splatna Dan
  Dan: Celkova Odpocitatelna Dan
  Dan: Odpocitavana Dan
  Lehota: Lehota 30 Dni Od Podania Danoveho Priznania
  Zaznam: Udaje Potrebne Na Vypocet Celkovej Splatnej Dane
  Zaznam: Udaje Potrebne Na Vypocet Celkovej Odpocitatelnej Dane
  Zaznam: Udaje V Danovom Priznani
  Hodnota: Celkova Hodnota Tovarov A Sluzieb
  Hodnota: Hodnota Tovarov A Sluzieb Oslobodenych Od Dane
  ZdanovacieObdobie: Zdanovacie Obdobie Vzniku Danovej Povinnosti
  Povinnost: Danova Povinnost
  Vypocet: Zaokruhlovanie Na Najblizsi Eurocent
  Podmienka: Zaokruhlenie Do 0,005 Eura Nadol
  Podmienka: Zaokruhlenie Od 0,005 Eura Vratane Nahor
  Euro: 0,005 Eura
  Mena: Eurocent


chunk: 517
page: 127
text: uplatňovania tejto osobitnej úpravy nezaniká. (5) V poslednom zdaňovacom období sa majetok, pri ktorého nadobudnutí alebo vytvorení vlastnou činnosťou bola daň úplne alebo čiastočne odpočítaná alebo ktorý platiteľ nadobudol bez dane (§ 10 ods. 1 alebo ods. 3) a pri ktorého nadobudnutí alebo vytvorení vlastnou činnosťou bola daň u predchádzajúceho vlastníka úplne alebo čiastočne odpočítaná, považuje za dodaný platiteľom podľa § 8 ods. 3. Daň sa vypočíta zo základu dane podľa § 22 ods. 5, najviac však do výšky odpočítanej dane, okrem dane vzťahujúcej sa na majetok, ktorého dodanie by bolo oslobodené od dane podľa § 38 ods. 1 alebo ods. 7, pri ktorom sa daň vypočíta vo výške, v akej by bol platiteľ povinný upraviť odpočítanú daň podľa § 54a z dôvodu, že zmenil rozsah použitia tohto majetku len na iný účel ako na podnikanie. Ak ide o investičný majetok podľa § 54 ods. 2 písm. d), v poslednom zdaňovacom období je platiteľ povinný upraviť odpočítanú daň postupom podľa
relationships:
  Povazovanie Majetku Za Dodany -> [MA_OBDOBIE] -> Posledne Zdanovacie Obdobie
  Majetok -> [POVAZUJE_SA_ZA] -> Dodanie Majetku Platitelom
  Platitel -> [USKUTOCNUJE] -> Dodanie Majetku Platitelom
  Dodanie Majetku Platitelom -> [TYKA_SA] -> Majetok
  Dodanie Majetku Platitelom -> [JE_PODLA] -> Paragraf § 8 Odsek 3

  Majetok -> [MA_PODMIENKU] -> Uplne Alebo Ciastocne Odpocitana Dan
  Vytvorenie Majetku Vlastnou Cinnostou -> [TYKA_SA] -> Majetok
  Nadobudnutie Majetku Bez Dane -> [TYKA_SA] -> Majetok
  Nadobudnutie Majetku Bez Dane -> [JE_PODLA] -> Paragraf § 10 Odsek 1
  Nadobudnutie Majetku Bez Dane -> [JE_PODLA] -> Paragraf § 10 Odsek 3
  Predchadzajuci Vlastnik -> [SUVISI_S] -> Uplne Alebo Ciastocne Odpocitana Dan

  Dan -> [MA_ZAKLAD_DANE] -> Zaklad Dane Podla Odseku 5 Paragrafu 22
  Zaklad Dane Podla Odseku 5 Paragrafu 22 -> [JE_PODLA] -> Paragraf § 22 Odsek 5
  Vypocet Dane -> [VYCHADZA_Z] -> Zaklad Dane Podla Odseku 5 Paragrafu 22
  Vypocet Dane -> [MA_PODMIENKU] -> Limit Do Vysky Odpocitanej Dane
  Limit Do Vysky Odpocitanej Dane -> [MA_HODNOTU] -> Vyska Odpocitanej Dane

  Vypocet Dane -> [MA_VYNIMKU] -> Vypocet Dane Pri Majetku S Oslobodenym Dodanim
  Vypocet Dane Pri Majetku S Oslobodenym Dodanim -> [VZTAHUJE_SA_NA] -> Majetok S Oslobodenym Dodanim
  Dodanie Majetku S Oslobodenym Dodanim -> [TYKA_SA] -> Majetok S Oslobodenym Dodanim
  Dodanie Majetku S Oslobodenym Dodanim -> [JE_OSLOBODENE_OD] -> Dan
  Dodanie Majetku S Oslobodenym Dodanim -> [JE_PODLA] -> Paragraf § 38 Odsek 1
  Dodanie Majetku S Oslobodenym Dodanim -> [JE_PODLA] -> Paragraf § 38 Odsek 7

  Vypocet Dane Pri Majetku S Oslobodenym Dodanim -> [VYCHADZA_Z] -> Uprava Odpocitanej Dane Podla Paragrafu 54a
  Uprava Odpocitanej Dane Podla Paragrafu 54a -> [JE_PODLA] -> Paragraf § 54a
  Uprava Odpocitanej Dane Podla Paragrafu 54a -> [MA_DOVOD] -> Zmena Rozsahu Pouzitia Majetku
  Zmena Rozsahu Pouzitia Majetku -> [TYKA_SA] -> Majetok
  Zmena Rozsahu Pouzitia Majetku -> [MA_UCEL] -> Iny Ucel Ako Podnikanie
  Iny Ucel Ako Podnikanie -> [NEVZTAHUJE_SA_NA] -> Podnikanie

  Investicny Majetok Podla Paragrafu 54 Odsek 2 Pismeno D -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno d)
  Platitel -> [MA_POVINNOST] -> Uprava Odpocitanej Dane Pri Investicnom Majetku
  Uprava Odpocitanej Dane Pri Investicnom Majetku -> [TYKA_SA] -> Investicny Majetok Podla Paragrafu 54 Odsek 2 Pismeno D
  Uprava Odpocitanej Dane Pri Investicnom Majetku -> [MA_OBDOBIE] -> Posledne Zdanovacie Obdobie

nodes:
  ZdanitelnaOsoba: Platitel
  Osoba: Predchadzajuci Vlastnik
  ZdanovacieObdobie: Posledne Zdanovacie Obdobie
  Cinnost: Povazovanie Majetku Za Dodany
  Majetok: Majetok
  Majetok: Majetok S Oslobodenym Dodanim
  Cinnost: Dodanie Majetku Platitelom
  Cinnost: Dodanie Majetku S Oslobodenym Dodanim
  Cinnost: Vytvorenie Majetku Vlastnou Cinnostou
  Cinnost: Nadobudnutie Majetku Bez Dane
  Dan: Dan
  Dan: Uplne Alebo Ciastocne Odpocitana Dan
  Suma: Vyska Odpocitanej Dane
  Hodnota: Zaklad Dane Podla Odseku 5 Paragrafu 22
  Vypocet: Vypocet Dane
  Vypocet: Vypocet Dane Pri Majetku S Oslobodenym Dodanim
  Limit: Limit Do Vysky Odpocitanej Dane
  Oprava: Uprava Odpocitanej Dane Podla Paragrafu 54a
  Oprava: Uprava Odpocitanej Dane Pri Investicnom Majetku
  Cinnost: Zmena Rozsahu Pouzitia Majetku
  Cinnost: Iny Ucel Ako Podnikanie
  Cinnost: Podnikanie
  InvesticnyMajetok: Investicny Majetok Podla Paragrafu 54 Odsek 2 Pismeno D
  Odsek: Paragraf § 8 Odsek 3
  Odsek: Paragraf § 10 Odsek 1
  Odsek: Paragraf § 10 Odsek 3
  Odsek: Paragraf § 22 Odsek 5
  Odsek: Paragraf § 38 Odsek 1
  Odsek: Paragraf § 38 Odsek 7
  Paragraf: Paragraf § 54a
  Pismeno: Paragraf § 54 Odsek 2 Pismeno d)


chunk: 568
page: 140
text: (4) Colný úrad na základe žiadosti vráti daň zaplatenú pri dovoze tovaru, ktorý bol dovezený v období od 1. januára 2021 do 31. decembra 2021 Európskou komisiou, agentúrou alebo orgánom zriadeným podľa práva Európskej únie, ak tento tovar je určený alebo bol určený a použitý na plnenie úloh, ktoré im boli zverené právom Európskej únie, súvisiacich s pandémiou ochorenia COVID-19; to neplatí, ak Európska komisia, agentúra alebo orgán zriadený podľa práva Európskej únie použila alebo požije dovezený tovar na účely ďalšieho dodania za protihodnotu. (5) Ak sa po podaní žiadosti podľa odseku 4 prestali plniť podmienky na vrátenie dane v súvislosti s dovozom tovaru podľa odseku 4, Európska komisia, agentúra alebo orgán zriadený podľa práva Európskej únie sú povinné o tejto skutočnosti bez zbytočného odkladu informovať colný úrad. (6) Ak sa prestali plniť podmienky na vrátenie dane podľa odseku 4 a rozhodnutie o vrátení dane už bolo vydané, colný úrad toto rozhodnutie zruší.
relationships:
  Colny Urad -> [VYKONAVA] -> Vratenie Dane
  Vratenie Dane -> [TYKA_SA] -> Dan Zaplatena Pri Dovoze Tovaru
  Vratenie Dane -> [VYCHADZA_Z] -> Ziadost

  Dan Zaplatena Pri Dovoze Tovaru -> [VZNIKA_PRI] -> Dovoz Tovaru
  Dovoz Tovaru -> [TYKA_SA] -> Tovar
  Dovoz Tovaru -> [MA_OBDOBIE] -> Obdobie Od 1. Januara 2021 Do 31. Decembra 2021

  Dovoz Tovaru -> [USKUTOCNUJE] -> Europska Komisia
  Dovoz Tovaru -> [USKUTOCNUJE] -> Agentura Zriadena Podla Prava Europskej Unie
  Dovoz Tovaru -> [USKUTOCNUJE] -> Organ Zriadeny Podla Prava Europskej Unie
  Agentura Zriadena Podla Prava Europskej Unie -> [JE_PODLA] -> Pravo Europskej Unie
  Organ Zriadeny Podla Prava Europskej Unie -> [JE_PODLA] -> Pravo Europskej Unie

  Tovar -> [MA_UCEL] -> Plnenie Uloh
  Plnenie Uloh -> [VYPLNYVA_Z] -> Pravo Europskej Unie
  Plnenie Uloh -> [SUVISI_S] -> Pandemia Ochorenia Covid-19

  Vratenie Dane -> [MA_PODMIENKU] -> Podmienky Na Vratenie Dane
  Podmienky Na Vratenie Dane -> [NEPLATI_PRE] -> Dalsie Dodanie Za Protihodnotu
  Dalsie Dodanie Za Protihodnotu -> [TYKA_SA] -> Dovezeny Tovar

  Europska Komisia -> [MA_POVINNOST] -> Informovanie Colneho Uradu
  Agentura Zriadena Podla Prava Europskej Unie -> [MA_POVINNOST] -> Informovanie Colneho Uradu
  Organ Zriadeny Podla Prava Europskej Unie -> [MA_POVINNOST] -> Informovanie Colneho Uradu
  Informovanie Colneho Uradu -> [TYKA_SA] -> Prestanie Plnenia Podmienok Na Vratenie Dane
  Informovanie Colneho Uradu -> [MA_LEHOTU] -> Bez Zbytocneho Odkladu
  Informovanie Colneho Uradu -> [TYKA_SA] -> Colny Urad

  Colny Urad -> [ZRUSUJE] -> Rozhodnutie O Vrateni Dane
  Zrusenie Rozhodnutia O Vrateni Dane -> [MA_PODMIENKU] -> Prestanie Plnenia Podmienok Na Vratenie Dane
  Zrusenie Rozhodnutia O Vrateni Dane -> [MA_PODMIENKU] -> Vydane Rozhodnutie O Vrateni Dane

nodes:
  Urad: Colny Urad
  Ziadost: Ziadost
  Cinnost: Vratenie Dane
  Dan: Dan Zaplatena Pri Dovoze Tovaru
  Cinnost: Dovoz Tovaru
  Tovar: Tovar
  Tovar: Dovezeny Tovar
  Obdobie: Obdobie Od 1. Januara 2021 Do 31. Decembra 2021
  Organizacia: Europska Komisia
  Agentura: Agentura Zriadena Podla Prava Europskej Unie
  StatnyOrgan: Organ Zriadeny Podla Prava Europskej Unie
  PravnyPredpis: Pravo Europskej Unie
  Cinnost: Plnenie Uloh
  Dovod: Pandemia Ochorenia Covid-19
  Podmienka: Podmienky Na Vratenie Dane
  Cinnost: Dalsie Dodanie Za Protihodnotu
  Povinnost: Informovanie Colneho Uradu
  Dovod: Prestanie Plnenia Podmienok Na Vratenie Dane
  Lehota: Bez Zbytocneho Odkladu
  Rozhodnutie: Rozhodnutie O Vrateni Dane
  Rozhodnutie: Vydane Rozhodnutie O Vrateni Dane
  Cinnost: Zrusenie Rozhodnutia O Vrateni Dane


chunk: 602
page: 152
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 153 Príloha č. 4 k zákonu č. 222/2004 Z. z. ŽIADOSŤ o vrátenie dane z pridanej hodnoty zahraničnému zástupcovi DynamicResources\6e3bf279-9846-46da-975f-431c66fedc07_3.pdf IADOS
relationships:
  Zakon 222/2004 Z Z -> [OBSAHUJE] -> Priloha C 4
  Priloha C 4 -> [JE_SUCASTOU] -> Zakon C 222/2004 Z Z
  Priloha C 4 -> [OBSAHUJE] -> Ziadost O Vratenie Dane Z Pridanej Hodnoty Zahranicnemu Zastupcovi
  Ziadost O Vratenie Dane Z Pridanej Hodnoty Zahranicnemu Zastupcovi -> [MA_UCEL] -> Vratenie Dane Z Pridanej Hodnoty
  Vratenie Dane Z Pridanej Hodnoty -> [TYKA_SA] -> Dan Z Pridanej Hodnoty
  Vratenie Dane Z Pridanej Hodnoty -> [PLATI_PRE] -> Zahranicny Zastupca

nodes:
  Zakon: Zakon 222/2004 Z Z
  Priloha: Priloha C 4
  Ziadost: Ziadost O Vratenie Dane Z Pridanej Hodnoty Zahranicnemu Zastupcovi
  Cinnost: Vratenie Dane Z Pridanej Hodnoty
  Dan: Dan Z Pridanej Hodnoty
  Zastupca: Zahranicny Zastupca


chunk: 60
page: 14
text: odoslaný alebo prepravený, posudzuje sa, že podmienky podľa odsekov 1 a 5 sa prestali plniť bezprostredne predtým, ako sa odoslanie alebo preprava tovaru do iného štátu začala. Ak dôjde k zničeniu, strate alebo ku krádeži tovaru, posudzuje sa, že podmienky podľa odsekov 1 a 5 sa prestali plniť v deň, keď došlo k zničeniu, strate alebo ku krádeži tovaru; ak tento deň nie je možné určiť, v deň, keď sa zistilo, že tovar je zničený alebo chýba. § 9 Dodanie služby (1) Dodaním služby je každé plnenie, ktoré nie je dodaním tovaru podľa § 8, vrátane a) prevodu práva k nehmotnému majetku vrátane poskytnutia práva k priemyselnému vlastníctvu alebo inému duševnému vlastníctvu, b) poskytnutia práva užívať hmotný majetok, c) prijatia záväzku zdržať sa konania alebo strpieť konanie alebo stav, d) služby dodanej na základe poverenia alebo rozhodnutia vydaného štátnym orgánom alebo na základe zákona. (2) Užívanie hmotného majetku, ktorý je vo vlastníctve platiteľa a pri ktorého kúpe alebo
relationships:
  Paragraf § 9 -> [DEFINUJE] -> Dodanie Sluzby
  Dodanie Sluzby -> [ROZUMIE_SA] -> Plnenie Ktore Nie Je Dodanim Tovaru
  Plnenie Ktore Nie Je Dodanim Tovaru -> [NEVZTAHUJE_SA_NA] -> Dodanie Tovaru
  Dodanie Tovaru -> [JE_PODLA] -> Paragraf § 8

  Dodanie Sluzby -> [ZAHRNUJE] -> Prevod Prava K Nehmotnemu Majetku
  Prevod Prava K Nehmotnemu Majetku -> [ZAHRNUJE] -> Poskytnutie Prava K Priemyselnemu Vlastnictvu
  Prevod Prava K Nehmotnemu Majetku -> [ZAHRNUJE] -> Poskytnutie Prava K Inemu Dusevnemu Vlastnictvu
  Dodanie Sluzby -> [ZAHRNUJE] -> Poskytnutie Prava Uzivat Hmotny Majetok
  Dodanie Sluzby -> [ZAHRNUJE] -> Prijatie Zavazku Zdrzat Sa Konania
  Dodanie Sluzby -> [ZAHRNUJE] -> Prijatie Zavazku Strpiet Konanie Alebo Stav
  Dodanie Sluzby -> [ZAHRNUJE] -> Sluzba Dodana Na Zaklade Poverenia
  Dodanie Sluzby -> [ZAHRNUJE] -> Sluzba Dodana Na Zaklade Rozhodnutia Statneho Organu
  Dodanie Sluzby -> [ZAHRNUJE] -> Sluzba Dodana Na Zaklade Zakona
  Sluzba Dodana Na Zaklade Rozhodnutia Statneho Organu -> [VYCHADZA_Z] -> Rozhodnutie Statneho Organu
  Sluzba Dodana Na Zaklade Zakona -> [VYCHADZA_Z] -> Zakon

  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [NASTAVA_PRI] -> Odoslanie Alebo Preprava Tovaru Do Ineho Statu
  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [MA_DATUM] -> Bezprostredne Pred Zacatim Odoslania Alebo Prepravy Tovaru
  Odoslanie Alebo Preprava Tovaru Do Ineho Statu -> [TYKA_SA] -> Tovar
  Odoslanie Alebo Preprava Tovaru Do Ineho Statu -> [MA_MIESTO] -> Iny Stat

  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [NASTAVA_PRI] -> Znicenie Tovaru
  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [NASTAVA_PRI] -> Strata Tovaru
  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [NASTAVA_PRI] -> Kradez Tovaru
  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [MA_DATUM] -> Den Znicenia Straty Alebo Kradeze Tovaru
  Prestanie Plnenia Podmienok Podla Odsekov 1 A 5 -> [MA_DATUM] -> Den Zistenia Ze Tovar Je Zniceny Alebo Chyba

nodes:
  Paragraf: Paragraf § 9
  Paragraf: Paragraf § 8
  Cinnost: Dodanie Sluzby
  Cinnost: Plnenie Ktore Nie Je Dodanim Tovaru
  Cinnost: Dodanie Tovaru
  Cinnost: Prevod Prava K Nehmotnemu Majetku
  Pravo: Poskytnutie Prava K Priemyselnemu Vlastnictvu
  Pravo: Poskytnutie Prava K Inemu Dusevnemu Vlastnictvu
  Pravo: Poskytnutie Prava Uzivat Hmotny Majetok
  Zavazok: Prijatie Zavazku Zdrzat Sa Konania
  Zavazok: Prijatie Zavazku Strpiet Konanie Alebo Stav
  Sluzba: Sluzba Dodana Na Zaklade Poverenia
  Sluzba: Sluzba Dodana Na Zaklade Rozhodnutia Statneho Organu
  Sluzba: Sluzba Dodana Na Zaklade Zakona
  Rozhodnutie: Rozhodnutie Statneho Organu
  Zakon: Zakon
  Podmienka: Prestanie Plnenia Podmienok Podla Odsekov 1 A 5
  Cinnost: Odoslanie Alebo Preprava Tovaru Do Ineho Statu
  Tovar: Tovar
  Stat: Iny Stat
  Datum: Bezprostredne Pred Zacatim Odoslania Alebo Prepravy Tovaru
  Cinnost: Znicenie Tovaru
  Cinnost: Strata Tovaru
  Cinnost: Kradez Tovaru
  Datum: Den Znicenia Straty Alebo Kradeze Tovaru
  Datum: Den Zistenia Ze Tovar Je Zniceny Alebo Chyba


chunk: 7
page: 2
text: ktoré je predmetom dane podľa § 2, uskutočneným zdaniteľnou osobou, ktorá má v tuzemsku výlučne prevádzkareň, ak nejde o malý podnik zahraničnej osoby, ktorý uplatňuje oslobodenie od dane podľa § 68f ods. 2. (2) Zdaniteľná osoba podľa odseku 1 je povinná podať daňovému úradu žiadosť o registráciu pre daň do piatich pracovných dní a) odo dňa, v ktorom bol presiahnutý obrat podľa odseku 1 písm. a), b) odo dňa, v ktorom bol presiahnutý obrat podľa odseku 1 písm. b), ak nebola povinná podať žiadosť o registráciu pre daň podľa písmena a), c) odo dňa, v ktorom bol presiahnutý obrat podľa odseku 1 písm. b), ak bol tento obrat presiahnutý po presiahnutí obratu podľa odseku 1 písm. a) do uplynutia lehoty na podanie žiadosti o registráciu pre daň podľa písmena a) a žiadosť o registráciu pre daň podľa písmena a) nebola ešte podaná, d) odo dňa, keď sa stala platiteľom podľa odseku 1 písm. c) až i) alebo e) odo dňa, keď sa stala platiteľom podľa odseku 1 písm. c) až i), ak táto skutočnosť nastala po
relationships:
  Predmet Dane -> [JE_PODLA] -> Paragraf § 2

  Zdanitelna Osoba -> [MA_PREVADZKAREN] -> Prevadzkaren
  Prevadzkaren -> [NACHADZA_SA_V] -> Tuzemsko

  Maly Podnik Zahranicnej Osoby -> [MA_NAROK_NA] -> Oslobodenie Od Dane Podla Paragrafu 68f Odsek 2
  Oslobodenie Od Dane Podla Paragrafu 68f Odsek 2 -> [JE_PODLA] -> Odsek 2
  Paragraf § 68f -> [MA_ODSEK] -> Odsek 2

  Zdanitelna Osoba Podla Odseku 1 -> [JE_PODLA] -> Odsek 1
  Zdanitelna Osoba Podla Odseku 1 -> [MA_POVINNOST] -> Podanie Ziadosti O Registraciu Pre Dan
  Podanie Ziadosti O Registraciu Pre Dan -> [TYKA_SA] -> Ziadost O Registraciu Pre Dan
  Podanie Ziadosti O Registraciu Pre Dan -> [TYKA_SA] -> Danovy Urad
  Podanie Ziadosti O Registraciu Pre Dan -> [MA_LEHOTU] -> Lehota Do Piatich Pracovnych Dni
  Ziadost O Registraciu Pre Dan -> [TYKA_SA] -> Registracia Pre Dan

  Lehota Do Piatich Pracovnych Dni -> [VYCHADZA_Z] -> Den Presiahnutia Obratu Podla Odseku 1 Pismeno A
  Lehota Do Piatich Pracovnych Dni -> [VYCHADZA_Z] -> Den Presiahnutia Obratu Podla Odseku 1 Pismeno B
  Lehota Do Piatich Pracovnych Dni -> [VYCHADZA_Z] -> Den Ked Sa Zdanitelna Osoba Stala Platitelom Podla Odseku 1 Pismeno C Az I

  Den Presiahnutia Obratu Podla Odseku 1 Pismeno A -> [TYKA_SA] -> Obrat Podla Odseku 1 Pismeno A
  Den Presiahnutia Obratu Podla Odseku 1 Pismeno B -> [TYKA_SA] -> Obrat Podla Odseku 1 Pismeno B
  Obrat Podla Odseku 1 Pismeno A -> [JE_PODLA] -> Pismeno a)
  Obrat Podla Odseku 1 Pismeno B -> [JE_PODLA] -> Pismeno b)

  Den Presiahnutia Obratu Podla Odseku 1 Pismeno B -> [MA_PODMIENKU] -> Nepodanie Povinnosti Podat Ziadost Podla Pismena A
  Den Presiahnutia Obratu Podla Odseku 1 Pismeno B -> [MA_PODMIENKU] -> Presiahnutie Obratu Po Presiahnuti Obratu Podla Pismena A Pred Uplynutim Lehoty
  Presiahnutie Obratu Po Presiahnuti Obratu Podla Pismena A Pred Uplynutim Lehoty -> [TYKA_SA] -> Lehota Na Podanie Ziadosti Podla Pismena A

nodes:
  Paragraf: Paragraf § 2
  Paragraf: Paragraf § 68f
  Odsek: Odsek 1
  Odsek: Odsek 2
  Pismeno: Pismeno a)
  Pismeno: Pismeno b)
  Pismeno: Pismeno c)
  Pismeno: Pismeno d)
  Pismeno: Pismeno e)
  Pismeno: Pismeno f)
  Pismeno: Pismeno g)
  Pismeno: Pismeno h)
  Pismeno: Pismeno i)
  Dan: Predmet Dane
  ZdanitelnaOsoba: Zdanitelna Osoba
  ZdanitelnaOsoba: Zdanitelna Osoba Podla Odseku 1
  Tuzemsko: Tuzemsko
  Prevazdkaren: Prevadzkaren
  Podnik: Maly Podnik Zahranicnej Osoby
  OslobodenieOdDane: Oslobodenie Od Dane Podla Paragrafu 68f Odsek 2
  Urad: Danovy Urad
  Ziadost: Ziadost O Registraciu Pre Dan
  Registracia: Registracia Pre Dan
  Povinnost: Podanie Ziadosti O Registraciu Pre Dan
  Lehota: Lehota Do Piatich Pracovnych Dni
  Lehota: Lehota Na Podanie Ziadosti Podla Pismena A
  Obrat: Obrat Podla Odseku 1 Pismeno A
  Obrat: Obrat Podla Odseku 1 Pismeno B
  Datum: Den Presiahnutia Obratu Podla Odseku 1 Pismeno A
  Datum: Den Presiahnutia Obratu Podla Odseku 1 Pismeno B
  Datum: Den Ked Sa Zdanitelna Osoba Stala Platitelom Podla Odseku 1 Pismeno C Az I
  Podmienka: Nepodanie Povinnosti Podat Ziadost Podla Pismena A
  Podmienka: Presiahnutie Obratu Po Presiahnuti Obratu Podla Pismena A Pred Uplynutim Lehoty


chunk: 82
page: 20
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 21 2. dodaným tovarom nie sú nové dopravné prostriedky ani tovar dodaný s inštaláciou alebo montážou uskutočnenou dodávateľom alebo na jeho účet, b) predajom tovaru na diaľku dovážaného z územia tretích štátov dodanie tovaru, ktorý je odoslaný alebo prepravený dodávateľom alebo na jeho účet, a to aj keď dodávateľ zasahuje nepriamo do odoslania alebo prepravy tovaru, z územia tretích štátov k zákazníkovi v členskom štáte, ak sú splnené tieto podmienky: 1. dodanie tovaru sa uskutočňuje pre zdaniteľnú osobu alebo pre právnickú osobu, ktorá nie je zdaniteľnou osobou, u ktorých nadobudnutie tovaru na území Európskej únie nie je predmetom dane podľa § 11 ods. 4, alebo pre osobu inú ako zdaniteľnú osobu, 2. dodaným tovarom nie sú nové dopravné prostriedky ani tovar dodaný s inštaláciou alebo montážou uskutočnenou dodávateľom alebo na jeho účet. (2) Miestom dodania tovaru pri predaji tovaru na diaľku na území Európskej únie je miesto, kde
relationships:
  Zakon C 222/2004 Z Z -> [OBSAHUJE] -> Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov
  Zakon C 222/2004 Z Z -> [OBSAHUJE] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie

  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov -> [ROZUMIE_SA] -> Dodanie Tovaru
  Dodanie Tovaru -> [TYKA_SA] -> Tovar
  Tovar -> [VYCHADZA_Z] -> Uzemie Tretich Statov
  Tovar -> [MA_MIESTO] -> Clensky Stat
  Odoslanie Alebo Preprava Tovaru -> [TYKA_SA] -> Tovar
  Odoslanie Alebo Preprava Tovaru -> [VYCHADZA_Z] -> Uzemie Tretich Statov
  Odoslanie Alebo Preprava Tovaru -> [TYKA_SA] -> Zakaznik V Clenskom State
  Dodavatel -> [VYKONAVA] -> Odoslanie Alebo Preprava Tovaru
  Odoslanie Alebo Preprava Tovaru -> [PLATI_PRE] -> Ucet Dodavatela
  Dodavatel -> [VYKONAVA] -> Nepriamy Zasah Do Odoslania Alebo Prepravy Tovaru

  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov -> [MA_PODMIENKU] -> Dodanie Pre Zdanitelnu Osobu
  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov -> [MA_PODMIENKU] -> Dodanie Pre Pravnicku Osobu Ktora Nie Je Zdanitelnou Osobou
  Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov -> [MA_PODMIENKU] -> Dodanie Pre Osobu Inu Ako Zdanitelnu Osobu

  Dodanie Pre Zdanitelnu Osobu -> [TYKA_SA] -> Zdanitelna Osoba
  Dodanie Pre Pravnicku Osobu Ktora Nie Je Zdanitelnou Osobou -> [TYKA_SA] -> Pravnicka Osoba Ktora Nie Je Zdanitelnou Osobou
  Dodanie Pre Osobu Inu Ako Zdanitelnu Osobu -> [TYKA_SA] -> Osoba Ina Ako Zdanitelna Osoba

  Nadobudnutie Tovaru Na Uzemi Europskej Unie -> [TYKA_SA] -> Tovar
  Nadobudnutie Tovaru Na Uzemi Europskej Unie -> [MA_MIESTO] -> Uzemie Europskej Unie
  Nadobudnutie Tovaru Na Uzemi Europskej Unie -> [NIE_JE_PREDMETOM] -> Dan
  Nadobudnutie Tovaru Na Uzemi Europskej Unie -> [JE_PODLA] -> Paragraf § 11 Odsek 4

  Tovar -> [NEVZTAHUJE_SA_NA] -> Novy Dopravny Prostriedok
  Tovar -> [NEVZTAHUJE_SA_NA] -> Tovar Dodany S Instalaciou Alebo Montazou
  Tovar Dodany S Instalaciou Alebo Montazou -> [TYKA_SA] -> Instalacia Alebo Montaz Uskutocnena Dodavatelom Alebo Na Jeho Ucet

  Predaj Tovaru Na Dialku Na Uzemi Europskej Unie -> [VZTAHUJE_SA_NA] -> Uzemie Europskej Unie
  Miesto Dodania Tovaru -> [TYKA_SA] -> Predaj Tovaru Na Dialku Na Uzemi Europskej Unie

nodes:
  Zakon: Zakon C 222/2004 Z Z
  Cinnost: Predaj Tovaru Na Dialku Dovazany Z Uzemia Tretich Statov
  Cinnost: Predaj Tovaru Na Dialku Na Uzemi Europskej Unie
  Cinnost: Dodanie Tovaru
  Tovar: Tovar
  Tovar: Tovar Dodany S Instalaciou Alebo Montazou
  Cinnost: Odoslanie Alebo Preprava Tovaru
  Cinnost: Nepriamy Zasah Do Odoslania Alebo Prepravy Tovaru
  Subjekt: Dodavatel
  Subjekt: Zakaznik V Clenskom State
  Uzemie: Uzemie Tretich Statov
  Uzemie: Uzemie Europskej Unie
  ClenskyStat: Clensky Stat
  ZdanitelnaOsoba: Zdanitelna Osoba
  PravnickaOsoba: Pravnicka Osoba Ktora Nie Je Zdanitelnou Osobou
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Podmienka: Dodanie Pre Zdanitelnu Osobu
  Podmienka: Dodanie Pre Pravnicku Osobu Ktora Nie Je Zdanitelnou Osobou
  Podmienka: Dodanie Pre Osobu Inu Ako Zdanitelnu Osobu
  Cinnost: Nadobudnutie Tovaru Na Uzemi Europskej Unie
  Dan: Dan
  Odsek: Paragraf § 11 Odsek 4
  Vozidlo: Novy Dopravny Prostriedok
  Cinnost: Instalacia Alebo Montaz Uskutocnena Dodavatelom Alebo Na Jeho Ucet
  Ucet: Ucet Dodavatela
  Lokacia: Miesto Dodania Tovaru


chunk: 89
page: 21
text: (6) Miestom dodania pri inej preprave tovaru ako je preprava tovaru medzi členskými štátmi podľa odseku 7 osobe inej ako zdaniteľnej osobe je miesto, kde sa preprava tovaru vykonáva, a ak sa vykonáva vo viac ako jednom štáte, považuje sa za vykonanú v týchto štátoch pomerne k prekonaným vzdialenostiam v týchto štátoch. (7) Na účely odsekov 5 a 6 je a) prepravou tovaru medzi členskými štátmi preprava tovaru, ktorej miesto začatia a miesto skončenia sa nachádzajú na územiach dvoch rôznych členských štátov, b) miestom začatia prepravy tovaru miesto, kde sa preprava tovaru skutočne začína, bez ohľadu na prekonanú vzdialenosť do miesta, kde sa tovar nachádza, c) miestom skončenia prepravy tovaru miesto, kde sa preprava tovaru skutočne skončí. (8) Miestom dodania doplnkových služieb pri preprave, napríklad nakladanie, vykladanie, manipulácia a podobné služby, ak sú tieto služby dodané osobe inej ako zdaniteľnej osobe, je
relationships:
  Odsek 6 -> [ODKAZUJE_NA] -> Odsek 7
  Odsek 7 -> [VZTAHUJE_SA_NA] -> Odsek 5
  Odsek 7 -> [VZTAHUJE_SA_NA] -> Odsek 6

  Ina Preprava Tovaru -> [NEVZTAHUJE_SA_NA] -> Preprava Tovaru Medzi Clenskymi Statmi
  Ina Preprava Tovaru -> [TYKA_SA] -> Osoba Ina Ako Zdanitelna Osoba
  Miesto Dodania Pri Inej Preprave Tovaru -> [ROZUMIE_SA] -> Miesto Vykonavania Prepravy Tovaru
  Miesto Vykonavania Prepravy Tovaru -> [TYKA_SA] -> Ina Preprava Tovaru

  Ina Preprava Tovaru Vykonavana Vo Viac Ako Jednom State -> [MA_MIESTO] -> Viac Statov
  Ina Preprava Tovaru Vykonavana Vo Viac Ako Jednom State -> [MA_PODMIENKU] -> Pomerne K Prekonanym Vzdialenostiam V Statoch

  Odsek 7 -> [VYMEDZUJE] -> Preprava Tovaru Medzi Clenskymi Statmi
  Preprava Tovaru Medzi Clenskymi Statmi -> [MA_MIESTO] -> Miesto Zacatia Prepravy Tovaru
  Preprava Tovaru Medzi Clenskymi Statmi -> [MA_MIESTO] -> Miesto Skoncenia Prepravy Tovaru
  Miesto Zacatia Prepravy Tovaru -> [NACHADZA_SA_V] -> Uzemie Prveho Clenskeho Statu
  Miesto Skoncenia Prepravy Tovaru -> [NACHADZA_SA_V] -> Uzemie Druheho Clenskeho Statu
  Uzemie Prveho Clenskeho Statu -> [SUVISI_S] -> Dva Rozne Clenske Staty
  Uzemie Druheho Clenskeho Statu -> [SUVISI_S] -> Dva Rozne Clenske Staty

  Odsek 7 -> [VYMEDZUJE] -> Miesto Zacatia Prepravy Tovaru
  Miesto Zacatia Prepravy Tovaru -> [ROZUMIE_SA] -> Miesto Skutocneho Zacatia Prepravy Tovaru
  Miesto Skutocneho Zacatia Prepravy Tovaru -> [NEVZTAHUJE_SA_NA] -> Prekonana Vzdialenost Do Miesta Kde Sa Tovar Nachadza

  Odsek 7 -> [VYMEDZUJE] -> Miesto Skoncenia Prepravy Tovaru
  Miesto Skoncenia Prepravy Tovaru -> [ROZUMIE_SA] -> Miesto Skutocneho Skoncenia Prepravy Tovaru

  Odsek 8 -> [VZTAHUJE_SA_NA] -> Miesto Dodania Doplnkovych Sluzieb Pri Preprave
  Doplnkove Sluzby Pri Preprave -> [ZAHRNUJE] -> Nakladanie
  Doplnkove Sluzby Pri Preprave -> [ZAHRNUJE] -> Vykladanie
  Doplnkove Sluzby Pri Preprave -> [ZAHRNUJE] -> Manipulacia
  Doplnkove Sluzby Pri Preprave -> [TYKA_SA] -> Osoba Ina Ako Zdanitelna Osoba

nodes:
  Odsek: Odsek 5
  Odsek: Odsek 6
  Odsek: Odsek 7
  Odsek: Odsek 8
  Cinnost: Ina Preprava Tovaru
  Cinnost: Ina Preprava Tovaru Vykonavana Vo Viac Ako Jednom State
  Cinnost: Preprava Tovaru Medzi Clenskymi Statmi
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Lokacia: Miesto Dodania Pri Inej Preprave Tovaru
  Lokacia: Miesto Vykonavania Prepravy Tovaru
  Stat: Viac Statov
  Podmienka: Pomerne K Prekonanym Vzdialenostiam V Statoch
  Lokacia: Miesto Zacatia Prepravy Tovaru
  Lokacia: Miesto Skoncenia Prepravy Tovaru
  Lokacia: Miesto Skutocneho Zacatia Prepravy Tovaru
  Lokacia: Miesto Skutocneho Skoncenia Prepravy Tovaru
  Uzemie: Uzemie Prveho Clenskeho Statu
  Uzemie: Uzemie Druheho Clenskeho Statu
  ClenskyStat: Dva Rozne Clenske Staty
  Mnozstvo: Prekonana Vzdialenost Do Miesta Kde Sa Tovar Nachadza
  Lokacia: Miesto Dodania Doplnkovych Sluzieb Pri Preprave
  Sluzba: Doplnkove Sluzby Pri Preprave
  Cinnost: Nakladanie
  Cinnost: Vykladanie
  Cinnost: Manipulacia
