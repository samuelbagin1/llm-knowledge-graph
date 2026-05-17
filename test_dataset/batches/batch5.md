chunk: 119
page: 29
text: Strana 30 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. orgánom alebo členom ich štatutárneho orgánu, e) osoby, ktoré sú členom platiteľa, ktorý dodáva tovar alebo službu, f) fyzické osoby, ktoré sú v pracovnoprávnom vzťahu voči platiteľovi, ktorý dodáva tovar alebo službu, g) fyzické osoby, ktoré sú osobou blízkou6ac) fyzickej osobe uvedenej v písmene a), b), c), d) alebo f), h) právnické osoby, ktorých štatutárny orgán, spoločník alebo člen je aj štatutárnym orgánom alebo spoločníkom platiteľa, ktorý dodáva tovar alebo službu, i) fyzické osoby, ktoré žijú s platiteľom, ktorý dodáva tovar alebo službu, v domácnosti,24a) j) osoby blízke6ac) platiteľovi, ktorý je fyzickou osobou a ktorý dodáva tovar alebo službu. (10) Základom dane pri dodaní tovaru alebo dodaní služby pri použití viacúčelového poukazu je protihodnota zaplatená za poukaz znížená o daň; ak dodávateľ nemá informáciu o tejto protihodnote, základom dane je peňažná hodnota, ktorá je uvedená na viacúčelovom poukaze
relationships:
- Zakon 222/2004 Z. z. -> [OBSAHUJE] -> Odsek 10

- Pismeno e -> [VYMEDZUJE] -> Osoby ktore su clenom platitela
- Osoby ktore su clenom platitela -> [JE_CLENOM] -> Platitel
- Platitel -> [DODAVA] -> Tovar
- Platitel -> [DODAVA] -> Sluzba

- Pismeno f -> [VYMEDZUJE] -> Fyzicke osoby v pracovnopravnom vztahu voci platitelovi
- Fyzicke osoby v pracovnopravnom vztahu voci platitelovi -> [SUVISI_S] -> Platitel
- Platitel -> [DODAVA] -> Tovar
- Platitel -> [DODAVA] -> Sluzba

- Pismeno g -> [VYMEDZUJE] -> Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [JE_TYPOM] -> Fyzicka osoba
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [ODKAZUJE_NA] -> Pismeno a
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [ODKAZUJE_NA] -> Pismeno b
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [ODKAZUJE_NA] -> Pismeno c
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [ODKAZUJE_NA] -> Pismeno d
- Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f -> [ODKAZUJE_NA] -> Pismeno f

- Pismeno h -> [VYMEDZUJE] -> Pravnicke osoby ktorych statutarny organ spolocnik alebo clen je aj statutarnym organom alebo spolocnikom platitela
- Pravnicke osoby ktorych statutarny organ spolocnik alebo clen je aj statutarnym organom alebo spolocnikom platitela -> [JE_TYPOM] -> Pravnicka osoba
- Pravnicka osoba -> [MA_ZASTUPCU] -> Statutarny organ
- Pravnicka osoba -> [JE_ZASTUPENA] -> Spolocnik
- Pravnicka osoba -> [JE_ZASTUPENA] -> Clen
- Statutarny organ -> [SUVISI_S] -> Platitel
- Spolocnik -> [SUVISI_S] -> Platitel
- Platitel -> [DODAVA] -> Tovar
- Platitel -> [DODAVA] -> Sluzba

- Pismeno i -> [VYMEDZUJE] -> Fyzicke osoby ktore ziju s platitelom v domacnosti
- Fyzicke osoby ktore ziju s platitelom v domacnosti -> [JE_TYPOM] -> Fyzicka osoba
- Fyzicke osoby ktore ziju s platitelom v domacnosti -> [SUVISI_S] -> Platitel
- Fyzicke osoby ktore ziju s platitelom v domacnosti -> [MA_MIESTO] -> Domacnost
- Platitel -> [DODAVA] -> Tovar
- Platitel -> [DODAVA] -> Sluzba

- Pismeno j -> [VYMEDZUJE] -> Osoby blizke platitelovi ktory je fyzickou osobou
- Osoby blizke platitelovi ktory je fyzickou osobou -> [SUVISI_S] -> Platitel
- Platitel -> [JE_TYPOM] -> Fyzicka osoba
- Platitel -> [DODAVA] -> Tovar
- Platitel -> [DODAVA] -> Sluzba

- Odsek 10 -> [URCUJE] -> Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [TYKA_SA] -> Dodanie tovaru
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [TYKA_SA] -> Dodanie sluzby
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [NASTAVA_PRI] -> Pouzitie viacuceloveho poukazu
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [MA_ZAKLAD_DANE] -> Protihodnota zaplatena za poukaz znizena o dan
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [MA_PODMIENKU] -> Dodavatel nema informaciu o protihodnote
- Dodavatel nema informaciu o protihodnote -> [PODMIENUJE] -> Penazna hodnota uvedena na viacucelovom poukaze
- Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu -> [MA_ZAKLAD_DANE] -> Penazna hodnota uvedena na viacucelovom poukaze
- Penazna hodnota uvedena na viacucelovom poukaze -> [JE_SUCASTOU] -> Viacucelovy poukaz

nodes:
- Zakon: Zakon 222/2004 Z. z.
- Odsek: Odsek 10

- Pismeno: Pismeno a
- Pismeno: Pismeno b
- Pismeno: Pismeno c
- Pismeno: Pismeno d
- Pismeno: Pismeno e
- Pismeno: Pismeno f
- Pismeno: Pismeno g
- Pismeno: Pismeno h
- Pismeno: Pismeno i
- Pismeno: Pismeno j

- Osoba: Osoby ktore su clenom platitela
- FyzickaOsoba: Fyzicke osoby v pracovnopravnom vztahu voci platitelovi
- FyzickaOsoba: Fyzicke osoby ktore su osobou blizkou fyzickej osobe uvedenej v pismene a b c d alebo f
- PravnickaOsoba: Pravnicke osoby ktorych statutarny organ spolocnik alebo clen je aj statutarnym organom alebo spolocnikom platitela
- FyzickaOsoba: Fyzicke osoby ktore ziju s platitelom v domacnosti
- Osoba: Osoby blizke platitelovi ktory je fyzickou osobou

- ZdanitelnaOsoba: Platitel
- FyzickaOsoba: Fyzicka osoba
- PravnickaOsoba: Pravnicka osoba
- Osoba: Statutarny organ
- ClenSkupiny: Spolocnik
- ClenSkupiny: Clen
- Lokacia: Domacnost

- Tovar: Tovar
- Sluzba: Sluzba
- Cinnost: Dodanie tovaru
- Cinnost: Dodanie sluzby
- Cinnost: Pouzitie viacuceloveho poukazu
- Poukaz: Viacucelovy poukaz

- Dan: Zaklad dane pri dodani tovaru alebo sluzby pri pouziti viacuceloveho poukazu
- Suma: Protihodnota zaplatena za poukaz znizena o dan
- Podmienka: Dodavatel nema informaciu o protihodnote
- Hodnota: Penazna hodnota uvedena na viacucelovom poukaze



chunk: 132
page: 32
text: f) bolo zverejnené uznesenie o skončení reštrukturalizácie a pohľadávka v tomto konaní nebola prihlásená, a to dňom zverejnenia tohto uznesenia v Obchodnom vestníku v rozsahu, v akom platiteľ preukáže, že by jeho pohľadávka nebola uspokojená, ani ak by bola prihlásená, za podmienky, že táto pohľadávka k tomuto dňu nebola premlčaná. (3) Platiteľ môže vykonať opravu základu dane podľa odseku 1 najviac vo výške, ktorá zodpovedá neprijatej protihodnote za dodanie tovaru alebo služby. Rozdiel medzi pôvodným základom dane a opraveným základom dane a rozdiel medzi pôvodnou daňou a opravenou daňou sa uvedie najskôr v daňovom priznaní za zdaňovacie obdobie, v ktorom sa pohľadávka stala nevymožiteľnou podľa odseku 2; pôvodný základ dane sa nemôže znížiť o sumu, ktorú platiteľ prijal v súvislosti s dodaním tovaru alebo služby po tom, ako sa pohľadávka stala nevymožiteľnou podľa odseku 2. (4) Platiteľ nemôže vykonať opravu základu dane pri nevymožiteľnej pohľadávke podľa odseku 2, ak
relationships:
- Uznesenie o skonceni restrukturalizacie -> [MA_DATUM] -> Den zverejnenia uznesenia v Obchodnom vestniku
- Uznesenie o skonceni restrukturalizacie -> [UVADZA] -> Obchodny vestnik
- Pohladavka -> [SUVISI_S] -> Restrukturalizacne konanie
- Pohladavka -> [NESPLNA_PODMIENKY] -> Prihlasenie pohladavky v restrukturalizacnom konani
- Platitel -> [PREUKAZUJE] -> Neuspokojenie pohladavky
- Neuspokojenie pohladavky -> [MA_PODMIENKU] -> Pohladavka by nebola uspokojena ani keby bola prihlasena
- Nevymozitelna pohladavka -> [MA_PODMIENKU] -> Pohladavka nebola premlcana ku dnu zverejnenia uznesenia
- Pohladavka -> [STAVA_SA] -> Nevymozitelna pohladavka
- Nevymozitelna pohladavka -> [NASTAVA_PRI] -> Zverejnenie uznesenia o skonceni restrukturalizacie v Obchodnom vestniku
- Platitel -> [MA_PRAVO] -> Oprava zakladu dane
- Oprava zakladu dane -> [MA_SUMU] -> Vyska zodpovedajuca neprijatej protihodnote
- Neprijata protihodnota -> [SUVISI_S] -> Dodanie tovaru alebo sluzby
- Rozdiel medzi povodnym zakladom dane a opravenym zakladom dane -> [UVADZA] -> Danove priznanie
- Rozdiel medzi povodnou danou a opravenou danou -> [UVADZA] -> Danove priznanie
- Danove priznanie -> [MA_OBDOBIE] -> Zdanovacie obdobie
- Zdanovacie obdobie -> [NASTAVA_PRI] -> Pohladavka sa stala nevymozitelnou
- Povodny zaklad dane -> [NEMA_NAROK_NA] -> Znizenie o sumu prijatu po vzniku nevymozitelnosti pohladavky
- Suma prijata po vzniku nevymozitelnosti pohladavky -> [SUVISI_S] -> Dodanie tovaru alebo sluzby

nodes:
- Doklad: Uznesenie o skonceni restrukturalizacie
- Doklad: Obchodny vestnik
- Datum: Den zverejnenia uznesenia v Obchodnom vestniku
- Pohladavka: Pohladavka
- Pohladavka: Nevymozitelna pohladavka
- Konanie: Restrukturalizacne konanie
- Podmienka: Prihlasenie pohladavky v restrukturalizacnom konani
- Podmienka: Pohladavka by nebola uspokojena ani keby bola prihlasena
- Podmienka: Pohladavka nebola premlcana ku dnu zverejnenia uznesenia
- ZdanitelnaOsoba: Platitel
- Oprava: Oprava zakladu dane
- Suma: Vyska zodpovedajuca neprijatej protihodnote
- Hodnota: Neprijata protihodnota
- Cinnost: Dodanie tovaru alebo sluzby
- Hodnota: Povodny zaklad dane
- Hodnota: Opraveny zaklad dane
- Dan: Povodna dan
- Dan: Opravena dan
- Hodnota: Rozdiel medzi povodnym zakladom dane a opravenym zakladom dane
- Hodnota: Rozdiel medzi povodnou danou a opravenou danou
- DanovePriznanie: Danove priznanie
- ZdanovacieObdobie: Zdanovacie obdobie
- Cinnost: Pohladavka sa stala nevymozitelnou
- Suma: Suma prijata po vzniku nevymozitelnosti pohladavky
- Cinnost: Zverejnenie uznesenia o skonceni restrukturalizacie v Obchodnom vestniku
- Cinnost: Znizenie o sumu prijatu po vzniku nevymozitelnosti pohladavky


chunk: 205
page: 52
text: daňový úrad určí na základe skladovacej kapacity osobitného skladu, sadzby dane prislúchajúcej k tovaru a poslednej zverejnenej priemernej nominálnej ceny tovaru na webovom sídle Svetovej banky pred podaním žiadosti podľa odseku 3 prepočítanej na eurá; na prepočet sa použije referenčný výmenný kurz určený a vyhlásený Európskou centrálnou bankou v prvý deň kalendárneho mesiaca, v ktorom daňový úrad vydal rozhodnutie, alebo nasledujúci deň, ak nebol v tento deň referenčný výmenný kurz určený a vyhlásený. Zábezpeka na daň sa skladá zložením peňažných prostriedkov na účet daňového úradu alebo bankovou zárukou poskytnutou bankou bez výhrad. Zo zloženej zábezpeky na daň platiteľ nemá nárok na úroky. (8) Daňový úrad pred vydaním povolenia na prevádzkovanie osobitného skladu preverí skutočnosti a údaje podľa odsekov 4 až 6, a ak sú pravdivé a správne a platiteľ zložil zábezpeku podľa odseku 7, vydá povolenie na prevádzkovanie osobitného skladu bezodkladne.
relationships:
- Danovy urad -> [URCUJE] -> Zabezpeka na dan
- Zabezpeka na dan -> [VYCHADZA_Z] -> Skladovacia kapacita osobitneho skladu
- Skladovacia kapacita osobitneho skladu -> [TYKA_SA] -> Osobitny sklad
- Zabezpeka na dan -> [VYCHADZA_Z] -> Sadzba dane prisluchajuca k tovaru
- Sadzba dane prisluchajuca k tovaru -> [TYKA_SA] -> Tovar
- Zabezpeka na dan -> [VYCHADZA_Z] -> Posledna zverejnena priemerna nominalna cena tovaru
- Posledna zverejnena priemerna nominalna cena tovaru -> [TYKA_SA] -> Tovar
- Posledna zverejnena priemerna nominalna cena tovaru -> [MA_MIESTO] -> Webove sidlo Svetovej banky
- Webove sidlo Svetovej banky -> [PATRI_DO] -> Svetova banka
- Posledna zverejnena priemerna nominalna cena tovaru -> [MA_DOBU] -> Pred podanim ziadosti
- Podanie ziadosti -> [JE_PODLA] -> Odsek 3
- Posledna zverejnena priemerna nominalna cena tovaru -> [MA_MENU] -> Euro
- Prepocet na eura -> [VYCHADZA_Z] -> Referencny vymenny kurz
- Europska centralna banka -> [URCUJE] -> Referencny vymenny kurz
- Europska centralna banka -> [VYDAVA] -> Referencny vymenny kurz
- Referencny vymenny kurz -> [MA_DATUM] -> Prvy den kalendarneho mesiaca
- Prvy den kalendarneho mesiaca -> [TYKA_SA] -> Kalendarny mesiac vydania rozhodnutia
- Danovy urad -> [VYDAVA] -> Rozhodnutie
- Rozhodnutie -> [MA_DATUM] -> Kalendarny mesiac vydania rozhodnutia
- Referencny vymenny kurz -> [MA_PODMIENKU] -> Kurz nebol urceny a vyhlaseny v prvy den kalendarneho mesiaca
- Kurz nebol urceny a vyhlaseny v prvy den kalendarneho mesiaca -> [PODMIENUJE] -> Pouzitie kurzu nasledujuci den

- Zabezpeka na dan -> [VZNIKA_PRI] -> Zlozenie penaznych prostriedkov
- Zlozenie penaznych prostriedkov -> [MA_MIESTO] -> Ucet danoveho uradu
- Ucet danoveho uradu -> [PATRI_DO] -> Danovy urad
- Zabezpeka na dan -> [VZNIKA_PRI] -> Poskytnutie bankovej zaruky
- Banka -> [POSKYTUJE] -> Bankova zaruka
- Bankova zaruka -> [MA_PODMIENKU] -> Bez vyhrad
- Platitel -> [NEMA_NAROK_NA] -> Uroky
- Uroky -> [TYKA_SA] -> Zlozena zabezpeka na dan

- Odsek 8 -> [UPRAVUJE] -> Vydanie povolenia na prevadzkovanie osobitneho skladu
- Danovy urad -> [VYKONAVA] -> Preverenie skutocnosti a udajov
- Preverenie skutocnosti a udajov -> [ODKAZUJE_NA] -> Odsek 4
- Preverenie skutocnosti a udajov -> [ODKAZUJE_NA] -> Odsek 5
- Preverenie skutocnosti a udajov -> [ODKAZUJE_NA] -> Odsek 6
- Vydanie povolenia na prevadzkovanie osobitneho skladu -> [MA_PODMIENKU] -> Skutocnosti a udaje su pravdive a spravne
- Vydanie povolenia na prevadzkovanie osobitneho skladu -> [MA_PODMIENKU] -> Platitel zlozil zabezpeku podla odseku 7
- Platitel zlozil zabezpeku podla odseku 7 -> [ODKAZUJE_NA] -> Odsek 7
- Danovy urad -> [VYDAVA] -> Povolenie na prevadzkovanie osobitneho skladu
- Povolenie na prevadzkovanie osobitneho skladu -> [TYKA_SA] -> Osobitny sklad
- Danovy urad -> [MA_LEHOTU] -> Bezodkladne

nodes:
- SpravcaDane: Danovy urad
- ZabezpekaNaDan: Zabezpeka na dan
- Hodnota: Skladovacia kapacita osobitneho skladu
- Majetok: Osobitny sklad
- SadzbaDane: Sadzba dane prisluchajuca k tovaru
- Tovar: Tovar
- Hodnota: Posledna zverejnena priemerna nominalna cena tovaru
- ElektronickyProstriedok: Webove sidlo Svetovej banky
- Organizacia: Svetova banka
- CasovyUdaj: Pred podanim ziadosti
- Cinnost: Podanie ziadosti
- Ziadost: Ziadost
- Mena: Euro
- Vypocet: Prepocet na eura
- Kurz: Referencny vymenny kurz
- Banka: Europska centralna banka
- Datum: Prvy den kalendarneho mesiaca
- Obdobie: Kalendarny mesiac vydania rozhodnutia
- Rozhodnutie: Rozhodnutie
- Podmienka: Kurz nebol urceny a vyhlaseny v prvy den kalendarneho mesiaca
- Cinnost: Pouzitie kurzu nasledujuci den
- Datum: Nasledujuci den
- Cinnost: Zlozenie penaznych prostriedkov
- Mnozstvo: Penazne prostriedky
- BankovyUcet: Ucet danoveho uradu
- Cinnost: Poskytnutie bankovej zaruky
- Zavazok: Bankova zaruka
- Banka: Banka
- Podmienka: Bez vyhrad
- ZdanitelnaOsoba: Platitel
- Urok: Uroky
- ZabezpekaNaDan: Zlozena zabezpeka na dan
- Odsek: Odsek 3
- Odsek: Odsek 4
- Odsek: Odsek 5
- Odsek: Odsek 6
- Odsek: Odsek 7
- Odsek: Odsek 8
- Cinnost: Vydanie povolenia na prevadzkovanie osobitneho skladu
- Doklad: Povolenie na prevadzkovanie osobitneho skladu
- Cinnost: Preverenie skutocnosti a udajov
- Podmienka: Skutocnosti a udaje su pravdive a spravne
- Podmienka: Platitel zlozil zabezpeku podla odseku 7
- Lehota: Bezodkladne


chunk: 23
page: 6
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 7 predpisu33) od ktoréhokoľvek člena skupiny. (5) Členom skupiny sa môže stať aj ďalšia zdaniteľná osoba, ak spĺňa podmienky podľa § 4a. Žiadosť o zmenu registrácie skupiny z dôvodu pristúpenia ďalšieho člena do skupiny podáva zástupca skupiny. Ak zdaniteľná osoba spĺňa podmienky podľa § 4a, daňový úrad vydá bezodkladne rozhodnutie o zmene registrácie skupiny, proti ktorému nemožno podať odvolanie; účinky zmeny registrácie skupiny nastávajú k prvému dňu tretieho kalendárneho mesiaca nasledujúceho po kalendárnom mesiaci, v ktorom bola žiadosť o zmenu registrácie skupiny podaná. Platnosť identifikačného čísla pre daň člena skupiny, ktorý pristúpil do skupiny, zaniká dňom, ktorý predchádza dňu, keď nastali účinky zmeny registrácie skupiny. Práva a povinnosti zdaniteľnej osoby, ktorá pristúpila do skupiny, vyplývajúce z tohto zákona prechádzajú na skupinu dňom, keď nastali účinky zmeny registrácie skupiny.
relationships:
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Odsek 5
- Odsek 5 -> [UPRAVUJE] -> Zmena Registracie Skupiny
- Zdanitelna Osoba -> [SPLNA_PODMIENKY] -> Podmienky
- Podmienky -> [ODKAZUJE_NA] -> § 4A
- Zdanitelna Osoba -> [STAVA_SA] -> Clen Skupiny
- Clen Skupiny -> [JE_CLENOM] -> Skupina

- Zastupca Skupiny -> [PODAVA] -> Ziadost O Zmenu Registracie Skupiny
- Ziadost O Zmenu Registracie Skupiny -> [MA_DOVOD] -> Pristupenie Dalsieho Clena Do Skupiny
- Pristupenie Dalsieho Clena Do Skupiny -> [TYKA_SA] -> Clen Skupiny
- Pristupenie Dalsieho Clena Do Skupiny -> [TYKA_SA] -> Skupina

- Danovy Urad -> [VYDAVA] -> Rozhodnutie O Zmene Registracie Skupiny
- Rozhodnutie O Zmene Registracie Skupiny -> [TYKA_SA] -> Zmena Registracie Skupiny
- Danovy Urad -> [MA_LEHOTU] -> Bezodkladne
- Rozhodnutie O Zmene Registracie Skupiny -> [NEMA_NAROK_NA] -> Odvolanie

- Ucinky Zmeny Registracie Skupiny -> [NASTAVA_PRI] -> Prvy Den Tretieho Kalendarneho Mesiaca
- Prvy Den Tretieho Kalendarneho Mesiaca -> [NASTAVA_PRI] -> Kalendarne Mesiace Nasledujuce Po Mesiaci Podania Ziadosti
- Ziadost O Zmenu Registracie Skupiny -> [MA_DATUM] -> Kalendarne Mesiace Podania Ziadosti

- Clen Skupiny -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan
- Identifikacne Cislo Pre Dan -> [ZANIKA] -> Den Predchadzajuci Dnu Ucinkov Zmeny Registracie Skupiny
- Den Predchadzajuci Dnu Ucinkov Zmeny Registracie Skupiny -> [TYKA_SA] -> Ucinky Zmeny Registracie Skupiny

- Zdanitelna Osoba -> [MA_PRAVO] -> Prava Vyplivajuce Zo Zakona
- Zdanitelna Osoba -> [MA_POVINNOST] -> Povinnosti Vyplivajuce Zo Zakona
- Prava Vyplivajuce Zo Zakona -> [VYPLNYVA_Z] -> Zakon 222/2004 Z. Z.
- Povinnosti Vyplivajuce Zo Zakona -> [VYPLNYVA_Z] -> Zakon 222/2004 Z. Z.
- Prava Vyplivajuce Zo Zakona -> [PRECHADZA_NA] -> Skupina
- Povinnosti Vyplivajuce Zo Zakona -> [PRECHADZA_NA] -> Skupina
- Prava Vyplivajuce Zo Zakona -> [MA_DATUM] -> Den Ucinkov Zmeny Registracie Skupiny
- Povinnosti Vyplivajuce Zo Zakona -> [MA_DATUM] -> Den Ucinkov Zmeny Registracie Skupiny

nodes:
- Zakon: Zakon 222/2004 Z. Z.
- Odsek: Odsek 5
- Paragraf: § 4A
- Podmienka: Podmienky
- ZdanitelnaOsoba: Zdanitelna Osoba
- ClenSkupiny: Clen Skupiny
- Skupina: Skupina
- Zastupca: Zastupca Skupiny
- Ziadost: Ziadost O Zmenu Registracie Skupiny
- Dovod: Pristupenie Dalsieho Clena Do Skupiny
- Registracia: Zmena Registracie Skupiny
- Rozhodnutie: Rozhodnutie O Zmene Registracie Skupiny
- SpravcaDane: Danovy Urad
- Lehota: Bezodkladne
- Konanie: Odvolanie
- Cinnost: Ucinky Zmeny Registracie Skupiny
- Datum: Prvy Den Tretieho Kalendarneho Mesiaca
- Obdobie: Kalendarne Mesiace Nasledujuce Po Mesiaci Podania Ziadosti
- Obdobie: Kalendarne Mesiace Podania Ziadosti
- IdentifikacneCislo: Identifikacne Cislo Pre Dan
- Datum: Den Predchadzajuci Dnu Ucinkov Zmeny Registracie Skupiny
- Datum: Den Ucinkov Zmeny Registracie Skupiny
- Pravo: Prava Vyplivajuce Zo Zakona
- Povinnost: Povinnosti Vyplivajuce Zo Zakona


chunk: 254
page: 63
text: účely podnikania, ako aj na iný účel ako na podnikanie. Platiteľ postupuje pri úprave odpočítanej dane podľa prílohy č. 1. Na účely výpočtu úpravy dane odpočítanej pri investičnom majetku sa
relationships:
- Platitel -> [VYKONAVA] -> Uprava Odpocitanej Dane
- Uprava Odpocitanej Dane -> [JE_PODLA] -> Priloha C. 1
- Uprava Odpocitanej Dane -> [TYKA_SA] -> Dan Odpocitana
- Vypocet Upravy Dane Odpocitanej Pri Investicnom Majetku -> [TYKA_SA] -> Uprava Odpocitanej Dane
- Vypocet Upravy Dane Odpocitanej Pri Investicnom Majetku -> [TYKA_SA] -> Investicny Majetok
- Investicny Majetok -> [MA_UCEL] -> Ucely Podnikania
- Investicny Majetok -> [MA_UCEL] -> Iny Ucel Ako Na Podnikanie

nodes:
- ZdanitelnaOsoba: Platitel
- Oprava: Uprava Odpocitanej Dane
- Priloha: Priloha C. 1
- Dan: Dan Odpocitana
- Vypocet: Vypocet Upravy Dane Odpocitanej Pri Investicnom Majetku
- InvesticnyMajetok: Investicny Majetok
- Cinnost: Ucely Podnikania
- Cinnost: Iny Ucel Ako Na Podnikanie


chunk: 27
page: 6
text: (2) Ak z podkladov získaných pri výkone správy daní alebo na základe výsledkov činnosti Finančného riaditeľstva Slovenskej republiky4f) (ďalej len „finančné riaditeľstvo“) vyplýva, že nastali dôvody na registráciu skupiny z úradnej moci, Daňový úrad Banská Bystrica vyzve predpokladaných členov skupiny, aby do ôsmich dní odo dňa, keď bola táto výzva doručená poslednému z týchto predpokladaných členov skupiny, spomedzi seba určili spoločného zástupcu na účely registračného konania z úradnej moci. Ak tak neurobia, Daňový úrad Banská Bystrica spomedzi nich určí spoločného zástupcu a oznámi to predpokladaným členom skupiny; proti
relationships:
- Odsek 2 -> [UPRAVUJE] -> Registracia Skupiny Z Uradnej Moci
- Dovody Na Registraciu Skupiny Z Uradnej Moci -> [VYPLNYVA_Z] -> Podklady Ziskane Pri Vykone Spravy Dani
- Dovody Na Registraciu Skupiny Z Uradnej Moci -> [VYPLNYVA_Z] -> Vysledky Cinnosti Financneho Riaditelstva Slovenskej Republiky
- Podklady Ziskane Pri Vykone Spravy Dani -> [NASTAVA_PRI] -> Vykon Spravy Dani
- Vysledky Cinnosti Financneho Riaditelstva Slovenskej Republiky -> [TYKA_SA] -> Financne Riaditelstvo Slovenskej Republiky
- Dovody Na Registraciu Skupiny Z Uradnej Moci -> [TYKA_SA] -> Registracia Skupiny Z Uradnej Moci

- Danovy Urad Banska Bystrica -> [VYDAVA] -> Vyzva Na Urcenie Spolocneho Zastupcu
- Vyzva Na Urcenie Spolocneho Zastupcu -> [TYKA_SA] -> Predpokladani Clenovia Skupiny
- Vyzva Na Urcenie Spolocneho Zastupcu -> [MA_OBSAH] -> Urcenie Spolocneho Zastupcu
- Vyzva Na Urcenie Spolocneho Zastupcu -> [MA_LEHOTU] -> Osem Dni
- Osem Dni -> [NASTAVA_PRI] -> Dorucenie Vyzvy Poslednemu Predpokladanemu Clenovi Skupiny
- Dorucenie Vyzvy Poslednemu Predpokladanemu Clenovi Skupiny -> [TYKA_SA] -> Predpokladani Clenovia Skupiny

- Predpokladani Clenovia Skupiny -> [MA_POVINNOST] -> Urcenie Spolocneho Zastupcu
- Urcenie Spolocneho Zastupcu -> [MA_UCEL] -> Registracne Konanie Z Uradnej Moci
- Spolocny Zastupca -> [MA_UCEL] -> Registracne Konanie Z Uradnej Moci

- Danovy Urad Banska Bystrica -> [URCUJE] -> Spolocny Zastupca
- Urcenie Spolocneho Zastupcu Danovym Uradom -> [MA_PODMIENKU] -> Predpokladani Clenovia Skupiny Neurcili Spolocneho Zastupcu
- Danovy Urad Banska Bystrica -> [OZNAMUJE] -> Urcenie Spolocneho Zastupcu Danovym Uradom
- Urcenie Spolocneho Zastupcu Danovym Uradom -> [TYKA_SA] -> Predpokladani Clenovia Skupiny

nodes:
- Odsek: Odsek 2
- SpravcaDane: Danovy Urad Banska Bystrica
- FinancneRiaditelstvo: Financne Riaditelstvo Slovenskej Republiky
- Registracia: Registracia Skupiny Z Uradnej Moci
- Dovod: Dovody Na Registraciu Skupiny Z Uradnej Moci
- Doklad: Podklady Ziskane Pri Vykone Spravy Dani
- Cinnost: Vykon Spravy Dani
- Zaznam: Vysledky Cinnosti Financneho Riaditelstva Slovenskej Republiky
- ClenSkupiny: Predpokladani Clenovia Skupiny
- Vyzva: Vyzva Na Urcenie Spolocneho Zastupcu
- Lehota: Osem Dni
- Cinnost: Dorucenie Vyzvy Poslednemu Predpokladanemu Clenovi Skupiny
- Cinnost: Urcenie Spolocneho Zastupcu
- Zastupca: Spolocny Zastupca
- Konanie: Registracne Konanie Z Uradnej Moci
- Podmienka: Predpokladani Clenovia Skupiny Neurcili Spolocneho Zastupcu
- Cinnost: Urcenie Spolocneho Zastupcu Danovym Uradom


chunk: 286
page: 71
text: dovoze tovaru za podmienok uvedených v odseku 2. (2) Zahraničná osoba z tretieho štátu má nárok na vrátenie dane, ak a) je identifikovaná pre daň alebo obdobnú všeobecnú daň zo spotreby v štáte, v ktorom má sídlo, miesto podnikania, prevádzkareň, bydlisko alebo v ktorom sa obvykle zdržiava, b) v období, za ktoré podáva žiadosť o vrátenie dane, nemala na území Európskej únie sídlo, miesto podnikania, prevádzkareň ani bydlisko a ani sa na tomto území obvykle nezdržiavala, c) v období, za ktoré podáva žiadosť o vrátenie dane, nedodala tovar ani službu v tuzemsku s výnimkou dodania 1. prepravných služieb a s nimi súvisiacich doplnkových služieb, ktoré sú oslobodené od dane podľa § 47 ods. 6, 8, 10 a 12 a § 48 ods. 8, 2. služieb a dodania tovaru, ak je osobou povinnou platiť daň príjemca (§ 69 ods. 2, 3 a 12), 3. tovaru podľa § 13 ods. 1 písm. e) a f), ak je osobou povinnou platiť daň osoba, ktorej je tento tovar dodaný (§ 69 ods. 9),
relationships:
- Odsek 2 -> [UPRAVUJE] -> Narok Na Vratenie Dane
- Odsek 2 -> [MA_PISMENO] -> Pismeno a)
- Odsek 2 -> [MA_PISMENO] -> Pismeno b)
- Odsek 2 -> [MA_PISMENO] -> Pismeno c)

- Zahranicna Osoba Z Tretieho Statu -> [MA_NAROK_NA] -> Vratenie Dane
- Zahranicna Osoba Z Tretieho Statu -> [PODAVA] -> Ziadost O Vratenie Dane
- Ziadost O Vratenie Dane -> [TYKA_SA] -> Vratenie Dane
- Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Obdobie Za Ktore Sa Podava Ziadost O Vratenie Dane

- Narok Na Vratenie Dane -> [MA_PODMIENKU] -> Identifikacia Pre Dan Alebo Obdobnu Vseobecnu Dan Zo Spotreby
- Identifikacia Pre Dan Alebo Obdobnu Vseobecnu Dan Zo Spotreby -> [MA_MIESTO] -> Stat Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania
- Pismeno A -> [VYMEDZUJE] -> Identifikacia Pre Dan Alebo Obdobnu Vseobecnu Dan Zo Spotreby

- Narok Na Vratenie Dane -> [MA_PODMIENKU] -> Absencia Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania Na Uzemi Europskej Unie
- Absencia Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania Na Uzemi Europskej Unie -> [MA_OBDOBIE] -> Obdobie Za Ktore Sa Podava Ziadost O Vratenie Dane
- Absencia Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania Na Uzemi Europskej Unie -> [TYKA_SA] -> Uzemie Europskej Unie
- Pismeno b) -> [VYMEDZUJE] -> Absencia Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania Na Uzemi Europskej Unie

- Narok Na Vratenie Dane -> [MA_PODMIENKU] -> Nedodanie Tovaru Ani Sluzby V Tuzemsku
- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [MA_OBDOBIE] -> Obdobie Za Ktore Sa Podava Ziadost O Vratenie Dane
- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [MA_MIESTO] -> Tuzemsko
- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [TYKA_SA] -> Tovar
- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [TYKA_SA] -> Sluzba
- Pismeno c) -> [VYMEDZUJE] -> Nedodanie Tovaru Ani Sluzby V Tuzemsku

- Pismeno c) -> [OBSAHUJE] -> Bod 1
- Pismeno c) -> [OBSAHUJE] -> Bod 2
- Pismeno c) -> [OBSAHUJE] -> Bod 3

- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [MA_VYNIMKU] -> Dodanie Prepravnych Sluzieb A Suvisiacich Doplnkovych Sluzieb
- Dodanie Prepravnych Sluzieb A Suvisiacich Doplnkovych Sluzieb -> [TYKA_SA] -> Prepravne Sluzby A Suvisiace Doplnkove Sluzby
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_OSLOBODENE_OD] -> Dan
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_PODLA] -> § 47 Odsek 6
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_PODLA] -> § 47 Odsek 8
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_PODLA] -> § 47 Odsek 10
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_PODLA] -> § 47 Odsek 12
- Prepravne Sluzby A Suvisiace Doplnkove Sluzby -> [JE_PODLA] -> § 48 Odsek 8

- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [MA_VYNIMKU] -> Dodanie Sluzieb A Tovaru Ak Dan Plati Prijemca
- Dodanie Sluzieb A Tovaru Ak Dan Plati Prijemca -> [TYKA_SA] -> Sluzba
- Dodanie Sluzieb A Tovaru Ak Dan Plati Prijemca -> [TYKA_SA] -> Tovar
- Prijemca -> [JE_POVINNY_PLATIT] -> Dan
- Prijemca -> [JE_PODLA] -> § 69 Odsek 2
- Prijemca -> [JE_PODLA] -> § 69 Odsek 3
- Prijemca -> [JE_PODLA] -> § 69 Odsek 12

- Nedodanie Tovaru Ani Sluzby V Tuzemsku -> [MA_VYNIMKU] -> Dodanie Tovaru
- Dodanie Tovaru -> [TYKA_SA] -> Tovar
- Dodanie Tovaru -> [JE_PODLA] -> § 13 Odsek 1 Pismeno e)
- Dodanie Tovaru -> [JE_PODLA] -> § 13 Odsek 1 Pismeno f)
- Osoba Ktorej Je Tovar Dodany -> [JE_POVINNY_PLATIT] -> Dan
- Osoba Ktorej Je Tovar Dodany -> [JE_PODLA] -> § 69 Odsek 9

- § 47 -> [MA_ODSEK] -> § 47 Odsek 6
- § 47 -> [MA_ODSEK] -> § 47 Odsek 8
- § 47 -> [MA_ODSEK] -> § 47 Odsek 10
- § 47 -> [MA_ODSEK] -> § 47 Odsek 12
- § 48 -> [MA_ODSEK] -> § 48 Odsek 8
- § 69 -> [MA_ODSEK] -> § 69 Odsek 2
- § 69 -> [MA_ODSEK] -> § 69 Odsek 3
- § 69 -> [MA_ODSEK] -> § 69 Odsek 12
- § 69 -> [MA_ODSEK] -> § 69 Odsek 9
- § 13 -> [MA_ODSEK] -> § 13 Odsek 1
- § 13 Ods. 1 -> [MA_PISMENO] -> § 13 Odsek 1 Pismeno e)
- § 13 Ods. 1 -> [MA_PISMENO] -> § 13 Odsek 1 Pismeno f)

nodes:
- Odsek: Odsek 2
- Pismeno: Pismeno A
- Pismeno: Pismeno B
- Pismeno: Pismeno C
- Bod: Bod 1
- Bod: Bod 2
- Bod: Bod 3
- Osoba: Zahranicna Osoba Z Tretieho Statu
- Pravo: Narok Na Vratenie Dane
- Dan: Vratenie Dane
- Dan: Dan
- Ziadost: Ziadost O Vratenie Dane
- ZdanovacieObdobie: Obdobie Za Ktore Sa Podava Ziadost O Vratenie Dane
- Podmienka: Identifikacia Pre Dan Alebo Obdobnu Vseobecnu Dan Zo Spotreby
- Stat: Stat Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania
- Podmienka: Absencia Sidla Miesta Podnikania Prevadzkarne Bydliska Alebo Obvykleho Zdrziavania Na Uzemi Europskej Unie
- Uzemie: Uzemie Europskej Unie
- Podmienka: Nedodanie Tovaru Ani Sluzby V Tuzemsku
- Tuzemsko: Tuzemsko
- Tovar: Tovar
- Sluzba: Sluzba
- Cinnost: Dodanie Prepravnych Sluzieb A Suvisiacich Doplnkovych Sluzieb
- Sluzba: Prepravne Sluzby A Suvisiace Doplnkove Sluzby
- Cinnost: Dodanie Sluzieb A Tovaru Ak Dan Plati Prijemca
- Subjekt: Prijemca
- Cinnost: Dodanie Tovaru Podla § 13 Odsek 1 Pismeno E A F
- Subjekt: Osoba Ktorej Je Tovar Dodany
- Paragraf: § 47
- Paragraf: § 48
- Paragraf: § 69
- Paragraf: § 13
- Odsek: § 47 Odsek 6
- Odsek: § 47 Odsek 8
- Odsek: § 47 Odsek 10
- Odsek: § 47 Odsek 12
- Odsek: § 48 Odsek 8
- Odsek: § 69 Odsek 2
- Odsek: § 69 Odsek 3
- Odsek: § 69 Odsek 12
- Odsek: § 69 Odsek 9
- Odsek: § 13 Odsek 1
- Pismeno: § 13 Odsek 1 Pismeno e)
- Pismeno: § 13 Odsek 1 Pismeno f)


chunk: 309
page: 77
text: umožnené vrátenie dane podľa odsekov 4, 5, 7, 9, 10 a 11, pričom nárok na vrátenie dane sa pomerne zníži. Podrobnosti uplatnenia oslobodenia od dane a vrátenia dane ustanoví opatrenie, ktoré vydá Ministerstvo financií Slovenskej republiky a ktoré sa vyhlasuje v Zbierke zákonov Slovenskej republiky. § 62 (1) Zahraničný zástupca uplatňuje nárok na vrátenie dane podaním žiadosti o vrátenie dane Daňovému úradu Bratislava na tlačive, ktorého vzor je uvedený v prílohe č. 4. Ministerstvo zahraničných vecí a európskych záležitostí Slovenskej republiky potvrdí elektronicky Daňovému úradu Bratislava splnenie podmienky vzájomnosti podľa § 61 ods. 3. Žiadosť o vrátenie dane sa podáva za obdobie kalendárneho štvrťroka do 30. dňa po skončení kalendárneho štvrťroka. (2) Zahraničný zástupca k žiadosti o vrátenie dane musí doložiť originál faktúry alebo iného dokladu o kúpe tovaru alebo služby od platiteľa, v ktorom je uvedená suma dane v eurách
relationships:
- Vratenie Dane -> [JE_PODLA] -> Odsek 4
- Vratenie Dane -> [JE_PODLA] -> Odsek 5
- Vratenie Dane -> [JE_PODLA] -> Odsek 7
- Vratenie Dane -> [JE_PODLA] -> Odsek 9
- Vratenie Dane -> [JE_PODLA] -> Odsek 10
- Vratenie Dane -> [JE_PODLA] -> Odsek 11
- Narok Na Vratenie Dane -> [MA_VLASTNOST] -> Pomerne Znizenie Naroku Na Vratenie Dane

- Opatrenie -> [UPRAVUJE] -> Podrobnosti Uplatnenia Oslobodenia Od Dane A Vratenia Dane
- Podrobnosti Uplatnenia Oslobodenia Od Dane A Vratenia Dane -> [TYKA_SA] -> Oslobodenie Od Dane
- Podrobnosti Uplatnenia Oslobodenia Od Dane A Vratenia Dane -> [TYKA_SA] -> Vratenie Dane
- Ministerstvo Financii Slovenskej Republiky -> [VYDAVA] -> Opatrenie
- Zbierka Zakonov Slovenskej Republiky -> [OBSAHUJE] -> Opatrenie

- Paragraf § 62 -> [OBSAHUJE] -> Odsek 1
- Paragraf § 62 -> [OBSAHUJE] -> Odsek 2
- Odsek 1 -> [JE_PODLA] -> Paragraf § 62
- Odsek 2 -> [JE_PODLA] -> Paragraf § 62

- Zahranicny Zastupca -> [MA_NAROK_NA] -> Vratenie Dane
- Zahranicny Zastupca -> [PODAVA] -> Ziadost O Vratenie Dane
- Ziadost O Vratenie Dane -> [TYKA_SA] -> Vratenie Dane
- Ziadost O Vratenie Dane -> [DORUCUJE] -> Danovy Urad Bratislava
- Ziadost O Vratenie Dane -> [MA_DOKLAD] -> Tlacivo Ziadosti O Vratenie Dane
- Tlacivo Ziadosti O Vratenie Dane -> [JE_PODLA] -> Priloha C. 4

- Ministerstvo Zahranicnych Veci A Europskych Zalezitosti Slovenskej Republiky -> [OZNAMUJE] -> Splnenie Podmienky Vzajomnosti
- Splnenie Podmienky Vzajomnosti -> [TYKA_SA] -> Danovy Urad Bratislava
- Splnenie Podmienky Vzajomnosti -> [JE_PODLA] -> § 61 Odsek 3
- Splnenie Podmienky Vzajomnosti -> [MA_VLASTNOST] -> Elektronicke Potvrdenie

- Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Kalendarny Stvrtrok
- Ziadost O Vratenie Dane -> [MA_LEHOTU] -> 30. Den Po Skonceni Kalendarneho Stvrtroka

- Zahranicny Zastupca -> [MA_POVINNOST] -> Dolozenie Originalu Faktury Alebo Ineho Dokladu O Kupe
- Ziadost O Vratenie Dane -> [MA_DOKLAD] -> Original Faktury
- Ziadost O Vratenie Dane -> [MA_DOKLAD] -> Iny Doklad O Kupe Tovaru Alebo Sluzby
- Original Faktury -> [TYKA_SA] -> Kupa Tovaru Alebo Sluzby Od Platitela
- Iny Doklad O Kupe Tovaru Alebo Sluzby -> [TYKA_SA] -> Kupa Tovaru Alebo Sluzby Od Platitela
- Kupa Tovaru Alebo Sluzby Od Platitela -> [JE_PREDMETOM] -> Tovar
- Kupa Tovaru Alebo Sluzby Od Platitela -> [JE_PREDMETOM] -> Sluzba
- Kupa Tovaru Alebo Sluzby Od Platitela -> [TYKA_SA] -> Platitel
- Original Faktury -> [MA_SUMU] -> Suma Dane V Eurach
- Iny Doklad O Kupe Tovaru Alebo Sluzby -> [MA_SUMU] -> Suma Dane V Eurach
- Suma Dane V Eurach -> [MA_VLASTNOST] -> Euro

nodes:
- Dan: Vratenie Dane
- Pravo: Narok Na Vratenie Dane
- Hodnota: Pomerne Znizenie Naroku Na Vratenie Dane
- PravnyPredpis: Opatrenie
- Ministerstvo: Ministerstvo Financii Slovenskej Republiky
- PravnyPredpis: Zbierka Zakonov Slovenskej Republiky
- Cinnost: Podrobnosti Uplatnenia Oslobodenia Od Dane A Vratenia Dane
- OslobodenieOdDane: Oslobodenie Od Dane
- Paragraf: Paragraf § 62
- Paragraf: Paragraf § 61
- Odsek: Odsek 1
- Odsek: Odsek 2
- Odsek: Odsek 4
- Odsek: Odsek 5
- Odsek: Odsek 7
- Odsek: Odsek 9
- Odsek: Odsek 10
- Odsek: Odsek 11
- Odsek: § 61 Odsek 3
- Zastupca: Zahranicny Zastupca
- Ziadost: Ziadost O Vratenie Dane
- SpravcaDane: Danovy Urad Bratislava
- Doklad: Tlacivo Ziadosti O Vratenie Dane
- Priloha: Priloha C. 4
- Ministerstvo: Ministerstvo Zahranicnych Veci A Europskych Zalezitosti Slovenskej Republiky
- Podmienka: Splnenie Podmienky Vzajomnosti
- ElektronickyProstriedok: Elektronicke Potvrdenie
- Obdobie: Kalendarny Stvrtrok
- Lehota: 30. Den Po Skonceni Kalendarneho Stvrtroka
- Povinnost: Dolozenie Originalu Faktury Alebo Ineho Dokladu O Kupe
- Doklad: Original Faktury
- Doklad: Iny Doklad O Kupe Tovaru Alebo Sluzby
- Cinnost: Kupa Tovaru Alebo Sluzby Od Platitela
- Tovar: Tovar
- Sluzba: Sluzba
- ZdanitelnaOsoba: Platitel
- Suma: Suma Dane V Eurach
- Mena: Euro


chunk: 341
page: 85
text: (3) Ak zdaniteľná osoba nespĺňa podmienky na uplatňovanie osobitnej úpravy, Daňový úrad Bratislava vydá rozhodnutie o tom, že jej nepovoľuje uplatňovanie osobitnej úpravy; proti tomuto rozhodnutiu môže zdaniteľná osoba podať odvolanie. (4) Zdaniteľná osoba neusadená na území Európskej únie, ktorá má povolenie podľa odseku 2, je povinná uplatňovať osobitnú úpravu na všetky služby podľa § 68 ods. 1 písm. a). (5) Každú zmenu údajov uvedených v oznámení o začatí činnosti podľa odseku 2 je zdaniteľná osoba neusadená na území Európskej únie povinná oznámiť Daňovému úradu Bratislava. (6) Zdaniteľná osoba neusadená na území Európskej únie je povinná oznámiť Daňovému úradu Bratislava skončenie činnosti alebo zmenu činnosti v takom rozsahu, že ďalej nebude spĺňať podmienky na uplatňovanie osobitnej úpravy. (7) Daňový úrad Bratislava zruší zdaniteľnej osobe neusadenej na území Európskej únie povolenie uplatňovať osobitnú úpravu a odníme osobitné identifikačné číslo pre daň, ak
relationships:
- Odsek 3 -> [UPRAVUJE] -> Nepovolenie Uplatnovania Osobitnej Upravy
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [NESPLNA_PODMIENKY] -> Podmienky Na Uplatnovanie Osobitnej Upravy
- Danovy Urad Bratislava -> [VYDAVA] -> Rozhodnutie O Nepovoleni Uplatnovania Osobitnej Upravy
- Rozhodnutie O Nepovoleni Uplatnovania Osobitnej Upravy -> [TYKA_SA] -> Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
- Rozhodnutie O Nepovoleni Uplatnovania Osobitnej Upravy -> [TYKA_SA] -> Uplatnovanie Osobitnej Upravy
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_PRAVO] -> Odvolanie
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [PODAVA] -> Odvolanie
- Odvolanie -> [TYKA_SA] -> Rozhodnutie O Nepovoleni Uplatnovania Osobitnej Upravy

- Odsek 4 -> [UPRAVUJE] -> Povinnost Uplatnovat Osobitnu Upravu
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_DOKLAD] -> Povolenie
- Povolenie -> [JE_PODLA] -> Odsek 2
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_POVINNOST] -> Povinnost Uplatnovat Osobitnu Upravu
- Povinnost Uplatnovat Osobitnu Upravu -> [TYKA_SA] -> Osobitna Uprava
- Povinnost Uplatnovat Osobitnu Upravu -> [VZTAHUJE_SA_NA] -> Vsetky Sluzby

- Vsetky Sluzby -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno a)
- Paragraf § 68 -> [MA_ODSEK] -> § 68 Odsek 1
- § 68 Odsek 1 -> [MA_PISMENO] -> § 68 Odsek 1 Pismeno a)

- Odsek 5 -> [UPRAVUJE] -> Oznamenie Zmeny Udajov
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_POVINNOST] -> Oznamenie Zmeny Udajov
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [OZNAMUJE] -> Zmena Udajov Uvedenych V Oznameni O Zacati Cinnosti
- Zmena Udajov Uvedenych V Oznameni O Zacati Cinnosti -> [TYKA_SA] -> Oznamenie O Zacati Cinnosti
- Oznamenie O Zacati Cinnosti -> [JE_PODLA] -> Odsek 2
- Zmena Udajov Uvedenych V Oznameni O Zacati Cinnosti -> [TYKA_SA] -> Danovy Urad Bratislava

- Odsek 6 -> [UPRAVUJE] -> Oznamenie Skoncenia Cinnosti Alebo Zmeny Cinnosti
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [MA_POVINNOST] -> Oznamenie Skoncenia Cinnosti Alebo Zmeny Cinnosti
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [OZNAMUJE] -> Skoncenie Cinnosti
- Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie -> [OZNAMUJE] -> Zmena Cinnosti
- Skoncenie Cinnosti -> [TYKA_SA] -> Danovy Urad Bratislava
- Zmena Cinnosti -> [TYKA_SA] -> Danovy Urad Bratislava
- Zmena Cinnosti -> [NESPLNA_PODMIENKY] -> Podmienky Na Uplatnovanie Osobitnej Upravy

- Odsek 7 -> [UPRAVUJE] -> Zrusenie Povolenia Uplatnovat Osobitnu Upravu
- Danovy Urad Bratislava -> [ZRUSUJE] -> Povolenie Uplatnovat Osobitnu Upravu
- Povolenie Uplatnovat Osobitnu Upravu -> [TYKA_SA] -> Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
- Povolenie Uplatnovat Osobitnu Upravu -> [TYKA_SA] -> Osobitna Uprava
- Danovy Urad Bratislava -> [ZRUSUJE] -> Osobitne Identifikacne Cislo Pre Dan
- Osobitne Identifikacne Cislo Pre Dan -> [TYKA_SA] -> Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie

nodes:
- SpravcaDane: Danovy Urad Bratislava
- ZdanitelnaOsoba: Zdanitelna Osoba Neusadena Na Uzemi Europskej Unie
- Uzemie: Uzemie Europskej Unie
- Status: Osobitna Uprava
- Cinnost: Uplatnovanie Osobitnej Upravy
- Podmienka: Podmienky Na Uplatnovanie Osobitnej Upravy
- Cinnost: Nepovolenie Uplatnovania Osobitnej Upravy
- Rozhodnutie: Rozhodnutie O Nepovoleni Uplatnovania Osobitnej Upravy
- Konanie: Odvolanie
- Povinnost: Povinnost Uplatnovat Osobitnu Upravu
- Doklad: Povolenie
- Doklad: Povolenie Uplatnovat Osobitnu Upravu
- Sluzba: Vsetky Sluzby
- Paragraf: Paragraf § 68
- Odsek: § 68 Odsek 1
- Pismeno: § 68 Odsek 1 Pismeno a)
- Oznamenie: Oznamenie O Zacati Cinnosti
- Povinnost: Oznamenie Zmeny Udajov
- Cinnost: Zmena Udajov Uvedenych V Oznameni O Zacati Cinnosti
- Povinnost: Oznamenie Skoncenia Cinnosti Alebo Zmeny Cinnosti
- Cinnost: Skoncenie Cinnosti
- Cinnost: Zmena Cinnosti
- Cinnost: Zrusenie Povolenia Uplatnovat Osobitnu Upravu
- IdentifikacneCislo: Osobitne Identifikacne Cislo Pre Dan
- Odsek: Odsek 2
- Odsek: Odsek 3
- Odsek: Odsek 4
- Odsek: Odsek 5
- Odsek: Odsek 6
- Odsek: Odsek 7


chunk: 389
page: 96
text: v ktorom svoje rozhodnutie písomne oznámi daňovému úradu. (11) Platiteľ je povinný skončiť uplatňovanie osobitnej úpravy, ak a) v prebiehajúcom kalendárnom roku dosiahne obrat 100 000 eur, a to posledným dňom zdaňovacieho obdobia, v ktorom dosiahol obrat, b) sa stane členom skupiny, a to dňom, ktorý predchádza dňu, keď sa stal členom skupiny, c) je naňho vyhlásený konkurz alebo vstúpil do likvidácie, a to dňom, ktorý predchádza vyhláseniu konkurzu alebo vstupu do likvidácie, d) sa zrušuje bez likvidácie, a to dňom, ktorý predchádza dňu jeho zániku, e) je fyzickou osobou pokračujúcou v podnikaní po úmrtí platiteľa podľa § 83, a to posledným dňom posledného zdaňovacieho obdobia, v ktorom sa skončí konanie o dedičstve, f) nastane skutočnosť na zmenu registrácie pre daň podľa § 6a ods. 2, a to dňom, ktorý predchádza dňu, keď táto skutočnosť nastala. (12) Dátum skončenia uplatňovania osobitnej úpravy podľa odseku 11 platiteľ alebo jeho právny
relationships:
- Odsek 11 -> [UPRAVUJE] -> Povinnost Skoncit Uplatnovanie Osobitnej Upravy
- Odsek 11 -> [MA_PISMENO] -> Pismeno a)
- Odsek 11 -> [MA_PISMENO] -> Pismeno b)
- Odsek 11 -> [MA_PISMENO] -> Pismeno c)
- Odsek 11 -> [MA_PISMENO] -> Pismeno d)
- Odsek 11 -> [MA_PISMENO] -> Pismeno e)
- Odsek 11 -> [MA_PISMENO] -> Pismeno f)

- Platitel -> [MA_POVINNOST] -> Povinnost Skoncit Uplatnovanie Osobitnej Upravy
- Povinnost Skoncit Uplatnovanie Osobitnej Upravy -> [TYKA_SA] -> Osobitna Uprava

- Pismeno a) -> [VYMEDZUJE] -> Dosiahnutie Obratu 100 000 Eur V Prebiehajucom Kalendarnom Roku
- Dosiahnutie Obratu 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [TYKA_SA] -> Obrat 100 000 Eur
- Obrat 100 000 Eur -> [MA_SUMU] -> 100 000 Eur
- Dosiahnutie Obratu 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [MA_OBDOBIE] -> Prebiehajuci Kalendarny Rok
- Dosiahnutie Obratu 100 000 Eur V Prebiehajucom Kalendarnom Roku -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Posledny Den Zdanovacieho Obdobia Dosiahnutia Obratu

- Pismeno b) -> [VYMEDZUJE] -> Vznik Clenstva Platitela V Skupine
- Platitel -> [STAVA_SA] -> Clen Skupiny
- Clen Skupiny -> [JE_CLENOM] -> Skupina
- Vznik Clenstva Platitela V Skupine -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Dnu Vzniku Clenstva V Skupine

- Pismeno c) -> [VYMEDZUJE] -> Vyhlasenie Konkurzu Alebo Vstup Do Likvidacie
- Platitel -> [PODLIEHA] -> Konkurz
- Platitel -> [PODLIEHA] -> Vstup Do Likvidacie
- Vyhlasenie Konkurzu Alebo Vstup Do Likvidacie -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Vyhlaseniu Konkurzu Alebo Vstupu Do Likvidacie

- Pismeno d) -> [VYMEDZUJE] -> Zrusenie Bez Likvidacie
- Platitel -> [ZANIKA] -> Zanik Platitela
- Zrusenie Bez Likvidacie -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Dnu Zaniku

- Pismeno e) -> [VYMEDZUJE] -> Fyzicka Osoba Pokracujuca V Podnikani Po Umrti Platitela
- Fyzicka Osoba Pokracujuca V Podnikani Po Umrti Platitela -> [JE_PODLA] -> Paragraf § 83
- Fyzicka Osoba Pokracujuca V Podnikani Po Umrti Platitela -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Konanie O Dedicstve -> [PODMIENUJE] -> Posledne Zdanovacie Obdobie Skoncenia Konania O Dedicstve
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Posledny Den Posledneho Zdanovacieho Obdobia Skoncenia Konania O Dedicstve

- Pismeno f) -> [VYMEDZUJE] -> Skutocnost Na Zmenu Registracie Pre Dan
- Skutocnost Na Zmenu Registracie Pre Dan -> [JE_PODLA] -> Paragraf § 6A Odsek 2
- Skutocnost Na Zmenu Registracie Pre Dan -> [TYKA_SA] -> Zmena Registracie Pre Dan
- Skutocnost Na Zmenu Registracie Pre Dan -> [PODMIENUJE] -> Skoncenie Uplatnovania Osobitnej Upravy
- Skoncenie Uplatnovania Osobitnej Upravy -> [MA_DATUM] -> Den Predchadzajuci Dnu Nastania Skutocnosti Na Zmenu Registracie Pre Dan

- Odsek 12 -> [TYKA_SA] -> Datum Skoncenia Uplatnovania Osobitnej Upravy
- Datum Skoncenia Uplatnovania Osobitnej Upravy -> [JE_PODLA] -> Odsek 11

nodes:
- Odsek: Odsek 11
- Odsek: Odsek 12
- Pismeno: Pismeno a)
- Pismeno: Pismeno b)
- Pismeno: Pismeno c)
- Pismeno: Pismeno d)
- Pismeno: Pismeno e)
- Pismeno: Pismeno f)
- ZdanitelnaOsoba: Platitel
- Status: Osobitna Uprava
- Povinnost: Povinnost Skoncit Uplatnovanie Osobitnej Upravy
- Cinnost: Skoncenie Uplatnovania Osobitnej Upravy
- Cinnost: Dosiahnutie Obratu 100 000 Eur V Prebiehajucom Kalendarnom Roku
- Obrat: Obrat 100 000 Eur
- Suma: 100 000 Eur
- Obdobie: Prebiehajuci Kalendarny Rok
- ZdanovacieObdobie: Zdanovacie Obdobie
- Datum: Posledny Den Zdanovacieho Obdobia Dosiahnutia Obratu
- Cinnost: Vznik Clenstva Platitela V Skupine
- ClenSkupiny: Clen Skupiny
- Skupina: Skupina
- Datum: Den Predchadzajuci Dnu Vzniku Clenstva V Skupine
- Konanie: Konkurz
- Konanie: Vstup Do Likvidacie
- Cinnost: Vyhlasenie Konkurzu Alebo Vstup Do Likvidacie
- Datum: Den Predchadzajuci Vyhlaseniu Konkurzu Alebo Vstupu Do Likvidacie
- Konanie: Zrusenie Bez Likvidacie
- Stav: Zanik Platitela
- Datum: Den Predchadzajuci Dnu Zaniku
- FyzickaOsoba: Fyzicka Osoba Pokracujuca V Podnikani Po Umrti Platitela
- Paragraf: Paragraf § 83
- Konanie: Konanie O Dedicstve
- ZdanovacieObdobie: Posledne Zdanovacie Obdobie Skoncenia Konania O Dedicstve
- Datum: Posledny Den Posledneho Zdanovacieho Obdobia Skoncenia Konania O Dedicstve
- Registracia: Zmena Registracie Pre Dan
- Cinnost: Skutocnost Na Zmenu Registracie Pre Dan
- Paragraf: Paragraf § 6A
- Odsek: § 6A Odsek 2
- Datum: Den Predchadzajuci Dnu Nastania Skutocnosti Na Zmenu Registracie Pre Dan
- Datum: Datum Skoncenia Uplatnovania Osobitnej Upravy


chunk: 414
page: 102
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 103 posledného mesiaca predchádzajúceho kalendárneho štvrťroka. Ak malý podnik tuzemskej osoby v oznámení podľa prvej vety vyznačil skutočnosť, že sa rozhodol skončiť uplatňovanie osobitnej úpravy, daňový úrad rozhodne o odňatí individuálneho identifikačného čísla s príponou EX ku dňu, počnúc ktorým malý podnik tuzemskej osoby nesmie uplatňovať osobitnú úpravu; proti tomuto rozhodnutiu nemožno podať odvolanie. (14) Ak hodnota bez dane dodaných tovarov a služieb, ktoré sa zahŕňajú do ročného obratu v Únii, zdaniteľnou osobou presiahne v prebiehajúcom kalendárnom roku 100 000 eur, je zdaniteľná osoba povinná prestať uplatňovať osobitnú úpravu počnúc dodaním, ktorým bol tento obrat presiahnutý; zdaniteľná osoba je súčasne povinná do 15 pracovných dní od tohto dňa podať výkaz podľa odseku 8 za obdobie od začiatku príslušného kalendárneho štvrťroka do dňa, v ktorom hodnota bez dane dodaných tovarov a služieb, ktoré sa zahŕňajú do ročného obratu
relationships:
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Odsek 14

- Maly Podnik Tuzemskej Osoby -> [OZNAMUJE] -> Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy
- Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy -> [TYKA_SA] -> Osobitna Uprava
- Danovy Urad -> [ROZHODUJE_O] -> Odnatie Individualneho Identifikacneho Cisla S Priponou EX
- Odnatie Individualneho Identifikacneho Cisla S Priponou EX -> [TYKA_SA] -> Individualne Identifikacne Cislo S Priponou EX
- Odnatie Individualneho Identifikacneho Cisla S Priponou EX -> [MA_DATUM] -> Den Od Ktoreho Maly Podnik Tuzemskej Osoby Nesmie Uplatnovat Osobitnu Upravu
- Odvolanie -> [NEPLATI_PRE] -> Rozhodnutie O Odnati Individualneho Identifikacneho Cisla S Priponou EX

- Odsek 14 -> [UPRAVUJE] -> Povinnost Prestat Uplatnovat Osobitnu Upravu
- Rocny Obrat V Unii -> [ZAHRNUJE] -> Hodnota Bez Dane Dodanych Tovarov A Sluzieb
- Hodnota Bez Dane Dodanych Tovarov A Sluzieb -> [TYKA_SA] -> Dodane Tovary
- Hodnota Bez Dane Dodanych Tovarov A Sluzieb -> [TYKA_SA] -> Dodane Sluzby
- Hodnota Bez Dane Dodanych Tovarov A Sluzieb -> [PRESAHUJE] -> 100 000 Eur
- Presiahnutie Rocneho Obratu V Unii -> [MA_OBDOBIE] -> Prebiehajuci Kalendarny Rok
- Presiahnutie Rocneho Obratu V Unii -> [PODMIENUJE] -> Povinnost Prestat Uplatnovat Osobitnu Upravu
- Zdanitelna Osoba -> [MA_POVINNOST] -> Povinnost Prestat Uplatnovat Osobitnu Upravu
- Povinnost Prestat Uplatnovat Osobitnu Upravu -> [TYKA_SA] -> Osobitna Uprava
- Povinnost Prestat Uplatnovat Osobitnu Upravu -> [PLATI_OD] -> Dodanie Ktorym Bol Rocny Obrat V Unii Presiahnuty

- Zdanitelna Osoba -> [MA_POVINNOST] -> Podanie Vykazu
- Podanie Vykazu -> [MA_DOKLAD] -> Vykaz
- Vykaz -> [JE_PODLA] -> Odsek 8
- Podanie Vykazu -> [JE_PODLA] -> Odsek 8
- Podanie Vykazu Podla -> [MA_LEHOTU] -> 15 Pracovnych Dni Od Dna Presiahnutia Rocneho Obratu V Unii
- Vykaz -> [MA_OBDOBIE] -> Obdobie Od Zaciatku Prislusneho Kalendarneho Stvrtroka Do Dna Presiahnutia Rocneho Obratu V Unii

nodes:
- Zakon: Zakon 222/2004 Z. Z.
- Odsek: Odsek 14
- Odsek: Odsek 8
- Podnik: Maly Podnik Tuzemskej Osoby
- Oznamenie: Oznamenie Podla Prvej Vety
- Cinnost: Rozhodnutie Skoncit Uplatnovanie Osobitnej Upravy
- Status: Osobitna Uprava
- SpravcaDane: Danovy Urad
- Rozhodnutie: Rozhodnutie O Odnati Individualneho Identifikacneho Cisla S Priponou EX
- Cinnost: Odnatie Individualneho Identifikacneho Cisla S Priponou EX
- IdentifikacneCislo: Individualne Identifikacne Cislo S Priponou EX
- Datum: Den Od Ktoreho Maly Podnik Tuzemskej Osoby Nesmie Uplatnovat Osobitnu Upravu
- Konanie: Odvolanie
- ZdanitelnaOsoba: Zdanitelna Osoba
- Povinnost: Povinnost Prestat Uplatnovat Osobitnu Upravu
- Obrat: Rocny Obrat V Unii
- Hodnota: Hodnota Bez Dane Dodanych Tovarov A Sluzieb
- Tovar: Dodane Tovary
- Sluzba: Dodane Sluzby
- Suma: 100 000 Eur
- Obdobie: Prebiehajuci Kalendarny Rok
- Cinnost: Presiahnutie Rocneho Obratu V Unii
- Cinnost: Dodanie Ktorym Bol Rocny Obrat V Unii Presiahnuty
- Povinnost: Podanie Vykazu
- Zaznam: Vykaz
- Lehota: 15 Pracovnych Dni Od Dna Presiahnutia Rocneho Obratu V Unii
- Obdobie: Obdobie Od Zaciatku Prislusneho Kalendarneho Stvrtroka Do Dna Presiahnutia Rocneho Obratu V Unii


chunk: 45
page: 11
text: Strana 12 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. deň, keď prestala spĺňať status zahraničnej osoby, a adresu sídla, miesta podnikania, prevádzkarne, bydliska alebo miesta, kde sa obvykle zdržiava, v tuzemsku. (2) Ak zdaniteľná osoba spĺňa podmienky na registráciu podľa § 5 a je registrovaná podľa § 4, považuje sa za platiteľa registrovaného podľa § 5 odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava; túto skutočnosť je povinná oznámiť daňovému úradu do desiatich dní odo dňa, keď prestala mať v tuzemsku sídlo, miesto podnikania, prevádzkareň, bydlisko alebo miesto, kde sa obvykle zdržiava. (3) Daňový úrad vydá nové rozhodnutie o registrácii pre daň ku dňu, keď nastala skutočnosť, na základe ktorej došlo k zmene registrácie podľa odseku 1 alebo podľa odseku 2, a to do 30 dní odo dňa doručenia oznámenia podľa odseku 1 alebo podľa odseku 2; proti tomuto rozhodnutiu nemožno podať odvolanie.
relationships:
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf § 4
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf § 5

- Odsek 2 -> [UPRAVUJE] -> Zmena Registracie Zdanitelnej Osoby
- Zdanitelna Osoba -> [SPLNA_PODMIENKY] -> Podmienky Registracie
- Podmienky Registracie -> [JE_PODLA] -> Paragraf § 5
- Zdanitelna Osoba -> [REGISTRUJE] -> Registracia
- Registracia -> [JE_PODLA] -> Paragraf § 4
- Zdanitelna Osoba -> [POVAZUJE_SA_ZA] -> Platitel Registrovany
- Platitel Registrovany -> [JE_PODLA] -> Paragraf § 5
- Platitel Registrovany -> [PLATI_OD] -> Den Ked Zdanitelna Osoba Prestala Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkaren Bydlisko Alebo Miesto Obvykleho Zdrziavania

- Zdanitelna Osoba -> [MA_POVINNOST] -> Oznamenie Skutocnosti Danovemu Uradu
- Oznamenie Skutocnosti Danovemu Uradu -> [TYKA_SA] -> Skutocnost Ze Zdanitelna Osoba Prestala Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkaren Bydlisko Alebo Miesto Obvykleho Zdrziavania
- Oznamenie Skutocnosti Danovemu Uradu -> [DORUCUJE] -> Danovy Urad
- Oznamenie Skutocnosti Danovemu Uradu -> [MA_LEHOTU] -> Desat Dni
- Desat Dni -> [PLATI_OD] -> Den Ked Zdanitelna Osoba Prestala Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkaren Bydlisko Alebo Miesto Obvykleho Zdrziavania

- Odsek 3 -> [UPRAVUJE] -> Nove Rozhodnutie O Registracii Pre Dan
- Danovy Urad -> [VYDAVA] -> Nove Rozhodnutie O Registracii Pre Dan
- Nove Rozhodnutie O Registracii Pre Dan -> [TYKA_SA] -> Registracia Pre Dan
- Nove Rozhodnutie O Registracii Pre Dan -> [MA_DATUM] -> Den Ked Nastala Skutocnost Sposobujuca Zmenu Registracie
- Skutocnost Sposobujuca Zmenu Registracie -> [TYKA_SA] -> Zmena Registracie
- Zmena Registracie -> [JE_PODLA] -> Odsek 1
- Zmena Registracie -> [JE_PODLA] -> Odsek 2
- Nove Rozhodnutie O Registracii Pre Dan -> [MA_LEHOTU] -> 30 Dni Od Dorucenia Oznamenia
- 30 Dni Od Dorucenia Oznamenia -> [PLATI_OD] -> Dorucenie Oznamenia Podla Odseku 1 Alebo Odseku 2
- Odvolanie -> [NEPLATI_PRE] -> Nove Rozhodnutie O Registracii Pre Dan

nodes:
- Zakon: Zakon 222/2004 Z. Z.
- Paragraf: Paragraf 4
- Paragraf: Paragraf 5
- Odsek: Odsek 1
- Odsek: Odsek 2
- Odsek: Odsek 3
- SpravcaDane: Danovy Urad
- ZdanitelnaOsoba: Zdanitelna Osoba
- ZdanitelnaOsoba: Platitel Registrovany Podla Paragrafu 5
- Podmienka: Podmienky Registracie Podla Paragrafu 5
- Registracia: Registracia Podla Paragrafu 4
- Registracia: Registracia Pre Dan
- Registracia: Zmena Registracie
- Oznamenie: Oznamenie Skutocnosti Danovemu Uradu
- Cinnost: Skutocnost Ze Zdanitelna Osoba Prestala Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkaren Bydlisko Alebo Miesto Obvykleho Zdrziavania
- Datum: Den Ked Zdanitelna Osoba Prestala Mat V Tuzemsku Sidlo Miesto Podnikania Prevadzkaren Bydlisko Alebo Miesto Obvykleho Zdrziavania
- Lehota: Desat Dni
- Rozhodnutie: Nove Rozhodnutie O Registracii Pre Dan
- Datum: Den Ked Nastala Skutocnost Sposobujuca Zmenu Registracie
- Cinnost: Skutocnost Sposobujuca Zmenu Registracie
- Lehota: 30 Dni Od Dorucenia Oznamenia
- Cinnost: Dorucenie Oznamenia Podla Odseku 1 Alebo Odseku 2
- Konanie: Odvolanie
- Tuzemsko: Tuzemsko
- Sidlo: Sidlo
- Lokacia: Miesto Podnikania
- Prevazdkaren: Prevadzkaren
- Adresa: Bydlisko
- Lokacia: Miesto Obvykleho Zdrziavania


chunk: 473
page: 117
text: ak a) má sídlo alebo miesto podnikania na území Európskej únie, b) je bezúhonná a jej štatutárny orgán alebo člen štatutárneho orgánu je bezúhonný; za bezúhonného sa na účely tohto zákona nepovažuje ten, kto bol právoplatne odsúdený za úmyselný trestný čin, c) preukáže splnenie podmienok podľa odseku 2, d) preukáže oprávnenie poskytovať doručovaciu službu v súlade s podmienkami európskeho doručovacieho štandardu a s použitím nástrojov a postupov podľa tohto štandardu. (9) Súčasťou žiadosti podľa odseku 8 sú doklady preukazujúce splnenie podmienok podľa odseku 8 písm. a), c) a d). Na účel preukázania bezúhonnosti podľa odseku 8 písm. b) poskytne osoba a) výpis z registra trestov29aad) nie starší ako tri mesiace, b) iný rovnocenný doklad výpisu z registra trestov vydaný príslušným orgánom členského štátu, v ktorom má táto osoba sídlo alebo miesto podnikania, ak ide o zahraničnú osobu. (10) Ak osoba, ktorá podala žiadosť podľa odseku 8, nespĺňa podmienky podľa odseku 8,
relationships:
- Odsek 8 -> [MA_PISMENO] -> Pismeno a)
- Odsek 8 -> [MA_PISMENO] -> Pismeno b)
- Odsek 8 -> [MA_PISMENO] -> Pismeno c)
- Odsek 8 -> [MA_PISMENO] -> Pismeno d)

- Pismeno a) -> [VYMEDZUJE] -> Sidlo Alebo Miesto Podnikania Na Uzemi Europskej Unie
- Osoba -> [MA_SIDLO] -> Sidlo Na Uzemi Europskej Unie
- Osoba -> [MA_MIESTO_PODNIKANIA] -> Miesto Podnikania Na Uzemi Europskej Unie

- Pismeno b) -> [VYMEDZUJE] -> Bezuhonnost Osoby A Jej Statutarneho Organu Alebo Clena Statutarneho Organu
- Osoba -> [SPLNA_PODMIENKY] -> Bezuhonnost Osoby A Jej Statutarneho Organu Alebo Clena Statutarneho Organu
- Statutarny Organ -> [SPLNA_PODMIENKY] -> Bezuhonnost
- Clen Statutarneho Organu -> [SPLNA_PODMIENKY] -> Bezuhonnost
- Bezuhonnost -> [NEPLATI_PRE] -> Osoba Pravoplatne Odsudena Za Umyselny Trestny Cin

- Pismeno c) -> [VYMEDZUJE] -> Splnenie Podmienok
- Osoba -> [PREUKAZUJE] -> Splnenie Podmienok
- Splnenie Podmienok -> [JE_PODLA] -> Odsek 2

- Pismeno d) -> [VYMEDZUJE] -> Opravnenie Poskytovat Dorucovaciu Sluzbu
- Osoba -> [PREUKAZUJE] -> Opravnenie Poskytovat Dorucovaciu Sluzbu
- Opravnenie Poskytovat Dorucovaciu Sluzbu -> [TYKA_SA] -> Dorucovacia Sluzba
- Opravnenie Poskytovat Dorucovaciu Sluzbu -> [JE_PODLA] -> Europsky Dorucovaci Standard
- Poskytovanie Dorucovacej Sluzby -> [VYCHADZA_Z] -> Nastroje A Postupy Podla Europskeho Dorucovacieho Standardu

- Odsek 9 -> [UPRAVUJE] -> Doklady K Ziadosti
- Osoba -> [PODAVA] -> Ziadost
- Ziadost Podla -> [JE_PODLA] -> Odsek 8
- Doklady K Ziadosti -> [JE_PODLA] -> Odsek 8
- Ziadost -> [MA_DOKLAD] -> Doklady Preukazujuce Splnenie Podmienok
- Doklady Preukazujuce Splnenie Podmienok -> [PREUKAZUJE] -> Splnenie Podmienok
- Doklady Preukazujuce Splnenie Podmienok -> [JE_PODLA] -> Odsek 8 Pismeno a) c) a) d)
- Splnenie Podmienok -> [JE_PODLA] -> Odsek 8 Pismeno a) c) a) d)

- Odsek 9 -> [MA_PISMENO] -> Odsek 9 Pismeno a)
- Odsek 9 -> [MA_PISMENO] -> Odsek 9 Pismeno b)
- Odsek 9 Pismeno a) -> [VYMEDZUJE] -> Vypis Z Registra Trestov
- Osoba -> [PREDKLADA] -> Vypis Z Registra Trestov
- Vypis Z Registra Trestov -> [MA_LEHOTU] -> Tri Mesiace

- Odsek 9 Pismeno b) -> [VYMEDZUJE] -> Iny Rovnocenny Doklad Vypisu Z Registra Trestov
- Osoba -> [PREDKLADA] -> Iny Rovnocenny Doklad Vypisu Z Registra Trestov
- Iny Rovnocenny Doklad Vypisu Z Registra Trestov -> [MA_PODMIENKU] -> Osoba Je Zahranicna Osoba
- Prislusny Organ Clenskeho Statu -> [VYDAVA] -> Iny Rovnocenny Doklad Vypisu Z Registra Trestov
- Prislusny Organ Clenskeho Statu -> [SPADA_POD] -> Clensky Stat
- Clensky Stat -> [TYKA_SA] -> Sidlo Alebo Miesto Podnikania Osoby

- Odsek 10 -> [UPRAVUJE] -> Nesplnenie Podmienok
- Nesplnenie Podmienok -> [JE_PODLA] -> Odsek 8
- Osoba Ktora Podala Ziadost -> [NESPLNA_PODMIENKY] -> Podmienky Podla Odseku 8

nodes:
- Odsek: Odsek 2
- Odsek: Odsek 8
- Odsek: Odsek 9
- Odsek: Odsek 10
- Pismeno: Pismeno a)
- Pismeno: Pismeno b)
- Pismeno: Pismeno c)
- Pismeno: Pismeno d)
- Pismeno: Odsek 9 Pismeno a)
- Pismeno: Odsek 9 Pismeno b)
- Osoba: Osoba
- Ziadost: Ziadost Podla Odseku 8
- Podmienka: Sidlo Alebo Miesto Podnikania Na Uzemi Europskej Unie
- Sidlo: Sidlo Na Uzemi Europskej Unie
- Lokacia: Miesto Podnikania Na Uzemi Europskej Unie
- Uzemie: Uzemie Europskej Unie
- Podmienka: Bezuhonnost Osoby A Jej Statutarneho Organu Alebo Clena Statutarneho Organu
- Podmienka: Bezuhonnost
- Osoba: Statutarny Organ
- Osoba: Clen Statutarneho Organu
- Osoba: Osoba Pravoplatne Odsudena Za Umyselny Trestny Cin
- Podmienka: Splnenie Podmienok Podla Odseku 2
- Pravo: Opravnenie Poskytovat Dorucovaciu Sluzbu
- Sluzba: Dorucovacia Sluzba
- PravnyPredpis: Europsky Dorucovaci Standard
- Cinnost: Poskytovanie Dorucovacej Sluzby
- Cinnost: Nastroje A Postupy Podla Europskeho Dorucovacieho Standardu
- Doklad: Doklady K Ziadosti Podla Odseku 8
- Doklad: Doklady Preukazujuce Splnenie Podmienok Podla Odseku 8 Pismeno A C A D
- Podmienka: Splnenie Podmienok Podla Odseku 8 Pismeno A C A D
- Doklad: Vypis Z Registra Trestov
- Lehota: Tri Mesiace
- Doklad: Iny Rovnocenny Doklad Vypisu Z Registra Trestov
- StatnyOrgan: Prislusny Organ Clenskeho Statu
- ClenskyStat: Clensky Stat
- Podmienka: Osoba Je Zahranicna Osoba
- Sidlo: Sidlo Alebo Miesto Podnikania Osoby
- Podmienka: Podmienky
- Cinnost: Nesplnenie Podmienok


chunk: 520
page: 128
text: (10) Ak bolo zdaniteľnej osobe registrovanej pre daň podľa § 4 alebo § 5 oznámené individuálne identifikačné číslo s príponou EX pre tuzemsko, táto osoba prestáva byť platiteľom dňom, ktorý predchádza dňu tohto oznámenia; túto skutočnosť daňový úrad bezodkladne oznámi zdaniteľnej osobe. Uplynutím dňa, kedy zdaniteľná osoba podľa prvej vety prestáva byť platiteľom, končí prebiehajúce zdaňovacie obdobie a zaniká platnosť identifikačného čísla pre daň; ustanovenia odsekov 5 až 9 týmto nie sú dotknuté. § 81a (1) Daňový úrad zruší registráciu skupiny podľa § 4b k 31. decembru kalendárneho roka, ak zástupca skupiny podá žiadosť o zrušenie registrácie skupiny najneskôr do 31. októbra kalendárneho roka. Ak je žiadosť o zrušenie registrácie skupiny podaná po 31. októbri kalendárneho roka, daňový úrad zruší registráciu skupiny najneskôr k 31. decembru kalendárneho roka nasledujúceho po podaní žiadosti. (2) Ak skupina podľa § 4b alebo § 4c prestane spĺňať podmienky podľa § 4a, zástupca skupiny je
relationships:
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Odsek 10
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf 81A
- Paragraf § 81A -> [OBSAHUJE] -> Odsek 1
- Paragraf § 81A -> [OBSAHUJE] -> Odsek 2

- Zdanitelna Osoba -> [MA_STATUS] -> Osoba Registrovana Pre Dan
- Osoba Registrovana Pre Dan -> [JE_PODLA] -> Paragraf § 4
- Osoba Registrovana Pre Dan -> [JE_PODLA] -> Paragraf § 5
- Zdanitelna Osoba -> [PRIJIMA] -> Oznamenie Individualneho Identifikacneho Cisla S Priponou EX Pre Tuzemsko
- Oznamenie Individualneho Identifikacneho Cisla S Priponou EX Pre Tuzemsko -> [TYKA_SA] -> Individualne Identifikacne Cislo S Priponou EX Pre Tuzemsko
- Individualne Identifikacne Cislo S Priponou EX Pre Tuzemsko -> [PLATI_PRE] -> Tuzemsko
- Prestanie Byt Platitelom -> [TYKA_SA] -> Zdanitelna Osoba
- Prestanie Byt Platitelom -> [MA_DATUM] -> Den Predchadzajuci Dnu Oznamenia Individualneho Identifikacneho Cisla S Priponou EX Pre Tuzemsko
- Danovy Urad -> [OZNAMUJE] -> Oznamenie O Prestani Byt Platitelom
- Oznamenie O Prestani Byt Platitelom -> [DORUCUJE] -> Zdanitelna Osoba
- Oznamenie O Prestani Byt Platitelom -> [MA_LEHOTU] -> Bezodkladne
- Oznamenie O Prestani Byt Platitelom -> [TYKA_SA] -> Prestanie Byt Platitelom
- Prebiehajuce Zdanovacie Obdobie -> [ZANIKA] -> Uplynutie Dna Ked Zdanitelna Osoba Prestava Byt Platitelom
- Platnost Identifikacneho Cisla Pre Dan -> [ZANIKA] -> Uplynutie Dna Ked Zdanitelna Osoba Prestava Byt Platitelom
- Prestanie Byt Platitelom Podla Odseku 10 -> [NEVZTAHUJE_SA_NA] -> Odsek 5
- Prestanie Byt Platitelom Podla Odseku 10 -> [NEVZTAHUJE_SA_NA] -> Odsek 6
- Prestanie Byt Platitelom Podla Odseku 10 -> [NEVZTAHUJE_SA_NA] -> Odsek 7
- Prestanie Byt Platitelom Podla Odseku 10 -> [NEVZTAHUJE_SA_NA] -> Odsek 8
- Prestanie Byt Platitelom Podla Odseku 10 -> [NEVZTAHUJE_SA_NA] -> Odsek 9

- Odsek 1 -> [UPRAVUJE] -> Zrusenie Registracie Skupiny
- Registracia Skupiny -> [JE_PODLA] -> Paragraf § 4B
- Zastupca Skupiny -> [PODAVA] -> Ziadost O Zrusenie Registracie Skupiny
- Ziadost O Zrusenie Registracie Skupiny -> [TYKA_SA] -> Zrusenie Registracie Skupiny
- Ziadost O Zrusenie Registracie Skupiny -> [MA_LEHOTU] -> Najneskor Do 31. Oktobra Kalendarneho Roka
- Danovy Urad -> [ZRUSUJE] -> Registracia Skupiny
- Zrusenie Registracie Skupiny -> [MA_DATUM] -> 31. December Kalendarneho Roka
- Zrusenie Registracie Skupiny -> [MA_PODMIENKU] -> Ziadost Podana Najneskor Do 31. Oktobra Kalendarneho Roka
- Zrusenie Registracie Skupiny Po 31. Oktobri -> [MA_PODMIENKU] -> Ziadost Podana Po 31. Oktobri Kalendarneho Roka
- Zrusenie Registracie Skupiny Po 31. Oktobri -> [MA_LEHOTU] -> Najneskor K 31. Decembru Kalendarneho Roka Nasledujuceho Po Podani Ziadosti

- Odsek 2 -> [UPRAVUJE] -> Nesplnenie Podmienok Skupiny
- Skupina -> [JE_PODLA] -> Paragraf § 4B
- Skupina -> [JE_PODLA] -> Paragraf § 4C
- Skupina -> [NESPLNA_PODMIENKY] -> Podmienky
- Podmienky -> [JE_PODLA] -> Paragraf § 4A

nodes:
- Zakon: Zakon 222/2004 Z. Z.
- Paragraf: Paragraf § 81A
- Paragraf: Paragraf § 4
- Paragraf: Paragraf § 5
- Paragraf: Paragraf § 4A
- Paragraf: Paragraf § 4B
- Paragraf: Paragraf § 4C
- Odsek: Odsek 1
- Odsek: Odsek 2
- Odsek: Odsek 5
- Odsek: Odsek 6
- Odsek: Odsek 7
- Odsek: Odsek 8
- Odsek: Odsek 9
- Odsek: Odsek 10
- ZdanitelnaOsoba: Zdanitelna Osoba
- Status: Osoba Registrovana Pre Dan
- IdentifikacneCislo: Individualne Identifikacne Cislo S Priponou EX Pre Tuzemsko
- IdentifikacneCislo: Identifikacne Cislo Pre Dan
- Status: Platnost Identifikacneho Cisla Pre Dan
- Tuzemsko: Tuzemsko
- Oznamenie: Oznamenie Individualneho Identifikacneho Cisla S Priponou EX Pre Tuzemsko
- Oznamenie: Oznamenie O Prestani Byt Platitelom
- Cinnost: Prestanie Byt Platitelom
- Cinnost: Prestanie Byt Platitelom Podla Odseku 10
- Datum: Den Predchadzajuci Dnu Oznamenia Individualneho Identifikacneho Cisla S Priponou EX Pre Tuzemsko
- CasovyUdaj: Uplynutie Dna Ked Zdanitelna Osoba Prestava Byt Platitelom
- ZdanovacieObdobie: Prebiehajuce Zdanovacie Obdobie
- SpravcaDane: Danovy Urad
- Lehota: Bezodkladne
- Skupina: Skupina
- Zastupca: Zastupca Skupiny
- Ziadost: Ziadost O Zrusenie Registracie Skupiny
- Registracia: Registracia Skupiny
- Cinnost: Zrusenie Registracie Skupiny
- Cinnost: Zrusenie Registracie Skupiny Po 31. Oktobri
- Datum: 31. December Kalendarneho Roka
- Datum: Najneskor Do 31. Oktobra Kalendarneho Roka
- Lehota: Najneskor K 31. Decembru Kalendarneho Roka Nasledujuceho Po Podani Ziadosti
- Podmienka: Ziadost Podana Najneskor Do 31. Oktobra Kalendarneho Roka
- Podmienka: Ziadost Podana Po 31. Oktobri Kalendarneho Roka
- Podmienka: Podmienky
- Cinnost: Nesplnenie Podmienok Skupiny


chunk: 552
page: 136
text: (1) V období od 1. januára 2011 do posledného dňa kalendárneho roka, v ktorom Európska komisia (Eurostat) uverejní údaje36) o tom, že aktuálny schodok verejnej správy Slovenskej republiky je menej ako 3 %, je základná sadzba dane na tovary a služby 20 % zo základu dane. Skončenie obdobia uplatňovania základnej sadzby dane 20 % podľa prvej vety vyhlási Ministerstvo financií Slovenskej republiky všeobecne záväzným právnym predpisom. (2) Ak platiteľ investičný majetok uvedený v § 54 ods. 2 písm. b) a c) nadobudol alebo vytvoril do 31. decembra 2010, uplatní § 9 ods. 2 predpisu účinného do 31. decembra 2010. (3) Ak bol platiteľ povinný vykonať jednu alebo viac úprav odpočítanej dane pri investičnom majetku uvedenom v § 54 ods. 2 písm. b) a c) za obdobie rokov 2004 až 2010, je obdobím na úpravu dane odpočítanej pri tomto majetku desať rokov podľa § 54 ods. 4 predpisu účinného do 31. decembra 2010 a platiteľ použije pri každej zmene účelu použitia postup podľa prílohy č. 1 predpisu účinného do 31. decembra 2010.
relationships:
- Odsek 1 -> [UPRAVUJE] -> Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent
- Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent -> [MA_SADZBU] -> 20 Percent
- Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent -> [TYKA_SA] -> Tovary
- Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent -> [TYKA_SA] -> Sluzby
- Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent -> [MA_ZAKLAD_DANE] -> Zaklad Dane
- Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent -> [MA_OBDOBIE] -> Obdobie Od 1. Januara 2011 Do Posledneho Dna Kalendarneho Roka Zverejnenia Udajov Eurostatu
- Europska Komisia Eurostat -> [UVADZA] -> Udaje O Aktualnom Schodku Verejnej Spravy Slovenskej Republiky Menej Ako 3 Percenta
- Udaje O Aktualnom Schodku Verejnej Spravy Slovenskej Republiky Menej Ako 3 Percenta -> [TYKA_SA] -> Aktualny Schodok Verejnej Spravy Slovenskej Republiky
- Aktualny Schodok Verejnej Spravy Slovenskej Republiky -> [MA_HODNOTU] -> Menej Ako 3 Percenta
- Ministerstvo Financii Slovenskej Republiky -> [VYDAVA] -> Vseobecne Zavazny Pravny Predpis
- Vseobecne Zavazny Pravny Predpis -> [UPRAVUJE] -> Skoncenie Obdobia Uplatnovania Zakladnej Sadzby Dane 20 Percent

- Odsek 2 -> [UPRAVUJE] -> Uplatnenie Paragraf § 9 Odsek 2 Predpisu Ucinneho Do 31. Decembra 2010
- Platitel -> [NADOBUDA] -> Investicny Majetok Uvedeny V Paragrafe § 54 Odsek 2 Pismeno b) A Pismeno c)
- Platitel -> [VYKONAVA] -> Vytvorenie Investicneho Majetku
- Investicny Majetok Uvedeny V Paragrafe § 54 Odsek 2 Pismeno b) A Pismeno c) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno b)
- Investicny Majetok Uvedeny V Paragrafe § 54 Odsek 2 Pismeno b) A Pismeno c) -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno c)
- Nadobudnutie Alebo Vytvorenie Investicneho Majetku Do 31. Decembra 2010 -> [PODMIENUJE] -> Uplatnenie Paragraf § 9 Odsek 2 Predpisu Ucinneho Do 31. Decembra 2010
- Uplatnenie Paragraf § 9 Odsek 2 Predpisu Ucinneho Do 31. Decembra 2010 -> [JE_PODLA] -> Paragraf § 9 Odsek 2

- Odsek 3 -> [UPRAVUJE] -> Uprava Dane Odpocitanej Pri Investicnom Majetku
- Platitel -> [MA_POVINNOST] -> Vykonanie Jednej Alebo Viacerych Uprav Odpocitanej Dane
- Vykonanie Jednej Alebo Viacerych Uprav Odpocitanej Dane -> [TYKA_SA] -> Investicny Majetok Uvedeny V § 54 Ods. 2 Pismeno B A Pismeno C
- Vykonanie Jednej Alebo Viacerych Uprav Odpocitanej Dane -> [MA_OBDOBIE] -> Obdobie Rokov 2004 Az 2010
- Obdobie Na Upravu Dane Odpocitanej Pri Investicnom Majetku -> [MA_DOBU] -> Desat Rokov
- Obdobie Na Upravu Dane Odpocitanej Pri Investicnom Majetku -> [JE_PODLA] -> Paragraf § 54 Odsek 4 Predpisu Ucinneho Do 31. Decembra 2010
- Platitel -> [VYKONAVA] -> Postup Podla Prilohy C. 1 Predpisu Ucinneho Do 31. Decembra 2010
- Postup Podla Prilohy C. 1 Predpisu Ucinneho Do 31. Decembra 2010 -> [NASTAVA_PRI] -> Kazda Zmena Ucelu Pouzitia
- Postup Podla Prilohy C. 1 Predpisu Ucinneho Do 31. Decembra 2010 -> [JE_PODLA] -> Priloha C. 1
- Priloha C. 1 -> [JE_SUCASTOU] -> Predpis Ucinny Do 31. Decembra 2010

- Paragraf § 54 -> [MA_ODSEK] -> Paragraf § 54 Odsek 2
- Paragraf § 54 Odsek 2 -> [MA_PISMENO] -> Paragraf § 54 Odsek 2 Pismeno b)
- Paragraf § 54 Odsek 2 -> [MA_PISMENO] -> Paragraf § 54 Odsek 2 Pismeno c)
- Paragraf § 54 -> [MA_ODSEK] -> Paragraf § 54 Odsek 4
- Paragraf § 9 -> [MA_ODSEK] -> Paragraf § 9 Odsek 2

nodes:
- Odsek: Odsek 1
- Odsek: Odsek 2
- Odsek: Odsek 3
- SadzbaDane: Zakladna Sadzba Dane Na Tovary A Sluzby 20 Percent
- Hodnota: 20 Percent
- Tovar: Tovary
- Sluzba: Sluzby
- Dan: Zaklad Dane
- Obdobie: Obdobie Od 1. Januara 2011 Do Posledneho Dna Kalendarneho Roka Zverejnenia Udajov Eurostatu
- Organizacia: Europska Komisia Eurostat
- Zaznam: Udaje O Aktualnom Schodku Verejnej Spravy Slovenskej Republiky Menej Ako 3 Percenta
- Hodnota: Aktualny Schodok Verejnej Spravy Slovenskej Republiky
- Hodnota: Menej Ako 3 Percenta
- Ministerstvo: Ministerstvo Financii Slovenskej Republiky
- PravnyPredpis: Vseobecne Zavazny Pravny Predpis
- Cinnost: Skoncenie Obdobia Uplatnovania Zakladnej Sadzby Dane 20 Percent
- ZdanitelnaOsoba: Platitel
- InvesticnyMajetok: Investicny Majetok Uvedeny V § 54 Ods. 2 Pismeno B A Pismeno C
- Cinnost: Vytvorenie Investicneho Majetku
- Cinnost: Nadobudnutie Alebo Vytvorenie Investicneho Majetku Do 31. Decembra 2010
- Cinnost: Uplatnenie § 9 Ods. 2 Predpisu Ucinneho Do 31. Decembra 2010
- Cinnost: Uprava Dane Odpocitanej Pri Investicnom Majetku
- Povinnost: Vykonanie Jednej Alebo Viacerych Uprav Odpocitanej Dane
- Obdobie: Obdobie Rokov 2004 Az 2010
- Obdobie: Obdobie Na Upravu Dane Odpocitanej Pri Investicnom Majetku
- Lehota: Desat Rokov
- Cinnost: Kazda Zmena Ucelu Pouzitia
- Cinnost: Postup Podla Prilohy C. 1 Predpisu Ucinneho Do 31. Decembra 2010
- PravnyPredpis: Predpis Ucinny Do 31. Decembra 2010
- Priloha: Priloha C. 1
- Paragraf: § 54
- Odsek: § 54 Ods. 2
- Pismeno: § 54 Ods. 2 Pismeno B
- Pismeno: § 54 Ods. 2 Pismeno C
- Odsek: § 54 Ods. 4
- Paragraf: § 9
- Odsek: § 9 Ods. 2


chunk: 58
page: 14
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 15 členského štátu. (3) Ak do 12 mesiacov po skončení prepravy tovaru v členskom štáte, do ktorého bol tovar odoslaný alebo prepravený, tento tovar nebol dodaný zdaniteľnej osobe, ktorej sa mal dodať podľa odseku 1 písm. c) alebo odseku 5, premiestnenie podľa § 8 ods. 4 prvej vety sa považuje za uskutočnené v deň, ktorý nasleduje po uplynutí 12 mesiacov, okrem situácií uvedených v odseku 6. (4) Premiestnenie podľa § 8 ods. 4 prvej vety sa nepovažuje za uskutočnené, ak a) nedošlo k prevodu práva nakladať s tovarom ako vlastník a tovar bol vrátený do tuzemska v lehote podľa odseku 3 a  b) platiteľ, ktorý tovar odoslal alebo prepravil podľa odseku 1 písm. a), uviedol vrátenie tovaru v záznamoch podľa § 70 ods. 2 písm. g). (5) Ak v lehote podľa odseku 3 zdaniteľnú osobu podľa odseku 1 písm. c) nahradila iná zdaniteľná osoba, premiestnenie podľa § 8 ods. 4 prvej vety sa nepovažuje za uskutočnené v okamihu tohto nahradenia, ak
relationships:
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf § 8
- Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf § 70
- Paragraf § 8 -> [MA_ODSEK] -> Paragraf § 8 Odsek 4
- Paragraf § 70 -> [MA_ODSEK] -> Paragraf § 70 Odsek 2
- Paragraf § 70 Odsek 2 -> [MA_PISMENO] -> Paragraf § 70 Odsek 2 Pismeno g)

- Odsek 3 -> [UPRAVUJE] -> Uskutocnenie Premiestnenia Tovaru Po Uplynuti 12 Mesiacov
- Premiestnenie Tovaru -> [JE_PODLA] -> Paragraf § 8 Odsek 4 Prva Veta
- Tovar -> [DODAVA] -> Zdanitelna Osoba
- Dodanie Tovaru Zdanitelnej Osobe -> [JE_PODLA] -> Odsek 1 Pismeno c)
- Dodanie Tovaru Zdanitelnej Osobe -> [JE_PODLA] -> Odsek 5
- Dodanie Tovaru Zdanitelnej Osobe -> [MA_LEHOTU] -> 12 Mesiacov Po Skonceni Prepravy Tovaru
- Preprava Tovaru -> [MA_MIESTO] -> Clensky Stat Odoslania Alebo Prepravy Tovaru
- Nedodanie Tovaru Zdanitelnej Osobe Do 12 Mesiacov -> [PODMIENUJE] -> Uskutocnenie Premiestnenia Tovaru
- Uskutocnenie Premiestnenia Tovaru -> [MA_DATUM] -> Den Nasledujuci Po Uplynuti 12 Mesiacov
- Uskutocnenie Premiestnenia Tovaru -> [MA_VYNIMKU] -> Situacie Uvedene V Odseku 6
- Situacie Uvedene V Odseku 6 -> [JE_PODLA] -> Odsek 6

- Odsek 4 -> [UPRAVUJE] -> Nepovazovanie Premiestnenia Tovaru Za Uskutocnene
- Odsek 4 -> [MA_PISMENO] -> Pismeno a)
- Odsek 4 -> [MA_PISMENO] -> Pismeno b)
- Nepovazovanie Premiestnenia Tovaru Za Uskutocnene -> [TYKA_SA] -> Premiestnenie Tovaru
- Nepovazovanie Premiestnenia Tovaru Za Uskutocnene -> [MA_PODMIENKU] -> Nedošlo K Prevodu Prava Nakladat S Tovarom Ako Vlastnik
- Nepovazovanie Premiestnenia Tovaru Za Uskutocnene -> [MA_PODMIENKU] -> Vratenie Tovaru Do Tuzemska V Lehote Podla Odseku 3
- Vratenie Tovaru Do Tuzemska V Lehote -> [MA_MIESTO] -> Tuzemsko
- Vratenie Tovaru Do Tuzemska V Lehote -> [JE_PODLA] -> Odseke 3
- Vratenie Tovaru Do Tuzemska V Lehote -> [MA_LEHOTU] -> Lehota
- Nepovazovanie Premiestnenia Tovaru Za Uskutocnene -> [MA_PODMIENKU] -> Uvedenie Vratenia Tovaru V Zaznamoch
- Platitel -> [UVADZA] -> Vratenie Tovaru V Zaznamoch
- Vratenie Tovaru V Zaznamoch -> [JE_PODLA] -> Paragraf § 70 Odsek 2 Pismeno g)
- Platitel -> [DODAVA] -> Tovar
- Odoslanie Alebo Preprava Tovaru Platitelom -> [JE_PODLA] -> Odsek 1 Pismeno a)

- Odsek 5 -> [UPRAVUJE] -> Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou
- Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou -> [MA_LEHOTU] -> Lehota Podla Odseku 3
- Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou -> [TYKA_SA] -> Zdanitelna Osoba
- Zdanitelna Osoba -> [JE_PODLA] -> Odsek 1 Pismeno c)
- Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou -> [TYKA_SA] -> Ina Zdanitelna Osoba
- Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou -> [PODMIENUJE] -> Nepovazovanie Premiestnenia Tovaru Za Uskutocnene V Okamihu Nahradenia

nodes:
- Zakon: Zakon 222/2004 Z. Z.
- Paragraf: Paragraf § 8
- Paragraf: Paragraf § 70
- Odsek: Paragraf § 8 Odsek 4
- Odsek: Paragraf § 8 Odsek 4 Prva Veta
- Odsek: Paragraf § 70 Odsek 2
- Pismeno: Paragraf § 70 Odsek 2 Pismeno G
- Odsek: Odsek 1
- Odsek: Odsek 3
- Odsek: Odsek 4
- Odsek: Odsek 5
- Odsek: Odsek 6
- Pismeno: Odsek 1 Pismeno A
- Pismeno: Odsek 1 Pismeno C
- Pismeno: Pismeno A
- Pismeno: Pismeno B
- Cinnost: Premiestnenie Tovaru
- Cinnost: Uskutocnenie Premiestnenia Tovaru
- Cinnost: Uskutocnenie Premiestnenia Tovaru Po Uplynuti 12 Mesiacov
- Cinnost: Nepovazovanie Premiestnenia Tovaru Za Uskutocnene
- Cinnost: Nepovazovanie Premiestnenia Tovaru Za Uskutocnene V Okamihu Nahradenia
- Tovar: Tovar
- ZdanitelnaOsoba: Zdanitelna Osoba
- ZdanitelnaOsoba: Zdanitelna Osoba
- ZdanitelnaOsoba: Ina Zdanitelna Osoba
- Cinnost: Dodanie Tovaru Zdanitelnej Osobe
- Podmienka: Nedodanie Tovaru Zdanitelnej Osobe Do 12 Mesiacov
- Lehota: 12 Mesiacov Po Skonceni Prepravy Tovaru
- Cinnost: Preprava Tovaru
- ClenskyStat: Clensky Stat Odoslania Alebo Prepravy Tovaru
- Datum: Den Nasledujuci Po Uplynuti 12 Mesiacov
- Cinnost: Situacie Uvedene V Odseku 6
- Podmienka: Nedošlo K Prevodu Prava Nakladat S Tovarom Ako Vlastnik
- Cinnost: Vratenie Tovaru Do Tuzemska V Lehote
- Tuzemsko: Tuzemsko
- Lehota: Lehota Podla Odseku 3
- Podmienka: Uvedenie Vratenia Tovaru V Zaznamoch
- Zaznam: Vratenie Tovaru V Zaznamoch
- ZdanitelnaOsoba: Platitel
- Cinnost: Odoslanie Alebo Preprava Tovaru Platitelom
- Cinnost: Nahradenie Zdanitelnej Osoby Inou Zdanitelnou Osobou


chunk: 601
page: 151
text: Strana 152 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. Príloha č. 3 k zákonu č. 222/2004 Z. z. POTVRDENIE O POSTAVENÍ DAŇOVÉHO SUBJEKTU DynamicResources\6e3bf279-9846-46da-975f-431c66fedc07_2.pdf 1 does not have a VAT iden ti fi ca tion num ber, the com pe tent aut ho ri ty shall sta te the re a son for this. POTVRDENIE O POSTAVENÍ DAÒOVÉHO SUBJEKTU CERTIFICATE OF STATUS OF TAXABLE PERSON                      ò) Ak GDRYÞVXEMHNW nemá identifikaèné èíslo pre daò z pridanej hodnoty, kompetentný orgán zdôvodní túto skutoènos/If the WD[DEOHSHUVRQ
relationships:
- Zakon C. 222/2004 Z. Z. -> [OBSAHUJE] -> Priloha C. 3
- Priloha C. 3 -> [OBSAHUJE] -> Potvrdenie O Postaveni Danoveho Subjektu
- Potvrdenie O Postaveni Danoveho Subjektu -> [VZTAHUJE_SA_NA] -> Danovy Subjekt
- Identifikacne Cislo Pre Dan Z Pridanej Hodnoty -> [TYKA_SA] -> Dan Z Pridanej Hodnoty
- Danovy Subjekt -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan Z Pridanej Hodnoty
- Danovy Subjekt Nema Identifikacne Cislo Pre Dan Z Pridanej Hodnoty -> [PODMIENUJE] -> Zdovodnenie Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty
- Kompetentny Organ -> [MA_POVINNOST] -> Zdovodnenie Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty
- Zdovodnenie Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty -> [MA_DOVOD] -> Dovod Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty
- Kompetentny Organ -> [UVADZA] -> Dovod Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty

nodes:
- Zakon: Zakon C. 222/2004 Z. Z.
- Priloha: Priloha C. 3
- Doklad: Potvrdenie O Postaveni Danoveho Subjektu
- Subjekt: Danovy Subjekt
- Dan: Dan Z Pridanej Hodnoty
- IdentifikacneCislo: Identifikacne Cislo Pre Dan Z Pridanej Hodnoty
- Podmienka: Danovy Subjekt Nema Identifikacne Cislo Pre Dan Z Pridanej Hodnoty
- Povinnost: Zdovodnenie Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty
- Dovod: Dovod Neexistencie Identifikacneho Cisla Pre Dan Z Pridanej Hodnoty
- StatnyOrgan: Kompetentny Organ


chunk: 69
page: 16
text: (4) Nadobudnutie tovaru v tuzemsku z iného členského štátu nie je predmetom dane, ak a) dodanie takého tovaru v tuzemsku by bolo oslobodené od dane podľa § 47 ods. 7 až 10, b) nadobúdateľom je zdaniteľná osoba, ktorá nie je platiteľom a ktorá nie je registrovaná pre daň podľa § 7, alebo právnická osoba, ktorá nie je zdaniteľnou osobou a ktorá nie je registrovaná pre daň podľa § 7, a súčasne celková hodnota nadobudnutého tovaru nedosiahla 14 000 eur za predchádzajúci kalendárny rok a ani v prebiehajúcom kalendárnom roku túto hodnotu nedosiahne,
relationships:
- Odsek 4 -> [UPRAVUJE] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
- Odsek 4 -> [MA_PISMENO] -> Pismeno a)
- Odsek 4 -> [MA_PISMENO] -> Pismeno b)

- Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_MIESTO] -> Tuzemsko
- Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [VYCHADZA_Z] -> Iny Clensky Stat
- Dan -> [NEVZTAHUJE_SA_NA] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu

- Pismeno A -> [VYMEDZUJE] -> Dodanie Takeho Tovaru V Tuzemsku Oslobodene Od Dane
- Dodanie Takeho Tovaru V Tuzemsku -> [JE_OSLOBODENE_OD] -> Dan
- Dodanie Takeho Tovaru V Tuzemsku -> [JE_PODLA] -> Paragraf § 47 Odsek 7
- Dodanie Takeho Tovaru V Tuzemsku -> [JE_PODLA] -> Paragraf § 47 Odsek 8
- Dodanie Takeho Tovaru V Tuzemsku -> [JE_PODLA] -> Paragraf § 47 Odsek 9
- Dodanie Takeho Tovaru V Tuzemsku -> [JE_PODLA] -> Paragraf § 47 Odsek 10

- Pismeno b) -> [VYMEDZUJE] -> Nadobudatel Nie Je Platitel Ani Registrovany Pre Dan
- Nadobudatel Nie Je Platitel Ani Registrovany Pre Dan -> [JE_TYPOM] -> Zdanitelna Osoba
- Zdanitelna Osoba -> [NEVZTAHUJE_SA_NA] -> Platitel
- Zdanitelna Osoba -> [NEVZTAHUJE_SA_NA] -> Registracia Pre Dan
- Registracia Pre Dan -> [JE_PODLA] -> Paragraf § 7

- Pismeno b) -> [VYMEDZUJE] -> Pravnicka Osoba Ktora Nie Je Zdanitelnou Osobou Ani Registrovana Pre Dan
- Pravnicka Osoba -> [NEVZTAHUJE_SA_NA] -> Zdanitelna Osoba
- Pravnicka Osoba -> [NEVZTAHUJE_SA_NA] -> Registracia Pre Dan

- Celkova Hodnota Nadobudnuteho Tovaru -> [TYKA_SA] -> Nadobudnuty Tovar
- Celkova Hodnota Nadobudnuteho Tovaru -> [NEPLATI_PRE] -> Hodnota 14 000 Eur Alebo Viac
- Celkova Hodnota Nadobudnuteho Tovaru -> [MA_OBDOBIE] -> Predchadzajuci Kalendarny Rok
- Celkova Hodnota Nadobudnuteho Tovaru -> [MA_OBDOBIE] -> Prebiehajuci Kalendarny Rok
- Hodnota 14 000 Eur Alebo Viac -> [MA_SUMU] -> 14 000 Eur

- Paragraf § 47 -> [MA_ODSEK] -> Paragraf § 47 Odsek 7
- Paragraf § 47 -> [MA_ODSEK] -> Paragraf § 47 Odsek 8
- Paragraf § 47 -> [MA_ODSEK] -> Paragraf § 47 Odsek 9
- Paragraf § 47 -> [MA_ODSEK] -> Paragraf § 47 Odsek 10

nodes:
- Odsek: Odsek 4
- Pismeno: Pismeno a)
- Pismeno: Pismeno b)
- Cinnost: Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
- Tuzemsko: Tuzemsko
- ClenskyStat: Iny Clensky Stat
- Dan: Dan
- Cinnost: Dodanie Takeho Tovaru V Tuzemsku
- Podmienka: Dodanie Takeho Tovaru V Tuzemsku Oslobodene Od Dane
- Paragraf: Paragraf § 47
- Odsek: Paragraf § 47 Odsek 7
- Odsek: Paragraf § 47 Odsek 8
- Odsek: Paragraf § 47 Odsek 9
- Odsek: Paragraf § 47 Odsek 10
- Podmienka: Nadobudatel Nie Je Platitel Ani Registrovany Pre Dan
- ZdanitelnaOsoba: Zdanitelna Osoba
- ZdanitelnaOsoba: Platitel
- Registracia: Registracia Pre Dan
- Paragraf: Paragraf § 7
- PravnickaOsoba: Pravnicka Osoba
- Podmienka: Pravnicka Osoba Ktora Nie Je Zdanitelnou Osobou Ani Registrovana Pre Dan
- Hodnota: Celkova Hodnota Nadobudnuteho Tovaru
- Tovar: Nadobudnuty Tovar
- Hodnota: Hodnota 14 000 Eur Alebo Viac
- Suma: 14 000 Eur
- Obdobie: Predchadzajuci Kalendarny Rok
- Obdobie: Prebiehajuci Kalendarny Rok


chunk: 97
page: 23
text: tovaru podľa § 14 ods. 2 a miesto dodania služby sa mení na miesto dodania služby podľa § 16 ods. 14. (3) Dodávateľ tovaru alebo služby, ktorý spĺňa podmienky podľa odseku 1, sa môže rozhodnúť pre miesto dodania tovaru podľa § 14 ods. 2 a miesto dodania služieb podľa § 16 ods. 14 a je povinný tieto miesta uplatňovať najmenej po dobu dvoch kalendárnych rokov.
relationships:
- Odsek 3 -> [UPRAVUJE] -> Rozhodnutie Pre Miesto Dodania Tovaru A Miesto Dodania Sluzby
- Dodavatel Tovaru Alebo Sluzby -> [SPLNA_PODMIENKY] -> Podmienky
- Podmienky -> [JE_PODLA] -> Odsek 1

- Dodavatel Tovaru Alebo Sluzby -> [MA_PRAVO] -> Rozhodnutie Pre Miesto Dodania Tovaru A Miesto Dodania Sluzby
- Rozhodnutie Pre Miesto Dodania Tovaru A Miesto Dodania Sluzby -> [TYKA_SA] -> Miesto Dodania Tovaru Podla
- Rozhodnutie Pre Miesto Dodania Tovaru A Miesto Dodania Sluzby -> [TYKA_SA] -> Miesto Dodania Sluzby Podla

- Miesto Dodania Tovaru -> [JE_PODLA] -> Paragraf § 14 Odsek 2
- Miesto Dodania Sluzby -> [JE_PODLA] -> Paragraf § 16 Odsek 14

- Dodavatel Tovaru Alebo Sluzby -> [MA_POVINNOST] -> Uplatnovanie Miesta Dodania Tovaru A Miesta Dodania Sluzby
- Uplatnovanie Miesta Dodania Tovaru A Miesta Dodania Sluzby -> [TYKA_SA] -> Paragraf § 14 Odsek 2
- Uplatnovanie Miesta Dodania Tovaru A Miesta Dodania Sluzby -> [TYKA_SA] -> Paragraf § 16 Odsek 14
- Uplatnovanie Miesta Dodania Tovaru A Miesta Dodania Sluzby -> [MA_DOBU] -> Najmenej Dva Kalendarne Roky

- Paragraf § 14 -> [MA_ODSEK] -> Paragraf § 14 Odsek 2
- Paragraf § 16 -> [MA_ODSEK] -> Paragraf § 16 Odsek 14

nodes:
- Odsek: Odsek 1
- Odsek: Odsek 3
- Paragraf: Paragraf § 14
- Paragraf: Paragraf § 16
- Odsek: Paragraf § 14 Odsek 2
- Odsek: Paragraf § 16 Odsek 14
- Subjekt: Dodavatel Tovaru Alebo Sluzby
- Podmienka: Podmienky Podla Odseku 1
- Cinnost: Rozhodnutie Pre Miesto Dodania Tovaru A Miesto Dodania Sluzby
- Lokacia: Miesto Dodania Tovaru
- Lokacia: Miesto Dodania Sluzby
- Povinnost: Uplatnovanie Miesta Dodania Tovaru A Miesta Dodania Sluzby
- Obdobie: Najmenej Dva Kalendarne Roky

