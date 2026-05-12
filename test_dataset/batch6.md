chunk: 105
page: 25
text: (4) Ak je platba prijatá pred dodaním tovaru alebo služby, vzniká daňová povinnosť z prijatej platby dňom prijatia platby. (5) Pri dodaní tovaru na základe komisionárskej zmluvy vzniká komitentovi daňová povinnosť v ten istý deň, keď vzniká daňová povinnosť komisionárovi. (6) Pri obstaraní služby podľa § 9 ods. 4 vrátane obstarania opakovane alebo čiastkovo dodávaných služieb sa služba obstaraná osobou, ktorá koná vo svojom mene, považuje za dodanú dňom vyhotovenia faktúry, ktorou obstarávateľ požaduje úhradu za službu, a ak faktúra nie je vyhotovená do konca tretieho kalendárneho mesiaca nasledujúceho po kalendárnom mesiaci, v ktorom bola služba dodaná, daňová povinnosť vzniká posledným dňom tretieho kalendárneho mesiaca nasledujúceho po kalendárnom mesiaci, v ktorom bola služba dodaná; deň dodania služby podľa tohto odseku sa neurčí, ak sa obstará služba s miestom dodania podľa § 15 ods. 1, pri ktorej je povinný platiť daň príjemca služby. Daňová povinnosť vzniká nositeľovi autorských
relationships:
  Danova Povinnost -> [VZNIKA_PRI:"z prijatej platby"] -> Prijatie Platby
  Prijatie Platby -> [PLATI_PRE:"pred dodanim tovaru alebo sluzby"] -> Platba
  Danova Povinnost -> [VZNIKA_PRI:"pri dodani tovaru na zaklade komisionarskej zmluvy"] -> Dodanie Tovaru
  Komitent -> [MA_POVINNOST:"pri dodani tovaru na zaklade komisionarskej zmluvy"] -> Danova Povinnost
  Komitent -> [JE_POVINNY_PLATIT:"daň"] -> Danova Povinnost
  Obstaranie Sluzby -> [VZTAHUJE_SA_NA:"podla § 9 ods. 4"] -> § 9 Ods. 4
  Sluzba -> [PODLIEHA:"obstarana sluzba"] -> § 9 Ods. 4
  Osoba, Ktora Kona Vo Svojom Mene -> [VYKONAVA:"konanie vo svojom mene"] -> Obstaranie Sluzby
  Faktura -> [MA_UCINOK:"sluzba sa povazuje za dodanu dnom vyhotovenia faktury"] -> Dodanie Sluzby
  Faktura -> [MA_LEHOTU:"ak nie je vyhotovena do konca tretieho kalendarneho mesiaca"] -> Posledny Den Tretieho Kalendarneho Mesiaca Nasledujuceho Po Kalendarnom Mesiaci, V Ktorom Bola Sluzba Dodana
  Danova Povinnost -> [VZNIKA_PRI:"ak faktura nie je vyhotovena do konca lehoty"] -> Posledny Den Tretieho Kalendarneho Mesiaca Nasledujuceho Po Kalendarnom Mesiaci, V Ktorom Bola Sluzba Dodana
  Sluzba -> [MA_MIESTO_DODANIA:"miesto dodania podla § 15 ods. 1"] -> § 15 Ods. 1
  Danova Povinnost -> [VZNIKA_PRI:"nositel autorskych prav"] -> Nositel Autorskych
  Danova Povinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Sluzba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Komisionarska Zmluva -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Komitent -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Komisionar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba, Ktora Kona Vo Svojom Mene -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Faktura -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Obstaravatel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Nositel Autorskych -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 9 Ods. 4 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 15 Ods. 1 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prijatie Platby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodanie Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodanie Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Obstaranie Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vyhotovenie Faktury -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Posledny Den Tretieho Kalendarneho Mesiaca Nasledujuceho Po Kalendarnom Mesiaci, V Ktorom Bola Sluzba Dodana -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Povinnost: Danova Povinnost
  Zaznam: Platba
  Tovar: Tovar
  Sluzba: Sluzba
  Zmluva: Komisionarska Zmluva
  Subjekt: Komitent
  Subjekt: Komisionar
  Osoba: Osoba, Ktora Kona Vo Svojom Mene
  Doklad: Faktura
  Subjekt: Obstaravatel
  Subjekt: Nositel Autorskych
  Paragraf: § 9 Ods. 4
  Paragraf: § 15 Ods. 1
  Cinnost: Prijatie Platby
  Cinnost: Dodanie Tovaru
  Cinnost: Dodanie Sluzby
  Cinnost: Obstaranie Sluzby
  Cinnost: Vyhotovenie Faktury
  Datum: Posledny Den Tretieho Kalendarneho Mesiaca Nasledujuceho Po Kalendarnom Mesiaci, V Ktorom Bola Sluzba Dodana

chunk: 137
page: 33
text: e) údaje podľa § 74 ods. 1 písm. a) až c) týkajúce sa faktúry o dodaní tovaru alebo služby, ktorého sa oprava základu dane týka. (10) Opravný doklad podľa odseku 7 písm. b) musí obsahovať a) číselnú identifikáciu opravného dokladu, b) dátum vyhotovenia opravného dokladu, c) údaje z opravného dokladu podľa odseku 9 písm. a) a b), d) slovnú informáciu „oprava základu dane podľa § 25a“, e) sumu, ktorú platiteľ prijal v súvislosti s nevymožiteľnou pohľadávkou podľa odseku 2 alebo jej časťou, a z toho sumu prislúchajúcej dane, f) dátum prijatia sumy podľa písmena e). (11) Platiteľ je povinný vyhotoviť a odoslať opravný doklad najneskôr do uplynutia lehoty na podanie daňového priznania za zdaňovacie obdobie, a) za ktoré vykonal opravu základu dane podľa odseku 3, b) v ktorom prijal platbu podľa odseku 6.
relationships:
  Paragraf 74 -> [OBSAHUJE] -> Odsek 10
  Paragraf 74 -> [OBSAHUJE] -> Odsek 11
  Opravny Doklad -> [JE_PODLA] -> Odsek 7
  Opravny Doklad -> [MA_POVINNOST:"obsahovat datum vyhotovenia"] -> Datum Vyhotovenia Opravneho Dokladu
  Opravny Doklad -> [MA_POVINNOST:"obsahovat sumu prijatu v suvislosti s nevymozitelnou pohladavkou alebo jej castou"] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Opravny Doklad -> [MA_POVINNOST:"obsahovat sumu prisluchajucej dane"] -> Suma Prisluchajucej Dane
  Opravny Doklad -> [MA_POVINNOST:"obsahovat danove udaje z opravneho dokladu"] -> Danie
  Faktura -> [SUVISI_S:"faktura o dodani tovaru alebo sluzby"] -> Dodaní Tovaru Alebo Služby
  Oprava Zakladu Dane -> [VZTAHUJE_SA_NA:"oprava zakladu dane"] -> Faktura
  Oprava Zakladu Dane -> [ODKAZUJE_NA] -> Paragraf 25A
  Platitel -> [MA_POVINNOST:"vyhotovit a odoslat opravny doklad"] -> Opravny Doklad
  Platitel -> [MA_LEHOTU:"najneskor do uplynutia"] -> Lehota Na Podanie Danoveho Priznania
  Lehota Na Podanie Danoveho Priznania -> [VZTAHUJE_SA_NA:"podanie danoveho priznania"] -> Danove Priznanie
  Lehota Na Podanie Danoveho Priznania -> [VZTAHUJE_SA_NA:"za zdanovacie obdobie"] -> Zdanovacie Obdobie
  Platitel -> [VYKONAVA:"vykona opravu zakladu dane"] -> Oprava Zakladu Dane
  Platitel -> [PRIJIMA:"prijata suma"] -> Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Opravny Doklad -> [SUVISI_S:"opravny doklad k oprave zakladu dane"] -> Oprava Zakladu Dane
  Odsek 10 -> [ODKAZUJE_NA] -> Paragraf 25A
  Odsek 11 -> [ODKAZUJE_NA] -> Paragraf 25A
  Paragraf 74 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 25A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 7 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 9 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 10 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 11 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno B -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno C -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno D -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno E -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno F -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Opravny Doklad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Faktura -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oprava Zakladu Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Suma Prisluchajucej Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Datum Vyhotovenia Opravneho Dokladu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Datum Prijatia Sumy -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Lehota Na Podanie Danoveho Priznania -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanovacie Obdobie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Nevymozitelnou Pohladavkou -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodaní Tovaru Alebo Služby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Paragraf: Paragraf 74
  Paragraf: Paragraf 25A
  Odsek: Odsek 7
  Odsek: Odsek 9
  Odsek: Odsek 10
  Odsek: Odsek 11
  Pismeno: Pismeno A
  Pismeno: Pismeno B
  Pismeno: Pismeno C
  Pismeno: Pismeno D
  Pismeno: Pismeno E
  Pismeno: Pismeno F
  Doklad: Opravny Doklad
  Doklad: Faktura
  Oprava: Oprava Zakladu Dane
  Suma: Suma Prijata V Suvislosti S Nevymozitelnou Pohladavkou Alebo Jej Castou
  Suma: Suma Prisluchajucej Dane
  Datum: Datum Vyhotovenia Opravneho Dokladu
  Datum: Datum Prijatia Sumy
  Lehota: Lehota Na Podanie Danoveho Priznania
  Danovepriznanie: Danove Priznanie
  Zdanovacieobdobie: Zdanovacie Obdobie
  Zdanitelnaosoba: Platitel
  Pohladavka: Nevymozitelnou Pohladavkou
  Dan: Danie
  Cinnost: Dodaní Tovaru Alebo Služby

chunk: 16
page: 4
text: podľa § 7 alebo § 7a, je zdaniteľná osoba povinná túto skutočnosť oznámiť daňovému úradu pred tým, ako táto skutočnosť nastane. Zdaniteľná osoba, ktorej ku dňu oznámenia skutočnosti podľa prvej vety a) bolo doručené rozhodnutie o registrácii pre daň podľa odseku 4 písm. a), sa odo dňa nasledujúceho po dni, v ktorom oznámila túto skutočnosť, do 31. decembra prebiehajúceho kalendárneho roka alebo do dňa, keď sa v prebiehajúcom kalendárnom roku stane platiteľom podľa odseku 1 písm. b) až i), odseku 8 písm. b) alebo podľa § 48c ods. 5, považuje na účely tohto zákona za osobu registrovanú pre daň podľa § 7 alebo § 7a; daňový úrad rozhodnutie nevydáva, b) ešte nebolo doručené rozhodnutie o registrácii pre daň podľa odseku 4 písm. a), sa odo dňa doručenia tohto rozhodnutia do 31. decembra prebiehajúceho kalendárneho roka alebo do dňa, keď sa v prebiehajúcom kalendárnom roku stane platiteľom podľa odseku 1 písm. b) až i), odseku 8 písm. b) alebo podľa § 48c ods. 5, považuje na účely tohto zákona za osobu
relationships:
  7 -> [ODKAZUJE_NA] -> 7a
  Zdanitelna Osoba -> [MA_POVINNOST:"oznamenie skutocnosti pred tym ako nastane"] -> Danovy Urad
  Zdanitelna Osoba -> [OZNAMUJE:"podla § 7 alebo § 7a"] -> Skutocnost
  Zdanitelna Osoba -> [MA_DOKLAD:"bolo dorucene alebo este nebolo dorucene"] -> Rozhodnutie O Registracii Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [PODLIEHA:"pre dan podla odseku 4 pism. a"] -> Registracia Pre Dan
  Zdanitelna Osoba -> [PODLA:"na ucely tohto zakona"] -> 7
  Zdanitelna Osoba -> [PODLA:"na ucely tohto zakona"] -> 7a
  Zdanitelna Osoba -> [MA_OBDOBIE:"do 31. decembra alebo do dna ked sa stane platitelom"] -> Prebiehajuci Kalendarny Rok
  Zdanitelna Osoba -> [MA_DATUM:"termin"] -> 31. decembra prebiehajuceho kalendarneho roka
  Zdanitelna Osoba -> [MA_DATUM:"zaciatok obdobia v pripade b"] -> Den Dorucenia Tohto Rozhodnutia
  Zdanitelna Osoba -> [MA_DATUM:"zaciatok obdobia v pripade a"] -> Den Nasledujuci Po Dni Oznamenia
  Zdanitelna Osoba -> [STAVA_SA:"podla odseku 1 pism. b az i alebo odseku 8 pism. b alebo podla § 48c ods. 5"] -> Platitel
  7 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  7a -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  48c ods. 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 4 Pism. A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 1 Pism. B Az I -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 8 Pism. B -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanitelna Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oznamenie Skutocnosti -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Registracia Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Rozhodnutie O Registracii Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prebiehajuci Kalendarny Rok -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  31. decembra prebiehajuceho kalendarneho roka -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Den Dorucenia Tohto Rozhodnutia -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Den Nasledujuci Po Dni Oznamenia -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Skutocnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Paragraf: 7
  Paragraf: 7a
  Odsek: 48c ods. 5
  Odsek: Odsek 4 Pism. A
  Odsek: Odsek 1 Pism. B Az I
  Odsek: Odsek 8 Pism. B
  Zdanitelnaosoba: Zdanitelna Osoba
  Urad: Danovy Urad
  Oznamenie: Oznamenie Skutocnosti
  Registracia: Registracia Pre Dan
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Obdobie: Prebiehajuci Kalendarny Rok
  Datum: 31. decembra prebiehajuceho kalendarneho roka
  Datum: Den Dorucenia Tohto Rozhodnutia
  Datum: Den Nasledujuci Po Dni Oznamenia
  Status: Skutocnost
  Status: Platitel

chunk: 171
page: 43
text: alebo na jeho účet do miesta určenia na území tretieho štátu. (2) Oslobodené od dane je dodanie tovaru, ktorý je odoslaný alebo prepravený kupujúcim alebo na jeho účet do miesta určenia na území tretieho štátu, ak kupujúci nemá v tuzemsku sídlo, miesto podnikania, prevádzkareň ani bydlisko, s výnimkou dodania tovaru, ktorý prepravil kupujúci na účely vybavenia, zásobenia pohonnými látkami a potravinami výletných lodí, súkromných lietadiel alebo akýchkoľvek dopravných prostriedkov na súkromné použitie. (3) Odoslanie alebo prepravu tovaru do miesta určenia na území tretieho štátu podľa odsekov 1 a 2 je platiteľ povinný preukázať dokladom o odoslaní alebo o preprave tovaru a  a) colným vyhlásením, v ktorom je colným orgánom potvrdený výstup tovaru z územia Európskej únie, alebo
relationships:
  Oslobodenie Od Dane -> [VZTAHUJE_SA_NA:"dodanie tovaru do miesta urcenia na uzemi tretieho statu"] -> Dodanie Tovaru
  Oslobodenie Od Dane -> [MA_PODMIENKU:"kupujuci nema v tuzemsku sidlo, miesto podnikania, prevadzkaren ani bydlisko"] -> Kupujuci
  Oslobodenie Od Dane -> [JE_VYNIMKOU:"dodanie tovaru, ktory prepravil kupujuci na ucely vybavenia, zasobenia pohonnymi latkami a potravinami vyletnych lodi, sukromnych lietadiel alebo dopravnych prostriedkov na sukromne pouzitie"] -> Dodanie Tovaru
  Platitel -> [MA_POVINNOST:"preukazat odoslanie alebo prepravu tovaru do miesta urcenia na uzemi tretieho statu"] -> Doklad O Odoslani Alebo O Preprave Tovaru
  Platitel -> [MA_POVINNOST:"preukazat odoslanie alebo prepravu tovaru colnym vyhlasenim s potvrdenym vystupom tovaru z uzemia Europskej unie"] -> Colne Vyhlasenie
  Odsek 2 -> [ODKAZUJE_NA:"odsek 2"] -> Oslobodenie Od Dane
  Odsek 3 -> [ODKAZUJE_NA:"odsek 3"] -> Doklad O Odoslani Alebo O Preprave Tovaru
  Odsek 3 -> [ODKAZUJE_NA:"odsek 3"] -> Colne Vyhlasenie
  Odsek 1 -> [ODKAZUJE_NA:"odsek 1"] -> Oslobodenie Od Dane
  Doklad O Odoslani Alebo O Preprave Tovaru -> [OBSAHUJE:"odoslanie alebo preprava tovaru"] -> Tovar
  Colne Vyhlasenie -> [OBSAHUJE:"potvrdeny vystup tovaru z uzemia Europskej unie"] -> Tovar
  Dodanie Tovaru -> [VZTAHUJE_SA_NA:"dodanie tovaru"] -> Tovar
  Dodanie Tovaru -> [VZTAHUJE_SA_NA:"miesto urcenia na uzemi tretieho statu"] -> Treti Stat
  Kupujuci -> [MA_SIDLO:"nemá sidlo v tuzemsku"] -> Tuzemsko
  Kupujuci -> [MA_MIESTO_PODNIKANIA:"nemá miesto podnikania v tuzemsku"] -> Tuzemsko
  Kupujuci -> [MA_PREVADZKAREN:"nemá prevadzkaren v tuzemsku"] -> Tuzemsko
  Kupujuci -> [MA_BYDLISKO:"nemá bydlisko v tuzemsku"] -> Tuzemsko
  Colne Vyhlasenie -> [ODKAZUJE_NA:"uzemie Europskej unie"] -> Eurpska Unia
  Oslobodenie Od Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodanie Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Doklad O Odoslani Alebo O Preprave Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Colne Vyhlasenie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 2 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 3 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 1 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Treti Stat -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tuzemsko -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Kupujuci -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Eurpska Unia -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Oslobodenieoddane: Oslobodenie Od Dane
  Cinnost: Dodanie Tovaru
  Tovar: Tovar
  Doklad: Doklad O Odoslani Alebo O Preprave Tovaru
  Doklad: Colne Vyhlasenie
  Odsek: Odsek 2
  Odsek: Odsek 3
  Odsek: Odsek 1
  Tretistat: Treti Stat
  Tuzemsko: Tuzemsko
  Subjekt: Kupujuci
  Subjekt: Platitel
  Uzemie: Eurpska Unia

chunk: 199
page: 51
text: Strana 52 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. § 48ca Oslobodenie od dane v colnom sklade (1) Osoba, ktorá dodá tovar s oslobodením od dane podľa § 48c ods. 1 písm. a), je povinná do desiatich dní odo dňa dodania tovaru oznámiť držiteľovi povolenia na prevádzkovanie colného skladu (ďalej len „prevádzkovateľ colného skladu“) obchodné meno alebo názov osoby, ktorej tovar dodala, dátum dodania tovaru a množstvo dodaného tovaru v metrických tonách. (2) Ak sa na tovar prestane vzťahovať colný režim colné uskladňovanie okrem ukončenia colného režimu colné uskladňovanie, ku ktorému dochádza v súvislosti s dodaním tovaru, oslobodenie od dane uplatnené na dodanie tovaru, ktoré tomu predchádzalo, a oslobodenie od dane na prijaté služby sa týmto okamihom zrušujú a osobou povinnou priznať a zaplatiť daň ku dňu, keď táto skutočnosť nastala, je osoba, ktorá spôsobí, že sa na tovar prestane vzťahovať colný režim colné
relationships:
  222/2004 Z. z. -> [OBSAHUJE] -> 48ca
  Oslobodenie Od Dane V Colnom Sklade -> [PODLA:"oslobodenie od dane v colnom sklade"] -> 48ca
  Osoba -> [JE_POVINNY_PLATIT:"dodanie tovaru s oslobodenim od dane"] -> Oslobodenie Od Dane V Colnom Sklade
  Osoba -> [MA_LEHOTU:"na oznamenie po dodani tovaru"] -> Desiatich Dni
  Osoba -> [OZNAMUJE:"obchodne meno alebo nazov osoby, datum dodania tovaru a mnozstvo dodaneho tovaru"] -> Prevadzkovatel Colneho Skladu
  Osoba -> [MA_DOKLAD:"oznamenie"] -> Obchodne Meno Alebo Nazov Osoby
  Osoba -> [MA_DATUM:"datum dodania tovaru"] -> Datum Dodania Tovaru
  Osoba -> [MA_MNOZSTVO:"dodaneho tovaru"] -> Mnozstvo Dodaneho Tovaru V Metrickych Tonach
  Dodanie Tovaru -> [TYKA_SA:"dodanie tovaru"] -> Tovar
  Osoba -> [VYKONAVA:"s oslobodenim od dane"] -> Dodanie Tovaru
  Oslobodenie Od Dane V Colnom Sklade -> [ZRUSUJE:"pri prestani vztaahu colneho rezimu"] -> Oslobodenie Od Dane Na Prijate Sluzby
  Osoba -> [JE_POVINNY_PLATIT:"priznat a zaplatit dan ku dnu ked skusutocnost nastala"] -> Oslobodenie Od Dane V Colnom Sklade
  Osoba -> [VZNIKA_PRI:"prestane vztaahovat na tovar"] -> Colny Rezim Colne Uskladnovanie
  222/2004 Z. z. -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  48ca -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oslobodenie Od Dane V Colnom Sklade -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prevadzkovatel Colneho Skladu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Obchodne Meno Alebo Nazov Osoby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Datum Dodania Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Mnozstvo Dodaneho Tovaru V Metrickych Tonach -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Desiatich Dni -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodanie Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oznamenit -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Priznat A Zaplatit Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oslobodenie Od Dane Na Prijate Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Colny Rezim Colne Uskladnovanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Pravnypredpis: 222/2004 Z. z.
  Paragraf: 48ca
  Dan: Oslobodenie Od Dane V Colnom Sklade
  Osoba: Osoba
  Tovar: Tovar
  Pravnickaosoba: Prevadzkovatel Colneho Skladu
  Doklad: Obchodne Meno Alebo Nazov Osoby
  Datum: Datum Dodania Tovaru
  Mnozstvo: Mnozstvo Dodaneho Tovaru V Metrickych Tonach
  Lehota: Desiatich Dni
  Cinnost: Dodanie Tovaru
  Cinnost: Oznamenit
  Cinnost: Priznat A Zaplatit Dan
  Dan: Oslobodenie Od Dane Na Prijate Sluzby
  Stav: Colny Rezim Colne Uskladnovanie

chunk: 240
page: 60
text: (6) Platiteľ, ktorý odviedol daň podľa odseku 5, má právo vykonať opravu opravenej odpočítanej dane, ak získa späť tovar, ktorý mu bol odcudzený, v rozsahu, v akom platiteľ použije späť získaný tovar na podnikanie s možnosťou odpočítania dane; opravu platiteľ vykoná v zdaňovacom období, v ktorom tovar získal späť, a to najviac do výšky dane odvedenej podľa odseku 5. (7) Ak bola vykonaná oprava sadzby dane, ktorá má za následok zníženie dane, je platiteľ, ktorý odpočítal daň, povinný opraviť odpočítanú daň v tom zdaňovacom období, v ktorom bola vykonaná oprava sadzby dane, alebo v prvom nasledujúcom zdaňovacom období. Ak bola vykonaná oprava sadzby dane, ktorá má za následok zvýšenie dane, má platiteľ, ktorý odpočítal daň, právo opraviť odpočítanú daň v tom zdaňovacom období, v ktorom bola vykonaná oprava sadzby dane, alebo v prvom nasledujúcom zdaňovacom období. Opravu sadzby dane a opravu odpočítanej dane nie je povinný vykonať platiteľ pri uplatnení nesprávnej sadzby dane pri nadobudnutí tovaru v tuzemsku
relationships:
  Odsek 6 -> [MA_PRAWO:"platitel ma pravo vykonat opravu opravenej odpoctanej dane"] -> Oprava Odpoctanej Dane
  Platitel -> [MA_PRAVO:"vykonat opravu opravenej odpoctanej dane"] -> Oprava Odpoctanej Dane
  Platitel -> [MA_POVINNOST:"opravi odpoctanu dan pri znizeni dane opravou sadzby dane"] -> Oprava Odpoctanej Dane
  Platitel -> [MA_PRAVO:"opravi odpoctanu dan pri zvyseni dane opravou sadzby dane"] -> Oprava Odpoctanej Dane
  Oprava Odpoctanej Dane -> [VYKONAVA:"v ktorom bolo vykonana oprava sadzby dane alebo tovar ziskany spat"] -> Zdanovacie Obdobie
  Oprava Odpoctanej Dane -> [MA_HODNOTU:"najviac do vysky dane odvedenej podla odseku 5"] -> Najviac Do Vysky Dane Odvedenej Podla Odseku 5
  Oprava Odpoctanej Dane -> [VYCHADZA_Z:"oprava sadzby dane ma za nasledok znizenie alebo zvysenie dane"] -> Oprava Sadzby Dane
  Oprava Odpoctanej Dane -> [VZTAHUJE_SA_NA:"ziskanie spat tovaru odcudzeneho platitelovi"] -> Tovar
  Oprava Odpoctanej Dane -> [VYKONAVA:"pouzitie spat ziskaneho tovaru na podnikanie s moznostou odpocitania dane"] -> Podnikanie
  Odsek 7 -> [MA_POVINNOST:"pri znizeni dane je platitel povinny opravit odpocitanu dan"] -> Oprava Odpoctanej Dane
  Odsek 7 -> [MA_PRAVO:"pri zvyseni dane ma platitel pravo opravit odpocitanu dan"] -> Oprava Odpoctanej Dane
  Odsek 7 -> [MA_VYNIMKU:"nepovinnost pri uplatneni nespravnej sadzby dane pri nadobudnuti tovaru v tuzemsku"] -> Oprava Sadzby Dane
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Opravena Odpoctitana Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oprava Odpoctanej Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oprava Sadzby Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Podnikanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanovacie Obdobie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 6 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 7 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Najviac Do Vysky Dane Odvedenej Podla Odseku 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Zdanitelnaosoba: Platitel
  Dan: Dan
  Dan: Opravena Odpoctitana Dan
  Oprava: Oprava Odpoctanej Dane
  Oprava: Oprava Sadzby Dane
  Tovar: Tovar
  Cinnost: Podnikanie
  Zdanovacieobdobie: Zdanovacie Obdobie
  Odsek: Odsek 5
  Odsek: Odsek 6
  Odsek: Odsek 7
  Limit: Najviac Do Vysky Dane Odvedenej Podla Odseku 5

chunk: 291
page: 72
text: b) údaje uvedené v žiadosti o vrátenie dane sú pravdivé, c) sa zaväzuje vrátiť späť akúkoľvek neoprávnene vrátenú daň. § 58 (1) Daňový úrad Bratislava rozhodne o žiadosti o vrátenie dane do šiestich mesiacov odo dňa podania žiadosti. Faktúry a dovozné doklady priložené k žiadosti o vrátenie dane Daňový úrad Bratislava vráti zahraničnej osobe z tretieho štátu do 60 dní od ich predloženia; tieto faktúry a dovozné doklady môže pred vrátením označiť. (2) Ak Daňový úrad Bratislava rozhodne o vrátení dane, vráti daň v lehote na rozhodnutie o žiadosti podľa odseku 1. Daň vráti v eurách na účet vedený v banke v tuzemsku alebo na základe žiadosti zahraničnej osoby z tretieho štátu na účet vedený v zahraničnej banke v inom
relationships:
  Ziadost O Vratenie Dane -> [MA_PREDMETOM:"ziadost o vratenie dane"] -> Vratenie Dane
  Daňovy Urad Bratislava -> [ROZHODUJE_O:"do siestich mesiacov odo dna podania ziadosti"] -> Ziadost O Vratenie Dane
  Ziadost O Vratenie Dane -> [MA_LEHOTU:"podanie ziadosti"] -> Siestich Mesiacov Odo Dna Podania Ziadosti
  Faktury -> [JE_SUCASTOU:"prilozene k ziadosti o vratenie dane"] -> Ziadost O Vratenie Dane
  Dovozne Doklady -> [JE_SUCASTOU:"prilozene k ziadosti o vratenie dane"] -> Ziadost O Vratenie Dane
  Daňovy Urad Bratislava -> [VRATI:"vrati zahranicnej osobe z tretieho statu"] -> Faktury
  Daňovy Urad Bratislava -> [VRATI:"vrati zahranicnej osobe z tretieho statu"] -> Dovozne Doklady
  Faktury -> [MA_LEHOTU:"pred vratenim"] -> 60 dni od ich predlozenia
  Dovozne Doklady -> [MA_LEHOTU:"pred vratenim"] -> 60 dni od ich predlozenia
  Daňovy Urad Bratislava -> [ROZHODUJE_O:"ak rozhodne o vrateni dane"] -> Vratenie Dane
  Daňovy Urad Bratislava -> [VRATI:"v lehote na rozhodnutie o ziadosti"] -> Vratenie Dane
  Vratenie Dane -> [MA_MIESTO:"ucet vedeny v banke v tuzemsku"] -> Banka V Tuzemsku
  Vratenie Dane -> [MA_MIESTO:"ucet vedeny v zahranicnej banke"] -> Zahranicna Banka
  Vratenie Dane -> [PLATI_PRE:"na zaklade ziadosti"] -> Zahranicna Osoba Z Tretieho Statu
  Vratenie Dane -> [MA_VLASTNOST:"vratenie dane v eurach"] -> V Eurach
  Ziadost O Vratenie Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vratenie Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Neopravnene Vratena Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Daňovy Urad Bratislava -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Faktury -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dovozne Doklady -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zahranicna Osoba Z Tretieho Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ucty Vedeny V Banke V Tuzemsku -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ucet Vedeny V Zahranicnej Banke -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Banka V Tuzemsku -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zahranicna Banka -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Siestich Mesiacov Odo Dna Podania Ziadosti -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  60 dni od ich predlozenia -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  V Eurach -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  O Vratenie Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Ziadost: Ziadost O Vratenie Dane
  Dan: Vratenie Dane
  Dan: Neopravnene Vratena Dan
  Urad: Daňovy Urad Bratislava
  Doklad: Faktury
  Doklad: Dovozne Doklady
  Osoba: Zahranicna Osoba Z Tretieho Statu
  Bankovyucet: Ucty Vedeny V Banke V Tuzemsku
  Bankovyucet: Ucet Vedeny V Zahranicnej Banke
  Banka: Banka V Tuzemsku
  Banka: Zahranicna Banka
  Casovyudaj: Siestich Mesiacov Odo Dna Podania Ziadosti
  Lehota: 60 dni od ich predlozenia
  Mena: V Eurach
  Povinnost: O Vratenie Dane

chunk: 318
page: 79
text: republiky, k žiadosti sa musia doložiť aj doklady preukazujúce uvedenie základu dane a dane z týchto tovarov a služieb v daňovom priznaní. § 64 Vrátenie dane neziskovým organizáciám poskytujúcim všeobecne prospešné služby a Slovenskému Červenému krížu (1) Nezisková organizácia poskytujúca všeobecne prospešné služby28) a Slovenský Červený kríž môžu požiadať o vrátenie dane zaplatenej v cene tovaru, ktorý vyviezli mimo územia Európskej únie na humanitárnu, dobročinnú alebo vzdelávaciu činnosť. (2) Nezisková organizácia poskytujúca všeobecne prospešné služby a Slovenský Červený kríž uplatňujú nárok na vrátenie dane podaním žiadosti Daňovému úradu Bratislava. K žiadosti o vrátenie dane sa musí doložiť a) doklad o kúpe tovaru od platiteľa, v ktorom je uvedená suma dane v eurách s potvrdením platby dane, b) colné vyhlásenie o vývoze tovaru. Osobitná úprava uplatňovania dane § 64a Na miesto dodania služby, vznik daňovej povinnosti, základ dane, opravu základu dane,
relationships:
  § 64 -> [OBSAHUJE:"upravuje vymoze vrateni dane"] -> Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu
  Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby -> [MA_PRAVO:"narok na vratenie dane"] -> Vratenie Dane
  Slovensky Cerveny Kriz -> [MA_PRAVO:"narok na vratenie dane"] -> Vratenie Dane
  Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby -> [PREDKLADA:"k ziadosti o vratenie dane"] -> Doklad O Kupe Tovaru
  Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby -> [PREDKLADA:"k ziadosti o vratenie dane"] -> Colne Vyhlasenie O Vyvoze Tovaru
  Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby -> [PREDKLADA:"doklad obsahuje sumu dane v eurach"] -> Suma Dane V Eurach
  Doklad O Kupe Tovaru -> [MA_SUMU:"uvedena suma dane v eurach"] -> Suma Dane V Eurach
  Suma Dane V Eurach -> [MA_MIESTO:"mena suma dane"] -> Eurach
  § 64 -> [VYPLNYVA_Z:"vratenie dane zaplatenej v cene tovaru"] -> Vratenie Dane Zaplatenej V Cene Tovaru
  § 64 -> [VYPLNYVA_Z:"dane z tychto tovarov a sluzieb v danovom priznani"] -> Dane Z Tychto Tovarov A Sluzieb
  Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu -> [MA_PODMINKU:"vyvoz tovaru mimo uzemia EU na humanitarna cinnost"] -> Humanitarna Cinnost
  Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu -> [MA_PODMINKU:"vyvoz tovaru mimo uzemia EU na dobrocinna cinnost"] -> Dobrocinna Cinnost
  Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu -> [MA_PODMINKU:"vyvoz tovaru mimo uzemia EU na vzdelavacia cinnost"] -> Vzdelavacia Cinnost
  Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu -> [PODLIEHA:"ziadost sa podava danovemu uradu Bratislava"] -> Danovy Urad Bratislava
  § 64A -> [OBSAHUJE:"osobitna uprava uplatnovania dane"] -> Vratenie Dane
  § 64 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 64A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Slovensky Cerveny Kriz -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy Urad Bratislava -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zostavovanie Dokladov Preukazujucich Uvedenie Zakladu Dane A Dane V Danovom Priznani -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Doklad O Kupe Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Colne Vyhlasenie O Vyvoze Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Daňove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vratenie Dane Zaplatenej V Cene Tovaru -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dane Z Tychto Tovarov A Sluzieb -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Suma Dane V Eurach -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Eurach -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Humanitarna Cinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dobrocinna Cinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vzdelavacia Cinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vratenie Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Paragraf: § 64
  Paragraf: § 64A
  Cinnost: Vratenie Dane Neziskovym Organizaciam Poskytujucim Vseobecne Prospesne Sluzby A Slovenskemu Cervenemu Krizu
  Organizacia: Neziskova Organizacia Poskytujuca Vseobecne Prospesne Sluzby
  Organizacia: Slovensky Cerveny Kriz
  Urad: Danovy Urad Bratislava
  Cinnost: Zostavovanie Dokladov Preukazujucich Uvedenie Zakladu Dane A Dane V Danovom Priznani
  Doklad: Doklad O Kupe Tovaru
  Doklad: Colne Vyhlasenie O Vyvoze Tovaru
  Danovepriznanie: Daňove Priznanie
  Dan: Vratenie Dane Zaplatenej V Cene Tovaru
  Dan: Dane Z Tychto Tovarov A Sluzieb
  Suma: Suma Dane V Eurach
  Euro: Eurach
  Cinnost: Humanitarna Cinnost
  Cinnost: Dobrocinna Cinnost
  Cinnost: Vzdelavacia Cinnost
  Pravo: Vratenie Dane

chunk: 356
page: 88
text: v odseku 2 povinná uviesť v nasledujúcom daňovom priznaní najneskôr do troch rokov odo dňa uplynutia lehoty na podanie pôvodného daňového priznania. V tomto nasledujúcom daňovom priznaní zdaniteľná osoba uvedie príslušný členský štát spotreby, zdaňovacie obdobie a sumu
relationships:
  Povinna -> [MA_POVINNOST:"uviest v nasledujucom danovom priznani do troch rokov odo dna uplynutia lehoty na podanie povodneho danoveho priznania"] -> Nasledujuce Danove Priznanie
  Nasledujuce Danove Priznanie -> [MA_LEHOTU:"najneskorsie do"] -> Tri Roky
  Tri Roky -> [VYCHADZA_Z:"od uplynutia"] -> Lehoty Na Podanie Povodneho Danoveho Priznania
  Zdanitelna Osoba -> [UVADZA:"v nasledujucom danovom priznani"] -> Prislusny Clensky Stat Spotreby
  Zdanitelna Osoba -> [UVADZA:"v nasledujucom danovom priznani"] -> Zdanovacie Obdobie
  Zdanitelna Osoba -> [UVADZA:"v nasledujucom danovom priznani"] -> Suma
  Odsek 2 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Povinna -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Nasledujuce Danove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Lehoty Na Podanie Povodneho Danoveho Priznania -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tri Roky -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanitelna Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prislusny Clensky Stat Spotreby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanovacie Obdobie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Suma -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Odsek: Odsek 2
  Pravnickaosoba: Povinna
  Danovepriznanie: Nasledujuce Danove Priznanie
  Lehota: Lehoty Na Podanie Povodneho Danoveho Priznania
  Casovyudaj: Tri Roky
  Zdanitelnaosoba: Zdanitelna Osoba
  Clenskystat: Prislusny Clensky Stat Spotreby
  Zdanovacieobdobie: Zdanovacie Obdobie
  Suma: Suma

chunk: 379
page: 94
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 95 6 písm. b) až d) týka nesprávne uvedenej výšky dane alebo dane neuvedenej v daňovom priznaní, konečnom daňovom priznaní alebo v daňovom priznaní podanom pred podaním konečného daňového priznania, časť osobitného tlačiva sa v rozsahu týchto opráv považuje za dodatočné daňové priznanie. (9) Osoba podľa odseku 6 je povinná v lehote na podanie osobitného tlačiva podľa odseku 6 písm. a) až c) zaplatiť daň vypočítanú z údajov uvedených podľa odseku 7 písm. a) alebo vyplývajúcu z opravy podľa odseku 7 písm. b). § 68cb Osobitná úprava na priznávanie a platbu dane pri dovoze tovaru (1) Ak sa na tovar, ktorý nie je predmetom spotrebnej dane, dovážaný do tuzemska v zásielke, ktorej vlastná hodnota nepresahuje 150 eur, neuplatní osobitná úprava podľa § 68c, a odoslanie alebo preprava tohto tovaru skončí v tuzemsku, colný úrad na základe žiadosti povolí osobe, ktorá mu predkladá tovar na účet osoby, pre ktorú je tovar určený, uplatňovanie osobitnej úpravy podľa
relationships:
  Zakon 222/2004 Z. Z. -> [OBSAHUJE:"§ 68cb"] -> Paragraf 68Cb
  Paragraf 68Cb -> [MA_NAROK_NA:"cast osobitneho tlaciva sa považuje za dodatocne daove priznanie"] -> Dodatocne Daove Priznanie
  Osoba Podla Odseku 6 -> [JE_POVINNY_PLATIT:"zaplatit dan vypocitanu z udajov alebo z opravy"] -> Dan
  Osoba Podla Odseku 6 -> [MA_LEHOTU:"lehota na podanie osobitneho tlaciva"] -> Lehota Na Podanie Osobitneho Tlaciva
  Paragraf 68Cb -> [VZTAHUJE_SA_NA:"tovar nie je predmetom spotrebnej dane"] -> Tovar
  Paragraf 68Cb -> [VZTAHUJE_SA_NA:"dovazany do tuzemska"] -> Zasielka
  Paragraf 68Cb -> [VZTAHUJE_SA_NA:"odoslanie alebo preprava skonci v tuzemske"] -> Tuzemsko
  Paragraf 68Cb -> [VZTAHUJE_SA_NA:"vlastna hodnota nepresahuje 150 eur"] -> Vlastna Hodnota 150 Eur
  Colny Urad -> [ROZHODUJE_O:"na zaklade ziadosti povoli uplatnovanie osobitnej upravy"] -> Ziadost
  Zakon 222/2004 Z. Z. -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 68Cb -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Daove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Konecne Daove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitne Tlacivo -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodatocne Daove Priznanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba Podla Odseku 6 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Lehota Na Podanie Osobitneho Tlaciva -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tuzemsko -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zasielka -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vlastna Hodnota 150 Eur -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ziadost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Colny Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Pravnypredpis: Zakon 222/2004 Z. Z.
  Paragraf: Paragraf 68Cb
  Danovepriznanie: Daove Priznanie
  Danovepriznanie: Konecne Daove Priznanie
  Doklad: Osobitne Tlacivo
  Danovepriznanie: Dodatocne Daove Priznanie
  Osoba: Osoba Podla Odseku 6
  Lehota: Lehota Na Podanie Osobitneho Tlaciva
  Dan: Dan
  Tovar: Tovar
  Tuzemsko: Tuzemsko
  Zasielka: Zasielka
  Hodnota: Vlastna Hodnota 150 Eur
  Ziadost: Ziadost
  Urad: Colny Urad

chunk: 406
page: 100
text: odvolanie. (6) Ak členský štát podľa odseku 2 písm. b) spôsobom podľa osobitného predpisu potvrdil, že malý podnik tuzemskej osoby nespĺňa podmienky pre uplatňovanie osobitnej úpravy na jeho území, na podanie opravného prostriedku malým podnikom tuzemskej osoby sa vzťahuje právny predpis tohto členského štátu. (7) Ak zdaniteľná osoba podľa odseku 2 nie je malým podnikom tuzemskej osoby, daňový úrad jej nepridelí individuálne identifikačné číslo s príponou EX. Rozhodnutie podľa prvej vety musí byť zdaniteľnej osobe doručené do 35 pracovných dní odo dňa doručenia oznámenia podľa odseku 2 alebo do 35 pracovných dní odo dňa doručenia opravného oznámenia podľa odseku 4. (8) Malý podnik tuzemskej osoby je povinný do konca kalendárneho mesiaca nasledujúceho po skončení kalendárneho štvrťroka, v ktorom uplatňoval osobitnú úpravu, podať výkaz na tlačive, ktorého vzor určí a uverejní finančné riaditeľstvo na webovom sídle finančného riaditeľstva; ak
relationships:
  Clensky_Stat -> [PODLA:"podla osobitneho predpisu"] -> Osobitny_Predpis
  Maly_Podnik_Tuzemskej_Osoby -> [JE_PODLA:"na podanie opravneho prostriedku"] -> Pravny_Predpis_Tohto_Clenskeho_Statu
  Danovy_Urad -> [NEPRIDELUJE:"ak zdanitelna osoba nie je malym podnikom tuzemskej osoby"] -> Individualne_Identifikacne_Cislo_S_Priponou_Ex
  Rozhodnutie -> [MA_LEHOTU:"dorucenie zdanitelnej osobe"] -> 35_pracovnych_dni
  Oznamenie -> [MA_LEHOTU:"odo dna dorucenia oznamnenia"] -> 35_pracovnych_dni
  Opravne_Oznamenie -> [MA_LEHOTU:"odo dna dorucenia opravneho oznamnenia"] -> 35_pracovnych_dni
  Maly_Podnik_Tuzemskej_Osoby -> [MA_POVINNOST:"podat vykaz"] -> Vykaz
  Vykaz -> [MA_DOKLAD:"na tlacive"] -> Tlacivo
  Financne_Riaditelstvo -> [VYDAVA:"urci a uverejni vzor tlaciva"] -> Tlacivo
  Financne_Riaditelstvo -> [UVADZA:"uverejnenie na webovom sidle"] -> Webove_Sidlo_Financneho_Riaditelstva
  Zdanitelna_Osoba -> [MA_STATUS:"uplatnovanie osobitnej upravy"] -> Osobitna_Uprava
  Podanie_Opravneho_Prostriedku -> [VZTAHUJE_SA_NA:"na jeho uzemi"] -> Vnutorne_Uzemie_Clenskeho_Statu
  Podanie_Opravneho_Prostriedku -> [PODLIEHA:"na podanie opravneho prostriedku malym podnikom tuzemskej osoby"] -> Pravny_Predpis_Tohto_Clenskeho_Statu
  Uplatnovanie_Osobitnej_Upravy -> [VZTAHUJE_SA_NA:"ak uplatnoval osobitnu upravu"] -> Osobitna_Uprava
  Odvolanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Clensky_Stat -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitny_Predpis -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tuzemska_Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Maly_Podnik_Tuzemskej_Osoby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pravny_Predpis_Tohto_Clenskeho_Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanitelna_Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy_Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Individualne_Identifikacne_Cislo_S_Priponou_Ex -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Rozhodnutie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oznamenie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Opravne_Oznamenie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  35_pracovnych_dni -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Koniec_Kalendarneho_Mesiaca_Nasledujuceho_Po_Skonceni_Kalendarneho_Stvrtroka -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vykaz -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tlacivo -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Financne_Riaditelstvo -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Webove_Sidlo_Financneho_Riaditelstva -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitna_Uprava -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Opravny_Prostriedok -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vnutorne_Uzemie_Clenskeho_Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Podanie_Opravneho_Prostriedku -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Uplatnovanie_Osobitnej_Upravy -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Doklad: Odvolanie
  Clenskystat: Clensky_Stat
  Pravnypredpis: Osobitny_Predpis
  Zdanitelnaosoba: Tuzemska_Osoba
  Podnik: Maly_Podnik_Tuzemskej_Osoby
  Pravnypredpis: Pravny_Predpis_Tohto_Clenskeho_Statu
  Zdanitelnaosoba: Zdanitelna_Osoba
  Urad: Danovy_Urad
  Identifikacnecislo: Individualne_Identifikacne_Cislo_S_Priponou_Ex
  Rozhodnutie: Rozhodnutie
  Oznamenie: Oznamenie
  Oznamenie: Opravne_Oznamenie
  Lehota: 35_pracovnych_dni
  Lehota: Koniec_Kalendarneho_Mesiaca_Nasledujuceho_Po_Skonceni_Kalendarneho_Stvrtroka
  Doklad: Vykaz
  Doklad: Tlacivo
  Financneriaditelstvo: Financne_Riaditelstvo
  Sidlo: Webove_Sidlo_Financneho_Riaditelstva
  Status: Osobitna_Uprava
  Doklad: Opravny_Prostriedok
  Uzemie: Vnutorne_Uzemie_Clenskeho_Statu
  Cinnost: Podanie_Opravneho_Prostriedku
  Cinnost: Uplatnovanie_Osobitnej_Upravy

chunk: 41
page: 10
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 11 b) sa stal platiteľom pred doručením rozhodnutia o registrácii pre daň podľa § 4 alebo § 5, a to počnúc dňom doručenia tohto rozhodnutia alebo c) je skupinou, počnúc dňom, ku ktorému daňový úrad vykoná registráciu skupiny. § 6 Osobitná oznamovacia povinnosť platiteľa (1) Platiteľ je povinný oznámiť spôsobom podľa odseku 5 finančnému riaditeľstvu každý vlastný účet vedený u poskytovateľa platobných služieb alebo u zahraničného poskytovateľa platobných služieb, ktorý bude používať na podnikanie, ktoré je predmetom dane podľa § 2 (ďalej len „bankový účet“), a to bezodkladne odo dňa, keď sa stal platiteľom, ktorý má pridelené identifikačné číslo pre daň podľa § 4, § 4b, § 4c alebo § 5, alebo odo dňa, keď si takýto bankový účet následne zriadil. (2) Ak okrem bankového účtu oznámeného podľa odseku 1 alebo podľa § 85kk chce platiteľ používať aj iný bankový účet, je povinný tento účet oznámiť spôsobom podľa odseku 5 finančnému
relationships:
  Zakon 222/2004 Z. Z. O Dani Z Pridanej Hodnoty -> [OBSAHUJE:"§ 6"] -> Paragraf 6
  Paragraf 6 -> [OBSAHUJE:"(1)"] -> Odsek 1
  Paragraf 6 -> [OBSAHUJE:"(2)"] -> Odsek 2
  Platitel -> [MA_POVINNOST:"oznamit kazdy vlastny ucet"] -> Osobitna Oznamovacia Povinnost Platitela
  Osobitna Oznamovacia Povinnost Platitela -> [PODLA:"§ 6 ods. 1"] -> Odsek 1
  Platitel -> [MA_POVINNOST:"oznamit aj iny bankovy ucet"] -> Osobitna Oznamovacia Povinnost Platitela
  Osobitna Oznamovacia Povinnost Platitela -> [PODLA:"§ 6 ods. 2"] -> Odsek 2
  Platitel -> [PODAVA:"oznamenie sposobu podla odseku 5"] -> Financnemu Riaditelstvu
  Platitel -> [MA_POVINNOST:"oznamit bankovy ucet na podnikanie"] -> Bankovy Ucet
  Bankovy Ucet -> [MA_UCEL:"na podnikanie, ktore je predmetom dane"] -> Podnikanie
  Podnikanie -> [VZTAHUJE_SA_NA:"predmet dane podla § 2"] -> Dan Podla Paragrafu 2
  Platitel -> [MA_IDENTIFIKACNE_CISLO:"podla § 4, § 4b, § 4c alebo § 5"] -> Identifikacne Cislo Pre Dan
  Zakon 222/2004 Z. Z. O Dani Z Pridanej Hodnoty -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 6 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitna Oznamovacia Povinnost Platitela -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Financnemu Riaditelstvu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Bankovy Ucet -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Podnikanie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dan Podla Paragrafu 2 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Identifikacne Cislo Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Poskytovatel Platobnych Sluzieb -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zahranicny Poskytovatel Platobnych Sluzieb -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Daňový Úrad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 4 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 85Kk -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 1 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 2 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Zakon: Zakon 222/2004 Z. Z. O Dani Z Pridanej Hodnoty
  Paragraf: Paragraf 6
  Cinnost: Osobitna Oznamovacia Povinnost Platitela
  Osoba: Platitel
  Financneriaditelstvo: Financnemu Riaditelstvu
  Bankovyucet: Bankovy Ucet
  Cinnost: Podnikanie
  Dan: Dan Podla Paragrafu 2
  Identifikacnecislo: Identifikacne Cislo Pre Dan
  Subjekt: Poskytovatel Platobnych Sluzieb
  Subjekt: Zahranicny Poskytovatel Platobnych Sluzieb
  Urad: Daňový Úrad
  Paragraf: § 4
  Paragraf: § 5
  Paragraf: § 85Kk
  Odsek: Odsek 1
  Odsek: Odsek 2
  Odsek: Odsek 5
  Osoba: Osoba

chunk: 436
page: 107
text: 4 začne daňovú kontrolu, tento daňový preplatok vráti do desiatich dní od skončenia daňovej kontroly, ak sa daňovou kontrolou nezistil rozdiel vo výške tohto daňového preplatku, inak do desiatich dní od nadobudnutia právoplatnosti rozhodnutia. (6) Daňový úrad vráti daňový preplatok podľa odseku 3 alebo podľa odseku 4 na bankový účet oznámený podľa § 6 ods. 1 až 3 alebo podľa § 85kk spôsobom podľa § 6 ods. 5 a ak dodávateľ nesplnil takúto povinnosť, daňový úrad vráti daňový preplatok do desiatich dní odo dňa, keď dodávateľ takúto povinnosť dodatočne splnil; postup podľa osobitného predpisu týmto nie je dotknutý.27bd) Povinnosti osôb povinných platiť daň § 70 Vedenie záznamov (1) Platiteľ je povinný viesť podrobné záznamy podľa jednotlivých zdaňovacích období o dodaných tovaroch a službách a o prijatých tovaroch a službách; osobitne vedie záznamy o dodaní tovarov a služieb do iného členského štátu, o nadobudnutí tovaru z iného členského
relationships:
  Danovy Urad -> [VYDAVA:"vratenie danoveho preplatku na bankovy ucet"] -> Danovy Preplatok
  Danovy Urad -> [VYDAVA:"vratenie danoveho preplatku na oznameny bankovy ucet"] -> Bankovy Ucet
  Danovy Preplatok -> [JE_PREDMETOM:"podla odseku 3"] -> Odsek 3
  Danovy Preplatok -> [JE_PREDMETOM:"podla odseku 4"] -> Odsek 4
  Danovy Preplatok -> [MA_LEHOTU:"vratenie do desiatich dni od skoncenia danovej kontroly alebo od nadobudnutia pravoplatnosti rozhodnutia"] -> Desiatich Dni
  § 70 -> [OBSAHUJE:"vedenie zaznamov"] -> Vedenie Zaznamov
  Platitel -> [MA_POVINNOST:"vies podrobne zaznamy"] -> Vedenie Zaznamov
  Vedenie Zaznamov -> [JE_PREDMETOM:"o dodanych tovaroch a sluzbach"] -> Dodane Tovary A Sluzby
  Vedenie Zaznamov -> [JE_PREDMETOM:"o prijatych tovaroch a sluzbach"] -> Prijate Tovary A Sluzby
  Vedenie Zaznamov -> [JE_PREDMETOM:"osobitne vedie zaznamy"] -> Dodaniz Tovarov A Sluzieb Do Ineho Clenskeho Statu
  Vedenie Zaznamov -> [JE_PREDMETOM:"osobitne vedie zaznamy"] -> Nadobudnutie Tovaru Z Ineho Clenskeho Statu
  Danovy Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Bankovy Ucet -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy Preplatok -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 3 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 4 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 6 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 70 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vedenie Zaznamov -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodane Tovary A Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prijate Tovary A Sluzby -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodaniz Tovarov A Sluzieb Do Ineho Clenskeho Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Nadobudnutie Tovaru Z Ineho Clenskeho Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Desiatich Dni -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Urad: Danovy Urad
  Bankovyucet: Bankovy Ucet
  Nadmernyodpocet: Danovy Preplatok
  Odsek: Odsek 3
  Odsek: Odsek 4
  Odsek: Odsek 6
  Paragraf: § 70
  Cinnost: Vedenie Zaznamov
  Zdanitelnaosoba: Platitel
  Tovar: Dodane Tovary A Sluzby
  Tovar: Prijate Tovary A Sluzby
  Cinnost: Dodaniz Tovarov A Sluzieb Do Ineho Clenskeho Statu
  Cinnost: Nadobudnutie Tovaru Z Ineho Clenskeho Statu
  Lehota: Desiatich Dni

chunk: 465
page: 115
text: (5) Za identifikačné číslo pre daň sa na účely vyhotovenia zjednodušenej faktúry podľa odseku 3 písm. b) považuje aj daňové identifikačné číslo, ktoré bolo platiteľovi pridelené podľa osobitného predpisu,29aaa) alebo identifikačné číslo, ktoré bolo platiteľovi pridelené podľa osobitného predpisu,29aab) ak ku dňu vyhotovenia tejto zjednodušenej faktúry platiteľ, ktorý splnil registračnú povinnosť, nemá pridelené identifikačné číslo pre daň podľa § 4 alebo § 5. (6) Ak tovar alebo službu dodáva skupina, uvedie sa vo faktúre ako údaj podľa odseku 1 písm. a) meno a adresa sídla, miesta podnikania, prípadne prevádzkarne člena skupiny, ktorý dodáva tovar alebo službu, a identifikačné číslo pre daň skupiny. (7) Ak elektronické faktúry sú poslané alebo sprístupnené rovnakému príjemcovi v jednom súbore, môžu sa údaje, ktoré sú spoločné pre jednotlivé faktúry, uviesť len raz, ak sú pre každú faktúru prístupné všetky informácie. § 75 Súhrnná faktúra
relationships:
  Zjednodusena Faktura -> [PODLA:"vyhotovenie zjednodusenej faktury"] -> Odsek 3
  Zjednodusena Faktura -> [PODLA:"odsek 3 pismeno b"] -> Pismeno B
  Zjednodusena Faktura -> [MA_PRAVO:"povazuje sa za identifikacne cislo pre dan"] -> Identifikacne Cislo Pre Dan
  Danove Identifikacne Cislo -> [JE_TYPOM:"pre dan"] -> Identifikacne Cislo Pre Dan
  Danove Identifikacne Cislo -> [PODLA:"pridelenie platitelovi"] -> Osobitny Predpis
  Identifikacne Cislo Pre Dan -> [PODLA:"pridelenie identifikacneho cisla pre dan"] -> § 4
  Identifikacne Cislo Pre Dan -> [PODLA:"pridelenie identifikacneho cisla pre dan"] -> § 5
  Platitel -> [MA_POVINNOST:"splnil registracnu povinnost"] -> Registacna Povinnost
  Tovar -> [DODA:"ak tovar dodava skupina"] -> Skupina
  Sluzba -> [DODA:"ak sluzbu dodava skupina"] -> Skupina
  Skupina -> [MA_ZASTUPCU:"clen skupiny, ktory dodava tovar alebo sluzbu"] -> Clen Skupiny
  Skupina -> [MA_IDENTIFIKACNE_CISLO:"identifikacne cislo pre dan skupiny"] -> Identifikacne Cislo Pre Dan
  Elektronicke Faktury -> [VZTAHUJE_SA_NA:"poslane alebo spristupnene rovnakemu prijemcovi v jednom subore"] -> Jeden Subor
  Elektronicke Faktury -> [MA_PRAVO:"rovnaky prijemca"] -> Prijemca
  § 75 -> [MA_NAZOV:"suhrrna faktura"] -> Suhrrna Faktura
  Identifikacne Cislo Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zjednodusena Faktura -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 3 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pismeno B -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danove Identifikacne Cislo -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitny Predpis -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Registacna Povinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 4 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tovar -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Sluzba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Skupina -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Clen Skupiny -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Faktura -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Elektronicke Faktury -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prijemca -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Jeden Subor -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 75 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Suhrrna Faktura -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Identifikacnecislo: Identifikacne Cislo Pre Dan
  Doklad: Zjednodusena Faktura
  Odsek: Odsek 3
  Pismeno: Pismeno B
  Identifikacnecislo: Danove Identifikacne Cislo
  Zdanitelnaosoba: Platitel
  Pravnypredpis: Osobitny Predpis
  Povinnost: Registacna Povinnost
  Paragraf: § 4
  Paragraf: § 5
  Tovar: Tovar
  Sluzba: Sluzba
  Skupina: Skupina
  Clenskupiny: Clen Skupiny
  Doklad: Faktura
  Doklad: Elektronicke Faktury
  Subjekt: Prijemca
  Doklad: Jeden Subor
  Paragraf: § 75
  Doklad: Suhrrna Faktura

chunk: 494
page: 122
text: (13) Ak nebol kontrolný výkaz podaný v lehote podľa odseku 1, daňový úrad vyzve platiteľa na jeho podanie. Ak vzniknú pochybnosti o správnosti, pravdivosti alebo úplnosti podaného kontrolného výkazu alebo o pravdivosti údajov v ňom uvedených, oznámi daňový úrad tieto pochybnosti platiteľovi, ktorý kontrolný výkaz podal, a vyzve ho, aby sa k nim vyjadril, neúplné údaje doplnil, nejasnosti vysvetlil a nepravdivé údaje opravil alebo pravdivosť údajov riadne preukázal. Na základe tejto výzvy je platiteľ povinný do piatich pracovných dní od doručenia výzvy nedostatky podaného kontrolného výkazu odstrániť. (14) Platiteľ, ktorý nesplnil registračnú povinnosť, je povinný podať kontrolný výkaz za každé zdaňovacie obdobie, za ktoré v dôsledku nesplnenia tejto povinnosti nepodal kontrolný výkaz v lehote podľa odseku 1, v chronologickom poradí počnúc prvým zdaňovacím obdobím. (15) Ak platiteľ nedoručí kontrolný výkaz daňovému úradu, doručí kontrolný výkaz oneskorene,
relationships:
  Danovy Urad -> [VYZIVA:"podanie kontrolneho vykazu"] -> Platitel
  Danovy Urad -> [OZNAMUJE:"pochybnosti o spravnosti, pravdivosti alebo uplnosti podaneho kontrolneho vykazu alebo o pravdivosti udajov v nom uvedenych"] -> Platitel
  Platitel -> [MA_POVINNOST:"vyjadrit sa, doplnit neuplne udaje, vysvetlit nejasnosti, opravit nepravdive udaje alebo preukazat pravdivost udajov riadne"] -> Vyzva
  Platitel -> [MA_POVINNOST:"podat kontrolny vykaz v lehote"] -> Lehota Podla Odseku 1
  Platitel -> [MA_POVINNOST:"odstranit nedostatky podaneho kontrolneho vykazu"] -> Piatich Pracovnych Dni Od Dorucenia Vyzvy
  Platitel -> [MA_POVINNOST:"podat kontrolny vykaz za kazde obdobie po nesplneni registracnej povinnosti"] -> Zdanovacie Obdobie
  Kontrolny Vykaz -> [PODLIEHA:"podanie kontrolneho vykazu"] -> Lehota Podla Odseku 1
  Kontrolny Vykaz -> [DORUCUJE:"doručenie kontrolneho vykazu"] -> Danovy Urad
  Kontrolny Vykaz -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Lehota Podla Odseku 1 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Piatich Pracovnych Dni Od Dorucenia Vyzvy -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Registračna Povinnost -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanovacie Obdobie -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vyzva -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pochybnosti O Spravnosti, Pravdivosti Alebo Uplnosti Podaneho Kontrolneho Vykazu Alebo O Pravdivosti Udajov V Nom Uvedenych -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Doklad: Kontrolny Vykaz
  Urad: Danovy Urad
  Zdanitelnaosoba: Platitel
  Lehota: Lehota Podla Odseku 1
  Lehota: Piatich Pracovnych Dni Od Dorucenia Vyzvy
  Povinnost: Registračna Povinnost
  Zdanovacieobdobie: Zdanovacie Obdobie
  Vyzva: Vyzva
  Dovod: Pochybnosti O Spravnosti, Pravdivosti Alebo Uplnosti Podaneho Kontrolneho Vykazu Alebo O Pravdivosti Udajov V Nom Uvedenych

chunk: 523
page: 129
text: Strana 130 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. moci, ak osoba podľa odseku 2 nesplnila povinnosť podať žiadosť o zrušenie registrácie pre daň; proti rozhodnutiu o zrušení registrácie nie je možné podať odvolanie. Daňový úrad zruší registráciu pre daň z úradnej moci, ak sa osoba registrovaná podľa § 7 alebo § 7a stala platiteľom, a to ku dňu, keď sa stala platiteľom; daňový úrad rozhodnutie o zrušení registrácie nevydáva. Zrušením registrácie zaniká platnosť identifikačného čísla pre daň; ak právnická osoba alebo fyzická osoba uplatňuje osobitnú úpravu podľa § 68b, platnosť identifikačného čísla pre daň na účely uplatňovania tejto osobitnej úpravy nezaniká. § 83 (1) Fyzická osoba, ktorá pokračuje v podnikaní po úmrtí platiteľa32) (ďalej len „osoba pokračujúca v podnikaní“), oznámi túto skutočnosť daňovému úradu, ktorý bol príslušný poručiteľovi, najneskôr do 30 dní odo dňa jeho úmrtia. Daňový úrad vykoná príslušné zmeny v registri,
relationships:
  Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf 83
  Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf 68B
  Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf 7
  Zakon 222/2004 Z. Z. -> [OBSAHUJE] -> Paragraf 7A
  Osoba Registrovana -> [JE_PODLA:"registracia pre dan"] -> Paragraf 7
  Osoba Registrovana -> [JE_PODLA:"registracia pre dan"] -> Paragraf 7A
  Osoba Registrovana -> [JE_PODLA:"osobitna uprava"] -> Paragraf 68B
  Daňový Úrad -> [ZRUSUJE:"z uradnej moci"] -> Registracia Pre Dan
  Registracia Pre Dan -> [ZANIKA:"platnost identifikacneho cisla pre dan"] -> Identifikacne Cislo Pre Dan
  Fyzicka Osoba -> [VYKONAVA:"ziadost o zrusenie registracie pre dan"] -> Podanie Ziadosti O Zrusenie Registracie Pre Dan
  Osoba Pokracujuca V Podnikani -> [OZNAMUJE:"daňovemu uradu"] -> Oznamenie Skutocnosti Danovemu Uradu
  Oznamenie Skutocnosti Danovemu Uradu -> [MA_LEHOTU:"najneskor do 30 dni odo dna umrtia"] -> 30 dni
  Daňový Úrad -> [VYKONAVA:"prislusne zmeny"] -> Zmeny V Registri
  Zakon 222/2004 Z. Z. -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 83 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 68B -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 7 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf 7A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Odsek 2 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Daňový Úrad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Registracia Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zrusenie Registracie Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Podanie Odvolania -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Podanie Ziadosti O Zrusenie Registracie Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba Registrovana -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pravnicka Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Fyzicka Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osoba Pokracujuca V Podnikani -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Uplatnovanie Tejto Osobitnej Upravy -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osobitna Uprava -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Identifikacne Cislo Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platnost Identifikacneho Cisla Pre Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oznamenie Skutocnosti Danovemu Uradu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zmeny V Registri -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  30 dni -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Zakon: Zakon 222/2004 Z. Z.
  Paragraf: Paragraf 83
  Paragraf: Paragraf 68B
  Paragraf: Paragraf 7
  Paragraf: Paragraf 7A
  Odsek: Odsek 2
  Urad: Daňový Úrad
  Registracia: Registracia Pre Dan
  Cinnost: Zrusenie Registracie Pre Dan
  Cinnost: Podanie Odvolania
  Ziadost: Podanie Ziadosti O Zrusenie Registracie Pre Dan
  Zdanitelnaosoba: Platitel
  Osoba: Osoba Registrovana
  Pravnickaosoba: Pravnicka Osoba
  Fyzickaosoba: Fyzicka Osoba
  Fyzickaosoba: Osoba Pokracujuca V Podnikani
  Cinnost: Uplatnovanie Tejto Osobitnej Upravy
  Cinnost: Osobitna Uprava
  Identifikacnecislo: Identifikacne Cislo Pre Dan
  Status: Platnost Identifikacneho Cisla Pre Dan
  Oznamenie: Oznamenie Skutocnosti Danovemu Uradu
  Zaznam: Zmeny V Registri
  Lehota: 30 dni

chunk: 547
page: 135
text: Strana 136 Zbierka zákonov Slovenskej republiky 222/2004 Z. z. § 85g Prechodné ustanovenia k úpravám účinným od 1. júla 2009 (1) Zdaniteľná osoba, ktorá dosiahla obrat 35 000 eur podľa § 4 ods. 1 alebo 2 v znení účinnom do 30. júna 2009 a stala sa platiteľom, môže požiadať o zrušenie registrácie pre daň, ak k poslednému dňu kalendárneho mesiaca, ktorý predchádza kalendárnemu mesiacu, v ktorom podala žiadosť o zrušenie registrácie pre daň, nedosiahla obrat 49 790 eur za najviac 12 predchádzajúcich po sebe nasledujúcich kalendárnych mesiacov. (2) Daňový úrad po prijatí žiadosti o zrušenie registrácie pre daň bezodkladne určí platiteľovi deň, ktorým platiteľ prestáva byť platiteľom, a týmto dňom zaniká platnosť osvedčenia o registrácii pre daň a platnosť identifikačného čísla pre daň. Platiteľ je povinný odovzdať daňovému úradu osvedčenie o registrácii pre daň do desiatich dní odo dňa, keď prestal byť platiteľom. (3) Platiteľ, ktorý požiada o zrušenie registrácie pre daň podľa odseku 1, nemôže odpočítať daň
relationships:
  Zakon_222_2004_Z_Z -> [OBSAHUJE] -> Paragraf_85G
  Paragraf_85G -> [MA_NAZOV] -> Prechodne_Ustanovenia_K_Upravam_Ucinnym_Od_1_Jula_2009
  Zdanitelna_Osoba -> [MA_HODNOTU:"dosiahla obrat"] -> Obrat_35_000_Eur
  Zdanitelna_Osoba -> [MA_PRAVO:"poziadat o zrusenie registracie pre dan"] -> Ziadost_O_Zrusenie_Registracie_Pre_Dan
  Zdanitelna_Osoba -> [MA_PODMIENKU:"nedosiahla obrat"] -> Obrat_49_790_Eur
  Zdanitelna_Osoba -> [MA_OBDOBIE:"za najviac 12 predchadzajucich po sebe nasledujucich kalendarnych mesiacov"] -> Dvanast_Predchadzajucich_Po_Sebe_Nasledujucich_Kalendarnych_Mesiacov
  Ziadost_O_Zrusenie_Registracie_Pre_Dan -> [PODAVA:"zrusenie registracie pre dan"] -> Danovy_Urad
  Danovy_Urad -> [URCUJE:"deň, ktorým platiteľ prestáva byť platiteľom"] -> Den_Prestatia_Byt_Platitelom
  Platitel -> [ZANIKA:"platnost osvedcenia o registracii pre dan"] -> Osvedcenie_O_Registracii_Pre_Dan
  Platitel -> [ZANIKA:"platnost identifikacneho cisla pre dan"] -> Identifikacne_Cislo_Pre_Dan
  Platitel -> [MA_POVINNOST:"odovzdat danovemu uradu do desiatich dni"] -> Osvedcenie_O_Registracii_Pre_Dan
  Platitel -> [MA_LEHOTU:"do desiatich dni odo dna, ked prestal byt platitelom"] -> Desat_Dni
  Zdanitelna_Osoba -> [JE_PODLA:"odsek 1"] -> Paragraf_85G
  Zakon_222_2004_Z_Z -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Paragraf_85G -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Prechodne_Ustanovenia_K_Upravam_Ucinnym_Od_1_Jula_2009 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanitelna_Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Obrat_35_000_Eur -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Obrat_49_790_Eur -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ziadost_O_Zrusenie_Registracie_Pre_Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Danovy_Urad -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Platitel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Den_Prestatia_Byt_Platitelom -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Osvedcenie_O_Registracii_Pre_Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Identifikacne_Cislo_Pre_Dan -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Desat_Dni -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dvanast_Predchadzajucich_Po_Sebe_Nasledujucich_Kalendarnych_Mesiacov -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Konec_Mesiaca_Predchadzajuceho_Mesiacu_Podania_Ziadosti -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Zakon: Zakon_222_2004_Z_Z
  Paragraf: Paragraf_85G
  Cinnost: Prechodne_Ustanovenia_K_Upravam_Ucinnym_Od_1_Jula_2009
  Zdanitelnaosoba: Zdanitelna_Osoba
  Obrat: Obrat_35_000_Eur
  Obrat: Obrat_49_790_Eur
  Ziadost: Ziadost_O_Zrusenie_Registracie_Pre_Dan
  Urad: Danovy_Urad
  Zdanitelnaosoba: Platitel
  Datum: Den_Prestatia_Byt_Platitelom
  Doklad: Osvedcenie_O_Registracii_Pre_Dan
  Identifikacnecislo: Identifikacne_Cislo_Pre_Dan
  Casovyudaj: Desat_Dni
  Obdobie: Dvanast_Predchadzajucich_Po_Sebe_Nasledujucich_Kalendarnych_Mesiacov
  Datum: Konec_Mesiaca_Predchadzajuceho_Mesiacu_Podania_Ziadosti

chunk: 574
page: 141
text: do 31. decembra 2022, § 53b ods. 8 v znení účinnom od 1. januára 2023 a na povinnosť vykázať opravu odpočítanej dane v kontrolnom výkaze sa uplatní § 78a v znení účinnom do 31. decembra
relationships:
  53b -> [MA_ODSEK] -> Ods 8
  Povinnost Vykazat Opravu Odpocitanej Dane V Kontrolnom Vykaze -> [MA_OBDOBIE:"v zneni ucinom od"] -> 1 januara 2023
  Povinnost Vykazat Opravu Odpocitanej Dane V Kontrolnom Vykaze -> [PODLA:"uplatni sa"] -> 78a
  Oprava Odpocitanej Dane -> [VZTAHUJE_SA_NA:"v kontrolnom vykaze"] -> Kontrolny Vykaz
  31 decembra 2022 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  1 januara 2023 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  53b -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ods 8 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  78a -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Povinnost Vykazat Opravu Odpocitanej Dane V Kontrolnom Vykaze -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Oprava Odpocitanej Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Kontrolny Vykaz -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Datum: 31 decembra 2022
  Datum: 1 januara 2023
  Paragraf: 53b
  Odsek: Ods 8
  Paragraf: 78a
  Povinnost: Povinnost Vykazat Opravu Odpocitanej Dane V Kontrolnom Vykaze
  Cinnost: Oprava Odpocitanej Dane
  Doklad: Kontrolny Vykaz

chunk: 596
page: 147
text: Z. z. a vyhlášky č. 424/2001 Z. z., 3. vyhláška Ministerstva financií Slovenskej republiky č. 94/1996 Z. z. o rozsahu a spôsobe vrátenia dane z pridanej hodnoty, ktoré vyplýva z medzinárodnej zmluvy v rámci projektov zahraničnej pomoci. § 87a Zrušovacie ustanovenie účinné od 1. januára 2021 Zrušuje sa opatrenie Ministerstva financií Slovenskej republiky z 28. októbra 2019 č. MF/014640/2019-731, ktorým sa ustanovuje vzor daňového priznania k dani z pridanej hodnoty (oznámenie č. 404/2019 Z. z.). § 88 Tento zákon nadobúda účinnosť dňom nadobudnutia platnosti zmluvy o pristúpení Slovenskej republiky k Európskej únii. Pavol Hrušovský v. r.   Mikuláš Dzurinda v. r.
relationships:
  Vyhlaska Ministerstva Financii Slovenskej Republiky C. 94/1996 Z. Z. -> [MA_NAZOV:"rozsah a sposob vratenia dane z pridanej hodnoty"] -> Vyplatenie Dane Z Pridanej Hodnoty
  Vyhlaska Ministerstva Financii Slovenskej Republiky C. 94/1996 Z. Z. -> [VYCHADZA_Z:"vratenie dane z pridanej hodnoty"] -> Medzinarodna Zmluva V Ramci Projektov Zahranicnej Pomoci
  Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731 -> [ZANIKA:"zrusuje sa od 1. januara 2021"] -> Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731
  Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731 -> [MA_NAZOV:"vzor danoveho priznania k dani z pridanej hodnoty"] -> Vyplatenie Dane Z Pridanej Hodnoty
  § 87A -> [MA_ODSEK:"ucinnost zrusovacieho ustanovenia"] -> 1. januara 2021
  § 87A -> [RUSI:"zrusuje sa"] -> Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731
  § 88 -> [MA_UCINOK:"nadobudnutie ucinnosti dnom nadobudnutia platnosti zmluvy o pristupeni Slovenskej republiky k Europskej unii"] -> Tento Zakon
  Tento Zakon -> [VZTAHUJE_SA_NA:"nadobudnutie ucinnosti"] -> Zmluva O Pristupeni Slovenskej Republiky K Europskej Unii
  Zmluva O Pristupeni Slovenskej Republiky K Europskej Unii -> [JE_PODLA:"pristupenie"] -> Slovenska Republika
  Zmluva O Pristupeni Slovenskej Republiky K Europskej Unii -> [JE_PODLA:"pristupenie"] -> Europska Unia
  Vyhlaska Ministerstva Financii Slovenskej Republiky C. 94/1996 Z. Z. -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vyhlaska C. 424/2001 Z. Z. -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ministerstvo Financii Slovenskej Republiky -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Vyplatenie Dane Z Pridanej Hodnoty -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dan Z Pridanej Hodnoty -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Medzinarodna Zmluva V Ramci Projektov Zahranicnej Pomoci -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 87A -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  § 88 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  1. januara 2021 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tento Zakon -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zmluva O Pristupeni Slovenskej Republiky K Europskej Unii -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Slovenska Republika -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Europska Unia -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Vyhlaska: Vyhlaska Ministerstva Financii Slovenskej Republiky C. 94/1996 Z. Z.
  Vyhlaska: Vyhlaska C. 424/2001 Z. Z.
  Ministerstvo: Ministerstvo Financii Slovenskej Republiky
  Nariadenie: Opatrenie Ministerstva Financii Slovenskej Republiky C. Mf/014640/2019-731
  Ziadost: Vyplatenie Dane Z Pridanej Hodnoty
  Dan: Dan Z Pridanej Hodnoty
  Zmluva: Medzinarodna Zmluva V Ramci Projektov Zahranicnej Pomoci
  Paragraf: § 87A
  Paragraf: § 88
  Datum: 1. januara 2021
  Zakon: Tento Zakon
  Zmluva: Zmluva O Pristupeni Slovenskej Republiky K Europskej Unii
  Stat: Slovenska Republika
  Skupina: Europska Unia

chunk: 68
page: 16
text: z iného členského štátu. (2) Nadobudnutie tovaru v tuzemsku z iného členského štátu je predmetom dane, ak a) nadobúdateľom je zdaniteľná osoba konajúca v postavení zdaniteľnej osoby, právnická osoba, ktorá nie je zdaniteľnou osobou, alebo zahraničná osoba, ktorá je identifikovaná pre daň v inom členskom štáte, a  b) dodávateľom je zdaniteľná osoba, ktorá je v inom členskom štáte, v ktorom sa začalo odoslanie alebo preprava tovaru dodaného za protihodnotu osobe podľa písmena a), identifikovaná pre daň podľa ustanovenia zákona platného v tomto inom členskom štáte zodpovedajúceho § 4, § 4b, § 4c alebo § 5 alebo sa od nej táto identifikácia vyžaduje, okrem dodania tovaru s inštaláciou alebo montážou dodávateľom alebo na jeho účet a okrem predaja tovaru na diaľku na území Európskej únie. (3) Predmetom dane je aj nadobudnutie nového dopravného prostriedku za protihodnotu v tuzemsku z iného členského štátu každou osobou. (4) Nadobudnutie tovaru v tuzemsku z iného členského štátu nie je predmetom dane, ak
relationships:
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [JE_PREDMETOM:"nadobudnutie tovaru v tuzemsku z ineho clenskeho statu"] -> Predmet Dane
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_MIESTO:"v tuzemsku"] -> Tuzemsko
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [PODLIEHA:"podla ustanovenia zakona"] -> Zakon
  Zakon -> [OBSAHUJE:"§ 4"] -> 4
  Zakon -> [OBSAHUJE:"§ 4b"] -> 4b
  Zakon -> [OBSAHUJE:"§ 4c"] -> 4c
  Zakon -> [OBSAHUJE:"§ 5"] -> 5
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU:"nadobudatel je zdanitelna osoba kona v postaveni zdanitelnej osoby"] -> Zdanitelna Osoba
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU:"nadobudatel je pravnicka osoba ktora nie je zdanitelnou osobou"] -> Pravnicka Osoba
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU:"nadobudatel je zahranicna osoba identifikovana pre dan v inom clenskom state"] -> Zahranicna Osoba
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_PODMIENKU:"dodavatel je zdanitelna osoba v inom clenskom state"] -> Dodavatel
  Dodavatel -> [MA_PODMIENKU:"identifikacia pre dan alebo sa vyzaduje v inom clenskom state"] -> Ina Clensky Stat
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [JE_PREDMETOM:"nadobudnutie noveho dopravneho prostriedku"] -> Novy Dopravny Prostriedok
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_HODNOTU:"za protihodnotu"] -> Protihodnota
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_MIESTO:"z ineho clenskeho statu"] -> Ina Clensky Stat
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zdanitelna Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Pravnicka Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zahranicna Osoba -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Dodavatel -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Ina Clensky Stat -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Tuzemsko -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Predmet Dane -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Novy Dopravny Prostriedok -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Protihodnota -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Zakon -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  4 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  4b -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  4c -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  5 -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
  Clensky Stat -> [IN_CHUNK] -> chunk_0_ZZ_222_2004
nodes:
  Cinnost: Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Zdanitelnaosoba: Zdanitelna Osoba
  Pravnickaosoba: Pravnicka Osoba
  Osoba: Zahranicna Osoba
  Zdanitelnaosoba: Dodavatel
  Clenskystat: Ina Clensky Stat
  Tuzemsko: Tuzemsko
  Dan: Predmet Dane
  Vozidlo: Novy Dopravny Prostriedok
  Hodnota: Protihodnota
  Pravnypredpis: Zakon
  Paragraf: 4
  Paragraf: 4b
  Paragraf: 4c
  Paragraf: 5
  Clenskystat: Clensky Stat
