je predmetom dane, je oslobodene od dane, nie je predmetom dane

chunk: 101
page: 24
text: (1) Daňová povinnosť vzniká dňom dodania tovaru. Dňom dodania tovaru je deň, keď kupujúci nadobudne právo nakladať s tovarom ako vlastník. Pri prevode alebo prechode nehnuteľnosti je dňom dodania deň odovzdania nehnuteľnosti do užívania, ak je tento deň skorší ako deň zápisu vlastníckeho práva k nehnuteľnosti do katastra nehnuteľností. Pri dodaní stavby na základe zmluvy o dielo alebo inej obdobnej zmluvy je dňom dodania deň odovzdania stavby. Pri dodaní tovaru podľa § 8 ods. 1 písm. c) je dňom dodania tovaru deň odovzdania tovaru nájomcovi.
relationships:
  Danova Povinnost -> [VZNIKA_PRI] -> Dodanie Tovaru
  Danova Povinnost -> [MA_DATUM] -> Den Dodania Tovaru

  Den Dodania Tovaru -> [POVAZUJE_SA_ZA] -> Den Nadobudnutia Prava Nakladat S Tovarom Ako Vlastnik
  Kupujuci -> [NADOBUDA] -> Pravo Nakladat S Tovarom Ako Vlastnik
  Pravo Nakladat S Tovarom Ako Vlastnik -> [VZTAHUJE_SA_NA] -> Tovar

  Prevod Alebo Prechod Nehnutelnosti -> [VZTAHUJE_SA_NA] -> Nehnutelnost
  Den Dodania Pri Prevode Alebo Prechode Nehnutelnosti -> [POVAZUJE_SA_ZA] -> Den Odovzdania Nehnutelnosti Do Uzivania
  Den Odovzdania Nehnutelnosti Do Uzivania -> [MA_PODMIENKU] -> Skorsi Den Ako Den Zapisu Vlastnickeho Prava Do Katastra Nehnutelnosti
  Zapis Vlastnickeho Prava K Nehnutelnosti Do Katastra Nehnutelnosti -> [VZTAHUJE_SA_NA] -> Nehnutelnost

  Dodanie Stavby -> [VYPLNYVA_Z] -> Zmluva O Dielo Alebo Ina Obdobna Zmluva
  Den Dodania Stavby -> [POVAZUJE_SA_ZA] -> Den Odovzdania Stavby

  Dodanie Tovaru Podla Paragrafu 8 Odseku 1 Pismena C -> [JE_PODLA] -> Paragraf § 8 Odsek 1 Pismeno c)
  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 1
  Paragraf § 8 Odsek 1 -> [OBSAHUJE] -> Paragraf § 8 Odsek 1 Pismeno c)
  Den Dodania Tovaru Podla Paragrafu 8 Odseku 1 Pismena C -> [POVAZUJE_SA_ZA] -> Den Odovzdania Tovaru Najomcovi

nodes:
  Povinnost: Danova Povinnost
  Cinnost: Dodanie Tovaru
  Datum: Den Dodania Tovaru
  Datum: Den Nadobudnutia Prava Nakladat S Tovarom Ako Vlastnik
  Subjekt: Kupujuci
  Pravo: Pravo Nakladat S Tovarom Ako Vlastnik
  Tovar: Tovar

  Cinnost: Prevod Alebo Prechod Nehnutelnosti
  Nehnutelnost: Nehnutelnost
  Datum: Den Dodania Pri Prevode Alebo Prechode Nehnutelnosti
  Datum: Den Odovzdania Nehnutelnosti Do Uzivania
  Podmienka: Skorsi Den Ako Den Zapisu Vlastnickeho Prava Do Katastra Nehnutelnosti
  Cinnost: Zapis Vlastnickeho Prava K Nehnutelnosti Do Katastra Nehnutelnosti

  Cinnost: Dodanie Stavby
  Stavba: Stavba
  Zmluva: Zmluva O Dielo Alebo Ina Obdobna Zmluva
  Datum: Den Dodania Stavby
  Datum: Den Odovzdania Stavby

  Cinnost: Dodanie Tovaru Podla Paragrafu 8 Odseku 1 Pismena C
  Datum: Den Dodania Tovaru Podla Paragrafu 8 Odseku 1 Pismena C
  Datum: Den Odovzdania Tovaru Najomcovi
  Subjekt: Najomca
  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 1
  Pismeno: Paragraf § 8 Odsek 1 Pismeno c)


chunk: 109
page: 26
text: uvádza na trh v tuzemsku ako prvý, a ktorý dodáva aj zálohované obaly spolu s tovarom, ktoré neuvádza na trh v tuzemsku ako prvý, nemôže uplatniť záporný rozdiel v daňovom priznaní. Základom dane je súčin zisteného rozdielu podľa prvej vety a výšky zálohy za zálohovaný obal určenej osobitným predpisom,6ab) ktorá je platná posledný deň príslušného kalendárneho roka, znížený o daň. (11) Pri dodaní tovaru zdaniteľnou osobou, ktorá uľahčuje dodanie tovaru na území Európskej únie a predaj tovaru na diaľku podľa § 8 ods. 7, a pri dodaní tohto tovaru tejto zdaniteľnej osobe sa za deň dodania tovaru a deň vzniku daňovej povinnosti považuje deň prijatia platby.6abc) § 20 Daňová povinnosť pri nadobudnutí tovaru v tuzemsku z iného členského štátu (1) Daňová povinnosť pri nadobudnutí tovaru v tuzemsku z iného členského štátu vzniká a) 15. deň kalendárneho mesiaca nasledujúceho po kalendárnom mesiaci, keď sa uskutočnilo nadobudnutie tovaru, alebo
relationships:
  Platitel Podla Prvej Vety -> [VYKONAVA] -> Uvedenie Zalohovanych Obalov Na Trh V Tuzemsku Ako Prvy
  Uvedenie Zalohovanych Obalov Na Trh V Tuzemsku Ako Prvy -> [MA_MIESTO] -> Tuzemsko
  Platitel Podla Prvej Vety -> [DODAVA] -> Zalohovane Obaly Spolu S Tovarom
  Platitel Podla Prvej Vety -> [DODAVA] -> Zalohovane Obaly Neuvadzane Na Trh V Tuzemsku Ako Prvy
  Platitel Podla Prvej Vety -> [NEMA_NAROK_NA] -> Uplatnenie Zaporneho Rozdielu V Danovom Priznani
  Uplatnenie Zaporneho Rozdielu V Danovom Priznani -> [VZTAHUJE_SA_NA] -> Danove Priznanie

  Zaklad Dane -> [VYCHADZA_Z] -> Vypocet Zakladu Dane
  Vypocet Zakladu Dane -> [VYCHADZA_Z] -> Zisteny Rozdiel Podla Prvej Vety
  Vypocet Zakladu Dane -> [VYCHADZA_Z] -> Vyska Zalohy Za Zalohovany Obal
  Vypocet Zakladu Dane -> [VYCHADZA_Z] -> Dan
  Zisteny Rozdiel Podla Prvej Vety -> [JE_PODLA] -> Prva Veta
  Osobitny Predpis -> [URCUJE] -> Vyska Zalohy Za Zalohovany Obal
  Vyska Zalohy Za Zalohovany Obal -> [VZTAHUJE_SA_NA] -> Zalohovany Obal
  Vyska Zalohy Za Zalohovany Obal -> [MA_DATUM] -> Posledny Den Prislusneho Kalendarneho Roka

  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 7
  Odsek 11 -> [UPRAVUJE] -> Den Dodania Tovaru A Den Vzniku Danovej Povinnosti
  Zdanitelna Osoba Ulahcujuca Dodanie Tovaru -> [VYKONAVA] -> Ulahcenie Dodania Tovaru Na Uzemi Europskej Unie
  Zdanitelna Osoba Ulahcujuca Dodanie Tovaru -> [VYKONAVA] -> Ulahcenie Predaja Tovaru Na Dialku
  Ulahcenie Dodania Tovaru Na Uzemi Europskej Unie -> [JE_PODLA] -> Paragraf § 8 Odsek 7
  Ulahcenie Predaja Tovaru Na Dialku -> [JE_PODLA] -> Paragraf § 8 Odsek 7
  Dodanie Tovaru Zdanitelnou Osobou Ulahcujucou Dodanie -> [MA_DATUM] -> Den Prijatia Platby
  Dodanie Tovaru Tejto Zdanitelnej Osobe -> [MA_DATUM] -> Den Prijatia Platby
  Den Dodania Tovaru -> [POVAZUJE_SA_ZA] -> Den Prijatia Platby
  Den Vzniku Danovej Povinnosti -> [POVAZUJE_SA_ZA] -> Den Prijatia Platby

  Paragraf § 20 -> [MA_NAZOV] -> Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Paragraf § 20 -> [OBSAHUJE] -> Paragraf § 20 Odsek 1
  Paragraf § 20 Odsek 1 -> [OBSAHUJE] -> Paragraf § 20 Odsek 1 Pismeno a)
  Paragraf § 20 Odsek 1 -> [UPRAVUJE] -> Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [VZNIKA_PRI] -> Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_DATUM] -> 15 Den Kalendarneho Mesiaca Nasledujuceho Po Mesiaci Nadobudnutia Tovaru
  Paragraf § 20 Odsek 1 Pismeno a) -> [VYMEDZUJE] -> 15 Den Kalendarneho Mesiaca Nasledujuceho Po Mesiaci Nadobudnutia Tovaru
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [MA_MIESTO] -> Tuzemsko
  Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu -> [TYKA_SA] -> Iny Clensky Stat

nodes:
  ZdanitelnaOsoba: Platitel Podla Prvej Vety
  Cinnost: Uvedenie Zalohovanych Obalov Na Trh V Tuzemsku Ako Prvy
  Tuzemsko: Tuzemsko
  Tovar: Zalohovane Obaly Spolu S Tovarom
  Tovar: Zalohovane Obaly Neuvadzane Na Trh V Tuzemsku Ako Prvy
  Tovar: Zalohovany Obal
  Cinnost: Uplatnenie Zaporneho Rozdielu V Danovom Priznani
  DanovePriznanie: Danove Priznanie
  Hodnota: Zaklad Dane
  Vypocet: Vypocet Zakladu Dane
  Hodnota: Zisteny Rozdiel Podla Prvej Vety
  Bod: Prva Veta
  Suma: Vyska Zalohy Za Zalohovany Obal
  Dan: Dan
  PravnyPredpis: Osobitny Predpis
  Datum: Posledny Den Prislusneho Kalendarneho Roka
  Obdobie: Prislusny Kalendarneho Rok

  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 7
  Odsek: Odsek 11
  ZdanitelnaOsoba: Zdanitelna Osoba Ulahcujuca Dodanie Tovaru
  Cinnost: Ulahcenie Dodania Tovaru Na Uzemi Europskej Unie
  Cinnost: Ulahcenie Predaja Tovaru Na Dialku
  Uzemie: Europska Unia
  Cinnost: Dodanie Tovaru Zdanitelnou Osobou Ulahcujucou Dodanie
  Cinnost: Dodanie Tovaru Tejto Zdanitelnej Osobe
  Datum: Den Prijatia Platby
  Datum: Den Dodania Tovaru
  Datum: Den Vzniku Danovej Povinnosti

  Paragraf: Paragraf § 20
  Hodnota: Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Odsek: Paragraf § 20 Odsek 1
  Pismeno: Paragraf § 20 Odsek 1 Pismeno a)
  Povinnost: Danova Povinnost Pri Nadobudnuti Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Cinnost: Nadobudnutie Tovaru V Tuzemsku Z Ineho Clenskeho Statu
  Datum: 15 Den Kalendarneho Mesiaca Nasledujuceho Po Mesiaci Nadobudnutia Tovaru
  ClenskyStat: Iny Clensky Stat


chunk: 11
page: 3
text: odseku 8 písm. a), pričom identifikačné číslo pre daň nadobúda platnosť dňom, keď sa zdaniteľná osoba stala platiteľom, 2. zdaniteľná osoba sa stala platiteľom podľa odseku 1 písm. c) až h) a doručenia dokladov podľa odseku 3, pričom identifikačné číslo pre daň nadobúda platnosť dňom, keď sa zdaniteľná osoba stala platiteľom. (5) Zdaniteľná osoba, ktorá podala žiadosť o registráciu pre daň podľa odseku 2 písm. a), je povinná bezodkladne oznámiť daňovému úradu skutočnosť, že sa stala platiteľom podľa odseku 1 písm. b) až i) alebo podľa § 48c ods. 5 do 31. decembra prebiehajúceho kalendárneho roka; v oznámení uvedie deň, keď k tejto skutočnosti došlo. Ak daňový úrad ku dňu oznámenia podľa prvej vety ešte nevydal rozhodnutie podľa odseku 4 písm. a), vydá rozhodnutie o registrácii pre daň podľa odseku 4 písm. b). (6) Platiteľom sa môže stať aj zdaniteľná osoba, ktorá má v tuzemsku sídlo, miesto podnikania, a ak nemá takéto miesto, ale má bydlisko v tuzemsku alebo sa v tuzemsku obvykle zdržiava, pred
relationships:
  Zdanitelna Osoba -> [PODAVA] -> Ziadost O Registraciu Pre Dan
  Ziadost O Registraciu Pre Dan -> [JE_PODLA] -> Odsek 2 Pismeno a)

  Zdanitelna Osoba -> [STAVA_SA] -> Platitel
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno c)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno d)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno e)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno f)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno g)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno h)
  Doručenie Dokladov -> [JE_PODLA] -> Odsek 3
  Identifikacne Cislo Pre Dan -> [PLATI_OD] -> Den Ked Sa Zdanitelna Osoba Stala Platitelom

  Zdanitelna Osoba -> [MA_POVINNOST] -> Oznamenie Skutocnosti Ze Sa Stala Platitelom
  Oznamenie Skutocnosti Ze Sa Stala Platitelom -> [DORUCUJE] -> Danovy Urad
  Oznamenie Skutocnosti Ze Sa Stala Platitelom -> [MA_OBSAH] -> Den Ked Sa Zdanitelna Osoba Stala Platitelom
  Oznamenie Skutocnosti Ze Sa Stala Platitelom -> [MA_LEHOTU] -> Do 31 Decembra Prebiehajuceho Kalendarneho Roka
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno b)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno c)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno d)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno e)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno f)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno g)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno h)
  Platitel -> [JE_PODLA] -> Odsek 1 Pismeno i)
  Platitel -> [JE_PODLA] -> Paragraf § 48c Odsek 5

  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Registracii Pre Dan
  Rozhodnutie O Registracii Pre Dan -> [JE_PODLA] -> Odsek 4 Pismeno b)
  Vydanie Rozhodnutia O Registracii Pre Dan -> [MA_PODMIENKU] -> Nevydanie Rozhodnutia Podla Odseku 4 Pismena A Ku Dnu Oznamenia

  Zdanitelna Osoba -> [MA_SIDLO] -> Sidlo V Tuzemsku
  Zdanitelna Osoba -> [MA_MIESTO_PODNIKANIA] -> Miesto Podnikania V Tuzemsku
  Zdanitelna Osoba -> [MA_BYDLISKO] -> Bydlisko V Tuzemsku
  Zdanitelna Osoba -> [MA_MIESTO] -> Miesto Obvykleho Zdrziavania V Tuzemsku

nodes:
  ZdanitelnaOsoba: Zdanitelna Osoba
  ZdanitelnaOsoba: Platitel
  Ziadost: Ziadost O Registraciu Pre Dan
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  Urad: Danovy Urad
  Oznamenie: Oznamenie Skutocnosti Ze Sa Stala Platitelom
  Datum: Den Ked Sa Zdanitelna Osoba Stala Platitelom
  Lehota: Do 31 Decembra Prebiehajuceho Kalendarneho Roka
  Rozhodnutie: Rozhodnutie O Registracii Pre Dan
  Cinnost: Vydanie Rozhodnutia O Registracii Pre Dan
  Podmienka: Nevydanie Rozhodnutia Podla Odseku 4 Pismena A Ku Dnu Oznamenia
  Doklad: Doručenie Dokladov
  Odsek: Odsek 2
  Odsek: Odsek 3
  Odsek: Odsek 4
  Odsek: Odsek 8
  Pismeno: Odsek 2 Pismeno a)
  Pismeno: Odsek 4 Pismeno a)
  Pismeno: Odsek 4 Pismeno b)
  Pismeno: Odsek 1 Pismeno b)
  Pismeno: Odsek 1 Pismeno c)
  Pismeno: Odsek 1 Pismeno d)
  Pismeno: Odsek 1 Pismeno e)
  Pismeno: Odsek 1 Pismeno f)
  Pismeno: Odsek 1 Pismeno g)
  Pismeno: Odsek 1 Pismeno h)
  Pismeno: Odsek 1 Pismeno i)
  Odsek: Paragraf § 48c Odsek 5
  Sidlo: Sidlo V Tuzemsku
  Lokacia: Miesto Podnikania V Tuzemsku
  Adresa: Bydlisko V Tuzemsku
  Lokacia: Miesto Obvykleho Zdrziavania V Tuzemsku
  Tuzemsko: Tuzemsko


# TODO
chunk: 153
page: 38
text: nebytového priestoru, alebo po piatich rokoch odo dňa začatia prvého užívania bytu, apartmánu alebo nebytového priestoru, a to podľa toho, čo nastane skôr, b) kolaudácie, ktorou sa povolila zmena účelu užívania bytu, apartmánu alebo nebytového priestoru, ku ktorej došlo v dôsledku vykonaných stavebných prác, ak náklady na tieto stavebné práce sú vo výške najmenej 40 % z hodnoty bytu, apartmánu alebo nebytového priestoru pred začatím stavebných prác; hodnotou bytu, apartmánu alebo nebytového priestoru pred začatím stavebných prác sa na účely tohto odseku rozumie hodnota, ktorá nie je nižšia ako cena porovnateľného bytu, apartmánu alebo nebytového priestoru na voľnom trhu v čase pred začatím stavebných prác,
relationships:
  Pet Rokov Odo Dna Zacatia Prveho Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru -> [VZTAHUJE_SA_NA] -> Zacatie Prveho Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru

  Kolaudacia -> [VYMEDZUJE] -> Zmena Ucelu Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru
  Zmena Ucelu Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru -> [VZNIKA_PRI] -> Vykonane Stavebne Prace
  Zmena Ucelu Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru -> [MA_PODMIENKU] -> Naklady Na Stavebne Prace Najmenej 40 Percent Z Hodnoty Pred Zacatim Stavebnych Prac

  Naklady Na Stavebne Prace -> [MA_HODNOTU] -> Najmenej 40 Percent Z Hodnoty Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac
  Najmenej 40 Percent Z Hodnoty Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac -> [VZTAHUJE_SA_NA] -> Hodnota Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac

  Hodnota Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac -> [ROZUMIE_SA] -> Hodnota Nie Nizsia Ako Cena Porovnatelneho Bytu Apartmanu Alebo Nebytoveho Priestoru Na Volnom Trhu
  Cena Porovnatelneho Bytu Apartmanu Alebo Nebytoveho Priestoru Na Volnom Trhu -> [MA_OBDOBIE] -> Cas Pred Zacatim Stavebnych Prac

nodes:
  CasovyUdaj: Pet Rokov Odo Dna Zacatia Prveho Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru
  Cinnost: Zacatie Prveho Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru
  Doklad: Kolaudacia
  Cinnost: Zmena Ucelu Uzivania Bytu Apartmanu Alebo Nebytoveho Priestoru
  Cinnost: Vykonane Stavebne Prace
  Suma: Naklady Na Stavebne Prace
  Podmienka: Naklady Na Stavebne Prace Najmenej 40 Percent Z Hodnoty Pred Zacatim Stavebnych Prac
  Hodnota: Najmenej 40 Percent Z Hodnoty Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac
  Hodnota: Hodnota Bytu Apartmanu Alebo Nebytoveho Priestoru Pred Zacatim Stavebnych Prac
  Hodnota: Hodnota Nie Nizsia Ako Cena Porovnatelneho Bytu Apartmanu Alebo Nebytoveho Priestoru Na Volnom Trhu
  Hodnota: Cena Porovnatelneho Bytu Apartmanu Alebo Nebytoveho Priestoru Na Volnom Trhu
  Obdobie: Cas Pred Zacatim Stavebnych Prac
  Nehnutelnost: Byt
  Nehnutelnost: Apartman
  Nehnutelnost: Nebytovy Priestor


chunk: 187
page: 48
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 49 spoločnosťou, ktorá je zodpovedná za jeho prepravu; pohonné látky iné ako v odseku 12 sa za osobnú batožinu nepovažujú, d) neobchodným dovozom dovoz tovaru, ak 1. je tovar určený na osobnú spotrebu cestujúceho alebo osobnú spotrebu jeho domácnosti,24a) alebo je určený ako dar, 2. povaha a množstvo tovaru nevzbudzuje podozrenie, že tovar sa dováža na obchodné účely a  3. dovoz sa uskutočňuje príležitostne, e) cigarkou cigara s jednotkovou hmotnosťou najviac 3 gramy. (2) Oslobodený od dane je neobchodný dovoz tovaru v osobnej batožine cestujúceho z územia tretích štátov. (3) Oslobodenie od dane podľa odseku 2 sa okrem tovaru uvedeného v odsekoch 6 až 12 uplatní na dovoz tovaru, ak jeho hodnota celkom nepresahuje a) 300 eur na osobu inú ako uvedenú v písmenách b) a c), b) 430 eur na osobu, ak cestuje leteckou dopravou, c) 150 eur na osobu mladšiu ako 15 rokov bez ohľadu na dopravný prostriedok.
relationships:
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Odsek 2
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Odsek 3

  Pohonne Latky Ine Ako V Odseku 12 -> [NEVZTAHUJE_SA_NA] -> Osobna Batozina
  Pohonne Latky Ine Ako V Odseku 12 -> [JE_PODLA] -> Odsek 12

  Neobchodny Dovoz -> [ROZUMIE_SA] -> Dovoz Tovaru
  Dovoz Tovaru -> [MA_PODMIENKU] -> Tovar Urceny Na Osobnu Spotrebu Cestujuceho
  Dovoz Tovaru -> [MA_PODMIENKU] -> Tovar Urceny Na Osobnu Spotrebu Domacnosti Cestujuceho
  Dovoz Tovaru -> [MA_PODMIENKU] -> Tovar Urceny Ako Dar
  Dovoz Tovaru -> [MA_PODMIENKU] -> Povaha A Mnozstvo Tovaru Nevzbudzuje Podozrenie Na Obchodne Ucely
  Dovoz Tovaru -> [MA_PODMIENKU] -> Prilezitostny Dovoz

  Cigarka -> [ROZUMIE_SA] -> Cigara S Jednotkovou Hmotnostou Najviac 3 Gramy

  Odsek 2 -> [UPRAVUJE] -> Oslobodenie Od Dane
  Neobchodny Dovoz Tovaru V Osobnej Batozine Cestujuceho Z Uzemia Tretich Statov -> [JE_OSLOBODENE_OD] -> Dan
  Neobchodny Dovoz Tovaru V Osobnej Batozine Cestujuceho Z Uzemia Tretich Statov -> [VZTAHUJE_SA_NA] -> Osobna Batozina Cestujuceho
  Neobchodny Dovoz Tovaru V Osobnej Batozine Cestujuceho Z Uzemia Tretich Statov -> [VYCHADZA_Z] -> Uzemie Tretich Statov

  Oslobodenie Od Dane Podla Odseku 2 -> [JE_PODLA] -> Odsek 2
  Oslobodenie Od Dane Podla Odseku 2 -> [MA_VYNIMKU] -> Tovar Uvedeny V Odsekoch 6 Az 12
  Oslobodenie Od Dane Podla Odseku 2 -> [VZTAHUJE_SA_NA] -> Dovoz Tovaru

  Dovoz Tovaru -> [MA_HODNOTU] -> Hodnota Tovaru Celkom
  Hodnota Tovaru Celkom -> [PRESAHUJE] -> Limit Hodnoty Tovaru

  Oslobodenie Od Dane Podla Odseku 2 -> [MA_SUMU] -> 300 eur na osobu
  300 eur na osobu -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Uvedena V Pismenach B A C

  Oslobodenie Od Dane Podla Odseku 2 -> [MA_SUMU] -> 430 eur na osobu
  430 eur na osobu -> [MA_PODMIENKU] -> Letecka Doprava

  Oslobodenie Od Dane Podla Odseku 2 -> [MA_SUMU] -> 150 eur na osobu
  150 eur na osobu -> [VZTAHUJE_SA_NA] -> Osoba Mladsia Ako 15 Rokov
  150 eur na osobu -> [NEVZTAHUJE_SA_NA] -> Dopravny Prostriedok

nodes:
  PravnyPredpis: 222/2004 Z. z. Zbierka zakonov Slovenskej republiky
  Odsek: Odsek 2
  Odsek: Odsek 3
  Odsek: Odsek 12
  Odsek: Odsek 6
  Odsek: Odsek 7
  Odsek: Odsek 8
  Odsek: Odsek 9
  Odsek: Odsek 10
  Odsek: Odsek 11
  Tovar: Pohonne Latky Ine Ako V Odseku 12
  Majetok: Osobna Batozina
  Majetok: Osobna Batozina Cestujuceho

  Cinnost: Neobchodny Dovoz
  Cinnost: Dovoz Tovaru
  Cinnost: Neobchodny Dovoz Tovaru V Osobnej Batozine Cestujuceho Z Uzemia Tretich Statov
  Podmienka: Tovar Urceny Na Osobnu Spotrebu Cestujuceho
  Podmienka: Tovar Urceny Na Osobnu Spotrebu Domacnosti Cestujuceho
  Podmienka: Tovar Urceny Ako Dar
  Podmienka: Povaha A Mnozstvo Tovaru Nevzbudzuje Podozrenie Na Obchodne Ucely
  Podmienka: Prilezitostny Dovoz

  Tovar: Cigarka
  Tovar: Cigara S Jednotkovou Hmotnostou Najviac 3 Gramy
  Mnozstvo: 3 Gramy

  OslobodenieOdDane: Oslobodenie Od Dane
  OslobodenieOdDane: Oslobodenie Od Dane Podla Odseku 2
  Dan: Dan
  TretiStat: Uzemie Tretich Statov
  Tovar: Tovar Uvedeny V Odsekoch 6 Az 12
  Hodnota: Hodnota Tovaru Celkom
  Limit: Limit Hodnoty Tovaru

  Euro: 300 eur na osobu
  Euro: 430 eur na osobu
  Euro: 150 eur na osobu
  Osoba: Osoba Ina Ako Uvedena V Pismenach B A C
  Osoba: Osoba Mladsia Ako 15 Rokov
  Cinnost: Letecka Doprava
  Majetok: Dopravny Prostriedok


# TODO
chunk: 226
page: 57
text: (9) Platiteľ, ktorý je zahraničnou osobou a spĺňa podmienky na vrátenie dane podľa § 55a alebo § 56, nemôže uplatňovať odpočítanie dane z tovarov a služieb prostredníctvom daňového priznania okrem odpočítania dane z tovarov a služieb, ktoré použije na dodávky tovarov a služieb, pri ktorých je osobou povinnou platiť daň podľa § 69 ods. 1 a okrem odpočítania ním uplatnenej dane na tovary a služby, pri ktorých je osobou povinnou platiť daň. (10) Ak zahraničná osoba uplatňuje osobitnú úpravu podľa § 68a až 68c alebo osobitnú úpravu podľa ustanovení zákona platného v inom členskom štáte zodpovedajúcich § 68a až 68c a súčasne vykonáva v tuzemsku aj činnosti, na ktoré sa tieto osobitné úpravy nevzťahujú a v súvislosti s ktorými je registrovaná ako platiteľ podľa § 5, má právo na odpočítanie dane uplatnenej pri tovaroch a službách, ktoré súvisia s dodaním tovarov a služieb uvedených v § 68 ods. 1. § 49a (1) Daň uplatnenú na investičný majetok uvedený v § 54 ods. 2 písm. b) a c), ktorý je zahrnutý
relationships:
  Platitel -> [JE_TYPOM] -> Zahranicna Osoba
  Platitel -> [SPLNA_PODMIENKY] -> Podmienky Na Vratenie Dane
  Podmienky Na Vratenie Dane -> [JE_PODLA] -> Paragraf § 55a
  Podmienky Na Vratenie Dane -> [JE_PODLA] -> Paragraf § 56

  Platitel -> [NEMA_NAROK_NA] -> Odpocitanie Dane Z Tovarov A Sluzieb Prostrednictvom Danoveho Priznania
  Odpocitanie Dane Z Tovarov A Sluzieb Prostrednictvom Danoveho Priznania -> [VZTAHUJE_SA_NA] -> Danove Priznanie
  Odpocitanie Dane Z Tovarov A Sluzieb Prostrednictvom Danoveho Priznania -> [MA_VYNIMKU] -> Odpocitanie Dane Z Tovarov A Sluzieb Pouzitych Na Dodavky Tovarov A Sluzieb Podla Paragrafu 69 Odseku 1
  Odpocitanie Dane Z Tovarov A Sluzieb Prostrednictvom Danoveho Priznania -> [MA_VYNIMKU] -> Odpocitanie Dane Uplatnenej Platitelom Na Tovary A Sluzby Pri Ktorych Je Osobou Povinnou Platit Dan
  Platitel -> [JE_POVINNY_PLATIT] -> Dan
  Dan -> [JE_PODLA] -> Paragraf § 69 Odsek 1

  Zahranicna Osoba -> [VYKONAVA] -> Uplatnovanie Osobitnej Upravy Podla Paragrafov 68a Az 68c
  Uplatnovanie Osobitnej Upravy Podla Paragrafov 68a Az 68c -> [JE_PODLA] -> Paragraf § 68a
  Uplatnovanie Osobitnej Upravy Podla Paragrafov 68a Az 68c -> [JE_PODLA] -> Paragraf § 68b
  Uplatnovanie Osobitnej Upravy Podla Paragrafov 68a Az 68c -> [JE_PODLA] -> Paragraf § 68c
  Zahranicna Osoba -> [VYKONAVA] -> Uplatnovanie Osobitnej Upravy Podla Zakona Ineho Clenskeho Statu
  Uplatnovanie Osobitnej Upravy Podla Zakona Ineho Clenskeho Statu -> [JE_PODLA] -> Ustanovenia Zakona Platneho V Inom Clenskom State
  Ustanovenia Zakona Platneho V Inom Clenskom State -> [SUVISI_S] -> Paragraf § 68a
  Ustanovenia Zakona Platneho V Inom Clenskom State -> [SUVISI_S] -> Paragraf § 68b
  Ustanovenia Zakona Platneho V Inom Clenskom State -> [SUVISI_S] -> Paragraf § 68c

  Zahranicna Osoba -> [VYKONAVA] -> Cinnosti V Tuzemsku
  Osobitne Upravy -> [NEVZTAHUJE_SA_NA] -> Cinnosti V Tuzemsku
  Cinnosti V Tuzemsku -> [MA_MIESTO] -> Tuzemsko
  Zahranicna Osoba -> [MA_STATUS] -> Platitel Registrovany Podla Paragrafu 5
  Platitel Registrovany Podla Paragrafu 5 -> [JE_PODLA] -> Paragraf § 5

  Zahranicna Osoba -> [MA_PRAVO] -> Odpocitanie Dane Uplatnenej Pri Tovaroch A Sluzbach
  Odpocitanie Dane Uplatnenej Pri Tovaroch A Sluzbach -> [VZTAHUJE_SA_NA] -> Tovary A Sluzby Suvisiace S Dodanim Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1
  Tovary A Sluzby Suvisiace S Dodanim Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1 -> [TYKA_SA] -> Dodanie Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1
  Dodanie Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1 -> [JE_PODLA] -> Paragraf § 68 Odsek 1

  Paragraf § 49a -> [OBSAHUJE] -> Paragraf § 49a Odsek 1
  Dan Uplatnena Na Investicny Majetok -> [VZTAHUJE_SA_NA] -> Investicny Majetok
  Investicny Majetok -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno b)
  Investicny Majetok -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno c)

nodes:
  ZdanitelnaOsoba: Platitel
  Osoba: Zahranicna Osoba
  Podmienka: Podmienky Na Vratenie Dane
  Paragraf: Paragraf § 55a
  Paragraf: Paragraf § 56
  Odsek: Paragraf § 69 Odsek 1
  Paragraf: Paragraf § 68a
  Paragraf: Paragraf § 68b
  Paragraf: Paragraf § 68c
  Paragraf: Paragraf § 5
  Odsek: Paragraf § 68 Odsek 1
  Paragraf: Paragraf § 49a
  Odsek: Paragraf § 49a Odsek 1
  Pismeno: Paragraf § 54 Odsek 2 Pismeno b)
  Pismeno: Paragraf § 54 Odsek 2 Pismeno c)

  Pravo: Odpocitanie Dane Z Tovarov A Sluzieb Prostrednictvom Danoveho Priznania
  Pravo: Odpocitanie Dane Z Tovarov A Sluzieb Pouzitych Na Dodavky Tovarov A Sluzieb Podla Paragrafu 69 Odseku 1
  Pravo: Odpocitanie Dane Uplatnenej Platitelom Na Tovary A Sluzby Pri Ktorych Je Osobou Povinnou Platit Dan
  Pravo: Odpocitanie Dane Uplatnenej Pri Tovaroch A Sluzbach
  DanovePriznanie: Danove Priznanie
  Dan: Dan
  Dan: Dan Uplatnena Na Investicny Majetok

  Cinnost: Uplatnovanie Osobitnej Upravy Podla Paragrafov 68a Az 68c
  Cinnost: Uplatnovanie Osobitnej Upravy Podla Zakona Ineho Clenskeho Statu
  Status: Osobitne Upravy
  PravnyPredpis: Ustanovenia Zakona Platneho V Inom Clenskom State
  ClenskyStat: Iny Clensky Stat
  Cinnost: Cinnosti V Tuzemsku
  Tuzemsko: Tuzemsko
  Status: Platitel Registrovany Podla Paragrafu 5

  Tovar: Tovary A Sluzby Suvisiace S Dodanim Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1
  Cinnost: Dodanie Tovarov A Sluzieb Podla Paragrafu 68 Odseku 1
  InvesticnyMajetok: Investicny Majetok


chunk: 246
page: 62
text: 222/2004 Z. z. Zbierka zákonov Slovenskej republiky Strana 63 podľa § 25a ods. 6, a to vo výške zodpovedajúcej sume protihodnoty, ktorá bola zaplatená za dodanie tovaru alebo služby. (5) Pri oprave opravenej odpočítanej dane podľa odseku 4 platiteľ zohľadní pomerné odpočítanie dane a vykonané úpravy odpočítanej dane; ak platiteľ opravil odpočítanú daň podľa odseku 1 alebo odseku 2 z dôvodu, že za dodanie tovaru alebo služby úplne nezaplatil, zohľadní aj úpravy odpočítanej dane, ktoré by bol povinný vykonať, ak by neopravil odpočítanú daň podľa odseku 1 alebo odseku 2. (6) Ak platiteľ opravil odpočítanú daň podľa odseku 1 alebo odseku 2 alebo opravil opravenú odpočítanú daň podľa odseku 4, zohľadní tieto opravy pri úprave odpočítanej dane podľa § 54, § 54a alebo § 54d. (7) Platiteľ je povinný uviesť opravu odpočítanej dane podľa odseku 1 alebo odseku 2 a opravu opravenej odpočítanej dane podľa odseku 4 v záznamoch podľa § 70 ods. 2 písm. j).
relationships:
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Paragraf § 25a
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Paragraf § 54
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Paragraf § 54a
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Paragraf § 54d
  222/2004 Z. z. Zbierka zakonov Slovenskej republiky -> [OBSAHUJE] -> Paragraf § 70

  Suma Protihodnoty Zaplatena Za Dodanie Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru Alebo Sluzby
  Suma Protihodnoty Zaplatena Za Dodanie Tovaru Alebo Sluzby -> [JE_PODLA] -> Paragraf § 25a Odsek 6

  Oprava Opravenej Odpocitanej Dane -> [JE_PODLA] -> Odsek 4
  Platitel -> [VYKONAVA] -> Oprava Opravenej Odpocitanej Dane
  Oprava Opravenej Odpocitanej Dane -> [VYCHADZA_Z] -> Pomerne Odpocitanie Dane
  Oprava Opravenej Odpocitanej Dane -> [VYCHADZA_Z] -> Vykonane Upravy Odpocitanej Dane

  Oprava Odpocitanej Dane -> [JE_PODLA] -> Odsek 1
  Oprava Odpocitanej Dane -> [JE_PODLA] -> Odsek 2
  Oprava Odpocitanej Dane -> [MA_DOVOD] -> Uplne Nezaplatenie Za Dodanie Tovaru Alebo Sluzby
  Uplne Nezaplatenie Za Dodanie Tovaru Alebo Sluzby -> [VZTAHUJE_SA_NA] -> Dodanie Tovaru Alebo Sluzby

  Platitel -> [VYKONAVA] -> Zohladnenie Oprav Pri Uprave Odpocitanej Dane
  Zohladnenie Oprav Pri Uprave Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Oprava Odpocitanej Dane
  Zohladnenie Oprav Pri Uprave Odpocitanej Dane -> [VZTAHUJE_SA_NA] -> Oprava Opravenej Odpocitanej Dane
  Uprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 54
  Uprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 54a
  Uprava Odpocitanej Dane -> [JE_PODLA] -> Paragraf § 54d

  Platitel -> [MA_POVINNOST] -> Uvedenie Oprav V Zaznamoch
  Uvedenie Oprav V Zaznamoch -> [VZTAHUJE_SA_NA] -> Oprava Odpocitanej Dane
  Uvedenie Oprav V Zaznamoch -> [VZTAHUJE_SA_NA] -> Oprava Opravenej Odpocitanej Dane
  Uvedenie Oprav V Zaznamoch -> [VZTAHUJE_SA_NA] -> Zaznamy Podla Paragrafu 70 Odseku 2 Pismena J
  Zaznamy Podla Paragrafu 70 Odseku 2 Pismena J -> [JE_PODLA] -> Paragraf § 70 Odsek 2 Pismeno j)

nodes:
  PravnyPredpis: 222/2004 Z. z. Zbierka zakonov Slovenskej republiky
  Paragraf: Paragraf § 25a
  Paragraf: Paragraf § 54
  Paragraf: Paragraf § 54a
  Paragraf: Paragraf § 54d
  Paragraf: Paragraf § 70
  Odsek: Paragraf § 25a Odsek 6
  Odsek: Odsek 1
  Odsek: Odsek 2
  Odsek: Odsek 4
  Pismeno: Paragraf § 70 Odsek 2 Pismeno j)

  ZdanitelnaOsoba: Platitel
  Suma: Suma Protihodnoty Zaplatena Za Dodanie Tovaru Alebo Sluzby
  Cinnost: Dodanie Tovaru Alebo Sluzby
  Oprava: Oprava Odpocitanej Dane
  Oprava: Oprava Opravenej Odpocitanej Dane
  Cinnost: Pomerne Odpocitanie Dane
  Oprava: Vykonane Upravy Odpocitanej Dane
  Dovod: Uplne Nezaplatenie Za Dodanie Tovaru Alebo Sluzby
  Oprava: Uprava Odpocitanej Dane
  Cinnost: Zohladnenie Oprav Pri Uprave Odpocitanej Dane
  Povinnost: Uvedenie Oprav V Zaznamoch
  Zaznam: Zaznamy Podla Paragrafu 70 Odseku 2 Pismena J


chunk: 261
page: 65
text: dane, akoby bol používaný len na podnikanie. § 55 Odpočítanie dane pri registrácii platiteľa a pri oneskorenej registrácii platiteľa (1) Platiteľ môže výlučne v prvom zdaňovacom období uplatniť právo na odpočítanie dane viažucej sa k tovarom a službám, ktoré nadobudol alebo prijal v postavení zdaniteľnej osoby pred dňom, keď sa stal platiteľom, alebo ktoré nadobudne alebo prijme v postavení platiteľa, ak daňová povinnosť vznikla podľa § 19 ods. 4 pred dňom, keď sa stal platiteľom, ak tieto prijaté plnenia okrem zásob neboli zahrnuté do daňových výdavkov podľa osobitného predpisu27a) v kalendárnych rokoch predchádzajúcich kalendárnemu roku, v ktorom sa stal platiteľom. Daň pri majetku, ktorý je podľa osobitného predpisu odpisovaným majetkom,26) platiteľ zníži o pomernú časť dane zodpovedajúcu odpisom; platiteľ, ktorý nie je účtovnou jednotkou, použije pri znížení odpočítateľnej dane postup ako platiteľ, ktorý je účtovnou jednotkou.
relationships:
  Zakon O Dani Z Pridanej Hodnoty -> [OBSAHUJE] -> Paragraf § 55
  Paragraf § 55 -> [MA_NAZOV] -> Odpocitanie Dane Pri Registracii Platitela A Pri Oneskorenej Registracii Platitela

  Platitel -> [MA_PRAVO] -> Pravo Na Odpocitanie Dane
  Pravo Na Odpocitanie Dane -> [MA_OBDOBIE] -> Prve Zdanovacie Obdobie
  Pravo Na Odpocitanie Dane -> [VZTAHUJE_SA_NA] -> Dan Viazuca Sa K Tovarom A Sluzbam
  Dan Viazuca Sa K Tovarom A Sluzbam -> [VZTAHUJE_SA_NA] -> Tovary A Sluzby

  Platitel -> [NADOBUDA] -> Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Zdanitelnej Osoby
  Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Zdanitelnej Osoby -> [MA_DATUM] -> Pred Dnom Ked Sa Zdanitelna Osoba Stala Platitelom

  Platitel -> [NADOBUDA] -> Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Platitela
  Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Platitela -> [MA_PODMIENKU] -> Danova Povinnost Vznikla Podla Paragrafu 19 Odseku 4 Pred Dnom Ked Sa Zdanitelna Osoba Stala Platitelom
  Danova Povinnost -> [JE_PODLA] -> Paragraf § 19 Odsek 4

  Prijate Plnenia Okrem Zasob -> [NESPLNA_PODMIENKY] -> Zahrnutie Do Danovych Vydavkov
  Zahrnutie Do Danovych Vydavkov -> [JE_PODLA] -> Osobitny Predpis
  Zahrnutie Do Danovych Vydavkov -> [MA_OBDOBIE] -> Kalendarne Roky Predchadzajuce Roku V Ktorom Sa Zdanitelna Osoba Stala Platitelom

  Odpisovany Majetok -> [JE_PODLA] -> Osobitny Predpis
  Dan Pri Odpisovanom Majetku -> [VZTAHUJE_SA_NA] -> Odpisovany Majetok
  Platitel -> [VYKONAVA] -> Znizenie Dane Pri Odpisovanom Majetku
  Znizenie Dane Pri Odpisovanom Majetku -> [MA_SUMU] -> Pomerna Cast Dane Zodpovedajuca Odpisom
  Pomerna Cast Dane Zodpovedajuca Odpisom -> [VYCHADZA_Z] -> Odpisy

  Platitel Ktory Nie Je Uctovnou Jednotkou -> [VYKONAVA] -> Znizenie Odpocitatelnej Dane
  Znizenie Odpocitatelnej Dane -> [VYCHADZA_Z] -> Postup Ako Pri Platitelovi Ktory Je Uctovnou Jednotkou

nodes:
  PravnyPredpis: Zakon O Dani Z Pridanej Hodnoty
  Paragraf: Paragraf § 55
  Hodnota: Odpocitanie Dane Pri Registracii Platitela A Pri Oneskorenej Registracii Platitela

  ZdanitelnaOsoba: Platitel
  ZdanitelnaOsoba: Platitel Ktory Nie Je Uctovnou Jednotkou
  ZdanitelnaOsoba: Platitel Ktory Je Uctovnou Jednotkou
  Pravo: Pravo Na Odpocitanie Dane
  ZdanovacieObdobie: Prve Zdanovacie Obdobie

  Dan: Dan Viazuca Sa K Tovarom A Sluzbam
  Tovar: Tovary A Sluzby
  Tovar: Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Zdanitelnej Osoby
  Tovar: Tovary A Sluzby Nadobudnute Alebo Prijate V Postaveni Platitela
  Datum: Pred Dnom Ked Sa Zdanitelna Osoba Stala Platitelom
  Podmienka: Danova Povinnost Vznikla Podla Paragrafu 19 Odseku 4 Pred Dnom Ked Sa Zdanitelna Osoba Stala Platitelom
  Povinnost: Danova Povinnost
  Odsek: Paragraf § 19 Odsek 4

  Cinnost: Prijate Plnenia Okrem Zasob
  Tovar: Zasoby
  Hodnota: Zahrnutie Do Danovych Vydavkov
  PravnyPredpis: Osobitny Predpis
  Obdobie: Kalendarne Roky Predchadzajuce Roku V Ktorom Sa Zdanitelna Osoba Stala Platitelom

  Majetok: Odpisovany Majetok
  Dan: Dan Pri Odpisovanom Majetku
  Cinnost: Znizenie Dane Pri Odpisovanom Majetku
  Cinnost: Znizenie Odpocitatelnej Dane
  Suma: Pomerna Cast Dane Zodpovedajuca Odpisom
  Vypocet: Odpisy
  Vypocet: Postup Ako Pri Platitelovi Ktory Je Uctovnou Jednotkou


chunk: 298
page: 74
text: dane. (4) Tlačivo na vrátenie dane uchováva platiteľ počas desiatich rokov od konca kalendárneho roka, v ktorom vrátenú daň uplatnil v daňovom priznaní. (5) Ak sa vrátenie dane uplatňuje u poverenej osoby podľa § 59 ods. 4 písm. c), táto osoba vráti daň cestujúcemu po overení oprávnenosti nároku na vrátenie dane (§ 59 ods. 1, 3 a 6) na základe elektronicky predloženého dokladu o kúpe tovaru a potvrdenia vývozu tovaru colným úradom. (6) Daň vrátenú podľa odseku 5 uplatní poverená osoba podľa § 59 ods. 4 písm. c) podaním žiadosti o vrátenie dane elektronickými prostriedkami Daňovému úradu Bratislava za kalendárny mesiac, v ktorom bola daň vrátená cestujúcemu. Prílohou k žiadosti o vrátenie dane je zoznam dokladov o kúpe tovaru, z ktorých bola vrátená daň. Zoznam dokladov o kúpe tovaru sa uvádza v členení podľa cestujúcich a obsahuje údaje v rozsahu dohodnutom podľa § 59 ods. 5 písm. d).
relationships:
  Platitel -> [UCHOVAVA] -> Tlacivo Na Vratenie Dane
  Tlacivo Na Vratenie Dane -> [MA_UCEL] -> Vratenie Dane
  Uchovanie Tlaciva Na Vratenie Dane -> [MA_LEHOTU] -> Desat Rokov Od Konca Kalendarneho Roka
  Desat Rokov Od Konca Kalendarneho Roka -> [PLATI_OD] -> Koniec Kalendarneho Roka V Ktorom Bola Vratena Dan Uplatnena V Danovom Priznani
  Platitel -> [VYKONAVA] -> Uplatnenie Vratenej Dane V Danovom Priznani

  Poverena Osoba -> [JE_PODLA] -> Paragraf § 59 Odsek 4 Pismeno c)
  Poverena Osoba -> [VYKONAVA] -> Vratenie Dane Cestujucemu
  Vratenie Dane Cestujucemu -> [VZTAHUJE_SA_NA] -> Cestujuci
  Vratenie Dane Cestujucemu -> [MA_PODMIENKU] -> Overenie Opravnenosti Naroku Na Vratenie Dane
  Overenie Opravnenosti Naroku Na Vratenie Dane -> [JE_PODLA] -> Paragraf § 59 Odsek 1
  Overenie Opravnenosti Naroku Na Vratenie Dane -> [JE_PODLA] -> Paragraf § 59 Odsek 3
  Overenie Opravnenosti Naroku Na Vratenie Dane -> [JE_PODLA] -> Paragraf § 59 Odsek 6
  Overenie Opravnenosti Naroku Na Vratenie Dane -> [VYCHADZA_Z] -> Elektronicky Predlozeny Doklad O Kupe Tovaru
  Overenie Opravnenosti Naroku Na Vratenie Dane -> [VYCHADZA_Z] -> Potvrdenie Vyvozu Tovaru Colnym Uradom

  Poverena Osoba -> [PODAVA] -> Ziadost O Vratenie Dane
  Ziadost O Vratenie Dane -> [DORUCUJE] -> Danovy Urad Bratislava
  Ziadost O Vratenie Dane -> [MA_VLASTNOST] -> Elektronicke Prostriedky
  Ziadost O Vratenie Dane -> [MA_OBDOBIE] -> Kalendarneho Mesiaca V Ktorom Bola Dan Vratena Cestujucemu
  Dan Vratena Podla Odseku 5 -> [JE_PODLA] -> Odsek 5

  Zoznam Dokladov O Kupe Tovaru -> [JE_SUCASTOU] -> Ziadost O Vratenie Dane
  Zoznam Dokladov O Kupe Tovaru -> [OBSAHUJE] -> Doklady O Kupe Tovaru
  Doklady O Kupe Tovaru -> [VZTAHUJE_SA_NA] -> Dan Vratena Cestujucemu
  Zoznam Dokladov O Kupe Tovaru -> [MA_VLASTNOST] -> Clenenie Podla Cestujucich
  Zoznam Dokladov O Kupe Tovaru -> [MA_OBSAH] -> Udaje V Rozsahu Dohodnutom Podla Paragrafu 59 Odseku 5 Pismena D
  Udaje V Rozsahu Dohodnutom Podla Paragrafu 59 Odseku 5 Pismena D -> [JE_PODLA] -> Paragraf § 59 Odsek 5 Pismeno d)

nodes:
  ZdanitelnaOsoba: Platitel
  Doklad: Tlacivo Na Vratenie Dane
  Cinnost: Vratenie Dane
  Cinnost: Uchovanie Tlaciva Na Vratenie Dane
  Lehota: Desat Rokov Od Konca Kalendarneho Roka
  Obdobie: Koniec Kalendarneho Roka V Ktorom Bola Vratena Dan Uplatnena V Danovom Priznani
  Dan: Vratena Dan
  Dan: Dan Vratena Podla Odseku 5
  Dan: Dan Vratena Cestujucemu
  DanovePriznanie: Danove Priznanie
  Cinnost: Uplatnenie Vratenej Dane V Danovom Priznani

  Subjekt: Poverena Osoba
  Osoba: Cestujuci
  Cinnost: Vratenie Dane Cestujucemu
  Cinnost: Overenie Opravnenosti Naroku Na Vratenie Dane
  Pravo: Narok Na Vratenie Dane
  Doklad: Elektronicky Predlozeny Doklad O Kupe Tovaru
  Doklad: Potvrdenie Vyvozu Tovaru Colnym Uradom

  Ziadost: Ziadost O Vratenie Dane
  ElektronickyProstriedok: Elektronicke Prostriedky
  Urad: Danovy Urad Bratislava
  Obdobie: Kalendarneho Mesiaca V Ktorom Bola Dan Vratena Cestujucemu
  Doklad: Zoznam Dokladov O Kupe Tovaru
  Doklad: Doklady O Kupe Tovaru
  Hodnota: Clenenie Podla Cestujucich
  Hodnota: Udaje V Rozsahu Dohodnutom Podla Paragrafu 59 Odseku 5 Pismena D

  Pismeno: Paragraf § 59 Odsek 4 Pismeno c)
  Odsek: Paragraf § 59 Odsek 1
  Odsek: Paragraf § 59 Odsek 3
  Odsek: Paragraf § 59 Odsek 6
  Odsek: Odsek 5
  Pismeno: Paragraf § 59 Odsek 5 Pismeno d)


chunk: 327
page: 81
text: c) iným obchodníkom, ktorý uplatňuje daň podľa osobitnej úpravy tohto zákona alebo zákona platného v inom členskom štáte. (3) Základom dane pri predaji tovaru podľa odseku 2 je kladný rozdiel medzi predajnou cenou a kúpnou cenou znížený o daň.
relationships:
  Iny Obchodnik -> [VYKONAVA] -> Uplatnovanie Dane Podla Osobitnej Upravy
  Uplatnovanie Dane Podla Osobitnej Upravy -> [JE_PODLA] -> Osobitna Uprava Tohto Zakona
  Uplatnovanie Dane Podla Osobitnej Upravy -> [JE_PODLA] -> Osobitna Uprava Zakona Platneho V Inom Clenskom State
  Osobitna Uprava Zakona Platneho V Inom Clenskom State -> [PLATI_PRE] -> Iny Clensky Stat

  Zaklad Dane Pri Predaji Tovaru Podla Odseku 2 -> [VZTAHUJE_SA_NA] -> Predaj Tovaru Podla Odseku 2
  Predaj Tovaru Podla Odseku 2 -> [JE_PODLA] -> Odsek 2
  Zaklad Dane Pri Predaji Tovaru Podla Odseku 2 -> [VYCHADZA_Z] -> Kladny Rozdiel Medzi Predajnou Cenou A Kupnou Cenou Znizeny O Dan
  Kladny Rozdiel Medzi Predajnou Cenou A Kupnou Cenou Znizeny O Dan -> [VYCHADZA_Z] -> Predajna Cena
  Kladny Rozdiel Medzi Predajnou Cenou A Kupnou Cenou Znizeny O Dan -> [VYCHADZA_Z] -> Kupna Cena
  Kladny Rozdiel Medzi Predajnou Cenou A Kupnou Cenou Znizeny O Dan -> [VYCHADZA_Z] -> Dan

nodes:
  Subjekt: Iny Obchodnik
  Dan: Dan
  Cinnost: Uplatnovanie Dane Podla Osobitnej Upravy
  Status: Osobitna Uprava Tohto Zakona
  Status: Osobitna Uprava Zakona Platneho V Inom Clenskom State
  ClenskyStat: Iny Clensky Stat
  Hodnota: Zaklad Dane Pri Predaji Tovaru Podla Odseku 2
  Cinnost: Predaj Tovaru Podla Odseku 2
  Odsek: Odsek 2
  Vypocet: Kladny Rozdiel Medzi Predajnou Cenou A Kupnou Cenou Znizeny O Dan
  Suma: Predajna Cena
  Suma: Kupna Cena


chunk: 365
page: 90
text: (9) Zdaniteľná osoba, ktorá má povolenie na uplatňovanie osobitnej úpravy podľa odsekov 4 až 25, je povinná uplatňovať osobitnú úpravu na všetky predaje tovaru na diaľku podľa § 68 ods. 1 písm. c).
relationships:
  Odsek 9 -> [UPRAVUJE] -> Povinnost Uplatnovat Osobitnu Upravu
  Zdanitelna Osoba -> [MA_DOKLAD] -> Povolenie Na Uplatnovanie Osobitnej Upravy
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 4
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 5
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 6
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 7
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 8
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 9
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 10
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 11
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 12
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 13
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 14
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 15
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 16
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 17
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 18
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 19
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 20
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 21
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 22
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 23
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 24
  Povolenie Na Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Odsek 25

  Zdanitelna Osoba -> [MA_POVINNOST] -> Povinnost Uplatnovat Osobitnu Upravu
  Povinnost Uplatnovat Osobitnu Upravu -> [VZTAHUJE_SA_NA] -> Vsetky Predaje Tovaru Na Dialku Podla Paragrafu 68 Odseku 1 Pismena C
  Osobitna Uprava -> [VZTAHUJE_SA_NA] -> Vsetky Predaje Tovaru Na Dialku Podla Paragrafu 68 Odseku 1 Pismena C

  Vsetky Predaje Tovaru Na Dialku Podla Paragrafu 68 Odseku 1 Pismena C -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno c)
  Paragraf § 68 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1
  Paragraf § 68 Odsek 1 -> [OBSAHUJE] -> Paragraf § 68 Odsek 1 Pismeno c)

nodes:
  Odsek: Odsek 9
  ZdanitelnaOsoba: Zdanitelna Osoba
  Povinnost: Povinnost Uplatnovat Osobitnu Upravu
  Doklad: Povolenie Na Uplatnovanie Osobitnej Upravy
  Status: Osobitna Uprava
  Odsek: Odsek 4
  Odsek: Odsek 5
  Odsek: Odsek 6
  Odsek: Odsek 7
  Odsek: Odsek 8
  Odsek: Odsek 10
  Odsek: Odsek 11
  Odsek: Odsek 12
  Odsek: Odsek 13
  Odsek: Odsek 14
  Odsek: Odsek 15
  Odsek: Odsek 16
  Odsek: Odsek 17
  Odsek: Odsek 18
  Odsek: Odsek 19
  Odsek: Odsek 20
  Odsek: Odsek 21
  Odsek: Odsek 22
  Odsek: Odsek 23
  Odsek: Odsek 24
  Odsek: Odsek 25
  Cinnost: Vsetky Predaje Tovaru Na Dialku Podla Paragrafu 68 Odseku 1 Pismena C
  Paragraf: Paragraf § 68
  Odsek: Paragraf § 68 Odsek 1
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)


chunk: 372
page: 92
text: nasledujúci deň, ak nebol v posledný deň zdaňovacieho obdobia kurz určený a vyhlásený. (23) Daň sa platí v eurách na príslušný účet daňového úradu najneskôr do konca lehoty na podanie daňového priznania s uvedením odkazu na príslušné daňové priznanie. Ak koniec lehoty na zaplatenie dane pripadne na sobotu, nedeľu alebo deň pracovného pokoja, posledným dňom lehoty je tento deň. Za deň platby sa považuje deň, keď platba bola pripísaná na účet daňového úradu. (24) Každá zmena údajov z pôvodného daňového priznania sa uvedie v nasledujúcom daňovom priznaní najneskôr do troch rokov odo dňa uplynutia lehoty na podanie pôvodného daňového priznania. V tomto nasledujúcom daňovom priznaní sa uvedie príslušný členský štát spotreby, zdaňovacie obdobie a suma dane, ktorá vyplýva z opravy. Ak sa pri predaji tovaru na diaľku podľa § 68 ods. 1 písm. c) úhrada uskutočnila v inej mene ako v eurách, použije sa pri oprave sumy dane kurz, ktorý sa mal použiť pri prepočte úhrady za dodaný tovar podľa odseku 22.
relationships:
  Dan -> [JE_POVINNY_PLATIT] -> Euro
  Dan -> [ZAPLATI] -> Prislusny Ucet Danoveho Uradu
  Dan -> [MA_LEHOTU] -> Koniec Lehoty Na Podanie Danoveho Priznania
  Platba Dane -> [MA_DOKLAD] -> Odkaz Na Prislusne Danove Priznanie
  Odkaz Na Prislusne Danove Priznanie -> [TYKA_SA] -> Prislusne Danove Priznanie

  Koniec Lehoty Na Zaplatenie Dane -> [MA_PODMIENKU] -> Sobota Nedela Alebo Den Pracovneho Pokoja
  Posledny Den Lehoty -> [POVAZUJE_SA_ZA] -> Sobota Nedela Alebo Den Pracovneho Pokoja
  Den Platby -> [POVAZUJE_SA_ZA] -> Den Pripisania Platby Na Ucet Danoveho Uradu
  Den Pripisania Platby Na Ucet Danoveho Uradu -> [VZTAHUJE_SA_NA] -> Prislusny Ucet Danoveho Uradu

  Zmena Udajov Z Povodneho Danoveho Priznania -> [UVADZA] -> Nasledujuce Danove Priznanie
  Uvedenie Zmeny Udajov -> [MA_LEHOTU] -> Tri Roky Od Uplynutia Lehoty Na Podanie Povodneho Danoveho Priznania
  Tri Roky Od Uplynutia Lehoty Na Podanie Povodneho Danoveho Priznania -> [PLATI_OD] -> Uplynutie Lehoty Na Podanie Povodneho Danoveho Priznania

  Nasledujuce Danove Priznanie -> [MA_OBSAH] -> Prislusny Clensky Stat Spotreby
  Nasledujuce Danove Priznanie -> [MA_OBSAH] -> Zdanovacie Obdobie
  Nasledujuce Danove Priznanie -> [MA_OBSAH] -> Suma Dane Vyplyvajuca Z Opravy

  Predaj Tovaru Na Dialku -> [JE_PODLA] -> Paragraf § 68 Odsek 1 Pismeno c)
  Uhrada V Inej Mene Ako V Eurach -> [MA_PODMIENKU] -> Ina Mena Ako Euro
  Oprava Sumy Dane -> [MA_PODMIENKU] -> Uhrada V Inej Mene Ako V Eurach
  Oprava Sumy Dane -> [VYCHADZA_Z] -> Kurz Podla Odseku 22
  Kurz Podla Odseku 22 -> [JE_PODLA] -> Odsek 22
  Kurz Podla Odseku 22 -> [VZTAHUJE_SA_NA] -> Prepocet Uhrady Za Dodany Tovar

nodes:
  Dan: Dan
  Mena: Euro
  BankovyUcet: Prislusny Ucet Danoveho Uradu
  Lehota: Koniec Lehoty Na Podanie Danoveho Priznania
  Cinnost: Platba Dane
  Doklad: Odkaz Na Prislusne Danove Priznanie
  DanovePriznanie: Prislusne Danove Priznanie
  Lehota: Koniec Lehoty Na Zaplatenie Dane
  Podmienka: Sobota Nedela Alebo Den Pracovneho Pokoja
  Datum: Posledny Den Lehoty
  Datum: Den Platby
  Datum: Den Pripisania Platby Na Ucet Danoveho Uradu

  DanovePriznanie: Povodne Danove Priznanie
  DanovePriznanie: Nasledujuce Danove Priznanie
  Hodnota: Zmena Udajov Z Povodneho Danoveho Priznania
  Cinnost: Uvedenie Zmeny Udajov
  Lehota: Tri Roky Od Uplynutia Lehoty Na Podanie Povodneho Danoveho Priznania
  CasovyUdaj: Uplynutie Lehoty Na Podanie Povodneho Danoveho Priznania
  ClenskyStat: Prislusny Clensky Stat Spotreby
  ZdanovacieObdobie: Zdanovacie Obdobie
  Suma: Suma Dane Vyplyvajuca Z Opravy

  Cinnost: Predaj Tovaru Na Dialku
  Pismeno: Paragraf § 68 Odsek 1 Pismeno c)
  Suma: Uhrada V Inej Mene Ako V Eurach
  Mena: Ina Mena Ako Euro
  Oprava: Oprava Sumy Dane
  Kurz: Kurz Podla Odseku 22
  Odsek: Odsek 22
  Vypocet: Prepocet Uhrady Za Dodany Tovar
  Tovar: Dodany Tovar


chunk: 38
page: 9
text: zahraničná osoba stala platiteľom; ak platiteľ má pridelené identifikačné číslo pre daň podľa § 7 alebo na účely uplatňovania osobitnej úpravy podľa § 68b ods. 2, daňový úrad mu pridelí to isté identifikačné číslo pre daň. Proti rozhodnutiu podľa druhej vety nemožno podať odvolanie. Ak Daňový úrad Bratislava zahraničnú osobu nezaregistruje z dôvodu, že sa nestala platiteľom, vydá o tom rozhodnutie, proti ktorému nemožno podať odvolanie. (4) Platiteľom sa stáva aj zahraničná osoba, na ktorú v tuzemsku prechádza hmotný majetok alebo nehmotný majetok platiteľa, ktorý a) zanikol bez likvidácie, a to dňom keď sa stala právnym nástupcom platiteľa, ak naďalej spĺňa status zahraničnej osoby, b) sa rozdelil odštiepením, a to dňom účinnosti premeny pri odštiepení podľa zákona platného v inom členskom štáte alebo zmluvnom štáte Dohody o Európskom hospodárskom priestore, ak naďalej spĺňa status zahraničnej osoby, alebo c) sa rozdelil cezhraničným odštiepením, a to dňom účinnosti premeny pri cezhraničnom
relationships:
  Zahranicna Osoba -> [STAVA_SA] -> Platitel
  Platitel -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan

  Identifikacne Cislo Pre Dan -> [JE_PODLA] -> Paragraf § 7
  Uplatnovanie Osobitnej Upravy -> [JE_PODLA] -> Paragraf § 68b Odsek 2
  Danovy Urad -> [PRIDELUJE] -> To Iste Identifikacne Cislo Pre Dan
  To Iste Identifikacne Cislo Pre Dan -> [JE_TYPOM] -> Identifikacne Cislo Pre Dan

  Rozhodnutie Podla Druhej Vety -> [NEVZTAHUJE_SA_NA] -> Odvolanie
  Danovy Urad Bratislava -> [VYDAVA] -> Rozhodnutie O Nezaregistrovani Zahranicnej Osoby
  Rozhodnutie O Nezaregistrovani Zahranicnej Osoby -> [MA_DOVOD] -> Zahranicna Osoba Sa Nestala Platitelom
  Rozhodnutie O Nezaregistrovani Zahranicnej Osoby -> [NEVZTAHUJE_SA_NA] -> Odvolanie

  Hmotny Majetok Platitela -> [PRECHADZA_NA] -> Zahranicna Osoba
  Nehmotny Majetok Platitela -> [PRECHADZA_NA] -> Zahranicna Osoba
  Prechod Majetku Platitela -> [MA_MIESTO] -> Tuzemsko
  Zahranicna Osoba -> [STAVA_SA] -> Platitel
  Zahranicna Osoba -> [SPLNA_PODMIENKY] -> Status Zahranicnej Osoby

  Platitel -> [ZANIKA] -> Zanik Bez Likvidacie
  Zahranicna Osoba -> [STAVA_SA] -> Pravny Nastupca Platitela
  Zahranicna Osoba -> [MA_DATUM] -> Den Ked Sa Stala Pravnym Nastupcom Platitela

  Rozdelenie Odstiepenim -> [VZTAHUJE_SA_NA] -> Platitel
  Zahranicna Osoba -> [MA_DATUM] -> Den Ucinnosti Premeny Pri Odstiepeni
  Premena Pri Odstiepeni -> [JE_PODLA] -> Zakon Platny V Inom Clenskom State Alebo Zmluvnom State Dohody O Europskom Hospodarskom Priestore
  Zakon Platny V Inom Clenskom State Alebo Zmluvnom State Dohody O Europskom Hospodarskom Priestore -> [PLATI_PRE] -> Iny Clensky Stat
  Zakon Platny V Inom Clenskom State Alebo Zmluvnom State Dohody O Europskom Hospodarskom Priestore -> [PLATI_PRE] -> Zmluvny Stat Dohody O Europskom Hospodarskom Priestore

  Rozdelenie Cezhranicnym Odstiepenim -> [VZTAHUJE_SA_NA] -> Platitel
  Zahranicna Osoba -> [MA_DATUM] -> Den Ucinnosti Premeny Pri Cezhranicnom Odstiepeni

nodes:
  Osoba: Zahranicna Osoba
  ZdanitelnaOsoba: Platitel
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  IdentifikacneCislo: To Iste Identifikacne Cislo Pre Dan
  Urad: Danovy Urad
  Urad: Danovy Urad Bratislava
  Paragraf: Paragraf § 7
  Odsek: Paragraf § 68b Odsek 2
  Cinnost: Uplatnovanie Osobitnej Upravy
  Rozhodnutie: Rozhodnutie Podla Druhej Vety
  Rozhodnutie: Rozhodnutie O Nezaregistrovani Zahranicnej Osoby
  Konanie: Odvolanie
  Dovod: Zahranicna Osoba Sa Nestala Platitelom
  Tuzemsko: Tuzemsko
  Majetok: Hmotny Majetok Platitela
  Majetok: Nehmotny Majetok Platitela
  Cinnost: Prechod Majetku Platitela
  Status: Status Zahranicnej Osoby
  Status: Zanik Bez Likvidacie
  PravnyNastupca: Pravny Nastupca Platitela
  Datum: Den Ked Sa Stala Pravnym Nastupcom Platitela
  Cinnost: Rozdelenie Odstiepenim
  Cinnost: Premena Pri Odstiepeni
  Datum: Den Ucinnosti Premeny Pri Odstiepeni
  Cinnost: Rozdelenie Cezhranicnym Odstiepenim
  Datum: Den Ucinnosti Premeny Pri Cezhranicnom Odstiepeni
  PravnyPredpis: Zakon Platny V Inom Clenskom State Alebo Zmluvnom State Dohody O Europskom Hospodarskom Priestore
  ClenskyStat: Iny Clensky Stat
  ClenskyStat: Zmluvny Stat Dohody O Europskom Hospodarskom Priestore
  Zmluva: Dohoda O Europskom Hospodarskom Priestore


# TODO
chunk: 448
page: 110
text: služieb poskytovateľa platby podľa odseku 3 písm. b) povinný zahrnúť všetky platobné služby poskytnuté poskytovateľovi platby, ktoré zodpovedajú cezhraničným platbám rovnakému príjemcovi platby. (5) Tuzemský poskytovateľ platobných služieb podľa odseku 3 je povinný a) viesť záznamy podľa odseku 8 v elektronickom formáte počas obdobia troch kalendárnych rokov od konca kalendárneho roka, v ktorom bola platba vykonaná, b) sprístupniť finančnému riaditeľstvu elektronickými prostriedkami záznamy podľa odseku 8 prostredníctvom elektronického formulára najneskôr do konca kalendárneho mesiaca nasledujúceho po kalendárnom štvrťroku, ktorého sa tieto záznamy týkajú podľa osobitného predpisu.28p) (6) Za miesto poskytovateľa platby, ktorý vykoná cezhraničnú platbu, sa na účely tohto ustanovenia považuje členský štát identifikovaný podľa a) IBAN platobného účtu poskytovateľa platby alebo akéhokoľvek iného identifikátora, ktorý určuje poskytovateľa platby a jeho miesto, alebo
relationships:
  Poskytovatel Platobnych Sluzieb Poskytovatela Platby -> [MA_POVINNOST] -> Zahrnutie Vsetkych Platobnych Sluzieb
  Zahrnutie Vsetkych Platobnych Sluzieb -> [JE_PODLA] -> Odsek 3 Pismeno b)
  Zahrnutie Vsetkych Platobnych Sluzieb -> [VZTAHUJE_SA_NA] -> Platobne Sluzby Poskytnute Poskytovatelovi Platby
  Platobne Sluzby Poskytnute Poskytovatelovi Platby -> [TYKA_SA] -> Cezhranicne Platby Rovnakemu Prijemcovi Platby

  Tuzemsky Poskytovatel Platobnych Sluzieb -> [JE_PODLA] -> Odsek 3
  Tuzemsky Poskytovatel Platobnych Sluzieb -> [MA_POVINNOST] -> Vedenie Zaznamov Podla Odseku 8 V Elektronickom Formate
  Vedenie Zaznamov Podla Odseku 8 V Elektronickom Formate -> [VZTAHUJE_SA_NA] -> Zaznamy Podla Odseku 8
  Vedenie Zaznamov Podla Odseku 8 V Elektronickom Formate -> [MA_DOBU] -> Tri Kalendarne Roky Od Konca Kalendarneho Roka V Ktorom Bola Platba Vykonana
  Tri Kalendarne Roky Od Konca Kalendarneho Roka V Ktorom Bola Platba Vykonana -> [PLATI_OD] -> Koniec Kalendarneho Roka V Ktorom Bola Platba Vykonana
  Platba -> [MA_DATUM] -> Kalendarneho Roka V Ktorom Bola Platba Vykonana

  Tuzemsky Poskytovatel Platobnych Sluzieb -> [MA_POVINNOST] -> Spristupnenie Zaznamov Financnemu Riaditelstvu
  Spristupnenie Zaznamov Financnemu Riaditelstvu -> [VZTAHUJE_SA_NA] -> Zaznamy Podla Odseku 8
  Spristupnenie Zaznamov Financnemu Riaditelstvu -> [DORUCUJE] -> Financne Riaditelstvo
  Spristupnenie Zaznamov Financnemu Riaditelstvu -> [MA_VLASTNOST] -> Elektronicke Prostriedky
  Spristupnenie Zaznamov Financnemu Riaditelstvu -> [MA_DOKLAD] -> Elektronicky Formular
  Spristupnenie Zaznamov Financnemu Riaditelstvu -> [MA_LEHOTU] -> Koniec Kalendarneho Mesiaca Nasledujuceho Po Kalendarom Stvrtroku
  Koniec Kalendarneho Mesiaca Nasledujuceho Po Kalendarom Stvrtroku -> [VYCHADZA_Z] -> Osobitny Predpis
  Zaznamy Podla Odseku 8 -> [TYKA_SA] -> Kalendarny Stvrtrok

  Miesto Poskytovatela Platby -> [POVAZUJE_SA_ZA] -> Clensky Stat Identifikovany Podla IBAN Alebo Ineho Identifikatora
  Clensky Stat Identifikovany Podla IBAN Alebo Ineho Identifikatora -> [VYCHADZA_Z] -> IBAN Platobneho Uctu Poskytovatela Platby
  Clensky Stat Identifikovany Podla IBAN Alebo Ineho Identifikatora -> [VYCHADZA_Z] -> Iny Identifikator Urcujuci Poskytovatela Platby A Jeho Miesto
  Poskytovatel Platby -> [VYKONAVA] -> Cezhranicna Platba

nodes:
  Odsek: Odsek 3
  Odsek: Odsek 8
  Pismeno: Odsek 3 Pismeno b)
  Subjekt: Poskytovatel Platobnych Sluzieb Poskytovatela Platby
  Subjekt: Tuzemsky Poskytovatel Platobnych Sluzieb
  Subjekt: Poskytovatel Platby
  Subjekt: Prijemca Platby
  Povinnost: Zahrnutie Vsetkych Platobnych Sluzieb
  Sluzba: Platobne Sluzby Poskytnute Poskytovatelovi Platby
  Cinnost: Cezhranicne Platby Rovnakemu Prijemcovi Platby
  Cinnost: Cezhranicna Platba

  Povinnost: Vedenie Zaznamov Podla Odseku 8 V Elektronickom Formate
  Zaznam: Zaznamy Podla Odseku 8
  ElektronickyProstriedok: Elektronicky Format
  Obdobie: Tri Kalendarne Roky Od Konca Kalendarneho Roka V Ktorom Bola Platba Vykonana
  CasovyUdaj: Koniec Kalendarneho Roka V Ktorom Bola Platba Vykonana
  Datum: Kalendarneho Roka V Ktorom Bola Platba Vykonana
  Cinnost: Platba

  Povinnost: Spristupnenie Zaznamov Financnemu Riaditelstvu
  FinancneRiaditelstvo: Financne Riaditelstvo
  ElektronickyProstriedok: Elektronicke Prostriedky
  Doklad: Elektronicky Formular
  Lehota: Koniec Kalendarneho Mesiaca Nasledujuceho Po Kalendarom Stvrtroku
  Obdobie: Kalendarny Stvrtrok
  PravnyPredpis: Osobitny Predpis

  Lokacia: Miesto Poskytovatela Platby
  ClenskyStat: Clensky Stat Identifikovany Podla IBAN Alebo Ineho Identifikatora
  BankovyUcet: IBAN Platobneho Uctu Poskytovatela Platby
  IdentifikacneCislo: Iny Identifikator Urcujuci Poskytovatela Platby A Jeho Miesto


chunk: 516
page: 127
text: 2. opakovane v kalendárnom roku nesplní povinnosť podať daňové priznanie alebo kontrolný výkaz, opakovane v kalendárnom roku nezaplatí vlastnú daňovú povinnosť, opakovane nie je zastihnuteľný na adrese sídla, miesta podnikania a ani na adrese prevádzkarne alebo opakovane porušuje povinnosti pri daňovej kontrole. (4) Daňový úrad o zrušení registrácie pre daň vydá rozhodnutie, v ktorom určí deň, uplynutím ktorého právnická osoba alebo fyzická osoba prestáva byť platiteľom; proti tomuto rozhodnutiu nie je možné podať odvolanie. Ak sa právnická osoba zrušuje bez likvidácie, daňový úrad rozhodnutie nevydá a právnická osoba prestáva byť platiteľom dňom jej zániku. Uplynutím dňa, kedy právnická osoba alebo fyzická osoba prestáva byť platiteľom, končí prebiehajúce zdaňovacie obdobie a zaniká platnosť identifikačného čísla pre daň; ak právnická osoba alebo fyzická osoba uplatňuje osobitnú úpravu podľa § 68b, platnosť identifikačného čísla pre daň na účely uplatňovania tejto osobitnej úpravy nezaniká.
relationships:
  Zrusenie Registracie Pre Dan -> [MA_DOVOD] -> Opakovane Nesplnenie Povinnosti Podat Danove Priznanie
  Zrusenie Registracie Pre Dan -> [MA_DOVOD] -> Opakovane Nesplnenie Povinnosti Podat Kontrolny Vykaz
  Zrusenie Registracie Pre Dan -> [MA_DOVOD] -> Opakovane Nezaplatenie Vlastnej Danovej Povinnosti
  Zrusenie Registracie Pre Dan -> [MA_DOVOD] -> Opakovana Nezastihnutelnost Na Adrese Sidla Miesta Podnikania Alebo Prevadzkarne
  Zrusenie Registracie Pre Dan -> [MA_DOVOD] -> Opakovane Porusovanie Povinnosti Pri Danovej Kontrole

  Opakovane Nesplnenie Povinnosti Podat Danove Priznanie -> [MA_OBDOBIE] -> Kalendarny Rok
  Opakovane Nesplnenie Povinnosti Podat Kontrolny Vykaz -> [MA_OBDOBIE] -> Kalendarny Rok
  Opakovane Nezaplatenie Vlastnej Danovej Povinnosti -> [MA_OBDOBIE] -> Kalendarny Rok

  Danovy Urad -> [VYDAVA] -> Rozhodnutie O Zruseni Registracie Pre Dan
  Rozhodnutie O Zruseni Registracie Pre Dan -> [TYKA_SA] -> Zrusenie Registracie Pre Dan
  Rozhodnutie O Zruseni Registracie Pre Dan -> [URCUJE] -> Den Prestania Byt Platitelom
  Rozhodnutie O Zruseni Registracie Pre Dan -> [NEVZTAHUJE_SA_NA] -> Odvolanie

  Pravnicka Osoba -> [MA_STATUS] -> Prestava Byt Platitelom
  Fyzicka Osoba -> [MA_STATUS] -> Prestava Byt Platitelom
  Prestava Byt Platitelom -> [MA_DATUM] -> Den Prestania Byt Platitelom

  Pravnicka Osoba -> [ZANIKA] -> Zanik Pravnickej Osoby Bez Likvidacie
  Zrusenie Pravnickej Osoby Bez Likvidacie -> [NEVZTAHUJE_SA_NA] -> Rozhodnutie O Zruseni Registracie Pre Dan
  Pravnicka Osoba -> [MA_STATUS] -> Prestava Byt Platitelom Dnom Zaniku

  Prebiehajuce Zdanovacie Obdobie -> [ZANIKA] -> Koniec Prebiehajuceho Zdanovacieho Obdobia
  Koniec Prebiehajuceho Zdanovacieho Obdobia -> [NASTAVA_PRI] -> Den Prestania Byt Platitelom

  Platnost Identifikacneho Cisla Pre Dan -> [ZANIKA] -> Zanik Platnosti Identifikacneho Cisla Pre Dan
  Zanik Platnosti Identifikacneho Cisla Pre Dan -> [NASTAVA_PRI] -> Den Prestania Byt Platitelom

  Pravnicka Osoba Alebo Fyzicka Osoba -> [VYKONAVA] -> Uplatnovanie Osobitnej Upravy Podla Paragrafu 68b
  Uplatnovanie Osobitnej Upravy Podla Paragrafu 68b -> [JE_PODLA] -> Paragraf § 68b
  Zanik Platnosti Identifikacneho Cisla Pre Dan -> [NEVZTAHUJE_SA_NA] -> Identifikacne Cislo Pre Dan Na Ucely Osobitnej Upravy Podla Paragrafu 68b

nodes:
  Obdobie: Kalendarny Rok
  DanovePriznanie: Danove Priznanie
  Doklad: Kontrolny Vykaz
  Povinnost: Vlastna Danova Povinnost
  Sidlo: Adresa Sidla
  Lokacia: Miesto Podnikania
  Prevazdkaren: Adresa Prevadzkarne
  Konanie: Danova Kontrola
  Urad: Danovy Urad
  Registracia: Zrusenie Registracie Pre Dan
  Rozhodnutie: Rozhodnutie O Zruseni Registracie Pre Dan
  PravnickaOsoba: Pravnicka Osoba
  FyzickaOsoba: Fyzicka Osoba
  ZdanitelnaOsoba: Pravnicka Osoba Alebo Fyzicka Osoba
  Status: Prestava Byt Platitelom
  Status: Prestava Byt Platitelom Dnom Zaniku
  Datum: Den Prestania Byt Platitelom
  Status: Zanik Pravnickej Osoby Bez Likvidacie
  Cinnost: Zrusenie Pravnickej Osoby Bez Likvidacie
  Konanie: Odvolanie
  ZdanovacieObdobie: Prebiehajuce Zdanovacie Obdobie
  Status: Koniec Prebiehajuceho Zdanovacieho Obdobia
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  Status: Platnost Identifikacneho Cisla Pre Dan
  Status: Zanik Platnosti Identifikacneho Cisla Pre Dan
  IdentifikacneCislo: Identifikacne Cislo Pre Dan Na Ucely Osobitnej Upravy Podla Paragrafu 68b
  Cinnost: Uplatnovanie Osobitnej Upravy Podla Paragrafu 68b
  Paragraf: Paragraf § 68b
  Dovod: Opakovane Nesplnenie Povinnosti Podat Danove Priznanie
  Dovod: Opakovane Nesplnenie Povinnosti Podat Kontrolny Vykaz
  Dovod: Opakovane Nezaplatenie Vlastnej Danovej Povinnosti
  Dovod: Opakovana Nezastihnutelnost Na Adrese Sidla Miesta Podnikania Alebo Prevadzkarne
  Dovod: Opakovane Porusovanie Povinnosti Pri Danovej Kontrole


chunk: 534
page: 131
text: (15) Pri tovare, ktorý je k 30. aprílu 2004 v tuzemsku a pri vstupe do tuzemska bol predložený colnému úradu a má postavenie dočasne uskladneného tovaru alebo je umiestnený do slobodného colného pásma alebo do slobodného colného skladu alebo prepustený do colného režimu uskladňovanie v colnom sklade, do colného režimu aktívny zušľachťovací styk, do colného režimu dočasné použitie s úplným oslobodením od dovozného cla a tento stav trvá k 1. máju 2004, platia doterajšie predpisy až do času, kým sa pre tovar dočasné uskladnenie alebo colne schválené určenie neskončí. (16) Pri tovare, ktorý bol do 30. apríla 2004 vrátane prepustený do spoločného tranzitného režimu34) alebo iného colného režimu tranzit a tento režim trvá k 1. máju 2004, platia doterajšie predpisy až do času, kým sa tento colný režim neskončí.
relationships:
  Docasne Uskladnenie Alebo Colne Schvalene Urcenie -> [ZAHRNUJE] -> Docasne Uskladnenie
  Docasne Uskladnenie Alebo Colne Schvalene Urcenie -> [ZAHRNUJE] -> Colne Schvalene Urcenie
  Docasne Uskladnenie Alebo Colne Schvalene Urcenie -> [VZTAHUJE_SA_NA] -> Tovar

  Uskladnovanie V Colnom Sklade -> [JE_DRUHOM] -> Colny Rezim
  Aktivny Zuslachcovaci Styk -> [JE_DRUHOM] -> Colny Rezim
  Docasne Pouzitie S Uplnym Oslobodenim Od Dovozneho Cla -> [JE_DRUHOM] -> Colny Rezim
  Spolocny Tranzitny Rezim -> [JE_DRUHOM] -> Colny Rezim
  Iny Colny Rezim Tranzit -> [JE_DRUHOM] -> Colny Rezim

  Colne Schvalene Urcenie -> [ZAHRNUJE] -> Uskladnovanie V Colnom Sklade
  Colne Schvalene Urcenie -> [ZAHRNUJE] -> Aktivny Zuslachcovaci Styk
  Colne Schvalene Urcenie -> [ZAHRNUJE] -> Docasne Pouzitie S Uplnym Oslobodenim Od Dovozneho Cla
  Colne Schvalene Urcenie -> [ZAHRNUJE] -> Spolocny Tranzitny Rezim
  Colne Schvalene Urcenie -> [ZAHRNUJE] -> Iny Colny Rezim Tranzit

  Doterajsie Predpisy -> [PLATI_DO] -> Skoncenie Docasneho Uskladnenia
  Doterajsie Predpisy -> [PLATI_DO] -> Skoncenie Colne Schvaleneho Urcenia
  Doterajsie Predpisy -> [PLATI_DO] -> Skoncenie Spolocneho Tranzitneho Rezimu
  Doterajsie Predpisy -> [PLATI_DO] -> Skoncenie Ineho Colneho Rezimu Tranzit

nodes:
  Tovar: Tovar
  Datum: 30. april 2004
  Datum: 1. maj 2004
  Urad: Colny Urad
  PravnyPredpis: Doterajsie Predpisy
  Cinnost: Docasne Uskladnenie Alebo Colne Schvalene Urcenie
  Cinnost: Docasne Uskladnenie
  Cinnost: Colne Schvalene Urcenie
  Konanie: Colny Rezim
  Cinnost: Uskladnovanie V Colnom Sklade
  Cinnost: Aktivny Zuslachcovaci Styk
  Cinnost: Docasne Pouzitie S Uplnym Oslobodenim Od Dovozneho Cla
  Cinnost: Spolocny Tranzitny Rezim
  Cinnost: Iny Colny Rezim Tranzit
  CasovyUdaj: Skoncenie Docasneho Uskladnenia
  CasovyUdaj: Skoncenie Colne Schvaleneho Urcenia
  CasovyUdaj: Skoncenie Spolocneho Tranzitneho Rezimu
  CasovyUdaj: Skoncenie Ineho Colneho Rezimu Tranzit
  ClenskyStat: Clensky Stat K 1. Maju 2004


chunk: 56
page: 13
text: c) zdaniteľná osoba, ktorej má byť tovar dodaný, je identifikovaná pre daň v členskom štáte, do ktorého je tovar odoslaný alebo prepravený, a platiteľ podľa písmena a) pozná v čase začatia odoslania alebo prepravy tovaru jej obchodné meno a identifikačné číslo pre daň pridelené týmto členským štátom, d) platiteľ uviedol premiestnenie tovaru v záznamoch podľa § 70 ods. 2 písm. g), e) platiteľ uviedol v súhrnnom výkaze podľa § 80 ods. 1 písm. e) identifikačné číslo pre daň pridelené zdaniteľnej osobe, ktorá nadobudne tovar, členským štátom, do ktorého je tovar odoslaný alebo prepravený. (2) Ak sú splnené podmienky podľa odseku 1 a prevod práva nakladať s tovarom ako vlastník sa uskutoční v lehote podľa odseku 3, v čase prevodu práva nakladať s tovarom ako vlastník na zdaniteľnú osobu podľa odseku 1 písm. c) alebo odseku 5 platí, že dodanie tovaru oslobodené od dane podľa § 43 ods. 1 sa považuje za uskutočnené platiteľom, ktorý tovar odoslal alebo prepravil
relationships:
  Zdanitelna Osoba Ktorej Ma Byt Tovar Dodany -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [PRIDELUJE] -> Clensky Stat Do Ktoreho Je Tovar Odoslany Alebo Prepraveny
  Zdanitelna Osoba Ktorej Ma Byt Tovar Dodany -> [MA_STATUS] -> Identifikovana Pre Dan V Clenskom State
  Clensky Stat Do Ktoreho Je Tovar Odoslany Alebo Prepraveny -> [TYKA_SA] -> Odoslanie Alebo Preprava Tovaru

  Platitel Podla Pismena A -> [MA_NAZOV] -> Obchodne Meno Zdanitelnej Osoby
  Platitel Podla Pismena A -> [MA_IDENTIFIKACNE_CISLO] -> Identifikacne Cislo Pre Dan Zdanitelnej Osoby
  Obchodne Meno Zdanitelnej Osoby -> [MA_OBDOBIE] -> Cas Zacatia Odoslania Alebo Prepravy Tovaru
  Identifikacne Cislo Pre Dan Zdanitelnej Osoby -> [MA_OBDOBIE] -> Cas Zacatia Odoslania Alebo Prepravy Tovaru

  Platitel -> [UVADZA] -> Premiestnenie Tovaru
  Premiestnenie Tovaru -> [NACHADZA_SA_V] -> Zaznamy
  Zaznamy -> [JE_PODLA] -> Paragraf § 70 Odsek 2 Pismeno g)

  Platitel -> [UVADZA] -> Identifikacne Cislo Pre Dan
  Identifikacne Cislo Pre Dan -> [NACHADZA_SA_V] -> Suhrnny Vykaz
  Suhrnny Vykaz -> [JE_PODLA] -> Paragraf § 80 Odsek 1 Pismeno e)
  Identifikacne Cislo Pre Dan -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba Ktora Nadobudne Tovar

  Podmienky -> [SPLNA_PODMIENKY] -> Odsek 1
  Prevod Prava Nakladat S Tovarom Ako Vlastnik -> [MA_LEHOTU] -> Lehota
  Lehota -> [JE_PODLA] -> Odsek 3
  Prevod Prava Nakladat S Tovarom Ako Vlastnik -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Zdanitelna Osoba -> [JE_PODLA] -> Odsek 1 Pismeno c)
  Zdanitelna Osoba -> [JE_PODLA] -> Odsek 5

  Dodanie Tovaru Oslobodene Od Dane -> [JE_PODLA] -> Paragraf § 43 Odsek 1
  Dodanie Tovaru Oslobodene Od Dane -> [POVAZUJE_SA_ZA] -> Dodanie Tovaru Uskutocnene Platitelom Ktory Tovar Odoslal Alebo Prepravil
  Platitel Ktory Tovar Odoslal Alebo Prepravil -> [DODAVA] -> Tovar

nodes:
  ZdanitelnaOsoba: Zdanitelna Osoba Ktorej Ma Byt Tovar Dodany
  ZdanitelnaOsoba: Zdanitelna Osoba Ktora Nadobudne Tovar
  ZdanitelnaOsoba: Zdanitelna Osoba
  ZdanitelnaOsoba: Platitel
  ZdanitelnaOsoba: Platitel Podla Pismena A
  ZdanitelnaOsoba: Platitel Ktory Tovar Odoslal Alebo Prepravil
  Tovar: Tovar
  Cinnost: Odoslanie Alebo Preprava Tovaru
  Cinnost: Premiestnenie Tovaru
  Cinnost: Prevod Prava Nakladat S Tovarom Ako Vlastnik
  Pravo: Pravo Nakladat S Tovarom Ako Vlastnik
  Cinnost: Dodanie Tovaru Uskutocnene Platitelom Ktory Tovar Odoslal Alebo Prepravil
  OslobodenieOdDane: Dodanie Tovaru Oslobodene Od Dane
  IdentifikacneCislo: Identifikacne Cislo Pre Dan
  IdentifikacneCislo: Identifikacne Cislo Pre Dan Zdanitelnej Osoby
  ClenskyStat: Clensky Stat Do Ktoreho Je Tovar Odoslany Alebo Prepraveny
  Status: Identifikovana Pre Dan V Clenskom State
  Hodnota: Obchodne Meno Zdanitelnej Osoby
  CasovyUdaj: Cas Zacatia Odoslania Alebo Prepravy Tovaru
  Zaznam: Zaznamy
  Zaznam: Suhrnny Vykaz
  Podmienka: Podmienky
  Lehota: Lehota
  Odsek: Odsek 1
  Odsek: Odsek 3
  Odsek: Odsek 5
  Pismeno: Odsek 1 Pismeno c)
  Pismeno: Pismeno d)
  Pismeno: Pismeno e)
  Pismeno: Pismeno a)
  Pismeno: Paragraf § 70 Odsek 2 Pismeno g)
  Pismeno: Paragraf § 80 Odsek 1 Pismeno e)
  Odsek: Paragraf § 43 Odsek 1


chunk: 578
page: 142
text: do 31. decembra 2024, na postup Daňového úradu Bratislava sa vzťahuje § 5 ods. 2 v znení účinnom do 31. decembra 2024. (4) Ustanovenie § 8 ods. 1 písm. c) v znení účinnom od 1. januára 2025 sa nepoužije na nájomnú zmluvu s dojednaným právom kúpy tovaru, ktorý je predmetom nájmu, ak bola táto zmluva uzavretá do 31. decembra 2024 vrátane. (5) Ustanovenia § 55 ods. 3, § 69 ods. 13 a § 78 ods. 9 v znení účinnom do 31. decembra 2024 sa vzťahujú na zdaniteľnú osobu, ktorej vznikla povinnosť podať daňové priznanie podľa § 78 ods. 9 v znení účinnom do 31. decembra 2024.
relationships:
  Paragraf § 5 -> [OBSAHUJE] -> Paragraf § 5 Odsek 2
  Postup Danoveho Uradu Bratislava -> [JE_PODLA] -> Paragraf § 5 Odsek 2 V Zneni Ucinnom Do 31 Decembra 2024
  Paragraf § 5 Odsek 2 V Zneni Ucinnom Do 31 Decembra 2024 -> [PLATI_DO] -> 31 Decembra 2024

  Paragraf § 8 -> [OBSAHUJE] -> Paragraf § 8 Odsek 1
  Paragraf § 8 Odsek 1 -> [OBSAHUJE] -> Paragraf § 8 Odsek 1 Pismeno c)
  Paragraf § 8 Odsek 1 Pismeno c) V Zneni Ucinnom Od 1 Januara 2025 -> [PLATI_OD] -> 1 Januara 2025
  Paragraf § 8 Odsek 1 Pismeno c) V Zneni Ucinnom Od 1 Januara 2025 -> [NEVZTAHUJE_SA_NA] -> Najomna Zmluva S Dojednanym Pravom Kupy Tovaru
  Najomna Zmluva S Dojednanym Pravom Kupy Tovaru -> [MA_PRAVO] -> Pravo Kupy Tovaru
  Tovar -> [JE_PREDMETOM] -> Najom
  Najomna Zmluva S Dojednanym Pravom Kupy Tovaru -> [MA_DATUM] -> Uzavretie Zmluvy Do 31 Decembra 2024 Vratane

  Paragraf § 55 -> [OBSAHUJE] -> Paragraf § 55 Odsek 3
  Paragraf § 69 -> [OBSAHUJE] -> Paragraf § 69 Odsek 13
  Paragraf § 78 -> [OBSAHUJE] -> Paragraf § 78 Odsek 9

  Paragraf § 55 Odsek 3 V Zneni Ucinnom Do 31 Decembra 2024 -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Paragraf § 69 Odsek 13 V Zneni Ucinnom Do 31 Decembra 2024 -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Paragraf § 78 Odsek 9 V Zneni Ucinnom Do 31 Decembra 2024 -> [VZTAHUJE_SA_NA] -> Zdanitelna Osoba
  Zdanitelna Osoba -> [MA_POVINNOST] -> Podanie Danoveho Priznania
  Podanie Danoveho Priznania -> [JE_PODLA] -> Paragraf § 78 Odsek 9 V Zneni Ucinnom Do 31 Decembra 2024

nodes:
  Urad: Danovy Urad Bratislava
  Cinnost: Postup Danoveho Uradu Bratislava
  Paragraf: Paragraf § 5
  Odsek: Paragraf § 5 Odsek 2
  Odsek: Paragraf § 5 Odsek 2 V Zneni Ucinnom Do 31 Decembra 2024
  Datum: 31 Decembra 2024
  Datum: 1 Januara 2025

  Paragraf: Paragraf § 8
  Odsek: Paragraf § 8 Odsek 1
  Pismeno: Paragraf § 8 Odsek 1 Pismeno c)
  Pismeno: Paragraf § 8 Odsek 1 Pismeno c) V Zneni Ucinnom Od 1 Januara 2025
  Zmluva: Najomna Zmluva S Dojednanym Pravom Kupy Tovaru
  Pravo: Pravo Kupy Tovaru
  Tovar: Tovar
  Cinnost: Najom
  Datum: Uzavretie Zmluvy Do 31 Decembra 2024 Vratane

  Paragraf: Paragraf § 55
  Odsek: Paragraf § 55 Odsek 3
  Odsek: Paragraf § 55 Odsek 3 V Zneni Ucinnom Do 31 Decembra 2024
  Paragraf: Paragraf § 69
  Odsek: Paragraf § 69 Odsek 13
  Odsek: Paragraf § 69 Odsek 13 V Zneni Ucinnom Do 31 Decembra 2024
  Paragraf: Paragraf § 78
  Odsek: Paragraf § 78 Odsek 9
  Odsek: Paragraf § 78 Odsek 9 V Zneni Ucinnom Do 31 Decembra 2024
  ZdanitelnaOsoba: Zdanitelna Osoba
  Povinnost: Podanie Danoveho Priznania
  DanovePriznanie: Danove Priznanie


chunk: 585
page: 144
text: podnikanie sa nepovažuje za dodanie služby za protihodnotu (§ 9 ods. 3). Ustanovenie § 49 ods. 4 týmto nie je dotknuté. (3) Platiteľ, ktorý v súvislosti s osobným motorovým vozidlom používaným na účely svojho podnikania, ako aj na iný účel ako na podnikanie, od 1. januára 2026 do 30. júna 2028 vrátane prijme služby alebo nadobudne tovar, ktorý nie je investičným majetkom podľa § 54 ods. 2 písm. a), odpočíta daň vzťahujúcu sa na tieto služby alebo tovar v rozsahu 50 %; ustanovenie § 49 ods. 4 týmto nie je dotknuté. (4) Odseky 1 až 3 sa, bez toho, aby boli dotknuté ustanovenia § 49 ods. 4 a § 54, neuplatnia na osobné motorové vozidlo, ktoré platiteľ a) nadobudol alebo používa výlučne na podnikanie, ktorým je 1. krátkodobý nájom alebo iný ako krátkodobý nájom osobného motorového vozidla, 2. doprava osôb a ich batožiny za protihodnotu vrátane taxislužby, 3. prevádzkovanie autoškoly, ak osobné motorové vozidlo je výcvikovým vozidlom,41)
relationships:
  Pouzitie Na Iny Ucel Ako Podnikanie -> [NEVZTAHUJE_SA_NA] -> Dodanie Sluzby Za Protihodnotu
  Dodanie Sluzby Za Protihodnotu -> [JE_PODLA] -> Paragraf § 9 Odsek 3
  Paragraf § 49 Odsek 4 -> [NEVZTAHUJE_SA_NA] -> Dotknutie Tymto Ustanovenim

  Platitel -> [MA_PRAVO] -> Odpocitanie Dane V Rozsahu 50 Percent
  Odpocitanie Dane V Rozsahu 50 Percent -> [MA_OBDOBIE] -> Od 1 Januara 2026 Do 30 Juna 2028 Vratane
  Odpocitanie Dane V Rozsahu 50 Percent -> [VZTAHUJE_SA_NA] -> Dan Vztahujuca Sa Na Sluzby Alebo Tovar
  Odpocitanie Dane V Rozsahu 50 Percent -> [MA_HODNOTU] -> 50 Percent

  Platitel -> [PRIJIMA] -> Sluzby
  Platitel -> [NADOBUDA] -> Tovar Ktory Nie Je Investicnym Majetkom
  Sluzby -> [VZTAHUJE_SA_NA] -> Osobne Motorove Vozidlo Pouzivane Na Podnikanie Aj Na Iny Ucel Ako Podnikanie
  Tovar Ktory Nie Je Investicnym Majetkom -> [VZTAHUJE_SA_NA] -> Osobne Motorove Vozidlo Pouzivane Na Podnikanie Aj Na Iny Ucel Ako Podnikanie
  Tovar Ktory Nie Je Investicnym Majetkom -> [NEVZTAHUJE_SA_NA] -> Investicny Majetok 
  Investicny Majetok -> [JE_PODLA] -> Paragraf § 54 Odsek 2 Pismeno a)
  Paragraf § 49 Odsek 4 -> [NEVZTAHUJE_SA_NA] -> Dotknutie Odsekom 3

  Odsek 1 -> [NEVZTAHUJE_SA_NA] -> Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie
  Odsek 2 -> [NEVZTAHUJE_SA_NA] -> Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie
  Odsek 3 -> [NEVZTAHUJE_SA_NA] -> Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie
  Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie -> [MA_UCEL] -> Podnikanie
  Platitel -> [NADOBUDA] -> Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie
  Platitel -> [VYKONAVA] -> Pouzivanie Osobneho Motoroveho Vozidla Vylucne Na Podnikanie

  Podnikanie -> [ZAHRNUJE] -> Kratkodoby Najom Osobneho Motoroveho Vozidla
  Podnikanie -> [ZAHRNUJE] -> Iny Ako Kratkodoby Najom Osobneho Motoroveho Vozidla
  Podnikanie -> [ZAHRNUJE] -> Doprava Osob A Ich Batoziny Za Protihodnotu Vratane Taxisluzby
  Podnikanie -> [ZAHRNUJE] -> Prevadzkovanie Autoskoly
  Prevadzkovanie Autoskoly -> [MA_PODMIENKU] -> Osobne Motorove Vozidlo Je Vycvikove Vozidlo
  Osobne Motorove Vozidlo -> [JE_TYPOM] -> Vycvikove Vozidlo

nodes:
  Cinnost: Pouzitie Na Iny Ucel Ako Podnikanie
  Cinnost: Dodanie Sluzby Za Protihodnotu
  Odsek: Paragraf § 9 Odsek 3
  Odsek: Paragraf § 49 Odsek 4
  Hodnota: Dotknutie Tymto Ustanovenim
  Hodnota: Dotknutie Odsekom 3

  ZdanitelnaOsoba: Platitel
  Pravo: Odpocitanie Dane V Rozsahu 50 Percent
  Obdobie: Od 1 Januara 2026 Do 30 Juna 2028 Vratane
  Hodnota: 50 Percent
  Dan: Dan Vztahujuca Sa Na Sluzby Alebo Tovar
  Sluzba: Sluzby
  Tovar: Tovar Ktory Nie Je Investicnym Majetkom
  InvesticnyMajetok: Investicny Majetok
  Pismeno: Paragraf § 54 Odsek 2 Pismeno a)
  Vozidlo: Osobne Motorove Vozidlo
  Vozidlo: Osobne Motorove Vozidlo Pouzivane Na Podnikanie Aj Na Iny Ucel Ako Podnikanie

  Odsek: Odsek 1
  Odsek: Odsek 2
  Odsek: Odsek 3
  Vozidlo: Osobne Motorove Vozidlo Nadobudnute Alebo Pouzivane Vylucne Na Podnikanie
  Cinnost: Podnikanie
  Cinnost: Pouzivanie Osobneho Motoroveho Vozidla Vylucne Na Podnikanie
  Cinnost: Kratkodoby Najom Osobneho Motoroveho Vozidla
  Cinnost: Iny Ako Kratkodoby Najom Osobneho Motoroveho Vozidla
  Cinnost: Doprava Osob A Ich Batoziny Za Protihodnotu Vratane Taxisluzby
  Cinnost: Prevadzkovanie Autoskoly
  Podmienka: Osobne Motorove Vozidlo Je Vycvikove Vozidlo
  Vozidlo: Vycvikove Vozidlo


chunk: 92
page: 22
text: (14) Miestom dodania telekomunikačných služieb, služieb rozhlasového vysielania a televízneho vysielania a elektronických služieb dodaných osobe inej ako zdaniteľnej osobe je miesto, kde má táto osoba sídlo, bydlisko alebo miesto, kde sa obvykle zdržiava, ak § 16a ods. 1 neustanovuje inak. (15) Miestom dodania služieb uvedených v odseku 16 vrátane prijatia záväzku zdržať sa zámeru ich vykonávania alebo zdržať sa ich vykonávania úplne alebo čiastočne, ak sú tieto služby dodané osobe inej ako zdaniteľnej osobe, ktorá má sídlo, bydlisko alebo sa obvykle zdržiava mimo územia Európskej únie, je miesto, kde má táto osoba sídlo, bydlisko alebo miesto, kde sa obvykle zdržiava. (16) Služby, pri ktorých sa určí miesto dodania podľa odseku 15, sú tieto: a) prevod a postúpenie autorských práv, patentov, licencií, ochranných známok a podobných práv, b) reklamné služby, c) poradenské, inžinierske, technické, právne, účtovné, audítorské, prekladateľské, tlmočnícke
relationships:
  Odsek 14 -> [UPRAVUJE] -> Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb
  Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba
  Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb -> [POVAZUJE_SA_ZA] -> Sidlo Osoby
  Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb -> [POVAZUJE_SA_ZA] -> Bydlisko Osoby
  Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb -> [POVAZUJE_SA_ZA] -> Miesto Kde Sa Osoba Obvykle Zdrziava
  Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb -> [MA_VYNIMKU] -> Paragraf § 16a Odsek 1

  Odsek 15 -> [UPRAVUJE] -> Miesto Dodania Sluzieb Uvedenych V Odseku 16
  Miesto Dodania Sluzieb Uvedenych V Odseku 16 -> [VZTAHUJE_SA_NA] -> Osoba Ina Ako Zdanitelna Osoba Mimo Uzemia Europskej Unie
  Miesto Dodania Sluzieb Uvedenych V Odseku 16 -> [POVAZUJE_SA_ZA] -> Sidlo Osoby Mimo Uzemia Europskej Unie
  Miesto Dodania Sluzieb Uvedenych V Odseku 16 -> [POVAZUJE_SA_ZA] -> Bydlisko Osoby Mimo Uzemia Europskej Unie
  Miesto Dodania Sluzieb Uvedenych V Odseku 16 -> [POVAZUJE_SA_ZA] -> Miesto Kde Sa Osoba Obvykle Zdrziava Mimo Uzemia Europskej Unie
  Osoba Ina Ako Zdanitelna Osoba Mimo Uzemia Europskej Unie -> [NACHADZA_SA_V] -> Uzemie Mimo Europskej Unie
  Sluzby Uvedene V Odseku 16 -> [ZAHRNUJE] -> Prijatie Zavazku Zdrzat Sa Zameru Vykonavania Sluzieb Alebo Zdrzat Sa Ich Vykonavania Uplne Alebo Ciastocne

  Odsek 16 -> [OBSAHUJE] -> Sluzby Uvedene V Odseku 16
  Sluzby Uvedene V Odseku 16 -> [ZAHRNUJE] -> Prevod A Postupenie Autorskych Prav Patentov Licencii Ochrannych Znamok A Podobnych Prav
  Sluzby Uvedene V Odseku 16 -> [ZAHRNUJE] -> Reklamne Sluzby
  Sluzby Uvedene V Odseku 16 -> [ZAHRNUJE] -> Poradenske Inzinierske Technicke Právne Uctovne Auditorske Prekladatelske A Tlmocnicke Sluzby

nodes:
  Odsek: Odsek 14
  Odsek: Odsek 15
  Odsek: Odsek 16
  Odsek: Paragraf § 16a Odsek 1

  Lokacia: Miesto Dodania Telekomunikacnych Sluzieb Sluzieb Rozhlasoveho Vysielania Sluzieb Televizneho Vysielania A Elektronickych Sluzieb
  Lokacia: Miesto Dodania Sluzieb Uvedenych V Odseku 16
  Osoba: Osoba Ina Ako Zdanitelna Osoba
  Osoba: Osoba Ina Ako Zdanitelna Osoba Mimo Uzemia Europskej Unie
  Sidlo: Sidlo Osoby
  Adresa: Bydlisko Osoby
  Lokacia: Miesto Kde Sa Osoba Obvykle Zdrziava
  Sidlo: Sidlo Osoby Mimo Uzemia Europskej Unie
  Adresa: Bydlisko Osoby Mimo Uzemia Europskej Unie
  Lokacia: Miesto Kde Sa Osoba Obvykle Zdrziava Mimo Uzemia Europskej Unie
  Uzemie: Uzemie Mimo Europskej Unie

  Sluzba: Telekomunikacne Sluzby
  Sluzba: Sluzby Rozhlasoveho Vysielania
  Sluzba: Sluzby Televizneho Vysielania
  Sluzba: Elektronicke Sluzby
  Sluzba: Sluzby Uvedene V Odseku 16
  Zavazok: Prijatie Zavazku Zdrzat Sa Zameru Vykonavania Sluzieb Alebo Zdrzat Sa Ich Vykonavania Uplne Alebo Ciastocne
  Sluzba: Prevod A Postupenie Autorskych Prav Patentov Licencii Ochrannych Znamok A Podobnych Prav
  Sluzba: Reklamne Sluzby
  Sluzba: Poradenske Inzinierske Technicke Právne Uctovne Auditorske Prekladatelske A Tlmocnicke Sluzby

