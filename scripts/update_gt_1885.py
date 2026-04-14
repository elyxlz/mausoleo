from __future__ import annotations

import json
import pathlib as pl

GT_PATH = pl.Path("eval/bootstrap_gt/1885-06-15/ground_truth.json")

gt = json.loads(GT_PATH.read_text())

new_articles = gt["articles"][:4]

new_articles.append({
    "headline": "Camera dei deputati",
    "unit_type": "article",
    "page_span": [2],
    "paragraphs": [{"text": "Seduta del 14 giugno\n\nL'intera seduta è stata occupata dalla discussione sulla proroga del corso legale dei biglietti di banca sino a tutto il 1885."}]
})

new_articles.append({
    "headline": "L'accoltellato della Lungara",
    "unit_type": "article",
    "page_span": [2],
    "paragraphs": [{"text": "Passatella sciagurata!\n\nMancavano pochi minuti a mezzanotte allorché una rissa furibonda s'impegnava nell'osteria al numero 47 in via della Lungara.\n\nStavano lì dentro giocando alla passatella in numerosa comitiva alcuni popolani fra i quali il barcarolo Pietro Guidetti, il fornaciaro Alfredo Batazzi, il facchino Serafino Del Signore, il barbiere Angelo Belloni, il bracciante Antonio Soverini e altri, tutti giovani sulla ventina.\n\nIl Guidetti, riuscito padrone della terza partita, fece tener l'olmo al Batazzi, il che nel gergo del giuoco significa che non gli diede da bere.\n\nIl Batazzi non seppe prendersela in santa pace; e più s'inviperì agli scherni del Guidetti, il quale abusò forse un po' troppo della sua posizione di vincitore.\n\nAvvinazzati com'erano, non ci volle più che tanto perché ne nascesse una cagnara d'inferno; cagnara a cui presero parte, in appoggio del Batazzi anche gli altri che ho nominati, ai quali pure ribolliva il sangue per aver subito lo stesso affronto.\n\nIl Batazzi, vistosi sostenuto dagli amici, perdette il lume degli occhi; cacciò il coltello e ne vibrò un colpo al Guidetti, che cadde sanguinante a terra chiamando aiuto.\n\nAl trambusto accorsero diversi militari, ma prima che essi giungessero, il Batazzi se l'era svignata.\n\nRimanevano i suoi compagni, Del Signore, Belloni e Soverini, che vennero tutti e tre arrestati.\n\nIl Guidetti fu portato al vicino ospedale di Santo Spirito: la ferita sembra pericolosa."}]
})

new_articles.append({
    "headline": "L'arresto di tre falsi monetari",
    "unit_type": "article",
    "page_span": [2],
    "paragraphs": [{"text": "La questura era stata avvertita che circolavano in città parecchi biglietti falsi da 50 lire del Banco di Napoli imitati a perfezione, tanto che i più esperti non li avrebbero riconosciuti.\n\nOrganizzato un servizio di spionaggio, si riuscì dopo lunghe ricerche all'arresto di due individui che spacciavano tali biglietti, il commesso di negozio Pietro Carpani e il fruttarolo Fernando Esposito; un terzo si trovava già in carcere per altro titolo, ed è precisamente il fratello del Carpani a nome Ugo.\n\nPerquisito anche costui in carcere, gli si trovarono cuciti nella fodera della giacca otto biglietti falsi da 50 lire.\n\nNon fu possibile sapere da nessuno dei tre chi li avesse dati: nè promesse, nè minacce nulla valse: essi si trincerarono nel più assoluto silenzio limitandosi a dichiarare che i biglietti erano stati loro consegnati da alcuni individui sconosciuti.\n\nLa polizia sospetta invece che gli arrestati siano agenti di qualche fabbrica di biglietti che avrebbe diramazioni a Venezia, a Firenze, a Napoli: in queste città proseguono le indagini, ma fin qui non diedero alcun frutto."}]
})

new_articles.append({
    "headline": "L'ammazzato di ieri",
    "unit_type": "article",
    "page_span": [2],
    "paragraphs": [{"text": "Quale orribile scena!\n\nVi presenterò anzitutto il colpevole, l'orzarolo Celestino Riconi, giovanotto di 24 anni e di carattere più che bollente.\n\nLa vittima poi — fremete! — è una capra mansueta dal folto pelo che formava la delizia di un altro bottegaio di quei dintorni ed era diventata familiare a tutto il vicinato.\n\nIeri dunque questa povera bestia, contando forse un po' troppo sulla confidenza con la quale l'orzarolo Riconi l'aveva trattata altre volte, si avvicinava al suo negozio, avanzava la testa in una canestra di pasta tenuta lì in mostra, e per diletto ne assaggiava qualche boccata.\n\nDisgraziatamente l'orzarolo si trovava in un cattivo quarto di luna.\n\nNon appena egli s'accorse della libertà che s'era presa la capra, le gridò furibondo dal banco: — Passa via!\n\nMa la capra non si mosse: e come meravigliata di quello sgarbo lo fissò con uno sguardo dolcissimo rispondendo nel suo arcano linguaggio: — Beeee....\n\nNon ebbe tempo di finire che il Riconi le si avventò addosso imbrandendo un coltello e glielo immerse tutto quanto nella schiena!\n\nDalla tremenda ferita zampillò istantaneamente una fontana di sangue; la povera bestia stravolti gli occhi, rovesciò a terra, e agitando per qualche istante le zampe fece sentire un ultimo straziante belato...\n\nPoi rimase inerte, stecchita.\n\nEra morta!\n\nAlla scena atrocissima tutti i vicini e così una folla di gente si raccolse davanti alla bottega dell'orzarolo, che inorridito egli medesimo di tanto scempio, e forse pentito troppo tardi d'essersi lasciato trasportare a tale eccesso dal suo furore, era rientrato nella bottega brandendo la lama del coltello rossa e fumante del sangue della sua vittima, e sternutendo.\n\nAccorsa anche la guardia di città, sequestrò l'arma omicida e lo trasse in arresto.\n\nIl cadavere della capra fu trasportato al mattatoio fuori porta del Popolo, dove quei veterinari dovranno procedere all'autopsia a tenor di legge."}]
})

new_articles.append({
    "headline": "L'usciere accoltellato a Tivoli",
    "unit_type": "article",
    "page_span": [2, 3],
    "paragraphs": [{"text": "Sul ferimento dell'usciere Alfonso Cesari avvenuto a Tivoli giovedì mattina, il signor Francesco Splendori genero dell'arrestato Angelo Cognetti, mi scrive:\n\n«Il movente della questione fu una esecuzione che il Cesari doveva fare tanto contro Domenico Cognetti mio suocero quanto contro di me.\n\n«Il Cesari, incominciò dal Cognetti, per la sua avanzata età, inabile a qualunque lavoro e per non posseder nulla di suo non poterlo soddisfare; a comprovare la sua povertà fornì un attestato rilasciatogli dal sindaco di Tivoli; mostratolo all'usciere, questi rispose che il certificato non contava nulla e si volse alla figlia nubile di esso Cognetti, chiedendo di sapere dove tenesse la poca roba del padre; approfittando quindi della mia assenza, fece consegnare da mia moglie Maddalena Cognetti lire 30 in danaro, e la fede nuziale del valore di lire 25.\n\n«Noti che se la condanna fu di lire 3 a testa, in 5 individui l'ammontare era di lire 15, e l'usciere aveva tolto alla mia consorte fra danaro e la fede d'oro la somma di lire 55.\n\n«Non pago di ciò il Cesari a ogni incontro insisteva presso il Cognetti pel pagamento: il vecchio alfine, giustamente irritato, lo colpì con un coltello a serramanico producendogli due lievi ferite.\n\n«Dal Cesari nessuno in Tivoli potrà dir bene; a San Polo de' Cavalieri giorni sono dovette fuggire fra una macchia per salvarsi la vita in seguito a una delle solite sue prepotenze.\n\n«Francesco Splendori.»"}]
})

new_articles.append({
    "headline": "La camorra nei municipi",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "Il governo si ostina a non voler procedere alla riforma comunale e provinciale, a non volere allargare il voto amministrativo.\n\nE intanto i municipi continuano a dare scandaloso spettacolo di sé, e a consiglieri comunali vengono eletti principi e baroni, proprietari e speculatori, affaristi di ogni genere che vanno su per tutelare non gli interessi del pubblico ma i propri, che vanno su per far degli affari.\n\nA Roma si sono visti consiglieri comunali i quali hanno subordinato il loro voto nei progetti edilizi alla considerazione del maggiore o minore vantaggio che ne ricavano le loro aree, i loro stabili; al Municipio di Roma si è allargata una strada piuttosto che un'altra, si è espropriata una zona a preferenza di un'altra onde acquistare la casa del tale o del tal altro consigliere, o parente od amico del medesimo.\n\nA Roma si è tentato di dare in appalto il servizio della nettezza pubblica a un circolo di beneficenza sacra ed affaristica, e solo le grida della stampa hanno impedito che il carrozzone avesse luogo."}]
})

new_articles.append({
    "headline": "Ne succedono delle graziose!",
    "unit_type": "article",
    "page_span": [3, 4],
    "paragraphs": [{"text": "E questa ha anche il merito della novità. È detta in quattro parole.\n\nL'altra sera si presentava alla caserma dei carabinieri a Termini un individuo di apparenza civile dicendo a quel maresciallo: — Vengo a costituirmi perchè mi mettano dentro.\n\n— Pronto a servirla — rispose il maresciallo — ma prima favorisca spiegare perchè dobbiamo metterla dentro.\n\n— Perchè io sono quel Giovanni Lianati, nativo di Gaeta e impiegato postale, che la questura di Napoli ricerca fin dal 22 dello scorso febbraio per la semplicissima ragione che quel giorno sono fuggito di là portandomi i quattrini del cassetto; venni a Roma sperando di far fortuna; invece non sono riuscito che a sciuparmi tutti i quattrini fino all'ultimo centesimo, e adesso, non avendone più, prego di favorirmi il vitto e l'alloggio gratis.\n\n— Ben volontieri! s'imagini!\n\nDetto questo, il bravo maresciallo passò in rivista le tante circolari per la ricerca di tanti che giacevano sul suo tavolo, ma non trovandone alcuna che si riferisse al Lianati, lo fece accompagnare da due carabinieri alla questura centrale a San Marcello.\n\nLa questura rinnovò le stesse ricerche col medesimo risultato: fino a che si venne a sapere, anche per confessione del Lianati, che s'era inventato quella frottola per ottenere il rimpatrio gratuito dopo averlo chiesto come una carità.\n\n— I signori delegati — disse il Lianati — avevano risposto che per ottenere il vitto gratuito bisogna farsi arrestare; e io, trovandomi qui in Roma solo, senza aiuto, sprovvisto di mezzi, ho ricorso a quello stratagemma per ritornare alla mia bella Napoli dove almeno non morirò di fame."}]
})

new_articles.append({
    "headline": "Il calcio del bue",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "Lo noto come un caso raro, poichè abitualmente il bue, animale pazientissimo, non scende al punto di lavorare di zoccolo.\n\nEppure ieri, alle 5 pom., nella tenuta Troili in via Aurelia fuori porta San Pancrazio, il campagnolo Santo Ricciotti d'Alpignano nello spingere avanti a suo modo due bovi attaccati a una barrozza, uno di essi gli diede un tal calcio che lo buttò a terra spezzandogli la gamba sinistra."}]
})

new_articles.append({
    "headline": "L'inginocchiatore di piazza Montanara",
    "unit_type": "article",
    "page_span": [3, 4],
    "paragraphs": [{"text": "Ieri mattina s'avvicinava a un gruppo di contadini che secondo il solito stavano seduti in piazza Montanara, un individuo di apparenza poco promettente, il quale, colto un pretesto qualunque per attaccar discorso, finiva domandando:\n\n— State dunque qua aspettando che vi venga l'occasione di occuparvi?\n\n— Per lo appunto — risposero i contadini.\n\n— Ebbene, se volete me ne incarico.\n\n— Magari!\n\n— Sicuro, solo soltanto ci vorrebbe una piccola senseria, perchè anch'io non ci sto a trattare gratis.\n\n— E noi ve la daremo, che diamine, siamo disposti a darvi anche la caparra, se bisogna.\n\n— Dal momento che ci trattiamo in buona fede proprio non vi faccio disturbo, mi darete qualche cosa...\n\n— Dite quanto e vi sarà dato.\n\n— Ecco... facendo il conto di una lira di senseria a testa, sarebbero dodici lire...\n\n— E sta bene, eccole qua.\n\nCiascuno dei contadini sborsò la sua parte e furono così riunite le 12 lire che vennero pagate all'incognito, il quale con nuovi inchini tentò subito d'allontanarsi.\n\nMa il proverbio «scarpe grosse cervello fino» doveva aver ragione anche stavolta; uno dei contadini era già rimasto impressionato dalla faccia sospetta di quel messere che nessuno aveva mai visto, e le diffidenze crebbero allorchè vide con quanta premura egli cercava di svignarsela.\n\nIncominciarono dei clamori, il sedicente sensale, accortosi che il suo gioco era scoperto, perdette la bussola, intervennero al basso le guardie e lo arrestarono.\n\nPure lui un campagnolo, certo Carmine Boccio; andava per bocciare e fu bocciato."}]
})

new_articles.append({
    "headline": "Un dito per un sigaro",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "Di rado succede che quella macchinetta che si trova presso tutti i tabaccai per spuntare e tagliare i sigari dia luogo a disgrazie.\n\nIeri alle 3 pom. ne avvenne una nella tabaccheria all'angolo di via della Consulta: il tipografo Napoleone Monti d'anni 35, operaio nello stabilimento Bottaccio, introdotto nella macchinetta un toscano da 8 per farlo spuntare, sbadatamente si fece saltar via la prima falange del dito indice della mano sinistra, e quel pezzo di carne con la relativa unghia volò fuori insieme al sigaro!\n\nIl signor Monti dovette andarsene alla Consolazione dove fu trasportato dal suo compagno di lavoro Riccardo Zerbola.\n\nI dottori giudicarono la ferita guaribile in 15 giorni con riserva."}]
})

new_articles.append({
    "headline": "Gara dei ragazzi pel borseggio di un orologio e di un anello",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "Domenica dell'altra settimana, 4 giugno, sei ragazzi uscirono in lieta comitiva fuori porta San Pancrazio.\n\nEssi: lo scolaro Pompilio Malatesta, il calzolaio Virginio Giacchetti, il macellaio Augusto Bresi, il canestraio Enrico Censi, il muratore Gaetano Barogi e il bracciante Alfredo.\n\nInoltratisi alquanto nella campagna, e giunti finalmente alla Marrana, pensarono di fare un bagno.\n\nFatto si spogliarono, e deposti i panni, si tuffarono nell'acqua intrattenendosi a lungo, ripigliando terra più volte, e infine si distesero l'un l'altro sul campo per asciugarsi.\n\nIl Malatesta aveva lasciato insieme ai suoi panni l'orologio d'argento e l'anello d'oro che s'era tolto temendo di perderli nell'acqua; quando fu per vestirsi, vide che l'orologio e l'anello erano spariti.\n\nChiese conto ai compagni, ma tutti dissero di non saperne niente mostrandosi offesi dai suoi sospetti: cerca di qua, cerca di là, il Malatesta dovette infine rassegnarsi a tornare a Roma senza averli trovati.\n\nI genitori, ai quali egli fu costretto a confessare la cosa, non informarono subito la polizia, ma fecero le indagini d'uso presso le case dei pegni, e si riuscì in tal modo a scoprire che l'orologio era stato impegnato nella bottega di via Urbana dove fu sequestrato.\n\nIn seguito a tale scoperta i cinque compagni furono tutti arrestati ieri mentre si trovavano nelle vicinanze di piazza Vittorio Emanuele."}]
})

new_articles.append({
    "headline": "Regalo della santa messa",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "Nella chiesa degli Agostiniani in via Merulana è avvenuta questa mattina una disgrazia.\n\nIl signor Luzzi, vecchio di 74 anni, pensionato, abitante in via dello Statuto n. 12 primo piano, per non smentire le sue devote abitudini, si recò alle 6 ad ascoltare la santa messa.\n\nNell'uscire, accostandosi alla pila dell'acqua santa, inciampò contro una donna e cadde voltando pesantemente a terra ferendosi gravemente al ginocchio.\n\nLa gente che trovavasi nella chiesa accorse e per cura di una guardia municipale fu messo in una vettura e accompagnato alla Consolazione.\n\nPer la strada il povero vecchio non faceva che esclamare fra sospiri e lamenti: — Non si può neppure stare in grazia di Dio senza uscire dalle disgrazie!\n\nI dottori di quell'ospedale gli consigliarono il letto almeno per una ventina di giorni."}]
})

new_articles.append({
    "headline": "Schiacciato da una trave di ferro in via Merulana",
    "unit_type": "article",
    "page_span": [3],
    "paragraphs": [{"text": "L'operaio Luigi Jacobelli d'anni 35, romano, abitante in via Principe Amedeo numero 2, che nel pomeriggio del 7 corrente lavorava nei lavori di costruzione che i frati d'Aracoeli fanno in via Merulana, colpito da una grossa trave di ferro lasciata troppo presto dai compagni che insieme la trasportavano, si spezzò il braccio destro e si ferì gravemente la testa, è morto stanotte alle 11 e mezza alla Consolazione."}]
})

new_articles.append({
    "headline": "I vetturini hanno sempre torto",
    "unit_type": "article",
    "page_span": [3, 4],
    "paragraphs": [{"text": "Alle 7 di ieri mattina il vetturino Luigi Giordano, abitante alla Salita dei Crocifissi n. 30, si trovava col suo legno n. 676 alla stazione dalla parte degli arrivi; gli si presentarono un uomo, una donna e una bambina per andare a San Pietro.\n\n— Laggiù — due passi di strada, come dice il vetturino — rimase di sale ricevendo che volessero soltanto una lira e quaranta; datemi una lira e buonanotte.\n\nIn ogni caso: quelli non gli volevano dare una lira di più.\n\nDopo un lungo battibecco l'individuo che pagava, il quale era il signor Francesco Taddo livornese, impiegato al ministero d'agricoltura e commercio, abitante in via Principe Umberto n. 92 piano primo, chiamò impazientito una guardia di pubblica sicurezza che era lì di piantone, e così il vetturino si vide costretto a seguirla alla vicina caserma di Borgo.\n\nQuando fu là dentro, si sentì dire dal graduato di servizio: — Voi siete un prepotente; il signore ha fatto il patto per una lira e dovete prender quella.\n\nResta inteso che questo famoso patto esisteva soltanto nel cervello del graduato.\n\nA farla corta, dopo aver perduto una quantità di tempo il Giordano se ne tornò indietro a mani vuote, poichè piuttosto che accettare quella miserabile lira egli non volle niente.\n\nNon basta: fu anche dichiarato in contravvenzione.\n\nE per che atto?\n\nOh, che gabbia di prepotenti!"}]
})

gt["articles"] = new_articles

GT_PATH.write_text(json.dumps(gt, indent=2, ensure_ascii=False))

print(f"Updated: {len(new_articles)} articles")
for i, a in enumerate(new_articles):
    print(f"  {i}: {a['headline']}")
