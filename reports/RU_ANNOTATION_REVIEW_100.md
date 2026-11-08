# Contextual review of 100 Russian examples

80 LOC-positive and 20 LOC-negative texts: separate full-file reservoir samples, seed 42. This ratio does not reflect positive prevalence. Codex performed the review, not independent experts; global semantic accuracy was not estimated. Gold is unchanged. Original NER text and entity strings are preserved.

## 001 · row_id=2900813 · doc_id=250643 · context_plausible_LOC

Перед голосованием спикер крымского парламента, большинство в котором имеют сторонники правящей на Украине Партии регионов и Компартии, Владимир Константинов подчеркнул, что правительство Украины приняло решение, отвечающее национальным интересам и «желанию подавляющего большинства крымчан».

**LOC:** ["Украине", "Украины"]

**ORG:** ["Партии регионов", "Компартии"]

**PER:** ["Владимир Константинов"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 002 · row_id=874105 · doc_id=78208 · context_plausible_LOC

В июне этого года билеты в обе стороны из Москвы обойдутся в среднем в 9 538 рублей (минус 12,4 процента).

**LOC:** ["Москвы"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 003 · row_id=5444108 · doc_id=461380 · context_plausible_LOC

Напомним, ранее в Москве уже не раз задерживали грузинских воров в законе.

**LOC:** ["Москве"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 004 · row_id=314255 · doc_id=27005 · metonymy

Президент Ирана Хасан Рухани в свою очередь заявил, что Тегеран не откажется от Совместного всеобъемлющего плана действий (СВПД) и продолжит соблюдать свои обязательства.

**LOC:** ["Ирана", "Тегеран"]

**ORG:** []

**PER:** ["Хасан Рухани"]

**Review:** Tehran is used as a state actor. This common news metonymy needs a convention consistent with the target labels.

## 005 · row_id=2145221 · doc_id=188161 · possible_type_error

Еврокомиссия пытается также консолидировать бюджет Греции и повысить занятость ее населения.

**LOC:** ["Еврокомиссия", "Греции"]

**ORG:** []

**PER:** []

**Review:** The European Commission is an institution here but is labeled LOC. This is a candidate ORG/LOC type error; gold is unchanged.

## 006 · row_id=7689440 · doc_id=695394 · metonymy

Напомним, в начале февраля Ватикан заявил об учреждении католических епархий в Москве, Саратове, Новосибирске и Иркутске.

**LOC:** ["Ватикан", "Москве", "Саратове", "Новосибирске", "Иркутске"]

**ORG:** []

**PER:** []

**Review:** Vatican can denote the city-state or its governing institution. A single context does not define a universal rule.

## 007 · row_id=5699106 · doc_id=487181 · context_plausible_LOC

Первый сингл диска - одноименная "Chinese Democracy" - стала доступна для скачивания через сеть интернет-радио США уже 22 октября.

**LOC:** ["США"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 008 · row_id=1846796 · doc_id=164668 · context_plausible_LOC

Мэр Нефтеюганска был убит 26 июня 1998.

**LOC:** ["Нефтеюганска"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 009 · row_id=4435403 · doc_id=373573 · context_plausible_LOC

Он является единственным аэропортом за Уралом с двумя взлетно-посадочными полосами.

**LOC:** ["Уралом"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 010 · row_id=7666962 · doc_id=692692 · context_plausible_LOC

Накануне министр обороны России Сергей Иванов сообщил, что обострение ситуации в Абхазии и возможное появление в Грузии американского спецназа могут повлиять на ход переговоров о сроках ликвидации баз.

**LOC:** ["России", "Абхазии", "Грузии"]

**ORG:** []

**PER:** ["Сергей Иванов"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 011 · row_id=3971840 · doc_id=334437 · text_fusion

Объем инвестиций в проект составил более 500 миллионов долларов.Объект является крупнейшим торгово-развлекательным центром Петербурга.

**LOC:** ["Петербурга"]

**ORG:** []

**PER:** []

**Review:** Two sentences are fused without a space after the period; the Petersburg LOC looks plausible.

## 012 · row_id=6552885 · doc_id=580900 · context_plausible_LOC

Президент РФ поручил соответствующим службам оказать всю возможную помощь пострадавшим во время катастрофы.

**LOC:** ["РФ"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 013 · row_id=4235606 · doc_id=356547 · context_plausible_LOC

Соответствующее послание мэр, как передает издание Salburger Nachrichten, направил мэру Вильнюса Артурасу Зуокасу.

**LOC:** ["Вильнюса"]

**ORG:** ["Salburger Nachrichten"]

**PER:** ["Артурасу Зуокасу"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 014 · row_id=5485444 · doc_id=465172 · text_fusion

Однако проведённая прокуратурой проверка показала, что все 19 самовольно возведённых построек (коттеджи и вспомогательные постройки) находятся на своих прежних местах.По результатам проверки в отношении Филатова возбуждено уголовное дело по части 2 статьи 292 УК РФ (служебный подлог, повлекший существенное нарушение прав и законных интересов граждан или организаций либо охраняемых законом интересов общества или государства).

**LOC:** ["РФ"]

**ORG:** []

**PER:** ["Филатова"]

**Review:** A sentence boundary lacks a space. Offsets are valid, but the row is not necessarily one clean sentence.

## 015 · row_id=4366213 · doc_id=367753 · context_plausible_LOC

Правительство Буркина-Фасо во главе с премьер-министром было отправлено в отставку после того, как в четверг, 14 апреля, в столице страны Уагадугу начались беспорядки с участием военных.

**LOC:** ["Буркина-Фасо", "Уагадугу"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 016 · row_id=7689993 · doc_id=695467 · context_plausible_LOC

Подсудимый был признан виновным в организации нескольких атак террористов-самоубийц, совершенных в 1997 году в Иерусалиме.

**LOC:** ["Иерусалиме"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 017 · row_id=1279827 · doc_id=115833 · context_plausible_LOC

29 сентября представитель верховного комиссара ООН по делам беженцев в Ираке Бруно Гедде заявил, что операция по освобождению Мосула может привести к гуманитарной катастрофе.

**LOC:** ["Ираке", "Мосула"]

**ORG:** ["ООН"]

**PER:** ["Бруно Гедде"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 018 · row_id=1055977 · doc_id=95171 · context_plausible_LOC

Хакеры выложили в сеть личную переписку бывшего футболиста сборной Англии Дэвида Бекхэма.

**LOC:** ["Англии"]

**ORG:** []

**PER:** ["Дэвида Бекхэма"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 019 · row_id=6696778 · doc_id=596114 · context_plausible_LOC

В частности, Америка и Россия будут поставлять для АЭС стран-клиентов обогащенный уран и забирать обратно отработанное топливо, для утилизации его на своей территории.

**LOC:** ["Америка", "Россия"]

**ORG:** ["АЭС"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 020 · row_id=2619916 · doc_id=228269 · context_plausible_LOC

Организаторы фестиваля Augustibluus («Августовский блюз») в Эстонии отменили запланированное выступление Стивена Сигала.

**LOC:** ["Эстонии"]

**ORG:** []

**PER:** ["Стивена Сигала"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 021 · row_id=1039962 · doc_id=93683 · context_plausible_LOC

По мнению авторов, доступная цена спиртного приводит к алкоголизации России.

**LOC:** ["России"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 022 · row_id=305443 · doc_id=26230 · context_plausible_LOC

Самая большая концентрация жилых комплексов с дешевыми квартирами в начале второго квартала 2018 года отмечается в ЮВАО, где реализуется пять новостроек с минимальной стоимостью квартир от 3,37 до 3,85 миллиона рублей.

**LOC:** ["ЮВАО"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 023 · row_id=2459972 · doc_id=214103 · context_plausible_LOC

Бывшему премьер-министру Казахстана Серику Ахметову предъявили обвинение в коррупции.

**LOC:** ["Казахстана"]

**ORG:** []

**PER:** ["Серику Ахметову"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 024 · row_id=3721110 · doc_id=314455 · context_plausible_LOC

На данный момент статья 23 устава МО предусматривает, что границами области "признаются границы, существующие на момент вступления в силу Устава Московской области", который был принят в 1996 году.

**LOC:** ["Московской области"]

**ORG:** ["МО"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 025 · row_id=5842529 · doc_id=502551 · context_plausible_LOC

В среду, 9 июля, на Кунцевском кладбище в Москве состоялись похороны актрисы Нонны Мордюковой.

**LOC:** ["Кунцевском кладбище", "Москве"]

**ORG:** []

**PER:** ["Нонны Мордюковой"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 026 · row_id=2563042 · doc_id=223256 · context_plausible_LOC

Представители Русской православной церкви осудили серию плакатов «Иконы дизайна», которые появились в барах и ресторанах Екатеринбурга.

**LOC:** ["Екатеринбурга"]

**ORG:** ["Русской православной церкви"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 027 · row_id=2577719 · doc_id=224546 · context_plausible_LOC

Небо над островом Хайнань неоднократно становилось ареной международных конфликтов.

**LOC:** ["Хайнань"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 028 · row_id=7011590 · doc_id=625545 · union_policy

В свою очередь, Евросоюз на тех же условиях предлагает Ирану гуманитарную помощь и возможность экспортировать иранские продукты на западные рынки.

**LOC:** ["Евросоюз", "Ирану"]

**ORG:** []

**PER:** []

**Review:** The European Union is labeled LOC. This may be the convention for country groupings and should be aligned with Ukrainian annotations.

## 029 · row_id=4892079 · doc_id=413103 · context_plausible_LOC

Новая станция будет работать на Москву и область на частоте 98,00 FM, на которой в настоящее время вещает радио "98 хитов".

**LOC:** ["Москву"]

**ORG:** ["98 хитов"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 030 · row_id=2706308 · doc_id=235808 · context_plausible_LOC

Следующая встреча пройдет 28 апреля в Праге.

**LOC:** ["Праге"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 031 · row_id=3271704 · doc_id=279351 · context_plausible_LOC

Иностранные туристы, посещающие КНДР, больше не могут выходить в интернет через сеть 3G местного оператора Koryolink, сообщает блог North Korea Tech со ссылкой на туристическое агентство Koryo Tours.

**LOC:** ["КНДР"]

**ORG:** ["Koryolink", "North Korea Tech", "Koryo Tours"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 032 · row_id=4431961 · doc_id=373277 · context_plausible_LOC

Сигнал о возгорании поступил в Центр управления в кризисных ситуациях Подмосковья в 03:40.

**LOC:** ["Подмосковья"]

**ORG:** ["Центр управления в кризисных ситуациях"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 033 · row_id=5275246 · doc_id=446123 · institution_geography_policy

Наконец, третья группа исследователей из Университета Миссури, статья которых появилась в журнале Human Nature, пыталась разобраться в вопросах, связанных с движущей силой роста человеческого мозга.

**LOC:** ["Миссури"]

**ORG:** ["Университета", "Human Nature"]

**PER:** []

**Review:** Missouri is labeled LOC, while the university head is separately ORG. Geographic components are not absorbed into one full institution span here.

## 034 · row_id=1542228 · doc_id=139471 · context_plausible_LOC

В Москве с ноября 2012 года вводится платная парковка.

**LOC:** ["Москве"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 035 · row_id=214203 · doc_id=18213 · context_plausible_LOC

— Я прошу США включить здравый смысл и уважать правила и своих союзников».

**LOC:** ["США"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 036 · row_id=4947684 · doc_id=417830 · context_plausible_LOC

По оценкам компании, такой способ доступа в США используют почти 6 миллионов человек.

**LOC:** ["США"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 037 · row_id=4211863 · doc_id=354567 · union_policy

Инвесторы вкладываются в японскую валюту в качестве альтернативы доллару и евро, которые в 2011 году снижаются из-за долговых кризисов в ЕС и США.

**LOC:** ["ЕС", "США"]

**ORG:** []

**PER:** []

**Review:** The EU is labeled LOC alongside the USA. Political-union conventions should be consistent across both corpora.

## 038 · row_id=7436611 · doc_id=667419 · hyphen_name

Алексею Ягудину в Нью-Йорке в пятницу была сделана артроскопическая операция на бедре.

**LOC:** ["Нью-Йорке"]

**ORG:** []

**PER:** ["Алексею Ягудину"]

**Review:** New York is preserved as one hyphenated LOC, contrasting with mixed whole/split treatment in the Ukrainian corpus.

## 039 · row_id=5136911 · doc_id=434158 · text_fusion

По его мнению, власти страны зачастую смешивают понятия религиозных и экстремистских организаций.Напомним, что в начале 2009 года, представляя концепцию национальной безопасности страны, президент Курманбек Бакиев в числе угроз Киргизии назвал деятельность экстремистских организаций, а также "растущую религиозность общества".

**LOC:** ["Киргизии"]

**ORG:** []

**PER:** ["Курманбек Бакиев"]

**Review:** A period joins two sentences without a space; the Kyrgyzstan LOC appears consistent with context.

## 040 · row_id=4851483 · doc_id=409709 · context_plausible_LOC

Однако на Евро-2008 сборная Франции не смогла выйти из группы.

**LOC:** ["Франции"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 041 · row_id=3434433 · doc_id=291930 · context_plausible_LOC

Первый штрихкод Вудленд нарисовал на песке на пляже во Флориде.

**LOC:** ["Флориде"]

**ORG:** []

**PER:** ["Вудленд"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 042 · row_id=6220937 · doc_id=544038 · context_plausible_LOC

На западном побережье Японии вечером 16 июля произошло третье за сутки землетрясение, сила которого на этот раз составила 6,6 балла по шкале Рихтера, передает агентство AFP.

**LOC:** ["Японии"]

**ORG:** ["AFP"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 043 · row_id=4951829 · doc_id=418167 · context_plausible_LOC

Позже Забелин уехал в Эстонию, где получил политическое убежище.

**LOC:** ["Эстонию"]

**ORG:** []

**PER:** ["Забелин"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 044 · row_id=2968132 · doc_id=255809 · text_fusion_and_aux

Об этом, как передаетРИА Новости, он заявил, открывая Совет регионов в Борисполе Киевской области.

**LOC:** ["Борисполе", "Киевской области"]

**ORG:** ["Совет регионов"]

**PER:** []

**Review:** The news-agency name is fused with the preceding verb and is missing from ORG. The city and region LOCs look plausible.

## 045 · row_id=2980673 · doc_id=256762 · context_plausible_LOC

В 2013 году Эрику Булатову исполнилось 80 лет, к его юбилею была приурочена выставка в Новом Национальном музее Монако.

**LOC:** ["Монако"]

**ORG:** ["Новом Национальном музее"]

**PER:** ["Эрику Булатову"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 046 · row_id=6042366 · doc_id=524143 · institution_geography_policy

Члены джамаата также намеревались убить председателя Духовного управления мусульман Санкт-Петербурга, муфтия Жафяра Пончаева.

**LOC:** ["Санкт-Петербурга"]

**ORG:** ["Духовного управления мусульман"]

**PER:** ["Жафяра Пончаева"]

**Review:** Saint Petersburg is separated as LOC from the head of a religious-institution name, which matters for type transfer.

## 047 · row_id=4658244 · doc_id=392890 · context_plausible_LOC

Термин LRIP (Low-Rate Initital Production) в США обозначает мелкосерийное начальное производство того или иного вида техники в интересах основного заказчика.

**LOC:** ["США"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 048 · row_id=2105948 · doc_id=185089 · context_plausible_LOC

Назначенная вице-губернатором Одесской области Мария Гайдар считает, что Крым должен быть возвращен в состав Украины.

**LOC:** ["Одесской области", "Крым", "Украины"]

**ORG:** []

**PER:** ["Мария Гайдар"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 049 · row_id=6919846 · doc_id=617143 · context_plausible_LOC

Известно, что жалобы на "торговую агрессию" Китая исходили, в частности, от Италии.

**LOC:** ["Китая", "Италии"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 050 · row_id=3547264 · doc_id=300729 · context_plausible_LOC

Между тем, ВВС России намерены принять на вооружение новый штурмовик, концепция применения которого походила бы на Су-25.

**LOC:** ["России"]

**ORG:** ["ВВС"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 051 · row_id=3909723 · doc_id=329469 · elliptical_name

Коммунисты соберутся на Театральной площади, а ЛДПР - на Пушкинской.

**LOC:** ["Театральной площади", "Пушкинской"]

**ORG:** ["ЛДПР"]

**PER:** []

**Review:** The second square has an elliptical name. The annotation contract should define whether such shortened names are included.

## 052 · row_id=2054630 · doc_id=181026 · context_plausible_LOC

Elect — австралийская компания, находящаяся в Мельбурне.

**LOC:** ["Мельбурне"]

**ORG:** ["Elect"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 053 · row_id=54419 · doc_id=4633 · context_plausible_LOC

В октябре сообщалось, что Путин познакомил с автомобилями проекта «Кортеж» президента Египта Абдель Фаттаха ас-Сиси, когда тот прибыл для переговоров в Россию.

**LOC:** ["Египта", "Россию"]

**ORG:** []

**PER:** ["Путин", "Абдель Фаттаха ас-Сиси"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 054 · row_id=3751150 · doc_id=316858 · context_plausible_LOC

ВТБ выдал правительству Украины два миллиарда долларов еще в июне 2010 года - деньги пошли на покрытие дефицита бюджета страны.

**LOC:** ["Украины"]

**ORG:** ["ВТБ"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 055 · row_id=5580641 · doc_id=474643 · context_plausible_LOC

Это уже второй эпизод с захватом танкера в Нигерии за последние несколько дней.

**LOC:** ["Нигерии"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 056 · row_id=6379489 · doc_id=561774 · context_plausible_LOC

За 2006 год золотовалютные резервы выросли России более, чем на 120 миллиардов долларов.

**LOC:** ["России"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 057 · row_id=2876773 · doc_id=248820 · institution_geography_policy

Министерство обороны США обнародовало концепцию оперативного глобального удара в 2009 году.

**LOC:** ["США"]

**ORG:** ["Министерство обороны"]

**PER:** []

**Review:** The USA is labeled LOC separately from the defense ministry head (ORG). Do not assume the Ukrainian nesting policy is identical.

## 058 · row_id=2200147 · doc_id=192500 · context_plausible_LOC

«Луганск — завершающая точка мотопробега.

**LOC:** ["Луганск"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 059 · row_id=167995 · doc_id=14206 · context_plausible_LOC

Крым и Севастополь вошли в состав России после проведенного в марте 2014 года референдума.

**LOC:** ["Крым", "Севастополь", "России"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 060 · row_id=1191667 · doc_id=107744 · context_plausible_LOC

А вот в чем: два дня назад спецслужбы Украины незаконно выкрали двух российских военнослужащих, а сегодня российские военнослужащие спасают украинских моряков», — написала дипломат.

**LOC:** ["Украины"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 061 · row_id=5755580 · doc_id=493269 · institution_geography_policy

Юридический комитет сената штата Аляска в пятницу постановил заслушать показания мужа губернатора Сары Пэйлин, претендующей на пост вице-президента США в паре с Джоном Маккейном от республиканцев, передает агентство Reuters.

**LOC:** ["Аляска", "США"]

**ORG:** ["Юридический комитет", "Reuters"]

**PER:** ["Сары Пэйлин", "Джоном Маккейном"]

**Review:** Alaska and the USA are geographic components; an institution need not absorb all such words into its span.

## 062 · row_id=4614717 · doc_id=389088 · context_plausible_LOC

После этого "Талибан" ушел в подполье, частично перебазировавшись в пакистанский регион Вазиристан.

**LOC:** ["Вазиристан"]

**ORG:** ["Талибан"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 063 · row_id=1035407 · doc_id=93267 · context_plausible_LOC

В Винницкой области дважды раненого участника боев за аэропорт Донецка Николая Лаврика приговорили к трем годам лишения свободы за кражу государственного флага Украины.

**LOC:** ["Винницкой области", "Донецка", "Украины"]

**ORG:** []

**PER:** ["Николая Лаврика"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 064 · row_id=3918629 · doc_id=330177 · context_plausible_LOC

В Сирию Энтони Шадид прибыл за неделю до смерти и занимался сбором информации о сопротивлении режиму президента Башара Асада, сообщили представители The New York Times.

**LOC:** ["Сирию"]

**ORG:** ["The New York Times"]

**PER:** ["Энтони Шадид", "Башара Асада"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 065 · row_id=2637084 · doc_id=229754 · context_plausible_LOC

Тем не менее в пресс-службе МВД подтвердили, что одного из пассажиров рейса из Испании, которым летел депутат, вывели из самолета сотрудники полиции.

**LOC:** ["Испании"]

**ORG:** ["МВД"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 066 · row_id=7471812 · doc_id=671223 · context_plausible_LOC

В посольстве подтвердили, что в ночь с воскресенья на понедельник столица Ирака подверглась массированной бомбардировке.

**LOC:** ["Ирака"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 067 · row_id=5620969 · doc_id=478871 · context_plausible_LOC

Кроме того, в 2007 году аль-Зайди был похищен в Багдаде, предположительно, "Аль-Каедой".

**LOC:** ["Багдаде"]

**ORG:** ["Аль-Каедой"]

**PER:** ["аль-Зайди"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 068 · row_id=7974326 · doc_id=732927 · abbreviation_and_quote

Об этом говорится в заявлении депутатов палаты представителей Народного собрания ЧР, текст которого был передан депутатами корреспонденту РИА "Новости" в Гудермесе.

**LOC:** ["ЧР", "Гудермесе"]

**ORG:** ["Народного собрания", "РИА \"Новости"]

**PER:** []

**Review:** The abbreviation needs document context for interpretation; the auxiliary ORG includes an opening quote. Not every label should be assumed perfect.

## 069 · row_id=6188353 · doc_id=540355 · address_scope

Однопалубный прогулочный корабль горел у берега Химкинского водохранилища вблизи дома 41 по Лодочной улице.

**LOC:** ["Химкинского водохранилища", "Лодочной улице"]

**ORG:** []

**PER:** []

**Review:** LOC includes the street name but not building number 41. Ukrainian address spans sometimes include numbers, so scope conventions differ.

## 070 · row_id=2566516 · doc_id=223556 · context_plausible_LOC

Авторы документа при этом указал, что «действия, предпринятые недавно Россией на востоке Украины противоречат основным принципам безопасности в Европе».

**LOC:** ["Россией", "Украины", "Европе"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 071 · row_id=3408313 · doc_id=289849 · context_plausible_LOC

Столкновение произошло на рынке в деревне Тамун во время задержания военнослужащими Израиля активиста исламистской группировки "Исламский джихад".

**LOC:** ["Тамун", "Израиля"]

**ORG:** ["Исламский джихад"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 072 · row_id=7461318 · doc_id=670023 · context_plausible_LOC

А бесшабашная Барселона наполняет их существование энергией и счастьем.

**LOC:** ["Барселона"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 073 · row_id=6276127 · doc_id=550212 · nonlocation_name

Лидеры "Другой России" Гарри Каспаров и Эдуард Лимонов решили принять участие в намеченном на 18 мая "Марше несогласных" в Самаре, сообщает сайт "Объединенного гражданского фронта" (ОГФ) со ссылкой на пресс-службу "Другой России".

**LOC:** ["Самаре"]

**ORG:** ["Другой России", "Объединенного гражданского фронта\" (ОГФ)", "Другой России"]

**PER:** ["Гарри Каспаров", "Эдуард Лимонов"]

**Review:** The movement name remains ORG, while Samara is LOC. This is a useful negative example of a geographic word inside another entity name.

## 074 · row_id=5385122 · doc_id=456028 · context_plausible_LOC

Питерский "Зенит" заинтересовался атакующим полузащитником сборной Португалии Рикарду Кварежмой, который с февраля 2009 года на правах аренды выступает за лондонский "Челси".

**LOC:** ["Португалии"]

**ORG:** ["Зенит", "Челси"]

**PER:** ["Рикарду Кварежмой"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 075 · row_id=4877442 · doc_id=411882 · context_plausible_LOC

У скоростного поезда ICE 105, направлявшегося из Амстердама в Базель, на полном ходу оторвало одну из дверей, сообщает Spiegel.

**LOC:** ["Амстердама", "Базель"]

**ORG:** ["ICE", "Spiegel"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 076 · row_id=6543736 · doc_id=579849 · context_plausible_LOC

В документе рекомендуется немедленно начать процедуры по выводу российских миротворцев из зон конфликта в Абхазии и Южной Осетии в связи с тем, что они не оправдали возложенных на них задач.

**LOC:** ["Абхазии", "Южной Осетии"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 077 · row_id=5885282 · doc_id=507128 · context_plausible_LOC

В размещенном на сайте украинского Министерства иностранных дел заявлении подчеркивается, что Украина заинтересована в добрососедских и партнерских отношениях с Россией, а оказание давления считает неприемлемым.

**LOC:** ["Украина", "Россией"]

**ORG:** ["Министерства иностранных дел"]

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 078 · row_id=6108249 · doc_id=531397 · context_plausible_LOC

Как пишет газета The Mirror со ссылкой на собственные источники, речь идет о любовнице Мьюрэта, уроженке Германии Микаэле Уолзух, и ее муже Луисе Антонио, с которым она проживает раздельно.

**LOC:** ["Германии"]

**ORG:** ["The Mirror"]

**PER:** ["Мьюрэта", "Микаэле Уолзух", "Луисе Антонио"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 079 · row_id=7638263 · doc_id=689401 · context_plausible_LOC

К 20 мая основными претендентами на пост губернатора считаются глава законодательного собрания Александр Усс, мэр Красноярска Петр Пимашков, губернатор Таймыра Александр Хлопонин, губернатор Эвенкии Борис Золотарев, вице-председатель Госдумы коммунист Петр Романов и исполняющий обязанности губернатора Красноярского края Николай Ашлапов.

**LOC:** ["Красноярска", "Таймыра", "Эвенкии", "Красноярского края"]

**ORG:** ["Госдумы"]

**PER:** ["Александр Усс", "Петр Пимашков", "Александр Хлопонин", "Борис Золотарев", "Петр Романов", "Николай Ашлапов"]

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 080 · row_id=1826031 · doc_id=163009 · context_plausible_LOC

В азиатской части Стамбула в районе Малтепе загорелся строящийся отель.

**LOC:** ["Стамбула", "Малтепе"]

**ORG:** []

**PER:** []

**Review:** The geographic labels look consistent with this context. This is AI review, not independent gold annotation.

## 081 · row_id=633910 · doc_id=55994 · context_plausible_negative

Прототипом орнамента послужила гоночная модель Pirelli Cinturato Blue Wet, используемая на экстремально влажных трассах.

**LOC:** []

**ORG:** ["Pirelli"]

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 082 · row_id=4957895 · doc_id=418697 · context_plausible_negative

Голы забили Патрик Хорнквист и Хенрик Зеттерберг.

**LOC:** []

**ORG:** []

**PER:** ["Патрик Хорнквист", "Хенрик Зеттерберг"]

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 083 · row_id=3202881 · doc_id=274124 · context_plausible_negative

Сама экс-премьер отвергает все обвинения в свой адрес, заявляя, что власти преследуют ее по политическим мотивам.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 084 · row_id=2763230 · doc_id=240328 · context_plausible_negative

По информации исполняющего обязанности главы МВД Арсена Авакова, у них при себе есть автоматы и пулеметы.

**LOC:** []

**ORG:** ["МВД"]

**PER:** ["Арсена Авакова"]

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 085 · row_id=2483080 · doc_id=216097 · context_plausible_negative

Всего было зафиксировано 13567 случаев заражения вирусом.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 086 · row_id=7678339 · doc_id=694027 · context_plausible_negative

Двое из нападавших были задержаны.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 087 · row_id=4564271 · doc_id=384622 · text_fusion

Кроме того, в регионе насчитывалось 2,5 тысячи аварийных многоквартирных домов.Фонд ЖКХ, созданный в 2007 году, на долевой основе с регионами финансирует капитальный ремонт многоквартирных домов и расселение аварийного жилья.

**LOC:** []

**ORG:** ["Фонд ЖКХ"]

**PER:** []

**Review:** Two statements are fused without a space. There is no LOC; the generic region phrase is not a named place.

## 088 · row_id=3973334 · doc_id=334559 · context_plausible_negative

При встрече обязательно погладь его и попробуй поцеловать".

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 089 · row_id=5586969 · doc_id=475286 · context_plausible_negative

В пояснительной записке к поправкам они объясняют, что нововведения помогут предотвратить уничтожение доказательств, поскольку в СИЗО не всегда возможно наблюдать за ходом свиданий обвиняемых и защитников.

**LOC:** []

**ORG:** ["СИЗО"]

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 090 · row_id=5565677 · doc_id=473110 · context_plausible_negative

Он был разработан ОАО "Конструкторское бюро "Луч".

**LOC:** []

**ORG:** ["ОАО \"Конструкторское бюро \"Луч\""]

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 091 · row_id=5366053 · doc_id=454260 · context_plausible_negative

Премьер-министр заявил, что результатом борьбы с контрабандой должны быть "посадки в тюрьму".

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 092 · row_id=6908371 · doc_id=616073 · context_plausible_negative

Около трех часов уйдет у королевы на то, чтобы объехать выстроенные в четыре линии судна королевских ВМС.

**LOC:** []

**ORG:** ["ВМС"]

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 093 · row_id=7706732 · doc_id=697557 · auxiliary_span_quality

Кроме того, она выступит в качестве продюсера фильма Джейн Кэмпион (Jane Campion) "На записи" (In the Cut).

**LOC:** []

**ORG:** []

**PER:** ["Джейн Кэмпион ("]

**Review:** PER includes an opening bracket after the name. LOC is absent, but auxiliary annotation quality also needs care.

## 094 · row_id=1181824 · doc_id=106834 · context_plausible_negative

Я очень уважаю ее и ее семью.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 095 · row_id=6301293 · doc_id=552973 · context_plausible_negative

Город разорвет договор с АРР до 30 апреля.

**LOC:** []

**ORG:** ["АРР"]

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 096 · row_id=4863722 · doc_id=410722 · context_plausible_negative

Ранее премию, учрежденную в 2004 году, вручали дирижеру Курту Мазуру, хореографу Джону Ноймайеру, скрипачу Гидону Кремеру и меццо-сопрано Кристе Людвиг.

**LOC:** []

**ORG:** []

**PER:** ["Курту Мазуру", "Джону Ноймайеру", "Гидону Кремеру", "Кристе Людвиг"]

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 097 · row_id=1264648 · doc_id=114436 · context_plausible_negative

Об этом они сообщили на пресс-конференции, которая прошла в четверг, 20 октября.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 098 · row_id=1599719 · doc_id=144453 · context_plausible_negative

Задумке Тайкур не было суждено осуществиться, поскольку вместо настоящего убийцы она вышла на связь с полицейским, работающим под прикрытием.

**LOC:** []

**ORG:** []

**PER:** ["Тайкур"]

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 099 · row_id=841070 · doc_id=75127 · context_plausible_negative

Также вручено две награды вне конкурса.

**LOC:** []

**ORG:** []

**PER:** []

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.

## 100 · row_id=2407390 · doc_id=209598 · context_plausible_negative

Джоли является обладательницей кинопремии «Оскар».

**LOC:** []

**ORG:** []

**PER:** ["Джоли"]

**Review:** The context does not clearly require a named location; the absence of LOC looks plausible.
