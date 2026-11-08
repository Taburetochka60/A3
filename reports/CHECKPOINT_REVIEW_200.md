# Contextual review of 200 checkpoint errors

100 FP and 100 FN focal cases from 200 distinct texts. This diagnostic sample emphasizes boundary coverage; its proportions do not estimate corpus-wide prevalence.
All contexts were reviewed by Codex. This is AI-assisted review, not independent expert annotation. Original gold is unchanged. Annotation-quality categories identify candidates for human review. Original NER text and entity strings are preserved in their source languages.

## 001. FN · row_id=1003092 · missing_name_material

Там уточнюють, що малюка близько 21:30, 3 травня, в Покровському районі Кривого Рогу, 25-річна дівчина і викликала поліцейських і медиків.

**Gold:** ["Покровському районі", "Кривого Рогу"]

**Prediction:** ["Покровському"]

**Focal surface:** Покровському районі; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 002. FN · row_id=1001115 · multiword_name

З урахуванням презентації варто нагадати, що єдиний на сьогодні п`ятизірковий готель в Києві - «Прем`єр Палас», має 289 номерів.

**Gold:** ["Києві", "Прем`єр Палас"]

**Prediction:** ["Києві", "Палас"]

**Focal surface:** Прем`єр Палас; pilot fit frequency: 0; UNK: True.

**Review:** The hotel name loses its first component and apostrophe; only the second component is returned.

## 003. FN · row_id=1006624 · granularity_policy

Митрополита Горлівського та Слов'янського Митрофана УПЦ МП затримали на контрольно-пропускному пункті та в супроводі конвою доправили до міського управління поліції Слов'янська Донецької області, після чого його відпустили.

**Gold:** ["Слов'янська Донецької області"]

**Prediction:** ["Донецької області"]

**Focal surface:** Слов'янська Донецької області; pilot fit frequency: 0; UNK: True.

**Review:** Gold combines the city and region into one entity; only the region is predicted. Specify the address-nesting convention.

## 004. FN · row_id=1008314 · address_boundary

У Яремчі на вулиці С. Ковпака та О. Довбуша мешканці самостійно ремонтували дорогу.

**Gold:** ["Яремчі", "вулиці С. Ковпака", "О. Довбуша"]

**Prediction:** ["Яремчі", "вулиці", "Ковпака"]

**Focal surface:** вулиці С. Ковпака; pilot fit frequency: 0; UNK: True.

**Review:** The street name with an initial is fragmented; the street-type word and surname are predicted separately.

## 005. FN · row_id=1001484 · entity_type_context

Як зазначив співрозмовник агентства, "замість українських підприємств технічний супровід програми "Дніпро" бере на себе Державний ракетний центр імені Макеєва (Міасс, Челябінська область) ".

**Gold:** ["Челябінська область"]

**Prediction:** ["Дніпро", "Міасс", "область"]

**Focal surface:** Челябінська область; pilot fit frequency: 0; UNK: True.

**Review:** Elsewhere in this sentence, Dnipro names a program and Miass is annotated as ORG. Geographic words can therefore occur under other entity types.

## 006. FN · row_id=1001664 · missing_name_material

В екосистемах Чорнобильської зони, на жаль, тривають пожежі.

**Gold:** ["Чорнобильської зони"]

**Prediction:** ["екосистемах", "зони"]

**Focal surface:** Чорнобильської зони; pilot fit frequency: 0; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 007. FN · row_id=1000779 · split_location

Нагадаємо, 15 жовтня, неподалік Повітрофлотського шляхопроводу в Києві спалахнув останній поверх 5-поверхової адмінбудівлі електротехнічного заводу "Транссигнал".

**Gold:** ["Повітрофлотського шляхопроводу", "Києві"]

**Prediction:** ["Повітрофлотського", "шляхопроводу", "Києві"]

**Focal surface:** Повітрофлотського шляхопроводу; pilot fit frequency: 0; UNK: True.

**Review:** One bridge name is split into two predicted entities.

## 008. FN · row_id=1008116 · possible_annotation_error

З іншого боку чимало хто з МП налаштовані проти Москви, у тому числі митрополити, зокрема Черкаський і Канівський Софроній.

**Gold:** ["Москви, у тому чи"]

**Prediction:** ["Москви"]

**Focal surface:** Москви, у тому чи; pilot fit frequency: 0; UNK: False.

**Review:** The gold span includes Moscow followed by a truncated fragment of the next phrase. The shorter prediction looks semantically more reasonable but is scored as wrong.

## 009. FN · row_id=1005408 · granularity_policy

Івано-Франківська область забезпечена в повному обсязі вакцинами КПК (кір, паротит,краснуха).

**Gold:** ["Івано", "Франківська область"]

**Prediction:** ["Івано", "-Франківська область забезпечена", "вакцинами КПК", "кір", "паротит", "краснуха"]

**Focal surface:** Франківська область; pilot fit frequency: 0; UNK: False.

**Review:** Gold splits the hyphenated city name; the model additionally absorbs the following predicate.

## 010. FN · row_id=1008088 · extra_neighbor_material

Нагадаємо, на поверхні в Антарктиді розпускаються мікроскопічні водорості і фарбують сніг у зелений колір.

**Gold:** ["Антарктиді"]

**Prediction:** ["Антарктиді розпускаються", "зелений колір"]

**Focal surface:** Антарктиді; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 011. FN · row_id=1006148 · extra_neighbor_material

При це екс-канцлер висловив сподівання, що після того, як президент РФ залишить свій пост на Заході говоритимуть, що період його правління був великим для Росії.

**Gold:** ["РФ", "Заході", "Росії"]

**Prediction:** ["РФ", "Заході говоритимуть", "Росії"]

**Focal surface:** Заході; pilot fit frequency: 7; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 012. FN · row_id=1006577 · missing_name_material

Найбільш напружена ситуація на Донецькому напрямку була на підступах до Мар’їнки та Новомихайлівки, де окупанти вели прицільний вогонь з застосуванням станкових протитанкових гранатометів та стрілецької зброї.

**Gold:** ["Донецькому напрямку", "Мар’їнки", "Новомихайлівки"]

**Prediction:** ["Донецькому", "підступах", "Мар’їнки", "Новомихайлівки"]

**Focal surface:** Донецькому напрямку; pilot fit frequency: 1; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 013. FN · row_id=1002893 · extra_neighbor_material

У штабі ООС також повідомили, що на території одного із приватних будинків в Дружківці на Донеччині виявили арсенал зі зброєю.

**Gold:** ["Дружківці", "Донеччині"]

**Prediction:** ["штабі", "Дружківці", "Донеччині виявили"]

**Focal surface:** Донеччині; pilot fit frequency: 5; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 014. FN · row_id=1001467 · granularity_policy

Елітна нерухомість в Івано-Франківську приваблює інвесторів, тримаючи продажі на позначці 4-5 % від загалу новобудов.

**Gold:** ["Івано", "-Франківську"]

**Prediction:** ["Елітна", "Івано", "-Франківську приваблює інвесторів", "позначці"]

**Focal surface:** -Франківську; pilot fit frequency: 1; UNK: False.

**Review:** Gold splits the hyphenated name, while the model also captures following words.

## 015. FN · row_id=1007340 · extra_neighbor_material

Раніше постійний представник України при ООН Сергій Кислиця повідомив, що в окупованому Криму переслідують власників українських паспортів і чинять тиск, для того щоб населення позбувалося українських документів.

**Gold:** ["України", "Криму"]

**Prediction:** ["України", "Криму переслідують"]

**Focal surface:** Криму; pilot fit frequency: 29; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 016. FN · row_id=1007301 · extra_neighbor_material

У Москві уточнили, що ініціатором був Київ. Війну на Донбасі у Кремлі традиційно називають "внутрішньоукраїнською кризою", не визнаючи, що саме Росія та підтримувані нею бойовики окупували частину Донбасу.

**Gold:** ["Москві", "Київ", "Донбасі", "Кремлі", "Росія", "Донбасу"]

**Prediction:** ["Москві уточнили", "Київ", "Донбасі", "Кремлі", "внутрішньоукраїнською", "Росія", "підтримувані", "Донбасу"]

**Focal surface:** Москві; pilot fit frequency: 11; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 017. FN · row_id=1005379 · missing_name_material

На фоні того, як Росія продовжує мілітаризувати Арктику, росте геостратегічна важливість Обєднаного королівства.

**Gold:** ["Росія", "Арктику", "Обєднаного королівства"]

**Prediction:** ["Росія", "королівства"]

**Focal surface:** Обєднаного королівства; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 018. FN · row_id=1004883 · multiword_name

Шартелла Сантьяго приїхала до Івано-Франківська аж із Карибських островів.

**Gold:** ["Івано", "-Франківська", "Карибських островів"]

**Prediction:** ["Івано", "-Франківська", "островів"]

**Focal surface:** Карибських островів; pilot fit frequency: 0; UNK: True.

**Review:** The modifier in the Caribbean Islands name is lost. The other city in gold is also split at its hyphen.

## 019. FN · row_id=1005367 · extra_neighbor_material

Половина населення просто не має можливості побувати і дивитися на Львів, на панораму Львова згори", - каже Ярослав Грибальський.

**Gold:** ["Львів", "Львова"]

**Prediction:** ["Львів", "панораму", "Львова згори"]

**Focal surface:** Львова; pilot fit frequency: 28; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 020. FN · row_id=1005667 · coordinated_locations

На період проведення заходів на Софіївській та Михайлівській площах до маршрутів деяких тролейбусів буде внесено зміни.

**Gold:** ["Софіївській та Михайлівській площах"]

**Prediction:** ["Софіївській"]

**Focal surface:** Софіївській та Михайлівській площах; pilot fit frequency: 0; UNK: True.

**Review:** Gold combines two squares and their conjunction into one entity; the model finds only the first modifier.

## 021. FN · row_id=1004857 · extra_neighbor_material

Його нинішнє місце розташування - на орбіті в районі Австралії приблизно в 35 тис. кілометрів над Землею.

**Gold:** ["Австралії", "Землею"]

**Prediction:** ["орбіті", "Австралії приблизно", "Землею"]

**Focal surface:** Австралії; pilot fit frequency: 4; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 022. FN · row_id=1003973 · merged_locations

Як зазначають, інцидент стався на вул. Зарічній в Рівному.

**Gold:** ["вул. Зарічній", "Рівному"]

**Prediction:** ["вул. Зарічній в Рівному."]

**Focal surface:** вул. Зарічній; pilot fit frequency: 0; UNK: True.

**Review:** The street and city are merged into one address fragment; gold requires two separate entities.

## 023. FN · row_id=1002212 · extra_neighbor_material

Сьогодні у Полтаві розпочинає роботу триденний ІІ Міжнародний фестиваль документального …

**Gold:** ["Полтаві"]

**Prediction:** ["Полтаві розпочинає", "триденний"]

**Focal surface:** Полтаві; pilot fit frequency: 1; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 024. FN · row_id=1005161 · coordinated_locations

У Волинській, Закарпатській і Харківській областях перевищено відсоток позитивних ПЛР-тестів на коронавірус.

**Gold:** ["Волинській, Закарпатській і Харківській областях"]

**Prediction:** ["Волинській", "Закарпатській", "Харківській областях"]

**Focal surface:** Волинській, Закарпатській і Харківській областях; pilot fit frequency: 0; UNK: False.

**Review:** Three coordinated regions form one gold entity, while the model emits three. This is a granularity issue, not complete failure to recognize geography.

## 025. FN · row_id=1007953 · extra_neighbor_material

На змаганнях у Мінську дівчина піднялася на п’єдестал пошани та вправах на колоді, де є переможницею загального заліку Кубка світу-2018.

**Gold:** ["Мінську"]

**Prediction:** ["Мінську дівчина піднялася", "колоді"]

**Focal surface:** Мінську; pilot fit frequency: 7; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 026. FN · row_id=1003664 · address_boundary

Тож, чекаємо вас у вівторок, 3 вересня 2019 року об 11:00 в приміщенні "Всеукраїнської асоціації поліграфологів" за адресою: Київ, вулиця Шовковична, 10, офіс 27. Ви славетний чемпіон.

**Gold:** ["Київ", "вулиця Шовковична, 10"]

**Prediction:** ["Київ", "вулиця"]

**Focal surface:** вулиця Шовковична, 10; pilot fit frequency: 0; UNK: True.

**Review:** Only the generic street-type word remains; the street name and number are not extracted.

## 027. FN · row_id=1009798 · merged_locations

У Маріуполі Донецької області на площі Перемоги у автомобілі Audi знайшли мертвого водія.

**Gold:** ["Маріуполі", "Донецької області", "площі Перемоги"]

**Prediction:** ["Маріуполі Донецької області", "площі Перемоги"]

**Focal surface:** Донецької області; pilot fit frequency: 1; UNK: False.

**Review:** The model merges Mariupol and Donetsk region, while gold keeps them separate.

## 028. FN · row_id=1001341 · boundary_and_aux_quality

Виготовлений Дідух із соломи та прикрашений різнобарвними стрічками майстрами Музею народної архітектури та побуту у Львові імені Климентія Шептицького.

**Gold:** ["Львові"]

**Prediction:** ["Львові імені"]

**Focal surface:** Львові; pilot fit frequency: 19; UNK: False.

**Review:** The word meaning named after is added to the city. The auxiliary ORG span also looks truncated; auxiliary labels should not be assumed error-free.

## 029. FN · row_id=1001371 · coordinated_locations

Глави терористичних організацій "ДНР" і "ЛНР", які за підтримки Росії окупували окремі території Луганської і Донецької областей, привітали підписання "формули Штайнмаєра".

**Gold:** ["Росії", "Луганської і Донецької областей"]

**Prediction:** ["ЛНР", "Росії", "Луганської", "Донецької областей"]

**Focal surface:** Луганської і Донецької областей; pilot fit frequency: 0; UNK: False.

**Review:** Gold joins two regions with a conjunction; the model separates the coordinated names.

## 030. FN · row_id=1000416 · possible_annotation_error

Українська тенісистка Дар'я Лопатецька поступилася представниці Нідерландів Аранчі Рус у фіналі турніру серії ITF World Tennis Tour в Санта Маргеріта ді Пула, Італія.

**Gold:** ["Нідерландів Аранчі Рус", "Санта Маргеріта ді Пула", "Італія"]

**Prediction:** ["Нідерландів", "Санта", "Пула", "Італія"]

**Focal surface:** Нідерландів Аранчі Рус; pilot fit frequency: 0; UNK: True.

**Review:** The LOC span combines the Netherlands with the tennis player's name. Review the gold type and end boundary.

## 031. FN · row_id=1007875 · extra_neighbor_material

Уже 2005-го року двоє з найближчих колег Путіна, олігархи Володимир Якунін і Костянтин Малофєєв, почали створювати організації, які будуть просувати "альтернативу" демократії та інтеграції в усій Європі.

**Gold:** ["Європі"]

**Prediction:** ["усій Європі"]

**Focal surface:** Європі; pilot fit frequency: 16; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 032. FN · row_id=1006541 · boundary_convention

З 1934 року фестиваль щорічно проводиться на острові Лідо у кінці літа – початку осені.

**Gold:** ["Лідо"]

**Prediction:** ["острові Лідо"]

**Focal surface:** Лідо; pilot fit frequency: 0; UNK: True.

**Review:** Lido island is recognized, but gold excludes the generic island word. A consistent rule for such descriptors is needed.

## 033. FN · row_id=1000722 · split_location

Експорт українських равликів до країн-членів Європейського Союзу суттєво зріс у 2019 році.

**Gold:** ["Європейського Союзу"]

**Prediction:** ["Європейського", "Союзу"]

**Focal surface:** Європейського Союзу; pilot fit frequency: 6; UNK: False.

**Review:** The established name European Union is split into two fragments.

## 034. FN · row_id=1001681 · boundary_and_granularity

Американський конвой з технікою та військовими прибув з Іраку до провінції Аль-Хасака на північному сході Сирії.

**Gold:** ["Іраку", "Аль", "-Хасака", "Сирії"]

**Prediction:** ["Іраку", "провінції Аль-Хасака на", "північному сході Сирії"]

**Focal surface:** Сирії; pilot fit frequency: 11; UNK: False.

**Review:** Extra spatial words are included; another hyphenated name in gold is split into parts.

## 035. FN · row_id=1006235 · multiword_name

Особливо вона посилилась тоді, як стало відомо, що під час доставки Олімпійського вогню в Афіни з Древньої Олімпії він гаснув аж 30 разів.

**Gold:** ["Олімпійського вогню", "Афіни", "Древньої Олімпії"]

**Prediction:** ["Афіни", "Древньої"]

**Focal surface:** Древньої Олімпії; pilot fit frequency: 0; UNK: True.

**Review:** Only the modifier remains from Ancient Olympia. The Olympic flame being labeled LOC also calls for a type-policy review.

## 036. FN · row_id=1000684 · possible_annotation_error

В Україні стверджують, що Шумков за власною ініціативою прибув в Росію, ховаючись від кримінального переслідування на батьківщині, і прямо на прикордонному пункті повідомив про членство в "Правому Секторі".

**Gold:** ["Україні", "Росію, ховаюч"]

**Prediction:** ["Україні", "Росію", "прикордонному"]

**Focal surface:** Росію, ховаюч; pilot fit frequency: 0; UNK: True.

**Review:** The gold span ends inside the following verb after Russia. Predicting just the country is not necessarily a semantic error.

## 037. FN · row_id=1005884 · missing_name_material

Найкраще астрономічне явище було видно в Австралії, Східній Азії, США і частині Росії.

**Gold:** ["Австралії", "Східній Азії", "США", "Росії"]

**Prediction:** ["Австралії", "Азії", "США", "Росії"]

**Focal surface:** Східній Азії; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 038. FN · row_id=1006483 · hyphen_granularity

Фестиваль здоров’я, який відбудеться в Івано-Франківську 23 червня, організований лікарями та медичними центрами міста з метою профілактики захворювань та задля того, щоб донести до містян: “бути здоровим – це модно”….

**Gold:** ["Івано", "-Франківську"]

**Prediction:** ["Івано-Франківську"]

**Focal surface:** -Франківську; pilot fit frequency: 1; UNK: False.

**Review:** The model emits the complete hyphenated city name, while gold requires two parts. Do not fix this by mechanically splitting every hyphenated name.

## 039. FN · row_id=1004933 · hyphen_granularity

У Івано-Франківську свідоцтво на таку діяльність має лише ДП “Івано-Франківськстандартметрологія”.

**Gold:** ["Івано", "-Франківську"]

**Prediction:** ["Івано-Франківську свідоцтво", "Івано", "Франківськстандартметрологія"]

**Focal surface:** -Франківську; pilot fit frequency: 1; UNK: False.

**Review:** Beyond the hyphen-granularity mismatch, a following noun is absorbed. These are two distinct issues.

## 040. FN · row_id=1001553 · quote_convention

"Вже за кілька хвилин патрульні розшукали та затримали ймовірних викрадачів на вулиці Кирилівській", - додають у поліції.

**Gold:** ["вулиці Кирилівській\""]

**Prediction:** ["вулиці Кирилівській"]

**Focal surface:** вулиці Кирилівській"; pilot fit frequency: 0; UNK: True.

**Review:** The model excludes a closing quotation mark that belongs to gold. The published regex does not remove it as one might expect.

## 041. FN · row_id=1006390 · hyphen_granularity

Нові вагони планують задіяти на Сирецько-Печерській лінії столичного метрополітену.

**Gold:** ["Сирецько", "Печерській лінії"]

**Prediction:** ["Нові вагони", "Сирецько-Печерській"]

**Focal surface:** Сирецько; pilot fit frequency: 0; UNK: True.

**Review:** The model partially combines a metro-line name; gold separates components and includes the generic line word.

## 042. FN · row_id=1000617 · missing_name_material

Останній уже пішов тріщинами і якщо влада не знайде мільйон гривень зараз, щоб зупиними зсув, то незабаром потрібні будуть сотні мільйонів, щоб відбудувати Лівадійський палац.

**Gold:** ["Лівадійський палац"]

**Prediction:** ["палац"]

**Focal surface:** Лівадійський палац; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 043. FN · row_id=1004149 · address_boundary

"Ми прокурору області сказали точні адреси цих ігрових клубів: пр. Леніна, 33а; пр. Леніна, 26; пр. Леніна, 37; вул.Молодіжна, 58; вул. Набережна енергетиків 7/15.

**Gold:** ["Леніна, 37", "вул.Молодіжна, 58", "вул. Набережна енергетиків 7/15."]

**Prediction:** ["вул", "Молодіжна", "вул. Набережна енергетиків"]

**Focal surface:** вул.Молодіжна, 58; pilot fit frequency: 0; UNK: True.

**Review:** The street abbreviation and number are omitted. Other addresses in this text lack LOC labels; annotation completeness needs review.

## 044. FN · row_id=1004834 · hyphen_granularity

У Івано-Франківську комунальне підприємство пропонує роботу за достойну плату - Pravda.

**Gold:** ["Івано", "-Франківську"]

**Prediction:** ["Івано-Франківську"]

**Focal surface:** -Франківську; pilot fit frequency: 1; UNK: False.

**Review:** A complete city name conflicts with split gold. This is not simply a wrong place name.

## 045. FN · row_id=1003360 · possible_text_corruption

МінТОТ ініціював запровадження санкцій проти хімкомпаній за діяльність в Кримужурнал «Український тиждень»

**Gold:** ["Кримужурнал"]

**Prediction:** ["хімкомпаній", "Кримужурнал «"]

**Focal surface:** Кримужурнал; pilot fit frequency: 0; UNK: True.

**Review:** A location word appears fused with a source/publication fragment. Bounds-valid offsets do not guarantee a clean training example.

## 046. FN · row_id=1004647 · missing_name_material

До 1931-го РСР Абхазія і РСР Грузія були рівноправними суб'єктами, пов'язаними союзним договором.

**Gold:** ["Абхазія", "РСР Грузія"]

**Prediction:** ["1931", "РСР", "Абхазія", "РСР"]

**Focal surface:** РСР Грузія; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: missing_name_material. Original offsets establish this relation, not its cause.

## 047. FN · row_id=1005787 · merged_locations

У селі Роздольне Зміївського району Харківської області внаслідок пожежі загинуло подружжя.

**Gold:** ["Роздольне Зміївського району", "Харківської області"]

**Prediction:** ["Роздольне Зміївського району Харківської області"]

**Focal surface:** Харківської області; pilot fit frequency: 0; UNK: False.

**Review:** The model merges the settlement, district and region; gold keeps the region separate.

## 048. FN · row_id=1003091 · vocabulary_cap

В Атлантичному океані на очах десятків відпочивальників розбився літак - Відео Новини - TCH.

**Gold:** ["Атлантичному океані"]

**Prediction:** ["Атлантичному"]

**Focal surface:** Атлантичному океані; pilot fit frequency: 1; UNK: True.

**Review:** The surface appeared in the pilot, but a component maps to UNK. A seen entity does not guarantee that all its words survived the vocabulary cap.

## 049. FN · row_id=1004029 · hyphen_split

Як повідомив в Буенос-Айресі міністр фінансів Бразилії Гуїдо Мантега, з червня поточного року всі розрахунки здійснюватимуться в національній валюті двох країн.

**Gold:** ["Буенос-Айресі", "Бразилії"]

**Prediction:** ["Буенос", "Айресі"]

**Focal surface:** Буенос-Айресі; pilot fit frequency: 0; UNK: True.

**Review:** Both parts of Buenos Aires are found separately; the full name should be retained.

## 050. FN · row_id=1005121 · merged_locations

У селі Мурафа Харківської області 9-річний хлопчик позбувся пальця після того, як підірвав у школі петарду.

**Gold:** ["Мурафа", "Харківської області"]

**Prediction:** ["Мурафа Харківської області"]

**Focal surface:** Харківської області; pilot fit frequency: 0; UNK: False.

**Review:** The village and region are merged; both geographic components are present in the prediction.

## 051. FN · row_id=1000126 · detection_miss

Але, попри сподівання Дорошенка, який оголосив себе гетьманом обох боків Дніпра, убивство Брюховецького на переговорах не посприяло возз'єднанню України.

**Gold:** ["Дніпра", "України"]

**Prediction:** ["України"]

**Focal surface:** Дніпра; pilot fit frequency: 2; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 2; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 052. FN · row_id=1002625 · possible_annotation_error

В інтерв’ю журналу New Yorker віце-президент США Джозеф Байден розповів про свій візит до Путіна в 2011 році, який на той час був прем’єр-міністром Росії.

**Gold:** ["США", "Путіна"]

**Prediction:** ["США", "Росії"]

**Focal surface:** Путіна; pilot fit frequency: 0; UNK: False.

**Review:** Putin refers to a person in this context but is included in LOC gold. Russia elsewhere in the same sentence has no LOC annotation.

## 053. FN · row_id=1009985 · building_or_ORG_policy

Зараз правоохоронці евакуювали людей з Верховної Ради, також посилили охорону в парламенті через події на Майдану Незалежності та протистояння на вулиці Інститутській, повідомляє в четвер, 20 лютого, телеканал Інтер, посилаючись на Володимира Литвина.

**Gold:** ["Ради", "Майдану Незалежності", "вулиці Інститутській"]

**Prediction:** ["Майдану Незалежності", "вулиці Інститутській"]

**Focal surface:** Ради; pilot fit frequency: 0; UNK: False.

**Review:** Gold contains only a fragment of the parliamentary name, while the context refers to a building. Clarify boundaries and the building/institution convention.

## 054. FN · row_id=1001927 · detection_miss

Від них буде залежати не тільки внутрішня політика Брюсселя, але і наскільки рішучий Євросоюз буде у підтримці України. Опитування показують зростання популярності євроскептиків, що традиційно більш прихильні до Москви.

**Gold:** ["Брюсселя", "України", "Москви"]

**Prediction:** ["України", "Москви"]

**Focal surface:** Брюсселя; pilot fit frequency: 2; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 2; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 055. FN · row_id=1000883 · detection_miss

«Незалежна Шотландія досить швидко вступить у ЄС»

**Gold:** ["Шотландія", "ЄС"]

**Prediction:** ["ЄС"]

**Focal surface:** Шотландія; pilot fit frequency: 0; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 056. FN · row_id=1008086 · detection_miss

Джерела повідомили виданню TMZ, яке першим повідомило цю новину, що подруга поїхала провідати 43-річну порнозірку у її будинку у Сан-Фернандо у Каліфорнії після того, як довго не могла вийти з нею на зв'язок.

**Gold:** ["Сан-Фернандо", "Каліфорнії"]

**Prediction:** ["Каліфорнії"]

**Focal surface:** Сан-Фернандо; pilot fit frequency: 0; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 057. FN · row_id=1005536 · detection_miss

На півночі колись єдиного Сомалі постав Сомаліленд, який до 1960 року був британською колонією.

**Gold:** ["Сомаліленд"]

**Prediction:** []

**Focal surface:** Сомаліленд; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 058. FN · row_id=1006833 · possible_annotation_omission

Живий ланцюг — символ та головна традиція Дня соборності України. Сьогодні це свято відзначають створенням живих ланцюгів, які об’єднують Західну і Східну частини країни.

**Gold:** ["Західну", "Східну частини"]

**Prediction:** ["України"]

**Focal surface:** Західну; pilot fit frequency: 0; UNK: False.

**Review:** The model misses the annotated parts of the country but also finds an unannotated Ukraine mention. This is a candidate for a gold-completeness check.

## 059. FN · row_id=1000596 · vocabulary_cap

У ЛЕО вже не вперше попереджали владу Лисичанська, що місто надовго може залишитися без водопостачання.

**Gold:** ["Лисичанська"]

**Prediction:** []

**Focal surface:** Лисичанська; pilot fit frequency: 1; UNK: True.

**Review:** The name appeared as LOC before, but its token is missing from the saved pilot vocabulary.

## 060. FN · row_id=1002510 · seen_context_miss

Львів — перше місто в світі, де проведуть Бієнале довіри.

**Gold:** ["Львів"]

**Prediction:** []

**Focal surface:** Львів; pilot fit frequency: 9; UNK: False.

**Review:** A seen Lviv surface is missed at the start of a short sentence. Word novelty alone cannot explain the error.

## 061. FN · row_id=1006326 · detection_miss

”Як ми неодноразово заявляли, ми очікуємо, що Росія забезпечить безперешкодне та вільне проходження через Керченську протоку в Азовське море та з нього відповідно до міжнародного права.

**Gold:** ["Росія", "Керченську протоку", "Азовське море"]

**Prediction:** ["Росія"]

**Focal surface:** Керченську протоку; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 062. FN · row_id=1000689 · detection_miss

Однак виконанню цього рішення завадила Російська Федерація, взявши екс-чекіста під свою опіку.

**Gold:** ["Російська Федерація"]

**Prediction:** []

**Focal surface:** Російська Федерація; pilot fit frequency: 2; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 2; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 063. FN · row_id=1002094 · generic_location_FP

Багато її потрапило і до річки Путилівка, яка входить в заповідну зону.

**Gold:** ["Путилівка"]

**Prediction:** ["заповідну зону"]

**Focal surface:** Путилівка; pilot fit frequency: 0; UNK: True.

**Review:** The river name is missed, and a generic protected-area description is predicted instead.

## 064. FN · row_id=1001794 · possible_annotation_error

Настільки, що дуже загальна англомовна назва для іноземного прокату "The Rising Hawk", "Схід Сокола", здається відповіднішою для фінального продукту, ніж "Захар Беркут".

**Gold:** ["Схід Сокола"]

**Prediction:** []

**Focal surface:** Схід Сокола; pilot fit frequency: 0; UNK: False.

**Review:** The context explicitly describes a film title, rather than a place, but it appears in LOC gold.

## 065. FN · row_id=1006068 · detection_miss

До Другої світової війни Дрогобич, як і решту міст Галичини, населяли поляки, українці та євреї, а також менш численні угорці та німці.

**Gold:** ["Дрогобич", "Галичини"]

**Prediction:** []

**Focal surface:** Дрогобич; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 066. FN · row_id=1002624 · detection_miss

Нині Київ устав з колін і разом з ним піднявся весь нескорений люд України. Впевнений, до нас сьогодні повернулась наша козацька демократія: ми мали Запорозьку Січ, відстоюємо волю, славу, честь.

**Gold:** ["Київ", "України", "Запорозьку Січ"]

**Prediction:** ["Київ", "України"]

**Focal surface:** Запорозьку Січ; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 067. FN · row_id=1006181 · detection_miss

Поки що інвестор розглядає пропозиції в Сарненському, Березнівському та Костопільському районах, жителі яких і працюють у нього в Польщі.

**Gold:** ["Сарненському", "Березнівському", "Костопільському", "Польщі"]

**Prediction:** ["Сарненському", "Польщі"]

**Focal surface:** Костопільському; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 068. FN · row_id=1005354 · administrative_policy

Але думаю, що зі створенням ОТГ, коли вся земля перейде до нас і ми будемо нею розпоряджатися, враховуючи землю Проценківської сільради, яка з нами об’єднується, варіанти вирішення цього питання знайдуться.

**Gold:** ["Проценківської сільради"]

**Prediction:** ["ОТГ"]

**Focal surface:** Проценківської сільради; pilot fit frequency: 0; UNK: True.

**Review:** A village council can refer to an institution or territory. Gold needs a consistent policy; the prediction instead names another administrative entity.

## 069. FN · row_id=1009334 · detection_miss

Часом він заявляє, що Албанія й Косово — це дві держави, а часом говорить про прискорену їх інтеграцію.

**Gold:** ["Албанія", "Косово"]

**Prediction:** []

**Focal surface:** Косово; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 070. FN · row_id=1006382 · possible_type_error

Зрештою, і коли у 2009 році Ворскла грала у фіналі із Шахтарем, більшість віддавала перевагу гірникам, проте кубок поїхав до Полтави.

**Gold:** ["Ворскла", "Полтави"]

**Prediction:** ["Полтави"]

**Focal surface:** Ворскла; pilot fit frequency: 0; UNK: True.

**Review:** Vorskla is playing in a football final here: context indicates a club, although gold labels LOC.

## 071. FN · row_id=1002340 · detection_miss

Але деякі спробували згадати головне місто 62 африканських країн, назвавши Єгипет і його столицю - Каїр.

**Gold:** ["Єгипет", "Каїр"]

**Prediction:** []

**Focal surface:** Каїр; pilot fit frequency: 0; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 072. FN · row_id=1001485 · detection_miss

І враз всі побачили диво: сліпі козаки, поранені, хворі, каліки і старці-кобзарі, що доживали свій вік на Січі, раптом стали в коло, і зазвучали кобзи.

**Gold:** ["Січі"]

**Prediction:** ["каліки", "старці"]

**Focal surface:** Січі; pilot fit frequency: 1; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 1; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 073. FN · row_id=1002916 · generic_gold_fragment

В департаменті соцполітики виконкому Криворізької міської ради розповіли, що пенсіонера можна оформити до Будинку нічного перебування для цього його сусідам потрібно звернутися до районного відділу соцзахисту населення.

**Gold:** ["Будинку"]

**Prediction:** ["соцполітики виконкому"]

**Focal surface:** Будинку; pilot fit frequency: 3; UNK: False.

**Review:** Gold contains only a generic building-word fragment from an institution name. Check whether such fragments are intended target names.

## 074. FN · row_id=1009522 · nested_ORG_policy

Не стане кафедральним собором православної церкви України і Софія Київська.

**Gold:** ["України"]

**Prediction:** []

**Focal surface:** України; pilot fit frequency: 266; UNK: False.

**Review:** A country inside a church name is labeled LOC, whereas geographic components inside institutions are excluded in other cases. Check policy consistency.

## 075. FN · row_id=1001633 · detection_miss

Відповідно, вартість нашої планети Земля поки більша всіх вивчених планет разом узятих.

**Gold:** ["Земля"]

**Prediction:** []

**Focal surface:** Земля; pilot fit frequency: 0; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 076. FN · row_id=1006770 · detection_miss

На "Ролс-ройсі" сюди вряди-годи заїжджав консул Нідерландів, а митрополит Віссаріон Пую, про якого подейкували, що він був однокашником самого Йосифа Сталіна по семінарії, хоча й мав лімузин, для приїзду на Панську використовував лише коней.

**Gold:** ["Нідерландів", "Панську"]

**Prediction:** ["Ролс", "вряди", "годи", "Нідерландів"]

**Focal surface:** Панську; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 077. FN · row_id=1004538 · detection_miss

У зв'язку з авіакатастрофою літака рейсу PS752 Тегеран-Київ, яка привела до численних людських жертв, ФЛАУ застерігає українських атлетів від виступів і тренувань у країнах Близького Сходу.

**Gold:** ["Тегеран", "Київ", "Близького Сходу"]

**Prediction:** ["Київ", "Близького", "Сходу"]

**Focal surface:** Тегеран; pilot fit frequency: 0; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 078. FN · row_id=1001304 · ORG_region_policy

Останній показник є найменшим поміж країн Східного партнерства.

**Gold:** ["Східного партнерства"]

**Prediction:** []

**Focal surface:** Східного партнерства; pilot fit frequency: 1; UNK: False.

**Review:** Eastern Partnership denotes a political initiative or grouping in this context; review the LOC-versus-ORG convention.

## 079. FN · row_id=1007999 · detection_miss

У Швейцарії виставлять на аукціон колекцію з 25 автомобілів класу люкс, конфіскованих у віце-президента та сина правителя Екваторіальної Гвінеї Теодоріна Обіанга Нгеми.

**Gold:** ["Швейцарії", "Екваторіальної Гвінеї"]

**Prediction:** ["Швейцарії виставлять"]

**Focal surface:** Екваторіальної Гвінеї; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 080. FN · row_id=1003075 · detection_miss

Відзначається, що в 2018 році у Слов'янську в 2018 році затримали майже 100 сепаратистів.

**Gold:** ["Слов'янську"]

**Prediction:** []

**Focal surface:** Слов'янську; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 081. FN · row_id=1008292 · detection_miss

Після групового зґвалтування дівчини в автобусі в Делі 2012 року, яке шокувало всю країну, було посилено покарання за подібні злочини.

**Gold:** ["Делі"]

**Prediction:** []

**Focal surface:** Делі; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 082. FN · row_id=1008331 · name_fragment_policy

Сульшер зустрівся з агентами Холанда, щоб переконати юного земляка обрати "Олд Траффорд", пише "

**Gold:** ["Траффорд"]

**Prediction:** []

**Focal surface:** Траффорд; pilot fit frequency: 0; UNK: True.

**Review:** The text contains Old Trafford, while gold includes only its second component. Check both object-name boundaries and metonymic usage.

## 083. FN · row_id=1007575 · detection_miss

Його знайшли у штаті Вайомінг 2009 року.

**Gold:** ["Вайомінг"]

**Prediction:** []

**Focal surface:** Вайомінг; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 084. FN · row_id=1008521 · possible_annotation_error

Переведення дітей на дистанційну освіту та подовження канікул мають враховувати можливості догляду за ними з боку батьків у цей період - Шлапак.

**Gold:** ["Шлапак"]

**Prediction:** []

**Focal surface:** Шлапак; pilot fit frequency: 0; UNK: True.

**Review:** The final surname in the headline looks like attribution to a speaker, rather than a geographic entity.

## 085. FN · row_id=1005340 · hyphen_granularity

Недалеко від Леона знаходиться вулкан Серро-Негро.

**Gold:** ["Серро", "Негро"]

**Prediction:** ["Негро"]

**Focal surface:** Серро; pilot fit frequency: 0; UNK: True.

**Review:** The volcano name is annotated as two entities, unlike some other hyphenated names.

## 086. FN · row_id=1009019 · possible_type_error

На кого ставлять букмекери в матчі Вашингтон Кепіталз - Сан-Хосе Шаркс?

**Gold:** ["Вашингтон Кепіталз", "Сан-Хосе Шаркс"]

**Prediction:** ["букмекери"]

**Focal surface:** Сан-Хосе Шаркс; pilot fit frequency: 0; UNK: True.

**Review:** Full sports-team names in a match context appear as LOC gold. This is not an unambiguous failure to find a city.

## 087. FN · row_id=1003834 · possible_annotation_omission

Дві в Каліфорнії, Канзасі та Орегоні, і по одній в Іллінойсі, Індіані, Міннесоті, Міссурі, Флориді, Джорджії, Міссісіпі та Небрасці.

**Gold:** ["Каліфорнії", "Канзасі", "Орегоні", "Іллінойсі", "Індіані", "Міннесоті"]

**Prediction:** ["Каліфорнії", "Іллінойсі", "Індіані", "Міннесоті", "Міссурі", "Флориді", "Джорджії", "Міссісіпі", "Небрасці"]

**Focal surface:** Орегоні; pilot fit frequency: 0; UNK: True.

**Review:** Only some state names in a long list are annotated. Additional reasonable predictions can therefore count as FP.

## 088. FN · row_id=1009861 · detection_miss

У Молдові тих, хто незаконно зберігав зброю, було зобов'язано звернутися до поліції для перевірки непричетності до здійснення злочинів, після чого повертали її власникам з відповідним дозволом.

**Gold:** ["Молдові"]

**Prediction:** []

**Focal surface:** Молдові; pilot fit frequency: 1; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 1; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 089. FN · row_id=1001890 · detection_miss

А коли має виходити якась книжка у видавництві, ходжу з нею по Бессарабці і показую людям: "Вам це більше подобається чи це?"

**Gold:** ["Бессарабці"]

**Prediction:** []

**Focal surface:** Бессарабці; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 090. FN · row_id=1001457 · detection_miss

Шість золотих зірок символізують шість містечок – центрів козацьких сотень - Седнівської, Олишівської, полкової сотні, Слабинської, Роїщенської, Білоуської.

**Gold:** ["Седнівської", "Олишівської", "Слабинської", "Роїщенської", "Білоуської"]

**Prediction:** []

**Focal surface:** Білоуської; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 091. FN · row_id=1000941 · detection_miss

Рєзвой став першим нашим співвітчизником і двадцятою людиною в світі, яка самотужки пересікла на веслах Атлантику.

**Gold:** ["Атлантику"]

**Prediction:** []

**Focal surface:** Атлантику; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 092. FN · row_id=1000090 · insufficient_context

У перекладі з тюркської Каракурт означає «чорний вовк».

**Gold:** ["Каракурт"]

**Prediction:** []

**Focal surface:** Каракурт; pilot fit frequency: 0; UNK: True.

**Review:** A single phrase explaining the translation of this name provides insufficient context to confirm its geographic type.

## 093. FN · row_id=1003290 · unusual_landmark_names

Кожна зі скель має свою незвичну назву: Голець, Колобок, Австрійка, Парус, Тюльпан, Бівачка, Броненосець та Шуршун.

**Gold:** ["Голець", "Колобок", "Австрійка", "Парус", "Тюльпан", "Бівачка", "Броненосець", "Шуршун"]

**Prediction:** ["Голець"]

**Focal surface:** Шуршун; pilot fit frequency: 0; UNK: True.

**Review:** Ordinary words are proper names of rocks in this context. A city-only dictionary or a blanket common-word ban would hurt these cases.

## 094. FN · row_id=1008876 · detection_miss

Проросійські бойовики обстріляли Орлівку під Авдіївкою - на село впало 7 снарядів.

**Gold:** ["Орлівку", "Авдіївкою"]

**Prediction:** []

**Focal surface:** Орлівку; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 095. FN · row_id=1006509 · detection_miss

На привілейоване становище жителів гончарного посаду Вишгорода вказує і Дмитро Бібіков.

**Gold:** ["Вишгорода"]

**Prediction:** ["привілейоване"]

**Focal surface:** Вишгорода; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 096. FN · row_id=1000580 · detection_miss

Високий суд Лондона визнав "нестандартними" операції Ігоря Коломойського щодо фінансування юридичних та інших витрат аеропорту Бельбек (Севастополь, АРК) і зобов'язав українського бізнесмена узгодити подібні платежі.

**Gold:** ["Лондона", "Бельбек", "Севастополь", "АРК"]

**Prediction:** []

**Focal surface:** Бельбек; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 097. FN · row_id=1002396 · detection_miss

"Отримані 45-річним жителем села Іванчі тілесні ушкодження у вигляді рубаних ран виявилися несумісні із життям.

**Gold:** ["Іванчі"]

**Prediction:** []

**Focal surface:** Іванчі; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 098. FN · row_id=1003222 · detection_miss

Подібним шляхом розвиває промоцію своєї країни й Грузія.

**Gold:** ["Грузія"]

**Prediction:** []

**Focal surface:** Грузія; pilot fit frequency: 3; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 3; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 099. FN · row_id=1002278 · detection_miss

Головні програмні постулати є такими: «ні» для «Брексіту» й «так» для незалежної Шотландії.

**Gold:** ["Шотландії"]

**Prediction:** []

**Focal surface:** Шотландії; pilot fit frequency: 1; UNK: False.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 1; UNK in the mention: False. Novelty/OOV are observed factors, not proven causes.

## 100. FN · row_id=1000740 · detection_miss

А ще фахівці переконані, що Віталій Марків, сидячи в своєму окопі на горі Карачун, не міг знати про приїзд цивільних.

**Gold:** ["Карачун"]

**Prediction:** []

**Focal surface:** Карачун; pilot fit frequency: 0; UNK: True.

**Review:** The annotated location has no overlap with any prediction. Pilot training frequency: 0; UNK in the mention: True. Novelty/OOV are observed factors, not proven causes.

## 101. FP · row_id=1006351 · possible_annotation_error

Відповідно до Постанови ВРУ від 12.09.91 р. №1545-XII «Про порядок тимчасової дії на території України окремих актів законодавства Союзу РСР».

**Gold:** ["України окр"]

**Prediction:** ["України окремих актів законодавства", "Союзу РСР"]

**Focal surface:** України окремих актів законодавства; pilot fit frequency: 0; UNK: False.

**Review:** Gold ends inside the word following Ukraine. The model also overextends the span; these two problems should be separated.

## 102. FP · row_id=1006206 · split_location

Президент Сполучених Штатів Америки Дональд Трамп не зміг завчасно визначитися зі святковим подарунком своїй дружині Меланії на свято Різдва, котре більшість християн світу відзначають 25 грудня.

**Gold:** ["Сполучених Штатів Америки"]

**Prediction:** ["Сполучених Штатів", "Америки"]

**Focal surface:** Америки; pilot fit frequency: 3; UNK: False.

**Review:** The complete United States name is split into two predictions.

## 103. FP · row_id=1004942 · extra_neighbor

Вперше з сольним концертом у Івано-Франківську вокально-інструментальний ансамбль LATEXFAUNA з романтичною програмою “Ти мені нравишся”.

**Gold:** ["Івано-Франківську"]

**Prediction:** ["сольним", "Івано-Франківську вокально", "інструментальний", "романтичною"]

**Focal surface:** Івано-Франківську вокально; pilot fit frequency: 0; UNK: True.

**Review:** The city absorbs the first component of the following hyphenated adjective.

## 104. FP · row_id=1007081 · quote_convention

За словами координаторки заходу Наталі Альохіної, акцію влаштували задля того, щоб "привернути увагу громадськості до російської агресії в Україні й нагадати австралійцям про геополітичний злочин Кремля – окупацію Криму".

**Gold:** ["Україні", "Кремля", "Криму"]

**Prediction:** ["Україні", "Кремля", "окупацію", "Криму\""]

**Focal surface:** Криму"; pilot fit frequency: 0; UNK: False.

**Review:** An extra closing quote causes a real string mismatch under the published scorer.

## 105. FP · row_id=1009719 · address_components

• З огляду на поверховість існуючої навколишньої забудови, планована поверховість адміністративно-виробничого центру з офісно-адміністративними приміщеннями на вул.Володимира Великого, 2-4 у м.Львові становить 7 поверхів.

**Gold:** ["вул.Володимира Великого, 2-4", "Львові"]

**Prediction:** ["поверховість", "вул.Володимира Великого", "Львові"]

**Focal surface:** вул.Володимира Великого; pilot fit frequency: 0; UNK: False.

**Review:** The address boundary excludes its comma and building-number suffix. Digit removal does not guarantee a match because punctuation also matters.

## 106. FP · row_id=1008775 · address_and_PER

Борис Егіазорян (Київ), Гарик Куригиян (Запоріжжя), Сейрак Маргариян (Полтава) ARTEast вул. Рейтарська, 13 тел. 229-65-52 “Годинник і миті”.

**Gold:** ["Київ", "Запоріжжя", "Полтава", "вул. Рейтарська, 13"]

**Prediction:** ["Егіазорян", "Київ", "Гарик", "Куригиян", "Запоріжжя", "Сейрак", "Маргариян", "вул. Рейтарська"]

**Focal surface:** вул. Рейтарська; pilot fit frequency: 0; UNK: True.

**Review:** The address loses its number and comma; artist names in the same sentence are also emitted as LOC.

## 107. FP · row_id=1000978 · quote_and_entity_type

В адміністративному корпусі аеропорту "Львів" імені Данила Галицького працівників поліції та Служби безпеки України у Львівській області провели обшуки.

**Gold:** ["Львів", "Львівській області"]

**Prediction:** ["адміністративному", "аеропорту", "Львів\"", "імені", "Данила", "Галицького", "України", "Львівській області"]

**Focal surface:** Львів"; pilot fit frequency: 0; UNK: False.

**Review:** An extra quote follows Lviv; person names and institution words also generate FP in this sentence.

## 108. FP · row_id=1005984 · missing_multiword_tail

В Міністерстві закордонних справ України очікують, що Російська Федерація швидко звільнить захоплених у листопаді 2018 року 24 українських моряків і три їхні судна.

**Gold:** ["Російська Федерація"]

**Prediction:** ["України", "Російська"]

**Focal surface:** Російська; pilot fit frequency: 0; UNK: False.

**Review:** Only the adjective remains from the full country name.

## 109. FP · row_id=1007792 · extra_neighbor

Незважаючи на обмеження помаранчевої зони, 5 жовтня в Палаці студентів на вулиці Пушкінській відбувся позаплановий концерт народної артистки, відділ комунікації поліції Харківської області.

**Gold:** ["студентів", "вулиці Пушкінській", "Харківської області"]

**Prediction:** ["помаранчевої зони", "Палаці студентів", "вулиці Пушкінській відбувся"]

**Focal surface:** вулиці Пушкінській відбувся; pilot fit frequency: 0; UNK: True.

**Review:** The street name absorbs the following verb; the object boundary is not separated from the next event.

## 110. FP · row_id=1005494 · hyphen_convention

В штабі антитерористичної операції спростовують інформацію, поширену в російських ЗМІ, що українські військові нібито зайняли ділянку у буферній зоні у траси Ясинувата-Горлівка і обстрілюють цивільний транспорт.

**Gold:** ["Ясинувата", "Горлівка"]

**Prediction:** ["штабі", "буферній", "траси", "Ясинувата-", "Горлівка", "обстрілюють"]

**Focal surface:** Ясинувата-; pilot fit frequency: 0; UNK: True.

**Review:** The route-separating hyphen is appended to the first city.

## 111. FP · row_id=1009674 · boundary_convention

Найбільше випадків захворювання виявили нині в Дніпровському районі міста — 21, у Дарницькому та Солом’янському – по 16 та в Шевченківському — 10 випадків.

**Gold:** ["Дніпровському районі міста", "Дарницькому", "Солом’янському", "Шевченківському"]

**Prediction:** ["Дніпровському районі", "Дарницькому", "Солом’янському", "Шевченківському"]

**Focal surface:** Дніпровському районі; pilot fit frequency: 0; UNK: True.

**Review:** Gold includes the generic city word, while the prediction stops at district. Clarify generic descriptors in the annotation rules.

## 112. FP · row_id=1000085 · possible_annotation_error

«Впродовж останніх засідань ТКГ в Мінську Україна наполягала на створенні постійного пункту моніторингу СММ в Керчі.

**Gold:** ["Мінську Україна", "Керчі"]

**Prediction:** ["Мінську", "Україна", "Керчі"]

**Focal surface:** Україна; pilot fit frequency: 92; UNK: False.

**Review:** Gold merges Minsk and Ukraine across an adverbial/subject boundary. These are candidates for separate entities.

## 113. FP · row_id=1009660 · extra_neighbor

Наступного дня в районі вулиці Молодіжна у місті Лозовій слідчі виявили на дорозі сліди, характерні для дорожньо-транспортної пригоди.

**Gold:** ["вулиці Молодіжна", "Лозовій"]

**Prediction:** ["вулиці Молодіжна", "Лозовій слідчі виявили"]

**Focal surface:** Лозовій слідчі виявили; pilot fit frequency: 0; UNK: True.

**Review:** The city absorbs the subject and verb of the following clause.

## 114. FP · row_id=1002635 · missing_modifier

Так само, як можна малоруську течію у Східній Україні.

**Gold:** ["Східній Україні"]

**Prediction:** ["Україні"]

**Focal surface:** Україні; pilot fit frequency: 187; UNK: False.

**Review:** The geographic modifier is dropped from Eastern Ukraine.

## 115. FP · row_id=1004613 · extra_neighbor_material

Рівень хокейного чемпіонату в Білорусі доволі хороший.

**Gold:** ["Білорусі"]

**Prediction:** ["Білорусі доволі"]

**Focal surface:** Білорусі доволі; pilot fit frequency: 0; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 116. FP · row_id=1000650 · hyphen_granularity

В автомобілі їхали двоє громадян України, 37-річний і 16-річний жителі Івано-Франківська.

**Gold:** ["України", "Івано-Франківська"]

**Prediction:** ["України", "Івано", "-Франківська"]

**Focal surface:** -Франківська; pilot fit frequency: 1; UNK: False.

**Review:** Gold requires a complete hyphenated city here, unlike several FN cases where the same city is split.

## 117. FP · row_id=1007643 · hyphen_split

Як повідомляв УНІАН, раніше в Тель-Авіві відбулися перші репетиції Євробачення 2019, після яких букмекери оновили прогнози щодо переможця конкурсу.

**Gold:** ["Тель-Авіві"]

**Prediction:** ["Тель"]

**Focal surface:** Тель; pilot fit frequency: 0; UNK: True.

**Review:** Only the first part of Tel Aviv remains.

## 118. FP · row_id=1000666 · boundary_convention

У місті Бені на сході Демократичної Республіки Конго, 25 листопада протестувальники увірвалися до будівлі миротворчого корпусу ООН і підпалили міську ратушу.

**Gold:** ["Бені", "Демократичної Республіки Конго"]

**Prediction:** ["місті", "Бені", "сході Демократичної Республіки Конго"]

**Focal surface:** сході Демократичної Республіки Конго; pilot fit frequency: 0; UNK: True.

**Review:** A generic directional component is added to the country name.

## 119. FP · row_id=1008399 · possible_annotation_error

Працівниця комісійного магазину в місті Кітті-Хок, знайшла серед товару, що надійшов до крамниці, картину Сальвадора Далі.

**Gold:** ["Кітті"]

**Prediction:** ["Кітті-Хок"]

**Focal surface:** Кітті-Хок; pilot fit frequency: 0; UNK: True.

**Review:** The model finds the complete hyphenated city after the word city; gold labels the first part LOC and the second part PER.

## 120. FP · row_id=1009811 · missing_name_material

"Передаємо Кривому Рогу і Криворізького району партію обладнання та засобів захисту для медиків.

**Gold:** ["Кривому Рогу", "Криворізького району"]

**Prediction:** ["Криворізького"]

**Focal surface:** Криворізького; pilot fit frequency: 0; UNK: True.

**Review:** The generic district word is dropped, and Kryvyi Rih is also missed.

## 121. FP · row_id=1001777 · hyphen_split

Мажоритарник з округу в Івано-Франківскій області 5 листопада порушив вимоги щодо персонального голосування.

**Gold:** ["Івано-Франківскій області"]

**Prediction:** ["Івано", "-Франківскій області"]

**Focal surface:** Івано; pilot fit frequency: 15; UNK: False.

**Review:** The hyphenated region name is split into two predictions.

## 122. FP · row_id=1005373 · missing_name_material

На сході країни та у Карпатському регіоні вночі на дорогах місцями ожеледиця.

**Gold:** ["Карпатському регіоні"]

**Prediction:** ["сході країни", "Карпатському"]

**Focal surface:** Карпатському; pilot fit frequency: 0; UNK: True.

**Review:** The generic region word is lost; a generic eastern-part description is additionally predicted as a name.

## 123. FP · row_id=1008203 · punctuation_gold_policy

Детективи НАБУ під процесуальним керівництвом САП здійснюють низку обшуків в Одесі, зокрема, у будівлі, де розташовано ТРК "Думська" через можливі зловживання посадовцями Одеської облради.

**Gold:** ["Одесі, "]

**Prediction:** ["Одесі", "будівлі"]

**Focal surface:** Одесі; pilot fit frequency: 15; UNK: False.

**Review:** Gold contains Odesa followed by a comma and a space. The city is found, but the scorer treats its string as different.

## 124. FP · row_id=1004470 · possible_annotation_error

Українська біатлоністкастала бронзовою призеркою гонки з масовим стартом на шостому етапі Кубка IBU в італійському Мартел-валь-Мартелло.

**Gold:** ["Мартел-валь-Мартел"]

**Prediction:** ["стартом", "шостому", "італійському", "валь", "Мартелло"]

**Focal surface:** валь; pilot fit frequency: 0; UNK: True.

**Review:** Gold ends inside the final component of the name; the model also fragments it. This cannot be reduced to an unknown-word problem.

## 125. FP · row_id=1001290 · extra_neighbor_material

Цього року відпочинок в Карпатах здорожчав на 15-30%.

**Gold:** ["Карпатах"]

**Prediction:** ["Карпатах здорожчав"]

**Focal surface:** Карпатах здорожчав; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 126. FP · row_id=1000550 · hyphen_split

Нагадаємо, 29 листопада стало відомо, що Українська православна церква (Московського патріархату) заявила, що Міністерство культури ініціювало перевірку наявних церковних коштовностей і святинь Києво-Печерської лаври.

**Gold:** ["Києво-Печерської лаври"]

**Prediction:** ["Києво", "Печерської лаври"]

**Focal surface:** Києво; pilot fit frequency: 1; UNK: False.

**Review:** The Kyiv-Pechersk Lavra name is split into two entities.

## 127. FP · row_id=1002601 · missing_modifier

На них населення Західної України мало вибрати з-поміж себе тих, хто від його імені проситиме Москву прийняти його, звісно – разом із територіями, до складу СРСР.

**Gold:** ["Західної України", "Москву", "СРСР"]

**Prediction:** ["України", "Москву", "територіями", "СРСР"]

**Focal surface:** України; pilot fit frequency: 266; UNK: False.

**Review:** The western modifier is omitted. An additional generic territory noun is not a named place in this context.

## 128. FP · row_id=1005344 · boundary_convention

Нагадаємо, під час обшуку приватного літака, на якому 21-річний виконавець прилетів в аеропот Чикаго, виявили понад 30 кілограмів марихуани.

**Gold:** ["Чикаго"]

**Prediction:** ["аеропот Чикаго"]

**Focal surface:** аеропот Чикаго; pilot fit frequency: 0; UNK: True.

**Review:** A misspelled generic airport word is included before Chicago.

## 129. FP · row_id=1007297 · boundary_convention

Скандал навколо Театру на Подолі - Korrespondent.

**Gold:** ["на Подолі"]

**Prediction:** ["Подолі"]

**Focal surface:** Подолі; pilot fit frequency: 2; UNK: False.

**Review:** Gold includes a preposition, while only the place name is predicted. Specify whether prepositions belong in names.

## 130. FP · row_id=1008497 · hyphen_split

Напруженою лишається ситуація в туніському місті Зіді-Бузід, звідки торік стартувала “арабська весна”.

**Gold:** ["Зіді-Бузід"]

**Prediction:** ["Зіді", "Бузід"]

**Focal surface:** Зіді; pilot fit frequency: 0; UNK: True.

**Review:** The hyphenated city name is split into two parts.

## 131. FP · row_id=1006710 · missing_modifier

"Якщо від Росії відділити території, які були приєднані до неї через кров, насилля, зраду і брехню, то "ісконно руські" болота, що залишаться після цього, можуть легко вміститись у межі однієї Московської області"

**Gold:** ["Росії", "Московської області"]

**Prediction:** ["Росії", "області"]

**Focal surface:** області; pilot fit frequency: 0; UNK: False.

**Review:** Only the generic region word remains; the proper modifier is lost.

## 132. FP · row_id=1005957 · hyphen_and_punctuation

Світова першість триватиме з 1 до 10 лютого в американському Парк-Сіті (штат Юта).

**Gold:** ["Парк-Сіті", "Юта"]

**Prediction:** ["Парк-", "Сіті (", "Юта"]

**Focal surface:** Сіті (; pilot fit frequency: 1; UNK: False.

**Review:** The hyphenated city is fragmented, and a bracket is added to one component.

## 133. FP · row_id=1008905 · quote_convention

Андрій Волокітін: "Маю надію, що залишусь у Львові"

**Gold:** ["Львові"]

**Prediction:** ["Львові\""]

**Focal surface:** Львові"; pilot fit frequency: 0; UNK: False.

**Review:** A closing quote is attached to the Lviv name.

## 134. FP · row_id=1008014 · extra_neighbor

Згідно з указом, найбільше таких матерів проживає у Волинській області - 116, а найменше - у Кіровоградській (двоє).

**Gold:** ["Волинській області", "Кіровоградській"]

**Prediction:** ["Волинській області", "Кіровоградській (двоє"]

**Focal surface:** Кіровоградській (двоє; pilot fit frequency: 0; UNK: False.

**Review:** The span captures the beginning of a parenthetical explanation about a count of people.

## 135. FP · row_id=1002823 · extra_neighbor

У Полтавському районі лісівники та мисливці хочуть збільшити популяцію пернатих.

**Gold:** ["Полтавському районі"]

**Prediction:** ["Полтавському районі лісівники"]

**Focal surface:** Полтавському районі лісівники; pilot fit frequency: 0; UNK: True.

**Review:** The district absorbs the subject of the next statement.

## 136. FP · row_id=1000210 · boundary_convention

Доставка по місту Харків 400 грн, межгород 5 грн/км в обидві сторони.

**Gold:** ["Харків"]

**Prediction:** ["місту Харків"]

**Focal surface:** місту Харків; pilot fit frequency: 0; UNK: False.

**Review:** A generic city descriptor is added, while gold contains only Kharkiv.

## 137. FP · row_id=1009279 · extra_neighbor

Під консульством РФ у Харкові запалили шини та фаєри.

**Gold:** ["РФ", "Харкові"]

**Prediction:** ["РФ", "Харкові запалили шини", "фаєри"]

**Focal surface:** Харкові запалили шини; pilot fit frequency: 0; UNK: True.

**Review:** The city absorbs a verb and the object of the event.

## 138. FP · row_id=1007683 · possible_type_error

У Діми Кийкова рецидив лейкозу виявили у серпні минулого року і це після першої боротьби, яка тривала ще у 2014 році.

**Gold:** ["Діми Кийкова"]

**Prediction:** ["Діми"]

**Focal surface:** Діми; pilot fit frequency: 0; UNK: True.

**Review:** The gold name refers to a person in a leukemia context, but it is labeled LOC.

## 139. FP · row_id=1001690 · extra_neighbor_material

"Згідно зі словами віцеголови, в Угорщині розбудовується "хвора демократія", — написав Орбан у листі до голови Єврокомісії Урсули фон дер Ляєн, оприлюдненому у вівторок, 29 вересня.

**Gold:** ["Угорщині"]

**Prediction:** ["Угорщині розбудовується"]

**Focal surface:** Угорщині розбудовується; pilot fit frequency: 0; UNK: True.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 140. FP · row_id=1005345 · missing_modifier

Вони пролетіли над Карпатами та Чорним морем, над херсонськими степами та лісами Житомирщини.

**Gold:** ["Карпатами", "Чорним морем", "Житомирщини"]

**Prediction:** ["Карпатами", "морем", "Житомирщини"]

**Focal surface:** морем; pilot fit frequency: 0; UNK: False.

**Review:** The proper modifier Black is omitted, leaving only the generic sea word.

## 141. FP · row_id=1007390 · extra_neighbor

Іномарка їхала по вул. Великій Морській з боку вул. Московської в напрямку Садовій.

**Gold:** ["вул. Великій Морській", "вул. Московської", "Садовій"]

**Prediction:** ["вул. Великій Морській", "вул. Московської в напрямку", "Садовій."]

**Focal surface:** вул. Московської в напрямку; pilot fit frequency: 0; UNK: True.

**Review:** The street absorbs a directional transition before another name.

## 142. FP · row_id=1003947 · coordinated_locations

Задуха, потяги без кондиціонерів та маршрутки без пасків безпеки - ось, що чекає відпочивальників, які планують поїхати на узбережжя Чорного або Азовського морів та у Карпати.

**Gold:** ["Чорного або Азовського морів", "Карпати"]

**Prediction:** ["Чорного", "Карпати"]

**Focal surface:** Чорного; pilot fit frequency: 0; UNK: False.

**Review:** Gold combines an alternative list of seas, while the model keeps only the first modifier.

## 143. FP · row_id=1005839 · extra_neighbor_material

В Японії свіжим і малосольним часником лікують рак, він є необхідною складовою частиною всіх протиракових лікувальних заходів.

**Gold:** ["Японії"]

**Prediction:** ["Японії свіжим"]

**Focal surface:** Японії свіжим; pilot fit frequency: 0; UNK: False.

**Review:** The focal span's boundary differs from its counterpart; observed relation: extra_neighbor_material. Original offsets establish this relation, not its cause.

## 144. FP · row_id=1006189 · generic_word_FP

В основному зміни стосуються напрямків всередині провінцій на сході Канади, а також вильотів в північно-східні райони США.

**Gold:** ["Канади", "США"]

**Prediction:** ["сході", "Канади", "вильотів", "північно-східні", "США"]

**Focal surface:** вильотів; pilot fit frequency: 0; UNK: True.

**Review:** The focal word describes departures/flights, not a named place.

## 145. FP · row_id=1006983 · generic_word_FP

"На переговорах з МВФ за новою програмою знову включають пункт про забезпечення ефективної роботи незалежних судів.

**Gold:** []

**Prediction:** ["переговорах"]

**Focal surface:** переговорах; pilot fit frequency: 0; UNK: False.

**Review:** Negotiations are an event; this sentence has no LOC gold.

## 146. FP · row_id=1000692 · generic_place_FP

Вона змайструвала собі саморобний парашут зі старої маминої парасолі та простирадла, піднялася з ним на горище і стрибнула.

**Gold:** []

**Prediction:** ["горище", "стрибнула"]

**Focal surface:** горище; pilot fit frequency: 0; UNK: True.

**Review:** An unnamed attic is a generic place, not a target geographic named entity.

## 147. FP · row_id=1000234 · generic_word_FP

Нападнику Динамо Артему Бєсєдіну зробили операцію на ахілловому сухожиллі.

**Gold:** []

**Prediction:** ["ахілловому"]

**Focal surface:** ахілловому; pilot fit frequency: 0; UNK: True.

**Review:** An adjective from an anatomical term is incorrectly emitted as LOC.

## 148. FP · row_id=1003276 · numeric_normalization

У тому числі оформили на 10 тисяч тонн більше імпортних вантажів та на 166,5 тисячі — експортних.

**Gold:** []

**Prediction:** ["166"]

**Focal surface:** 166; pilot fit frequency: 1; UNK: True.

**Review:** The number 166 is incorrectly extracted. It normalizes to an empty string, which still counts as FP rather than an empty list.

## 149. FP · row_id=1000594 · generic_word_FP

Але варто пам’ятати, що при проблемах зі шлунком і кишечником, а також захворюваннях підшлункової, нирок і серцево-судинної системи вживати редьку потрібно з великою обережністю, так як це може привести до виникнення болю.

**Gold:** []

**Prediction:** ["шлунком", "кишечником"]

**Focal surface:** шлунком; pilot fit frequency: 0; UNK: True.

**Review:** The stomach and intestine words are anatomical terms, not geography.

## 150. FP · row_id=1003995 · nested_ORG_policy

У МВС Казахстану повідомили, що на місці катастрофи працюють близько тисячі людей і 70 одиниць техніки, а також 33 бригади швидкої допомоги.

**Gold:** []

**Prediction:** ["Казахстану"]

**Focal surface:** Казахстану; pilot fit frequency: 2; UNK: False.

**Review:** Kazakhstan is part of the annotated ministry name. This may reflect a non-nested LOC policy rather than confusion of the entire institution with a place.

## 151. FP · row_id=1001590 · generic_word_FP

Для тих, хто працював тоді в журналі, вона була «літературною мамою»: підказувала, багато допомагала.

**Gold:** []

**Prediction:** ["підказувала"]

**Focal surface:** підказувала; pilot fit frequency: 0; UNK: True.

**Review:** The focal verb does not denote a place.

## 152. FP · row_id=1004493 · generic_word_FP

Бачила, як старі жінки у пічці-буржуйці сушать яблука на компот.

**Gold:** []

**Prediction:** ["пічці", "компот"]

**Focal surface:** компот; pilot fit frequency: 0; UNK: True.

**Review:** Compote is an ordinary noun in a food-preparation context.

## 153. FP · row_id=1006836 · generic_word_FP

Євро і долар знову подешевшали в курсах НБУ - Гроші - TCH.

**Gold:** []

**Prediction:** ["курсах"]

**Focal surface:** курсах; pilot fit frequency: 0; UNK: True.

**Review:** The focal word refers to an exchange rate, not a geographic object.

## 154. FP · row_id=1005037 · generic_word_FP

А морозива цього року на святі було багато видів.

**Gold:** []

**Prediction:** ["святі"]

**Focal surface:** святі; pilot fit frequency: 0; UNK: False.

**Review:** The focal word denotes an event without a proper place name.

## 155. FP · row_id=1008120 · generic_region_FP

Соціологи опитали телефоном 2 472 респондентів в усіх областях, крім окупованих Росією Криму та районів Донеччини і Луганщини.

**Gold:** ["Росією", "Криму", "Донеччини", "Луганщини"]

**Prediction:** ["усіх областях", "Росією", "Криму", "Донеччини", "Луганщини."]

**Focal surface:** усіх областях; pilot fit frequency: 0; UNK: False.

**Review:** An all-regions phrase is a generic description, not the proper name of a region.

## 156. FP · row_id=1002808 · ORG_as_LOC

У селі Онишківці Шумського району на Тернопільщині 17-річний хлопець викрав із двору ВАЗ-2106 і поїхав додому в райцентр.

**Gold:** ["Онишківці", "Шумського району", "Тернопільщині"]

**Prediction:** ["Онишківці", "Шумського району", "Тернопільщині", "ВАЗ", "2106", "райцентр"]

**Focal surface:** ВАЗ; pilot fit frequency: 0; UNK: False.

**Review:** The vehicle brand/model is annotated ORG; the model emits a fragment as LOC.

## 157. FP · row_id=1004789 · geographic_adjective_FP

Втримається 27 вересня в Україні й сильний південно-східний вітер.

**Gold:** ["Україні"]

**Prediction:** ["Україні", "південно"]

**Focal surface:** південно; pilot fit frequency: 0; UNK: False.

**Review:** A south-easterly wind describes direction, not a named location.

## 158. FP · row_id=1001672 · generic_word_FP

Експерти, як розуміються на специфіці регіону, вважають, що за минулі п'ять років в області не стільки додалося прибічників федералізму, скільки зменшилося тих, хто боїться відповідати на це питання чесно.

**Gold:** []

**Prediction:** ["специфіці"]

**Focal surface:** специфіці; pilot fit frequency: 0; UNK: True.

**Review:** The focal abstract noun does not become a place name merely because a region is mentioned nearby.

## 159. FP · row_id=1008393 · generic_word_FP

У центрі Києва чоловіка прикували наручниками до поручнів та пограбували - Київ - TCH.

**Gold:** ["Києва", "Київ"]

**Prediction:** ["Києва", "поручнів", "пограбували", "Київ"]

**Focal surface:** поручнів; pilot fit frequency: 0; UNK: True.

**Review:** Railings are an object, not a geographic name.

## 160. FP · row_id=1005941 · generic_place_FP

Цього так бракує особливо взимку тим, хто живе на вулиці.

**Gold:** []

**Prediction:** ["вулиці."]

**Focal surface:** вулиці.; pilot fit frequency: 2; UNK: False.

**Review:** An unnamed street is not a named address.

## 161. FP · row_id=1006381 · generic_word_FP

«Патрулювання правоохоронці провели в столиці на багатолюдних об’єктах.

**Gold:** []

**Prediction:** ["багатолюдних"]

**Focal surface:** багатолюдних; pilot fit frequency: 0; UNK: True.

**Review:** The focal adjective describes crowded objects rather than naming a place.

## 162. FP · row_id=1000562 · generic_word_FP

Після того чоловіка в ЛКП до паспортистки по довідку форми №2 звернулася онука дідуся, але вона не мала при собі документів, тому паспортистка не могла видати їй довідку, але розповіла, що таку довідку вже видавала на ім’я іншої людини.

**Gold:** []

**Prediction:** ["ЛКП", "паспортистки", "довідку"]

**Focal surface:** паспортистки; pilot fit frequency: 0; UNK: True.

**Review:** A profession is not LOC; the neighboring institution abbreviation and certificate word also trigger extra predictions.

## 163. FP · row_id=1003404 · generic_word_FP

Я дійсно вважаю, що довідка у нарколога, у психолога для людей, які влаштовуються на держслужбу, вони не потрібні.

**Gold:** []

**Prediction:** ["нарколога", "держслужбу"]

**Focal surface:** нарколога; pilot fit frequency: 0; UNK: True.

**Review:** A medical specialist is a person by occupation, not a location.

## 164. FP · row_id=1007360 · generic_place_FP

На Камчатці (Росія) однієї з частин викинули в болото невитрачені після стрільб боєприпаси “з метою створення видимості якісного проведення занять”.

**Gold:** ["Камчатці", "Росія"]

**Prediction:** ["Камчатці", "Росія", "болото"]

**Focal surface:** болото; pilot fit frequency: 0; UNK: True.

**Review:** The swamp word is a generic terrain type without a proper name.

## 165. FP · row_id=1006727 · ORG_as_LOC

Міністр фінансів також наголосив що у Держбюджеті на 2021 рік вперше передбачені кошти на забезпечення транспортної інфраструктури (будівництво доріг), купівлю нових пасажирських вагонів та збільшення коштів на децентралізацію та оборону.

**Gold:** []

**Prediction:** ["Держбюджеті"]

**Focal surface:** Держбюджеті; pilot fit frequency: 0; UNK: True.

**Review:** The focal budget term matches an auxiliary ORG label; the context concerns a budget, not a place.

## 166. FP · row_id=1006654 · generic_word_FP

В обласному музеї, який після переміщення з Луганська розташувався у Старобільську, можна оглянути спеціальну виставку «Гордість твоя, Луганщино», яку працівники створили до 80-річчя утворення області.

**Gold:** ["Луганська", "Старобільську"]

**Prediction:** ["Луганська", "Старобільську", "твоя", "утворення області"]

**Focal surface:** твоя; pilot fit frequency: 0; UNK: False.

**Review:** A possessive pronoun in an exhibition title is not LOC.

## 167. FP · row_id=1003031 · nonlocation_named_item

"Є запис командира корабля "Бердянська", де він указує: "Я не роблю протиправні дії, я не повертаю свої установки проти вас.

**Gold:** []

**Prediction:** ["Бердянська"]

**Focal surface:** Бердянська; pilot fit frequency: 1; UNK: False.

**Review:** The focal geographic-looking name refers to a ship here, not the city.

## 168. FP · row_id=1002968 · possible_annotation_omission

Пасажирський літак, що летів з Києва, здійснив аварійну посадку в талліннському аеропорту через руйнування покришки шасі повітряного судна.

**Gold:** ["Києва"]

**Prediction:** ["Києва", "талліннському аеропорту", "покришки"]

**Focal surface:** талліннському аеропорту; pilot fit frequency: 0; UNK: True.

**Review:** Tallinn airport looks semantically like a location but is absent from gold. Check completeness and the treatment of objects with geographic adjectives.

## 169. FP · row_id=1006499 · ORG_as_LOC

Про це повідомив народний депутат від ПППУ (фракція «Наша Україна») Лев ГНАТЕНКО сьогодні на прес-конференції в УНІАН.

**Gold:** []

**Prediction:** ["ПППУ", "Україна", "УНІАН"]

**Focal surface:** УНІАН; pilot fit frequency: 0; UNK: False.

**Review:** The news agency is annotated ORG and has no LOC gold.

## 170. FP · row_id=1002810 · ORG_metonymy_policy

У Кабміні також нагадали, що для отримання субсидії необхідно подати заяву і декларацію про доходи до місцевих управлінь праці та соціальної політики, органи соцзахисту або через інтернет.

**Gold:** []

**Prediction:** ["Кабміні"]

**Focal surface:** Кабміні; pilot fit frequency: 0; UNK: True.

**Review:** The cabinet is an institution, sometimes used metonymically for its building. Clarify the policy rather than treating every such string as a city.

## 171. FP · row_id=1004964 · nested_ORG_policy

Голова Адміністрації президента України Андрій Богдан заявив, що влада допускає винесення на всенародний референдум питання про укладення мирних домовленостей з РФ.

**Gold:** ["РФ"]

**Prediction:** ["України", "РФ"]

**Focal surface:** України; pilot fit frequency: 266; UNK: False.

**Review:** The country is included only within the complete presidential-administration ORG span.

## 172. FP · row_id=1007522 · possible_annotation_omission

Президент України Петро Порошенко закликав Норвегію приєднатися до міжнародної кампанії зі звільнення українських політв’язнів, які перебувають за ґратами в Росії.

**Gold:** ["Росії"]

**Prediction:** ["України", "Росії"]

**Focal surface:** України; pilot fit frequency: 266; UNK: False.

**Review:** The country in a presidential title is not labeled LOC, and Norway is included in PER. This sentence contains candidates for type-label noise.

## 173. FP · row_id=1009823 · generic_word_FP

Туреччина ґрунтується на праві на самооборону.

**Gold:** []

**Prediction:** ["Туреччина", "праві", "самооборону"]

**Focal surface:** праві; pilot fit frequency: 0; UNK: True.

**Review:** The focal word denotes a legal right, not a place. Turkey at the start is also unannotated, so a possible gold omission coexists in this text.

## 174. FP · row_id=1000142 · generic_word_FP

Пожежа у "Вікторії": працівники табору вирішували, в яких корпусах розміщувати дітей - Укрaїнa - TCH.

**Gold:** ["Укрaїнa"]

**Prediction:** ["працівники", "Укрaїнa"]

**Focal surface:** працівники; pilot fit frequency: 0; UNK: False.

**Review:** Camp employees are people, not a place name.

## 175. FP · row_id=1008146 · PER_as_LOC

У Данії знову вибухнув скандал через карикатури на пророка Мохаммеда.

**Gold:** ["Данії"]

**Prediction:** ["Данії", "Мохаммеда"]

**Focal surface:** Мохаммеда; pilot fit frequency: 0; UNK: True.

**Review:** The prophet's name matches a PER annotation but is predicted as LOC.

## 176. FP · row_id=1004418 · generic_word_FP

У туриста було необхідне спорядження для того, щоб переночувати в гірській місцевості без шкоди для здоров’я.

**Gold:** []

**Prediction:** ["туриста"]

**Focal surface:** туриста; pilot fit frequency: 0; UNK: True.

**Review:** A tourist is a person by role, not a location.

## 177. FP · row_id=1000671 · generic_place_FP

Перша доба після приймання ЛСД була для неї вкрай неприємною – її сусіди по кімнаті розповіли, що вона провела 24 години, сидячи в кріслі, часто непритомнюючи і блюючи, перш ніж прийшла до тями.

**Gold:** []

**Prediction:** ["кріслі"]

**Focal surface:** кріслі; pilot fit frequency: 0; UNK: True.

**Review:** A chair is an object/generic place, not named geography.

## 178. FP · row_id=1007618 · generic_word_FP

В тодішніх умовах послабити соціальне напруження можна було лише за рахунок чужинців.

**Gold:** []

**Prediction:** ["тодішніх"]

**Focal surface:** тодішніх; pilot fit frequency: 0; UNK: True.

**Review:** The temporal modifier is not a location named entity.

## 179. FP · row_id=1005242 · generic_word_FP

Так, на новому фото Леся Нікітюк постала у красивому пляжному образі.

**Gold:** []

**Prediction:** ["красивому"]

**Focal surface:** красивому; pilot fit frequency: 0; UNK: True.

**Review:** The evaluative adjective is not a geographic name.

## 180. FP · row_id=1009845 · generic_word_FP

Іван тікає й опиняється у затишному й безпечному домі своєї матері в Ужгороді, де його українські потуги швидко слабнуть.

**Gold:** ["Ужгороді"]

**Prediction:** ["затишному", "Ужгороді"]

**Focal surface:** затишному; pilot fit frequency: 0; UNK: True.

**Review:** The descriptive adjective before a house is not a name.

## 181. FP · row_id=1008575 · generic_word_FP

У цій статті відмітити, що наукова робота його, що направляється певною ідеологією дрібнобуржуазного націоналізму, штовхнула Грушевського в галузі політики на шлях, який призвів до краху.

**Gold:** []

**Prediction:** ["краху"]

**Focal surface:** краху; pilot fit frequency: 0; UNK: True.

**Review:** The focal word is an abstract consequence, not a location.

## 182. FP · row_id=1001476 · nonlocation_named_item

У Репортері – детальний гороскоп для всіх знаків.

**Gold:** []

**Prediction:** ["Репортері"]

**Focal surface:** Репортері; pilot fit frequency: 0; UNK: True.

**Review:** In the horoscope context, the focal name is more likely a publication or section title; LOC is not established.

## 183. FP · row_id=1006149 · generic_word_FP

Ця риба містить мінімум калорій, проте багато простагландину – сполуки, яка може регулювати обмін речовин у клітинах та запобігати появі здуття живота.

**Gold:** []

**Prediction:** ["клітинах"]

**Focal surface:** клітинах; pilot fit frequency: 0; UNK: True.

**Review:** The focal word refers to biological cells, not geography.

## 184. FP · row_id=1004753 · numeric_normalization

Анна народилася в 1984-му в Києві, в сім’ї зірки Динамо Володимира Безсонова і чемпіонки світу з художньої гімнастики в групових вправах Вікторії Сєдих.

**Gold:** ["Києві"]

**Prediction:** ["1984", "Києві", "’", "групових"]

**Focal surface:** 1984; pilot fit frequency: 1; UNK: True.

**Review:** 1984 is a birth year, not LOC; normalization to an empty string does not remove the FP from the metric.

## 185. FP · row_id=1007024 · generic_word_FP

Низка енергоекспертів заявляли, що наміри знизити рrice caps суттєво ускладнять можливість закупівлі вугілля й можуть спровокувати кризу в енергетиці.

**Gold:** []

**Prediction:** ["енергетиці"]

**Focal surface:** енергетиці; pilot fit frequency: 0; UNK: True.

**Review:** Energy is an industry here, not a named place.

## 186. FP · row_id=1008661 · PER_context_FP

Головна героїня мадам Жекіль - сором'язлива професорка фізики з коледжу в передмісті.

**Gold:** []

**Prediction:** ["мадам", "Жекіль", "передмісті"]

**Focal surface:** мадам; pilot fit frequency: 0; UNK: False.

**Review:** Madame is a form of address before a character's name, not a location.

## 187. FP · row_id=1004770 · ORG_as_LOC

Повідомляємо, що за ініціативи ПУ в АР у тісній взаємодії з Виконавчою владою Закатальского району АР здійснено заміну російськомовної таблиці на барельєфі Т.Г.Шевченка у місті Закатала на нову з написами українською та азербайджанською мовами.

**Gold:** ["Закатальского району", "Закатала"]

**Prediction:** ["АР", "тісній", "Виконавчою", "Закатальского району", "барельєфі Т", "Шевченка", "Закатала"]

**Focal surface:** Виконавчою; pilot fit frequency: 0; UNK: True.

**Review:** The adjective starts the name of a government body. The auxiliary ORG boundary also looks truncated.

## 188. FP · row_id=1007943 · generic_word_FP

За словами Голови Верховної Ради, робота над ними вже розпочалася у профільному комітеті.

**Gold:** []

**Prediction:** ["профільному"]

**Focal surface:** профільному; pilot fit frequency: 0; UNK: True.

**Review:** The adjective describes a committee rather than a geographic entity.

## 189. FP · row_id=1008786 · PER_as_LOC

У понеділок, 6 липня, президент Бразилії Жаїр Болсонару заявив, що здав тест на після рентгенографії легень.

**Gold:** ["Бразилії"]

**Prediction:** ["Бразилії", "Жаїр", "Болсонару"]

**Focal surface:** Болсонару; pilot fit frequency: 0; UNK: False.

**Review:** The president's surname matches PER; the predicted type is wrong.

## 190. FP · row_id=1003192 · ORG_context_FP

У комунальному підприємстві "Київський метрополітен" попереджають, що в період з 24 вересня до 5 жовтня зміниться режим роботи станції "Політехнічний інститут" (М1, "Червона лінія") через необхідність ремонту ескалатора.

**Gold:** []

**Prediction:** ["комунальному"]

**Focal surface:** комунальному; pilot fit frequency: 0; UNK: True.

**Review:** The adjective before a municipal enterprise is not LOC. Station names are annotated ORG in this gold, requiring a policy clarification.

## 191. FP · row_id=1009813 · nested_ORG_policy

Через ситуацію Медична асоціація Республіки Корея запропонувала призупинити програму вакцинації до 29 жовтня.

**Gold:** []

**Prediction:** ["Республіки"]

**Focal surface:** Республіки; pilot fit frequency: 0; UNK: False.

**Review:** The focal word is only part of a country name inside a complete medical-association name.

## 192. FP · row_id=1000452 · generic_word_FP

Раніше у Луцьку силовики евакуювали мешканців будинків, біля яких терорист "Максим Плохой" 21 липня захопив автобус із заручниками і погрожує зброєю.

**Gold:** ["Луцьку"]

**Prediction:** ["Луцьку", "заручниками"]

**Focal surface:** заручниками; pilot fit frequency: 0; UNK: True.

**Review:** Hostages are people by role, not a location.

## 193. FP · row_id=1004693 · generic_word_FP

Судячи з їхніх світлин, він, хоч як дивно, непогано зберігся – вціліли також настінні розписи ХІХ століття, тож із перекритим наново дахом може знову виконувати сакральні функції, була б тільки чиясь добра воля.

**Gold:** []

**Prediction:** ["воля"]

**Focal surface:** воля; pilot fit frequency: 1; UNK: False.

**Review:** The focal word denotes an abstract concept in this context.

## 194. FP · row_id=1009280 · geographic_adjective_policy

Учасник першої вистави, соліст нью-йоркського балету Ніколай Хюбер був захоплений сценою: "Це надзвичайно!

**Gold:** []

**Prediction:** ["нью", "йоркського"]

**Focal surface:** йоркського; pilot fit frequency: 0; UNK: True.

**Review:** The geographic adjective modifies a ballet institution rather than directly naming the city.

## 195. FP · row_id=1002311 · nested_title_policy

Згідно з ст. 9 Конституції України міжнародні договори є частиною національного законодавства України і тому обов'язкові для застосування судами.

**Gold:** []

**Prediction:** ["України"]

**Focal surface:** України; pilot fit frequency: 266; UNK: False.

**Review:** Ukraine occurs within a constitution title and a general legislative phrase. Specify the treatment of geographic names embedded in document titles.

## 196. FP · row_id=1005298 · generic_place_FP

Хокейний збір сільських аматорівУ Драбинівці Новосанжарського району на хокейному майданчику, побудованому за фінансової …

**Gold:** ["Драбинівці", "Новосанжарського району"]

**Prediction:** ["Новосанжарського району", "майданчику"]

**Focal surface:** майданчику; pilot fit frequency: 0; UNK: False.

**Review:** An unnamed hockey rink is not the target named location.

## 197. FP · row_id=1001913 · geographic_adjective_policy

Якщо маємо зовнішню агресію РФ, Мінські домовленості не годяться.

**Gold:** ["РФ"]

**Prediction:** ["РФ", "Мінські"]

**Focal surface:** Мінські; pilot fit frequency: 0; UNK: False.

**Review:** The Minsk agreements are the name of agreements, not a direct reference to the city.

## 198. FP · row_id=1001884 · generic_word_FP

На світ народилася донечка Даринка вагою 1230 грамів.

**Gold:** []

**Prediction:** ["світ"]

**Focal surface:** світ; pilot fit frequency: 15; UNK: False.

**Review:** The focal phrase is an idiom about birth, not a geographic entity.

## 199. FP · row_id=1004106 · ORG_as_LOC

Це, вважають у Мінкульті, поставить крапку у навмисно створених дискусіях щодо етнічної батьківщини рецепту страви.

**Gold:** []

**Prediction:** ["Мінкульті"]

**Focal surface:** Мінкульті; pilot fit frequency: 0; UNK: True.

**Review:** The ministry abbreviation matches ORG; it is an institution rather than LOC under this gold.

## 200. FP · row_id=1002941 · generic_place_FP

"Прибувши на місце події, правоохоронці спробували пройти в будинок, щоб надати жінці домедичну допомогу, проте на них накинувся з сокирою невідомий - як з'ясувалось згодом, 23-річний чоловік, котрий телефонував на оперативну лінію.

**Gold:** []

**Prediction:** ["будинок"]

**Focal surface:** будинок; pilot fit frequency: 0; UNK: False.

**Review:** An unnamed house is a generic object, not a named address.
