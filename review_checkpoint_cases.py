"""Contextual AI review of a frozen, fingerprinted set of 200 checkpoint cases.

Notes are hypotheses for a human handoff, never corrected target annotations.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent
EXPECTED='24ab61351826d3c9c0057f07b5820c8cc87ed90833096a79e3a51a60eaa044e2'

# Notes were written after reading the contexts of all 200 frozen cases.
SPECIAL = {
    2: ('multiword_name', 'The hotel name loses its first component and apostrophe; only the second component is returned.'),
    3: ('granularity_policy', 'Gold combines the city and region into one entity; only the region is predicted. Specify the address-nesting convention.'),
    4: ('address_boundary', 'The street name with an initial is fragmented; the street-type word and surname are predicted separately.'),
    5: ('entity_type_context', 'Elsewhere in this sentence, Dnipro names a program and Miass is annotated as ORG. Geographic words can therefore occur under other entity types.'),
    7: ('split_location', 'One bridge name is split into two predicted entities.'),
    8: ('possible_annotation_error', 'The gold span includes Moscow followed by a truncated fragment of the next phrase. The shorter prediction looks semantically more reasonable but is scored as wrong.'),
    9: ('granularity_policy', 'Gold splits the hyphenated city name; the model additionally absorbs the following predicate.'),
    14: ('granularity_policy', 'Gold splits the hyphenated name, while the model also captures following words.'),
    18: ('multiword_name', 'The modifier in the Caribbean Islands name is lost. The other city in gold is also split at its hyphen.'),
    20: ('coordinated_locations', 'Gold combines two squares and their conjunction into one entity; the model finds only the first modifier.'),
    22: ('merged_locations', 'The street and city are merged into one address fragment; gold requires two separate entities.'),
    24: ('coordinated_locations', 'Three coordinated regions form one gold entity, while the model emits three. This is a granularity issue, not complete failure to recognize geography.'),
    26: ('address_boundary', 'Only the generic street-type word remains; the street name and number are not extracted.'),
    27: ('merged_locations', 'The model merges Mariupol and Donetsk region, while gold keeps them separate.'),
    28: ('boundary_and_aux_quality', 'The word meaning named after is added to the city. The auxiliary ORG span also looks truncated; auxiliary labels should not be assumed error-free.'),
    29: ('coordinated_locations', 'Gold joins two regions with a conjunction; the model separates the coordinated names.'),
    30: ('possible_annotation_error', "The LOC span combines the Netherlands with the tennis player's name. Review the gold type and end boundary."),
    32: ('boundary_convention', 'Lido island is recognized, but gold excludes the generic island word. A consistent rule for such descriptors is needed.'),
    33: ('split_location', 'The established name European Union is split into two fragments.'),
    34: ('boundary_and_granularity', 'Extra spatial words are included; another hyphenated name in gold is split into parts.'),
    35: ('multiword_name', 'Only the modifier remains from Ancient Olympia. The Olympic flame being labeled LOC also calls for a type-policy review.'),
    36: ('possible_annotation_error', 'The gold span ends inside the following verb after Russia. Predicting just the country is not necessarily a semantic error.'),
    38: ('hyphen_granularity', 'The model emits the complete hyphenated city name, while gold requires two parts. Do not fix this by mechanically splitting every hyphenated name.'),
    39: ('hyphen_granularity', 'Beyond the hyphen-granularity mismatch, a following noun is absorbed. These are two distinct issues.'),
    40: ('quote_convention', 'The model excludes a closing quotation mark that belongs to gold. The published regex does not remove it as one might expect.'),
    41: ('hyphen_granularity', 'The model partially combines a metro-line name; gold separates components and includes the generic line word.'),
    43: ('address_boundary', 'The street abbreviation and number are omitted. Other addresses in this text lack LOC labels; annotation completeness needs review.'),
    44: ('hyphen_granularity', 'A complete city name conflicts with split gold. This is not simply a wrong place name.'),
    45: ('possible_text_corruption', 'A location word appears fused with a source/publication fragment. Bounds-valid offsets do not guarantee a clean training example.'),
    47: ('merged_locations', 'The model merges the settlement, district and region; gold keeps the region separate.'),
    48: ('vocabulary_cap', 'The surface appeared in the pilot, but a component maps to UNK. A seen entity does not guarantee that all its words survived the vocabulary cap.'),
    49: ('hyphen_split', 'Both parts of Buenos Aires are found separately; the full name should be retained.'),
    50: ('merged_locations', 'The village and region are merged; both geographic components are present in the prediction.'),
    52: ('possible_annotation_error', 'Putin refers to a person in this context but is included in LOC gold. Russia elsewhere in the same sentence has no LOC annotation.'),
    53: ('building_or_ORG_policy', 'Gold contains only a fragment of the parliamentary name, while the context refers to a building. Clarify boundaries and the building/institution convention.'),
    58: ('possible_annotation_omission', 'The model misses the annotated parts of the country but also finds an unannotated Ukraine mention. This is a candidate for a gold-completeness check.'),
    59: ('vocabulary_cap', 'The name appeared as LOC before, but its token is missing from the saved pilot vocabulary.'),
    60: ('seen_context_miss', 'A seen Lviv surface is missed at the start of a short sentence. Word novelty alone cannot explain the error.'),
    63: ('generic_location_FP', 'The river name is missed, and a generic protected-area description is predicted instead.'),
    64: ('possible_annotation_error', 'The context explicitly describes a film title, rather than a place, but it appears in LOC gold.'),
    68: ('administrative_policy', 'A village council can refer to an institution or territory. Gold needs a consistent policy; the prediction instead names another administrative entity.'),
    70: ('possible_type_error', 'Vorskla is playing in a football final here: context indicates a club, although gold labels LOC.'),
    73: ('generic_gold_fragment', 'Gold contains only a generic building-word fragment from an institution name. Check whether such fragments are intended target names.'),
    74: ('nested_ORG_policy', 'A country inside a church name is labeled LOC, whereas geographic components inside institutions are excluded in other cases. Check policy consistency.'),
    78: ('ORG_region_policy', 'Eastern Partnership denotes a political initiative or grouping in this context; review the LOC-versus-ORG convention.'),
    82: ('name_fragment_policy', 'The text contains Old Trafford, while gold includes only its second component. Check both object-name boundaries and metonymic usage.'),
    84: ('possible_annotation_error', 'The final surname in the headline looks like attribution to a speaker, rather than a geographic entity.'),
    85: ('hyphen_granularity', 'The volcano name is annotated as two entities, unlike some other hyphenated names.'),
    86: ('possible_type_error', 'Full sports-team names in a match context appear as LOC gold. This is not an unambiguous failure to find a city.'),
    87: ('possible_annotation_omission', 'Only some state names in a long list are annotated. Additional reasonable predictions can therefore count as FP.'),
    92: ('insufficient_context', 'A single phrase explaining the translation of this name provides insufficient context to confirm its geographic type.'),
    93: ('unusual_landmark_names', 'Ordinary words are proper names of rocks in this context. A city-only dictionary or a blanket common-word ban would hurt these cases.'),
    101: ('possible_annotation_error', 'Gold ends inside the word following Ukraine. The model also overextends the span; these two problems should be separated.'),
    102: ('split_location', 'The complete United States name is split into two predictions.'),
    103: ('extra_neighbor', 'The city absorbs the first component of the following hyphenated adjective.'),
    104: ('quote_convention', 'An extra closing quote causes a real string mismatch under the published scorer.'),
    105: ('address_components', 'The address boundary excludes its comma and building-number suffix. Digit removal does not guarantee a match because punctuation also matters.'),
    106: ('address_and_PER', 'The address loses its number and comma; artist names in the same sentence are also emitted as LOC.'),
    107: ('quote_and_entity_type', 'An extra quote follows Lviv; person names and institution words also generate FP in this sentence.'),
    108: ('missing_multiword_tail', 'Only the adjective remains from the full country name.'),
    109: ('extra_neighbor', 'The street name absorbs the following verb; the object boundary is not separated from the next event.'),
    110: ('hyphen_convention', 'The route-separating hyphen is appended to the first city.'),
    111: ('boundary_convention', 'Gold includes the generic city word, while the prediction stops at district. Clarify generic descriptors in the annotation rules.'),
    112: ('possible_annotation_error', 'Gold merges Minsk and Ukraine across an adverbial/subject boundary. These are candidates for separate entities.'),
    113: ('extra_neighbor', 'The city absorbs the subject and verb of the following clause.'),
    114: ('missing_modifier', 'The geographic modifier is dropped from Eastern Ukraine.'),
    116: ('hyphen_granularity', 'Gold requires a complete hyphenated city here, unlike several FN cases where the same city is split.'),
    117: ('hyphen_split', 'Only the first part of Tel Aviv remains.'),
    118: ('boundary_convention', 'A generic directional component is added to the country name.'),
    119: ('possible_annotation_error', 'The model finds the complete hyphenated city after the word city; gold labels the first part LOC and the second part PER.'),
    120: ('missing_name_material', 'The generic district word is dropped, and Kryvyi Rih is also missed.'),
    121: ('hyphen_split', 'The hyphenated region name is split into two predictions.'),
    122: ('missing_name_material', 'The generic region word is lost; a generic eastern-part description is additionally predicted as a name.'),
    123: ('punctuation_gold_policy', 'Gold contains Odesa followed by a comma and a space. The city is found, but the scorer treats its string as different.'),
    124: ('possible_annotation_error', 'Gold ends inside the final component of the name; the model also fragments it. This cannot be reduced to an unknown-word problem.'),
    126: ('hyphen_split', 'The Kyiv-Pechersk Lavra name is split into two entities.'),
    127: ('missing_modifier', 'The western modifier is omitted. An additional generic territory noun is not a named place in this context.'),
    128: ('boundary_convention', 'A misspelled generic airport word is included before Chicago.'),
    129: ('boundary_convention', 'Gold includes a preposition, while only the place name is predicted. Specify whether prepositions belong in names.'),
    130: ('hyphen_split', 'The hyphenated city name is split into two parts.'),
    131: ('missing_modifier', 'Only the generic region word remains; the proper modifier is lost.'),
    132: ('hyphen_and_punctuation', 'The hyphenated city is fragmented, and a bracket is added to one component.'),
    133: ('quote_convention', 'A closing quote is attached to the Lviv name.'),
    134: ('extra_neighbor', 'The span captures the beginning of a parenthetical explanation about a count of people.'),
    135: ('extra_neighbor', 'The district absorbs the subject of the next statement.'),
    136: ('boundary_convention', 'A generic city descriptor is added, while gold contains only Kharkiv.'),
    137: ('extra_neighbor', 'The city absorbs a verb and the object of the event.'),
    138: ('possible_type_error', 'The gold name refers to a person in a leukemia context, but it is labeled LOC.'),
    140: ('missing_modifier', 'The proper modifier Black is omitted, leaving only the generic sea word.'),
    141: ('extra_neighbor', 'The street absorbs a directional transition before another name.'),
    142: ('coordinated_locations', 'Gold combines an alternative list of seas, while the model keeps only the first modifier.'),
    144: ('generic_word_FP', 'The focal word describes departures/flights, not a named place.'),
    145: ('generic_word_FP', 'Negotiations are an event; this sentence has no LOC gold.'),
    146: ('generic_place_FP', 'An unnamed attic is a generic place, not a target geographic named entity.'),
    147: ('generic_word_FP', 'An adjective from an anatomical term is incorrectly emitted as LOC.'),
    148: ('numeric_normalization', 'The number 166 is incorrectly extracted. It normalizes to an empty string, which still counts as FP rather than an empty list.'),
    149: ('generic_word_FP', 'The stomach and intestine words are anatomical terms, not geography.'),
    150: ('nested_ORG_policy', 'Kazakhstan is part of the annotated ministry name. This may reflect a non-nested LOC policy rather than confusion of the entire institution with a place.'),
    151: ('generic_word_FP', 'The focal verb does not denote a place.'),
    152: ('generic_word_FP', 'Compote is an ordinary noun in a food-preparation context.'),
    153: ('generic_word_FP', 'The focal word refers to an exchange rate, not a geographic object.'),
    154: ('generic_word_FP', 'The focal word denotes an event without a proper place name.'),
    155: ('generic_region_FP', 'An all-regions phrase is a generic description, not the proper name of a region.'),
    156: ('ORG_as_LOC', 'The vehicle brand/model is annotated ORG; the model emits a fragment as LOC.'),
    157: ('geographic_adjective_FP', 'A south-easterly wind describes direction, not a named location.'),
    158: ('generic_word_FP', 'The focal abstract noun does not become a place name merely because a region is mentioned nearby.'),
    159: ('generic_word_FP', 'Railings are an object, not a geographic name.'),
    160: ('generic_place_FP', 'An unnamed street is not a named address.'),
    161: ('generic_word_FP', 'The focal adjective describes crowded objects rather than naming a place.'),
    162: ('generic_word_FP', 'A profession is not LOC; the neighboring institution abbreviation and certificate word also trigger extra predictions.'),
    163: ('generic_word_FP', 'A medical specialist is a person by occupation, not a location.'),
    164: ('generic_place_FP', 'The swamp word is a generic terrain type without a proper name.'),
    165: ('ORG_as_LOC', 'The focal budget term matches an auxiliary ORG label; the context concerns a budget, not a place.'),
    166: ('generic_word_FP', 'A possessive pronoun in an exhibition title is not LOC.'),
    167: ('nonlocation_named_item', 'The focal geographic-looking name refers to a ship here, not the city.'),
    168: ('possible_annotation_omission', 'Tallinn airport looks semantically like a location but is absent from gold. Check completeness and the treatment of objects with geographic adjectives.'),
    169: ('ORG_as_LOC', 'The news agency is annotated ORG and has no LOC gold.'),
    170: ('ORG_metonymy_policy', 'The cabinet is an institution, sometimes used metonymically for its building. Clarify the policy rather than treating every such string as a city.'),
    171: ('nested_ORG_policy', 'The country is included only within the complete presidential-administration ORG span.'),
    172: ('possible_annotation_omission', 'The country in a presidential title is not labeled LOC, and Norway is included in PER. This sentence contains candidates for type-label noise.'),
    173: ('generic_word_FP', 'The focal word denotes a legal right, not a place. Turkey at the start is also unannotated, so a possible gold omission coexists in this text.'),
    174: ('generic_word_FP', 'Camp employees are people, not a place name.'),
    175: ('PER_as_LOC', "The prophet's name matches a PER annotation but is predicted as LOC."),
    176: ('generic_word_FP', 'A tourist is a person by role, not a location.'),
    177: ('generic_place_FP', 'A chair is an object/generic place, not named geography.'),
    178: ('generic_word_FP', 'The temporal modifier is not a location named entity.'),
    179: ('generic_word_FP', 'The evaluative adjective is not a geographic name.'),
    180: ('generic_word_FP', 'The descriptive adjective before a house is not a name.'),
    181: ('generic_word_FP', 'The focal word is an abstract consequence, not a location.'),
    182: ('nonlocation_named_item', 'In the horoscope context, the focal name is more likely a publication or section title; LOC is not established.'),
    183: ('generic_word_FP', 'The focal word refers to biological cells, not geography.'),
    184: ('numeric_normalization', '1984 is a birth year, not LOC; normalization to an empty string does not remove the FP from the metric.'),
    185: ('generic_word_FP', 'Energy is an industry here, not a named place.'),
    186: ('PER_context_FP', "Madame is a form of address before a character's name, not a location."),
    187: ('ORG_as_LOC', 'The adjective starts the name of a government body. The auxiliary ORG boundary also looks truncated.'),
    188: ('generic_word_FP', 'The adjective describes a committee rather than a geographic entity.'),
    189: ('PER_as_LOC', "The president's surname matches PER; the predicted type is wrong."),
    190: ('ORG_context_FP', 'The adjective before a municipal enterprise is not LOC. Station names are annotated ORG in this gold, requiring a policy clarification.'),
    191: ('nested_ORG_policy', 'The focal word is only part of a country name inside a complete medical-association name.'),
    192: ('generic_word_FP', 'Hostages are people by role, not a location.'),
    193: ('generic_word_FP', 'The focal word denotes an abstract concept in this context.'),
    194: ('geographic_adjective_policy', 'The geographic adjective modifies a ballet institution rather than directly naming the city.'),
    195: ('nested_title_policy', 'Ukraine occurs within a constitution title and a general legislative phrase. Specify the treatment of geographic names embedded in document titles.'),
    196: ('generic_place_FP', 'An unnamed hockey rink is not the target named location.'),
    197: ('geographic_adjective_policy', 'The Minsk agreements are the name of agreements, not a direct reference to the city.'),
    198: ('generic_word_FP', 'The focal phrase is an idiom about birth, not a geographic entity.'),
    199: ('ORG_as_LOC', 'The ministry abbreviation matches ORG; it is an institution rather than LOC under this gold.'),
    200: ('generic_place_FP', 'An unnamed house is a generic object, not a named address.'),
}


def main():
    cases=json.loads((BASE/'reports/checkpoint_review_200.json').read_text(encoding='utf-8'))
    payload=json.dumps([(e['row_id'],e['kind'],e['surface']) for e in cases],ensure_ascii=False).encode('utf-8')
    assert hashlib.sha256(payload).hexdigest()==EXPECTED,'Review notes belong to a different frozen case set'
    for e in cases:
        if e['case_id'] in SPECIAL:
            tag,note=SPECIAL[e['case_id']]
        elif e['category']=='boundary_overlap':
            tag=e['boundary_relation']
            note=f"The focal span's boundary differs from its counterpart; observed relation: {tag}. Original offsets establish this relation, not its cause."
        elif e['kind']=='FN':
            tag='detection_miss'
            note=f"The annotated location has no overlap with any prediction. Pilot training frequency: {e['pilot_training_mentions']}; UNK in the mention: {e['contains_UNK']}. Novelty/OOV are observed factors, not proven causes."
        else:
            tag='generic_unannotated_FP'
            note='The focal span has no LOC support or overlap; contextual review identifies it as an extra fragment requiring precision control.'
        e.update(context_review_tag=tag,context_review_note=note,
                 reviewed_by='Codex: AI-assisted contextual review, not independent human adjudication',
                 gold_changed=False)
    (BASE/'reports/checkpoint_review_200_annotated.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    flat=[]
    for e in cases:
        flat.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in e.items()})
    pd.DataFrame(flat).to_csv(BASE/'reports/checkpoint_review_200_annotated.csv',index=False,encoding='utf-8-sig')
    lines=['# Contextual review of 200 checkpoint errors','',
           '100 FP and 100 FN focal cases from 200 distinct texts. This diagnostic sample emphasizes boundary coverage; its proportions do not estimate corpus-wide prevalence.',
           'All contexts were reviewed by Codex. This is AI-assisted review, not independent expert annotation. Original gold is unchanged. Annotation-quality categories identify candidates for human review. Original NER text and entity strings are preserved in their source languages.','']
    for e in cases:
        lines.extend([f"## {e['case_id']:03d}. {e['kind']} · row_id={e['row_id']} · {e['context_review_tag']}",'',
                      e['text'],'',
                      '**Gold:** '+json.dumps(e['gold'],ensure_ascii=False),'',
                      '**Prediction:** '+json.dumps(e['pred'],ensure_ascii=False),'',
                      f"**Focal surface:** {e['surface']}; pilot fit frequency: {e['pilot_training_mentions']}; UNK: {e['contains_UNK']}.",'',
                      '**Review:** '+e['context_review_note'],''])
    (BASE/'reports/CHECKPOINT_REVIEW_200.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Enriched 200 frozen cases with English contextual notes; gold unchanged')



if __name__=='__main__':main()
