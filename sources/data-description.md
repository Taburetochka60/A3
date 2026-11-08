# Name Entity Recognition dataset

## uk_geo_dataset
Ukrainian dataset [Corpora Ukrainian](https://huggingface.co/datasets/ukr-models/Ukr-Synth). 
Approximately contains 1M text samples with location and organization entities.
Structure:
- text (str): text sample.
- loc_markers (List[tuple]): list of markers (spans) of location entities. Left and right indices.
- org_markers (List[tuple]): list of markers (spans) of origanization entities. Left and right indices.
- per_markers (List[tuple]): list of markers (spans) of person entities. Left and right indices.
- is_valid (int): whether it was taken from the validation split.

## ru_geo_dataset
russian dataset based on [nerus dataset](https://github.com/natasha/nerus). 
Approximately contains 8M text samples with location and organization entities.
Structure:
- text (str): text sample.
- loc_markers (List[tuple]): list of markers (spans) of location entities. Left and right indices.
- org_markers (List[tuple]): list of markers (spans) of origanization entities. Left and right indices.
- per_markers (List[tuple]): list of markers (spans) of person entities. Left and right indices.
- doc_id (int): document id.
- sent_id (int): sentence id within a document.

## labeling_sample
sample from the same distribution as test
Structure:
- text_id: id of text sample
- text: text sample
- locations: emply list that should be modified in your submission
- loc_markers: list of markers (spans) of location entities. Left and right indices.

## test
Test dataset
Structure:
- text_id: id of text sample
- text: text sample
- locations: emply list that should be modified in your submission, you can check how to do this in this [kernel](https://www.kaggle.com/code/abazdyrev/spacyner-baseline)
