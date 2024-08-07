""" TASK A LLM challenge"""
wordnet_dataset = "llm_challenge/dataset/train/wordnet_output.csv"
wordnet_annoted = "llm_challenge/annotated/word_net_annotated.csv"

test_wordnet_dataset = "llm_challenge/dataset/test/wordnet/A.1(FS)_WordNet_Test.csv"
test_wordnet_annoted = "llm_challenge/annotated/word_net_annotated.csv"

test_geonames_dataset = "llm_challenge/dataset/test/geonames/A.2(FS)_GeoNames_Test.csv"
test_geonames_annotated = "llm_challenge/annotated/geoname/geoname_annotated_18.csv"

test_biological_dataset = "llm_challenge/dataset/test/biological/A.4(FS)_GO_Biological_Process_Test.csv"
test_biological_annotated = "llm_challenge/annotated/biological/biological_annotated.csv"

test_cellular_dataset = "llm_challenge/dataset/test/cellular/A.4(FS)_GO_Cellular_Component_Test.csv"
test_cellular_annotated = "llm_challenge/annotated/cellular/cellular_annotated_1.csv"

test_molecular_dataset = "llm_challenge/dataset/test/molecular/A.4(FS)_GO_Molecular_Function_Test.csv"
test_molecular_annotated = "llm_challenge/annotated/molecular/molecular_annotated.csv"

""" ===============
    OTHER
    ===============
"""
# Other information
""" cea """
cea_full_json_path_folder = "data/json/full/cea"
cea_full_json_train_dataset = "data/json/cea/train_semtab_2024_llm_1.jsonl"
cea_full_json_val_dataset = "data/json/cea/val_semtab_2024_llm_1.jsonl"
raw_cea = "data/result/cea/cea.csv"
raw_cta = "data/result/cta/cta.csv"
raw_cpa = "data/result/cpa/cpa.csv"
raw_ra = "data/result/ra/ra.csv"
raw_td = "data/result/td/td.csv"

""" cta"""
cta_full_json_path_folder = "data/json/full/cta"
cta_full_json_train_dataset = "data/json/cta/train_semtab_cta_2024.jsonl"
cta_full_json_val_dataset = "data/json/cta/val_semtab_cta_2024.jsonl"

base_model = "gpt-3.5-turbo-0613"
model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator:9JLG3i7Q"
cta_model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator-cta:9R9nlEKE"
cta_model_finetuned_1 = "ft:gpt-3.5-turbo-0613:tib:annotator-cta-1:9TqtfaMG"
llm_model_finetuned = "ft:gpt-3.5-turbo-0613:tib::9RQqOGhi"
cea_model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator-cea:9TBiCM7t"
cea_model_finetuned_2 = "ft:gpt-3.5-turbo-0613:tib:annotator-cea-001:9WFgKQkJ"
cea_model_finetuned_3 = "ft:gpt-3.5-turbo-0613:tib:annotator-cea:9Wpe5X7u"
cea_model_finetuned_4 = "ft:gpt-3.5-turbo-0613:tib:annotator-cea:9Xo3Twdr"
ra_model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator-ra:9azlDasr"
td_model_finetuned = "ft:gpt-3.5-turbo-0613:tib:annotator-td:9bYhZGqb"

""" ===============
    CEA TASK 2023 and 2024 semtab challenge
    ===============
"""

""" ============ LLM Track ==============="""
#  LLMs dataset semtab 2024
cea_target_llm = "data/csv/semtab2024/llm/train/gt/cea_gt.csv"
cea_dataset_table_llm = "data/csv/semtab2024/llm/train/tables"
cea_llm = "data/result/cea/annotate/2024/llm/cea_an.csv"
cea_dataset_llm = "data/result/cea/dataset/2024/llm/update_cea.csv"
cea_dataset_json_llm = "data/json/cea/llm/llm_train.json"
cea_llm_target = "data/csv/semtab2024/llm/train/gt/cea_gt.csv"

""" ============== Acuracy TRack ============"""

""" ------2023---------------------"""
# wikidata
cea_target_wikidata_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/gt/cea_gt.csv"
cea_dataset_table_path_wikidata_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/tables"
cea_wikidata_23 = "data/result/cea/annotate/wikidata/cea_an.csv"
cea_dataset_wikidata_23 = "data/result/cea/dataset/wikidata/update_cea.csv"
cea_dataset_json_wikidata_23 = "data/json/app2/wikidata/train_semtab2023_wikidata.jsonl"
cea_wikidata_target_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/targets/cea_targets.csv"
# dataset_wikidata_folder = "data/result/cea/dataset/wikidata"

# tfood_entity
cea_target_tfood_entity = "data/csv/tables/semtab2023/tfood/entity/val/gt/cea_gt.csv"
cea_dataset_table_tfood_entity = "data/csv/tables/semtab2023/tfood/entity/val/tables"
cea_tfood_entity = "data/result/cea/annotate/tfood/entity/tfood_entity_cea.csv"
cea_dataset_tfood_entity = "data/result/cea/dataset/tfood/entity/tfood_dataset_entity_cea.csv"
cea_dataset_json_tfood_entity = "data/json/app2/tfood/entity/tfood_dataset_horizontal_cea.jsonl"
cea_tfood_entity_target = "data/csv/tables/semtab2023/tfood/entity/val/targets/cea_targets.csv"

# tfood_horizontal
cea_target_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/gt/cea_gt.csv"
cea_dataset_table_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/tables"
cea_tfood_hor = "data/result/cea/annotate/tfood/horizontal/tfood_horizontal_cea.csv"
cea_dataset_tfood_hor = "data/result/cea/dataset/tfood/horizontal/tfood_dataset_horizontal_cea.csv"
cea_dataset_json_tfood_hor = "data/json/app2/tfood/horizontal/tfood_dataset_horizontal_cea.jsonl"
cea_tfood_hor_target = "data/csv/tables/semtab2023/tfood/horizontal/val/targets/cea_targets.csv"

""" -----------2024----------"""
"""=================Train Set=============================="""
# wikidata
cea_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/gt/cea_gt.csv"
cea_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/tables"
cea_wikidata_24 = "data/result/cea/annotate/2024/accuracy/wikidata/wikidata_cea_24.csv"
cea_dataset_wikidata_24 = "data/result/cea/dataset/2024/accuracy/wikidata/wikidata_dataset_24.csv"
cea_dataset_json_wikidata_24 = "data/json/app2/2024/accuracy/cea/wikidata/train_semtab2024_wikidata.jsonl"
cea_wikidata_target_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/targets/cea_targets.csv"


""" tbiomed entity"""
cea_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/gt/cea_gt.csv"
cea_tbiomed_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/targets/cea_targets.csv"
cea_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/tables"
cea_tbiomed_entity = "data/result/cea/annotate/2024/accuracy/tbiomed/entity/tbiomed_cea.csv"
cea_dataset_tbiomed_entity = "data/result/cea/dataset/2024/accuracy/tbiomed/entity/tbiomed_cea_dataset.csv"
cea_dataset_json_tbiomed_entity = "data/json/app2/2024/accuracy/cea/tbiomed/entity/train_semtab2024_tbiomed_entity.jsonl"

""" tbiomed Horizontal"""
cea_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/gt/cea_gt.csv"
cea_tbiomed_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/targets/cea_targets.csv"
cea_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/tables"
cea_tbiomed_hor = "data/result/cea/annotate/2024/accuracy/tbiomed/horizontal/tbiomed_cea.csv"
cea_dataset_tbiomed_hor = "data/result/cea/dataset/2024/accuracy/tbiomed/horizontal/tbiomed_dataset.csv"
cea_dataset_json_tbiomed_hor = "data/json/app2/2024/accuracy/cea/tbiomed/horizontal/train_semtab2024_tbiomed_hor.jsonl"


""" tbiodiv entity"""
cea_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/gt/cea_gt.csv"
cea_tbiodiv_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/targets/cea_targets.csv"
cea_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/tables"
cea_tbiodiv_entity = "data/result/cea/annotate/2024/accuracy/tbiodiv/entity/tbiodiv_cea.csv"
cea_dataset_tbiodiv_entity = "data/result/cea/dataset/2024/accuracy/tbiodiv/entity/tbiodiv_dataset.csv"
cea_dataset_json_tbiodiv_entity = "data/json/app2/2024/accuracy/cea/tbiodiv/entity/train_semtab2024_tbiodiv_entity.jsonl"

""" tbiodiv Horizontal"""
cea_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/gt/cea_gt.csv"
cea_tbiodiv_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/targets/cea_targets.csv"
cea_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/tables"
cea_tbiodiv_hor = "data/result/cea/annotate/2024/accuracy/tbiodiv/horizontal/tbiodiv_cea.csv"
cea_dataset_tbiodiv_hor = "data/result/cea/dataset/2024/accuracy/tbiodiv/horizontal/tbiodiv_dataset.csv"
cea_dataset_json_tbiodiv_hor = "data/json/app2/2024/accuracy/cea/tbiodiv/horizontal/train_semtab2024_tbiodiv_hor.jsonl"

"""=================CEA Test Set=============================="""
""" LLM"""
# test_cea_target_llm = "data/csv/tables/semtab2024/llm/test/gt/target.csv"
test_cea_target_llm = "data/csv/tables/semtab2024/llm/test/gt/cea_target.csv"
test_cea_dataset_table_llm = "data/csv/tables/semtab2024/llm/test/tables"
test_cea_llm = "data/result/cea/annotate/2024/llm/llm_test_cea.csv"
test_cea_dataset_llm = "data/result/cea/dataset/2024/llm/llm_test_cea_dataset.csv"
# test_cea_dataset_json_llm = "data/json/app2/2024/llm/train_semtab2024_llm.jsonl"

""" Accuracy Track"""
# wikidata test
test_cea_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/targets/cea_targets.csv"
test_cea_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/tables"
test_cea_wikidata_24 = "data/result/cea/annotate/2024/accuracy/wikidata/wikidata_test_cea_24_symb1.csv"
test_cea_dataset_wikidata_24 = "data/result/cea/dataset/2024/accuracy/wikidata/wikidata_test_dataset_24.csv"

# tbiomed test entity
test_cea_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/targets/cea_targets.csv"
test_cea_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/tables"
test_cea_tbiomed_entity = "data/result/cea/annotate/2024/accuracy/tbiomed/entity/tbiomed_cea_test.csv"
test_cea_dataset_tbiomed_entity = "data/result/cea/dataset/2024/accuracy/tbiomed/entity/tbiomed_cea_dataset_test.csv"

""" tbiomed Horizontal test"""
test_cea_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/targets/cea_targets.csv"
test_cea_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/tables"
test_cea_tbiomed_hor = "data/result/cea/annotate/2024/accuracy/tbiomed/horizontal/tbiomed_cea_test_10.csv"
test_cea_dataset_tbiomed_hor = "data/result/cea/dataset/2024/accuracy/tbiomed/horizontal/tbiomed_dataset_test.csv"


""" tbiodiv entity test"""
test_cea_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/targets/cea_targets.csv"
test_cea_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/tables"
test_cea_tbiodiv_entity = "data/result/cea/annotate/2024/accuracy/tbiodiv/entity/tbiodiv_cea_test.csv"
test_cea_dataset_tbiodiv_entity = "data/result/cea/dataset/2024/accuracy/tbiodiv/entity/tbiodiv_dataset_test.csv"

""" tbiodiv Horizontal"""
test_cea_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/targets/cea_targets.csv"
test_cea_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/tables"
test_cea_tbiodiv_hor = "data/result/cea/annotate/2024/accuracy/tbiodiv/horizontal/tbiodiv_cea_test_1.csv"
test_cea_dataset_tbiodiv_hor = "data/result/cea/dataset/2024/accuracy/tbiodiv/horizontal/tbiodiv_dataset_test.csv"

""" ===============
    CTA TASK 2023 and 2024 semtab challenge
    ===============
"""

""" -----------------2023--------------------"""
# wikidata
cta_target_wikidata_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/gt/cta_gt.csv"
cta_dataset_table_path_wikidata_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/tables"
cta_wikidata_23 = "data/result/cta/annotate/wikidata/wikidata_cta_23.csv"
cta_dataset_wikidata_23 = "data/result/cta/dataset/wikidata/wikidata_dataset_23.csv"
cta_dataset_json_wikidata_23 = "data/json/app2/wikidata/train_semtab2023_wikidata.jsonl"
cta_wikidata_target_23 = "data/csv/tables/semtab2023/WikidataTables/Valid/targets/cta_targets.csv"

# tfood_horizontal
cta_target_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/gt/cta_gt.csv"
cta_dataset_table_tfood_hor = "data/csv/tables/semtab2023/tfood/horizontal/val/tables"
cta_tfood_hor = "data/result/cta/annotate/tfood/horizontal/tfood_horizontal_cta.csv"
cta_dataset_tfood_hor = "data/result/cta/dataset/tfood/horizontal/tfood_dataset_horizontal_cta.csv"
cta_dataset_json_tfood_hor = "data/json/app2/tfood/horizontal/tfood_horizontal_cta.jsonl"
cta_tfood_hor_target = "data/csv/tables/semtab2023/tfood/horizontal/val/targets/cta_targets.csv"

""" --------------------2024------------------------------------"""
"""=================CTA train Set=============================="""
# wikidata
cta_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/gt/cta_gt.csv"
cta_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/tables"
cta_wikidata_24 = "data/result/cta/annotate/2024/accuracy/wikidata/wikidata_cta_24.csv"
cta_wikidata_1_24 = "data/result/cta/annotate/2024/accuracy/wikidata/wikidata_cta_1_24.csv"
cta_dataset_wikidata_24 = "data/result/cta/dataset/2024/accuracy/wikidata/wikidata_dataset_24.csv"
cta_dataset_wikidata_1_24 = "data/result/cta/dataset/2024/accuracy/wikidata/wikidata_dataset_1_24.csv"
cta_dataset_json_wikidata_24 = "data/json/app2/2024/accuracy/cta/wikidata/train_semtab2024_wikidata.jsonl"
cta_wikidata_target_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/targets/cta_targets.csv"

""" tbiomedical Horizontal"""
cta_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/gt/cta_gt.csv"
cta_tbiomed_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/targets/cta_targets.csv"
cta_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/tables"
cta_tbiomed_hor = "data/result/cta/annotate/2024/accuracy/tbiomed/horizontal/tbiomed_cta.csv"
cta_dataset_tbiomed_hor = "data/result/cta/dataset/2024/accuracy/tbiomed/horizontal/tbiomed_dataset.csv"
cta_dataset_json_tbiomed_hor = "data/json/app2/2024/accuracy/cta/tbiomed/horizontal/train_semtab2024_tbiomed_hor.jsonl"

""" tbiodiv Horizontal"""
cta_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/gt/cta_gt.csv"
cta_tbiodiv_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/targets/cta_targets.csv"
cta_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/tables"
cta_tbiodiv_hor = "data/result/cta/annotate/2024/accuracy/tbiodiv/horizontal/tbiodiv_cta.csv"
cta_dataset_tbiodiv_hor = "data/result/cta/dataset/2024/accuracy/tbiodiv/horizontal/tbiodiv_dataset.csv"
cta_dataset_json_tbiodiv_hor = "data/json/app2/2024/accuracy/cta/tbiodiv/horizontal/train_semtab2024_tbiodiv_hor.jsonl"

"""=================CTA Test Set=============================="""
# wikidata test
test_cta_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/tables"
test_cta_wikidata_24 = "data/result/cta/annotate/2024/accuracy/wikidata/wikidata_cta_24_test_combine_2.csv"
test_cta_dataset_wikidata_24 = "data/result/cta/dataset/2024/accuracy/wikidata/wikidata_dataset_24_test.csv"
test_cta_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/targets/cta_targets.csv"

R2_test_cta_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/Round2/Test/tables"
R2_test_cta_wikidata_24 = "data/result/Round2/cta/annotate/wikidata/wikidata_cta_24_1_part3.csv"
R2_test_cta_dataset_wikidata_24 = "data/result/Round2/cta/dataset/wikidata/wikidata_dataset_24_test.csv"
R2_test_cta_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/Round2/Test/targets/cta_targets.csv"


""" tbiomedical Horizontal test"""
test_cta_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/targets/cta_targets.csv"
test_cta_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/tables"
test_cta_tbiomed_hor = "data/result/cta/annotate/2024/accuracy/tbiomed/horizontal/tbiomed_cta_test_combine_2.csv"
test_cta_dataset_tbiomed_hor = "data/result/cta/dataset/2024/accuracy/tbiomed/horizontal/tbiomed_dataset_test.csv"

""" tbiodiv Horizontal test"""
test_cta_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/targets/cta_targets.csv"
test_cta_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/tables"
test_cta_tbiodiv_hor = "data/result/cta/annotate/2024/accuracy/tbiodiv/horizontal/tbiodiv_cta_test_combine_2.csv"
test_cta_dataset_tbiodiv_hor = "data/result/cta/dataset/2024/accuracy/tbiodiv/horizontal/tbiodiv_dataset_test.csv"


""" ===============
    RA TASK 2024 semtab challenge
    ===============
"""
ra_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/gt/ra_gt.csv"
ra_tbiomed_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/targets/ra_targets.csv"
ra_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/tables"
ra_tbiomed_hor = "data/result/ra/annotate/2024/tbiomed/tbiomed_ra.csv"
ra_dataset_tbiomed_hor = "data/result/ra/dataset/tbiomed/tbiomed_dataset_ra.csv"
ra_dataset_json_tbiomed_hor = "data/json/app2/2024/accuracy/ra/tbiomed/train_semtab2024_tbiomed_ra.jsonl"

ra_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/gt/ra_gt.csv"
ra_tbiodiv_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/targets/ra_targets.csv"
ra_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/tables"
ra_tbiodiv_hor = "data/result/ra/annotate/tbiodiv/tbiodiv_ra.csv"
ra_dataset_tbiodiv_hor = "data/result/ra/dataset/tbiodiv/tbiodiv_dataset.csv"
ra_dataset_json_tbiodiv_hor = "data/json/app2/2024/accuracy/ra/tbiodiv/train_semtab2024_tbiodiv_hor_ra.jsonl"

""" ===============
    RA TEST TASK 2024 semtab challenge
    ===============
"""

test_ra_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/targets/ra_targets.csv"
test_ra_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/tables"
test_ra_tbiomed_hor = "data/result/ra/annotate/tbiomed/tbiomed_ra_test_1.csv"
test_ra_dataset_tbiomed_hor = "data/result/ra/dataset/tbiomed/tbiomed_dataset_ra_test.csv"

test_ra_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/targets/ra_targets.csv"
test_ra_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/tables"
test_ra_tbiodiv_hor = "data/result/ra/annotate/tbiodiv/tbiodiv_ra_test_1_part11.csv"
test_ra_dataset_tbiodiv_hor = "data/result/ra/dataset/tbiodiv/tbiodiv_dataset_test.csv"

""" ===============
    TD TASK 2024 semtab challenge
    ===============
"""
""" tbiomed entity"""
td_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/gt/td_gt.csv"
td_tbiomed_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/targets/td_targets.csv"
td_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/tables"
td_tbiomed_entity = "data/result/td/annotate/tbiomed/entity/tbiomed_td.csv"
td_dataset_tbiomed_entity = "data/result/td/dataset/tbiomed/entity/tbiomed_td_dataset.csv"
td_dataset_json_tbiomed_entity = "data/json/app2/2024/accuracy/td/tbiomed/entity/train_semtab2024_tbiomed_entity_td.jsonl"

""" tbiomed Horizontal"""
td_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/gt/td_gt.csv"
td_tbiomed_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/targets/td_targets.csv"
td_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/tables"
td_tbiomed_hor = "data/result/td/annotate/tbiomed/horizontal/tbiomed_td.csv"
td_dataset_tbiomed_hor = "data/result/td/dataset/tbiomed/horizontal/tbiomed_dataset_td.csv"
td_dataset_json_tbiomed_hor = "data/json/app2/2024/accuracy/td/tbiomed/horizontal/train_semtab2024_tbiomed_hor_td.jsonl"


""" tbiodiv entity"""
td_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/gt/td_gt.csv"
td_tbiodiv_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/targets/td_targets.csv"
td_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/tables"
td_tbiodiv_entity = "data/result/td/annotate/tbiodiv/entity/tbiodiv_td.csv"
td_dataset_tbiodiv_entity = "data/result/td/dataset/tbiodiv/entity/tbiodiv_dataset_td.csv"
td_dataset_json_tbiodiv_entity = "data/json/app2/2024/accuracy/td/tbiodiv/entity/train_semtab2024_tbiodiv_entity_td.jsonl"

""" tbiodiv Horizontal"""
td_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/gt/td_gt.csv"
td_tbiodiv_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/targets/td_targets.csv"
td_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/tables"
td_tbiodiv_hor = "data/result/td/annotate/tbiodiv/horizontal/tbiodiv_td.csv"
td_dataset_tbiodiv_hor = "data/result/td/dataset/tbiodiv/horizontal/tbiodiv_dataset_td.csv"
td_dataset_json_tbiodiv_hor = "data/json/app2/2024/accuracy/td/tbiodiv/horizontal/train_semtab2024_tbiodiv_hor_td.jsonl"

""" ===============
    TD TEST TASK 2024 semtab challenge
    ===============
"""
""" tbiomed entity"""
test_td_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/targets/td_targets.csv"
test_td_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/tables"
test_td_tbiomed_entity = "data/result/td/annotate/tbiomed/entity/tbiomed_td_test_1.csv"
test_td_dataset_tbiomed_entity = "data/result/td/dataset/tbiomed/entity/tbiomed_td_dataset_test.csv"

""" tbiomed Horizontal"""
test_td_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/targets/td_targets.csv"
test_td_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/tables"
test_td_tbiomed_hor = "data/result/td/annotate/tbiomed/horizontal/tbiomed_td_test_1.csv"
test_td_dataset_tbiomed_hor = "data/result/td/dataset/tbiomed/horizontal/tbiomed_dataset_td_test.csv"

""" tbiodiv entity"""
test_td_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/targets/td_targets.csv"
test_td_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/tables"
test_td_tbiodiv_entity = "data/result/td/annotate/tbiodiv/entity/tbiodiv_td_test_1.csv"
test_td_dataset_tbiodiv_entity = "data/result/td/dataset/tbiodiv/entity/tbiodiv_dataset_td_test.csv"

""" tbiodiv Horizontal"""
test_td_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/targets/td_targets.csv"
test_td_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/tables"
test_td_tbiodiv_hor = "data/result/td/annotate/tbiodiv/horizontal/tbiodiv_td_test_1.csv"
test_td_dataset_tbiodiv_hor = "data/result/td/dataset/tbiodiv/horizontal/tbiodiv_dataset_td_test.csv"

""" ===============
    CPA TASK 2024 semtab challenge
    ===============
"""

cpa_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/gt/cpa_gt.csv"
cpa_wikidata_target_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/targets/cpa_targets.csv"
cpa_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Valid/tables"
cpa_wikidata_24 = "data/result/cpa/annotate/wikidata/wikidata_cpa_24.csv"
cpa_dataset_wikidata_24 = "data/result/cpa/dataset/wikidata/wikidata_dataset_cpa_24.csv"

""" tbiomed entity"""
cpa_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/gt/cpa_gt.csv"
cpa_tbiomed_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/targets/cpa_targets.csv"
cpa_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/val/tables"
cpa_tbiomed_entity = "data/result/cpa/annotate/tbiomed/entity/tbiomed_cpa.csv"
cpa_dataset_tbiomed_entity = "data/result/cpa/dataset/tbiomed/entity/tbiomed_cpa_dataset.csv"

""" tbiomed Horizontal"""
cpa_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/gt/cpa_gt.csv"
cpa_tbiomed_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/targets/cpa_targets.csv"
cpa_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/val/tables"
cpa_tbiomed_hor = "data/result/cpa/annotate/tbiomed/horizontal/tbiomed_cpa.csv"
cpa_dataset_tbiomed_hor = "data/result/cpa/dataset/tbiomed/horizontal/tbiomed_dataset_cpa.csv"

""" tbiodiv entity"""
cpa_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/gt/cpa_gt.csv"
cpa_tbiodiv_entity_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/targets/cpa_targets.csv"
cpa_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/val/tables"
cpa_tbiodiv_entity = "data/result/cpa/annotate/tbiodiv/entity/tbiodiv_cpa.csv"
cpa_dataset_tbiodiv_entity = "data/result/cpa/dataset/tbiodiv/entity/tbiodiv_dataset_cpa.csv"

""" tbiodiv Horizontal"""
cpa_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/gt/cpa_gt.csv"
cpa_tbiodiv_hor_target = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/targets/cpa_targets.csv"
cpa_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/val/tables"
cpa_tbiodiv_hor = "data/result/cpa/annotate/tbiodiv/horizontal/tbiodiv_cpa.csv"
cpa_dataset_tbiodiv_hor = "data/result/cpa/dataset/tbiodiv/horizontal/tbiodiv_dataset_cpa.csv"

""" ===============
    CPA TEST TASK 2024 semtab challenge
    ===============
"""

test_cpa_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/targets/cpa_targets.csv"
test_cpa_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/wikidata/Test/tables"
test_cpa_wikidata_24 = "data/result/cpa/annotate/wikidata/wikidata_cpa_24_test_2_part1_2.csv"
test_cpa_dataset_wikidata_24 = "data/result/cpa/dataset/wikidata/wikidata_dataset_cpa_test_1.csv"

R2_test_cpa_target_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/Round2/Test/targets/cpa_targets.csv"
R2_test_cpa_dataset_table_path_wikidata_24 = "data/csv/tables/semtab2024/AcuracyTrack/Round2/Test/tables"
R2_test_cpa_wikidata_24 = "data/result/Round2/cpa/annotatewikidata_cpa_24_test.csv"
R2_test_cpa_dataset_wikidata_24 = "data/result/Round2/cpa/dataset/wikidata_dataset_cpa_test.csv"

""" tbiomed entity"""
test_cpa_target_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/targets/cpa_targets.csv"
test_cpa_dataset_table_tbiomed_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/entity/test/tables"
test_cpa_tbiomed_entity = "data/result/cpa/annotate/tbiomed/entity/tbiomed_cpa_test_1.csv"
test_cpa_dataset_tbiomed_entity = "data/result/cpa/dataset/tbiomed/entity/tbiomed_cpa_dataset_test.csv"

""" tbiomed Horizontal"""
test_cpa_target_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/targets/cpa_targets.csv"
test_cpa_dataset_table_tbiomed_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiomedical2/horizontal/test/tables"
test_cpa_tbiomed_hor = "data/result/cpa/annotate/tbiomed/horizontal/tbiomed_cpa_test_2.csv"
test_cpa_dataset_tbiomed_hor = "data/result/cpa/dataset/tbiomed/horizontal/tbiomed_dataset_cpa_test.csv"

""" tbiodiv entity"""
test_cpa_target_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/targets/cpa_targets.csv"
test_cpa_dataset_table_tbiodiv_entity = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/entity/test/tables"
test_cpa_tbiodiv_entity = "data/result/cpa/annotate/tbiodiv/entity/tbiodiv_cpa_test_2.csv"
test_cpa_dataset_tbiodiv_entity = "data/result/cpa/dataset/tbiodiv/entity/tbiodiv_dataset_cpa_test.csv"

""" tbiodiv Horizontal"""
test_cpa_target_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/targets/cpa_targets.csv"
test_cpa_dataset_table_tbiodiv_hor = "data/csv/tables/semtab2024/AcuracyTrack/tbiodiv2/horizontal/test/tables"
test_cpa_tbiodiv_hor = "data/result/cpa/annotate/tbiodiv/horizontal/tbiodiv_cpa_test_2.csv"
test_cpa_dataset_tbiodiv_hor = "data/result/cpa/dataset/tbiodiv/horizontal/tbiodiv_dataset_cpa_test.csv"


# from SPARQLWrapper import SPARQLWrapper, JSON

# sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# query = """
# SELECT DISTINCT ?property ?propertyLabel WHERE {
#   ?entity ?property wd:Q422211.
#   FILTER (?property != wd:P31).
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
# }
# """

# sparql.setQuery(query)
# sparql.setReturnFormat(JSON)

# results = sparql.query().convert()

# for result in results["results"]["bindings"]:
#     property_id = result["property"]["value"]
#     if "direct" in property_id:
#         property_label = result["propertyLabel"]["value"]
#         print("Property ID:", property_id)
#         print("Property Label:", property_label)
#         print()