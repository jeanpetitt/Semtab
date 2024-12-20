from finetuning.utils import Gpt3FinetuningProcess
from task.cea import CEATask
from task.cta import CTATask
from task.ra import RATask
from task.cpa import CPATask
from task.td import TDTask
from path_data.utils import *

gpttuner = Gpt3FinetuningProcess(
    name="annotator-ra-02-24", 
    # base_model="gpt-4o-mini-2024-07-18",
    base_model=ra_model_finetuned_1,
    hyperparameter = {
        'n_epochs': 6,
        'batch_size': 2,
        "learning_rate_multiplier": 8,
    },
    # hyperparameter= {}
)
gpttuner.get_api_key()

""" ===============
    TASK A 2024 LLM challenge
    ===============
"""
# task_a_wordnet = TaskA(
#     output_dataset=wordnet_dataset,
#     file_annotated=wordnet_annoted,

# )

# test_task_a_wordnet = TaskA(
#     output_dataset=test_wordnet_dataset,
#     file_annotated=test_wordnet_annoted,
# )
# test_task_a_geoname = TaskA(
#     output_dataset=test_geonames_dataset,
#     file_annotated=test_geonames_annotated,
# )
# test_task_a_biological = TaskA(
#     output_dataset=test_biological_dataset,
#     file_annotated=test_biological_annotated,
# )
# test_task_a_cellular = TaskA(
#     output_dataset=test_cellular_dataset,
#     file_annotated=test_cellular_annotated,
# )

# test_task_a_molecular = TaskA(
#     output_dataset=test_molecular_dataset,
#     file_annotated=test_molecular_annotated,
# )


""" ----------------2024----------------"""
"""==============TRAIN SET================"""
""" semtab LLM Track 2024 """
# cea_task_llm = CEATask(
#     output_dataset=cea_dataset_llm,
#     raw_output_dataset=raw_cea,
#     target_file=cea_target_llm,
#     table_path=cea_dataset_table_llm,
#     file_annotated=cea_llm,
#     target_file_to_annotate=cea_llm_target
# )
# """ semtab Accuracy Track 2024 """
# # wikidata
# cea_task_wikidata_23 = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_wikidata_23,
#     target_file=cea_target_wikidata_23,
#     table_path=cea_dataset_table_path_wikidata_23,
#     file_annotated=cea_wikidata_23,
#     target_file_to_annotate=cea_wikidata_target_23
# )

# # tfood horizontal
# cea_task_tfood_hor = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tfood_hor,
#     target_file=cea_target_tfood_hor,
#     table_path=cea_dataset_table_tfood_hor,
#     file_annotated=cea_tfood_hor,
#     target_file_to_annotate=cea_tfood_hor_target
# )
# cea_task_tfood_entity = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tfood_entity,
#     target_file=cea_target_tfood_entity,
#     table_path=cea_dataset_table_tfood_entity,
#     file_annotated=cea_tfood_entity,
#     target_file_to_annotate=cea_tfood_entity_target
# )

# """ ------2024-------"""
# # wikidata 24
# cea_task_wikidata_24 = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_wikidata_24,
#     target_file=cea_target_wikidata_24,
#     table_path=cea_dataset_table_path_wikidata_24,
#     file_annotated=cea_wikidata_24,
#     target_file_to_annotate=cea_wikidata_target_24
# )
# # tbiomed horizontal
# cea_task_tbiomed_hor = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tbiomed_hor,
#     target_file=cea_target_tbiomed_hor,
#     table_path=cea_dataset_table_tbiomed_hor,
#     file_annotated=cea_tbiomed_hor,
#     target_file_to_annotate=cea_tbiomed_hor_target
# )
# # tbiomed entity
# cea_task_tbiomed_entity = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tbiomed_entity,
#     target_file=cea_target_tbiomed_entity,
#     table_path=cea_dataset_table_tbiomed_entity,
#     file_annotated=cea_tbiomed_entity,
#     target_file_to_annotate=cea_tbiomed_entity_target
# )
# # tbiodiv horizontal
# cea_task_tbiodiv_hor = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tbiodiv_hor,
#     target_file=cea_target_tbiodiv_hor,
#     table_path=cea_dataset_table_tbiodiv_hor,
#     file_annotated=cea_tbiodiv_hor,
#     target_file_to_annotate=cea_tbiodiv_hor_target
# )
# # tbiodiv entity
# cea_task_tbiodiv_entity = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=cea_dataset_tbiodiv_entity,
#     target_file=cea_target_tbiodiv_entity,
#     table_path=cea_dataset_table_tbiodiv_entity,
#     file_annotated=cea_tbiodiv_entity,
#     target_file_to_annotate=cea_tbiodiv_entity_target
# )

# """==============TEST SET================"""
# test_cea_task_wikidata_24 = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=test_cea_dataset_wikidata_24,
#     target_file=test_cea_target_wikidata_24,
#     table_path=test_cea_dataset_table_path_wikidata_24,
#     file_annotated=test_cea_wikidata_24,
#     target_file_to_annotate=test_cea_target_wikidata_24
# )
# # LLM test
# test_cea_task_llm = CEATask(
#     output_dataset=test_cea_dataset_llm,
#     raw_output_dataset=raw_cea,
#     target_file=test_cea_target_llm,
#     table_path=test_cea_dataset_table_llm,
#     file_annotated=test_cea_llm,
#     target_file_to_annotate=test_cea_target_llm
# )

# # tbiomed horizontal
# test_cea_task_tbiomed_hor = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=test_cea_dataset_tbiomed_hor,
#     target_file=test_cea_target_tbiomed_hor,
#     table_path=test_cea_dataset_table_tbiomed_hor,
#     file_annotated=test_cea_tbiomed_hor,
#     target_file_to_annotate=test_cea_target_tbiomed_hor
# )
# # tbiomed entity
# test_cea_task_tbiomed_entity = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=test_cea_dataset_tbiomed_entity,
#     target_file=test_cea_target_tbiomed_entity,
#     table_path=test_cea_dataset_table_tbiomed_entity,
#     file_annotated=test_cea_tbiomed_entity,
#     target_file_to_annotate=test_cea_target_tbiomed_entity
# )
# # tbiodiv horizontal
# test_cea_task_tbiodiv_hor = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=test_cea_dataset_tbiodiv_hor,
#     target_file=test_cea_target_tbiodiv_hor,
#     table_path=test_cea_dataset_table_tbiodiv_hor,
#     file_annotated=test_cea_tbiodiv_hor,
#     target_file_to_annotate=test_cea_target_tbiodiv_hor
# )
# # tbiodiv entity
# test_cea_task_tbiodiv_entity = CEATask(
#     raw_output_dataset=raw_cea,
#     output_dataset=test_cea_dataset_tbiodiv_entity,
#     target_file=test_cea_target_tbiodiv_entity,
#     table_path=test_cea_dataset_table_tbiodiv_entity,
#     file_annotated=test_cea_tbiodiv_entity,
#     target_file_to_annotate=test_cea_target_tbiodiv_entity
# )



""" ===============
    CTA TASK 2023 and 2024 semtab challenge
    ===============
"""

""" ------2023-------"""
""" =========train set==============="""
# wikidata
# cta_task_wikidata_23 = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=cta_dataset_wikidata_23,
#     target_file=cta_target_wikidata_23,
#     table_path=cta_dataset_table_path_wikidata_23,
#     file_annotated=cta_wikidata_23,
#     target_file_to_annotate=cta_wikidata_target_23
# )

# # tfood horizontal
# cta_task_tfood_hor = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=cta_dataset_tfood_hor,
#     target_file=cta_target_tfood_hor,
#     table_path=cta_dataset_table_tfood_hor,
#     file_annotated=cta_tfood_hor,
#     target_file_to_annotate=cta_tfood_hor_target
# )

# """ ------2024-------"""
# """ ==================================================="""
# """ ==================Round 1================================="""
# """ ==================================================="""

# cta_task_wikidata_24 = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=cta_dataset_wikidata_24,
#     target_file=cta_target_wikidata_24,
#     table_path=cta_dataset_table_path_wikidata_24,
#     file_annotated=cta_wikidata_1_24,
#     target_file_to_annotate=cta_wikidata_target_24
# )

# # tbiomed horizontal
# cta_task_tbiomed_hor = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=cta_dataset_tbiomed_hor,
#     target_file=cta_target_tbiomed_hor,
#     table_path=cta_dataset_table_tbiomed_hor,
#     file_annotated=cta_tbiomed_hor,
#     target_file_to_annotate=cta_tbiomed_hor_target
# )
# # tbiodiv horizontal
# cta_task_tbiodiv_hor = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=cta_dataset_tbiodiv_hor,
#     target_file=cta_target_tbiodiv_hor,
#     table_path=cta_dataset_table_tbiodiv_hor,
#     file_annotated=cta_tbiodiv_hor,
#     target_file_to_annotate=cta_tbiodiv_hor_target
# )
# """ =========test set==============="""
# test_cta_task_wikidata_24 = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=test_cta_dataset_wikidata_24,
#     target_file=test_cta_target_wikidata_24,
#     table_path=test_cta_dataset_table_path_wikidata_24,
#     file_annotated=test_cta_wikidata_24,
#     target_file_to_annotate=test_cta_target_wikidata_24
# )


# # tbiomed horizontal
# test_cta_task_tbiomed_hor = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=test_cta_dataset_tbiomed_hor,
#     target_file=test_cta_target_tbiomed_hor,
#     table_path=test_cta_dataset_table_tbiomed_hor,
#     file_annotated=test_cta_tbiomed_hor,
#     target_file_to_annotate=test_cta_target_tbiomed_hor
# )
# # tbiodiv horizontal
# test_cta_task_tbiodiv_hor = CTATask(
#     raw_output_dataset=raw_cta,
#     output_dataset=test_cta_dataset_tbiodiv_hor,
#     target_file=test_cta_target_tbiodiv_hor,
#     table_path=test_cta_dataset_table_tbiodiv_hor,
#     file_annotated=test_cta_tbiodiv_hor,
#     target_file_to_annotate=test_cta_target_tbiodiv_hor
# )


""" ===============
    CPA 2024 semtab challenge
    ===============
"""
cpa_task_wikidata_24 = CPATask(
    dataset_name="wikidata_24",
    output_dataset=cpa_dataset_wikidata_24,
    target_file_gt=cpa_target_wikidata_24,
    target_file=cpa_wikidata_target_24,
    table_path=cpa_dataset_table_path_wikidata_24,
    file_annotated=cpa_wikidata_24
)

cpa_task_tbiodiv_entity = CPATask(
    dataset_name="tbiodiv_entity",
    output_dataset=cpa_dataset_tbiodiv_entity,
    target_file_gt=cpa_target_tbiodiv_entity,
    table_path=cpa_dataset_table_tbiodiv_entity,
    file_annotated=cpa_tbiodiv_entity,
    target_file=cpa_tbiodiv_entity_target
)

cpa_task_tbiomed_entity = CPATask(
    dataset_name="tbiomed_entity",
    output_dataset=cpa_dataset_tbiomed_entity,
    target_file_gt=cpa_target_tbiomed_entity,
    table_path=cpa_dataset_table_tbiomed_entity,
    file_annotated=cpa_tbiomed_entity,
    target_file=cpa_tbiomed_entity_target
)

cpa_task_tbiomed_hor = CPATask(
    dataset_name="tbiomed_hor",
    output_dataset=cpa_dataset_tbiomed_hor,
    target_file_gt=cpa_target_tbiomed_hor,
    table_path=cpa_dataset_table_tbiomed_hor,
    file_annotated=cpa_tbiomed_hor,
    target_file=cpa_tbiomed_hor_target
)

cpa_task_tbiodiv_hor = CPATask(
    dataset_name="tbiodiv_horizontal",
    output_dataset=cpa_dataset_tbiodiv_hor,
    target_file_gt=cpa_target_tbiodiv_hor,
    table_path=cpa_dataset_table_tbiodiv_hor,
    file_annotated=cpa_tbiodiv_hor,
    target_file=cpa_tbiodiv_hor_target
)
""" ===============
    CPA TEST 2024 semtab challenge
    ===============
"""
test_cpa_task_tbiodiv_entity = CPATask(
    dataset_name="test_tbiodiv_entity",
    output_dataset=test_cpa_dataset_tbiodiv_entity,
    target_file_gt=test_cpa_target_tbiodiv_entity,
    table_path=test_cpa_dataset_table_tbiodiv_entity,
    file_annotated=test_cpa_tbiodiv_entity,
    target_file=test_cpa_target_tbiodiv_entity
)

test_cpa_task_tbiomed_entity = CPATask(
    dataset_name="test_tbiomed_entity",
    output_dataset=test_cpa_dataset_tbiomed_entity,
    target_file_gt=test_cpa_target_tbiomed_entity,
    table_path=test_cpa_dataset_table_tbiomed_entity,
    file_annotated=test_cpa_tbiomed_entity,
    target_file=test_cpa_target_tbiomed_entity
)

test_cpa_task_wikidata_24 = CPATask(
    dataset_name="test_wikidata_24",
    output_dataset=test_cpa_dataset_wikidata_24,
    target_file_gt=test_cpa_target_wikidata_24,
    table_path=test_cpa_dataset_table_path_wikidata_24,
    file_annotated=test_cpa_wikidata_24,
    target_file=test_cpa_target_wikidata_24
)

test_cpa_task_tbiomed_hor = CPATask(
    dataset_name="test_tbiomed_horizontal",
    output_dataset=test_cpa_dataset_tbiomed_hor,
    target_file_gt=test_cpa_target_tbiomed_hor,
    table_path=test_cpa_dataset_table_tbiomed_hor,
    file_annotated=test_cpa_tbiomed_hor,
    target_file=test_cpa_target_tbiomed_hor
)

test_cpa_task_tbiodiv_hor = CPATask(
    dataset_name="test_tbiodiv_horizontal",
    output_dataset=test_cpa_dataset_tbiodiv_hor,
    target_file_gt=test_cpa_target_tbiodiv_hor,
    table_path=test_cpa_dataset_table_tbiodiv_hor,
    file_annotated=test_cpa_tbiodiv_hor,
    target_file=test_cpa_target_tbiodiv_hor
)


""" =============================="""
""" ===============
    RA TASK 2024 semtab challenge
    ===============
"""
ra_task_tbiodiv_hor = RATask(
    output_dataset=ra_dataset_tbiodiv_hor,
    target_file_gt=ra_target_tbiodiv_hor,
    table_path=ra_dataset_table_tbiodiv_hor,
    file_annotated=ra_tbiodiv_hor,
    target_file=ra_tbiodiv_hor_target
)
ra_task_tbiomed_hor = RATask(
    output_dataset=ra_dataset_tbiomed_hor,
    target_file_gt=ra_target_tbiomed_hor,
    table_path=ra_dataset_table_tbiomed_hor,
    file_annotated=ra_tbiomed_hor,
    target_file=ra_tbiomed_hor_target
)

""" ===============
    RA TEST TASK 2024 semtab challenge
    ===============
"""
test_ra_task_tbiodiv_hor = RATask(
    output_dataset=test_ra_dataset_tbiodiv_hor,
    target_file_gt=test_ra_target_tbiodiv_hor,
    table_path=test_ra_dataset_table_tbiodiv_hor,
    file_annotated=test_ra_tbiodiv_hor,
    target_file=test_ra_target_tbiodiv_hor
)
test_ra_task_tbiomed_hor = RATask(
    output_dataset=test_ra_dataset_tbiomed_hor,
    target_file_gt=test_ra_target_tbiomed_hor,
    table_path=test_ra_dataset_table_tbiomed_hor,
    file_annotated=test_ra_tbiomed_hor,
    target_file=test_ra_target_tbiomed_hor
)

""" ===============
    TD TASK 2024 semtab challenge
    ===============
"""
# td_task_tbiodiv_hor = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=td_dataset_tbiodiv_hor,
#     target_file=td_target_tbiodiv_hor,
#     table_path=td_dataset_table_tbiodiv_hor,
#     file_annotated=td_tbiodiv_hor,
#     target_file_to_annotate=td_tbiodiv_hor_target
# )
# td_task_tbiodiv_entity = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=td_dataset_tbiodiv_entity,
#     target_file=td_target_tbiodiv_entity,
#     table_path=td_dataset_table_tbiodiv_entity,
#     file_annotated=td_tbiodiv_entity,
#     target_file_to_annotate=td_tbiodiv_entity_target
# )

# td_task_tbiomed_hor = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=td_dataset_tbiomed_hor,
#     target_file=td_target_tbiomed_hor,
#     table_path=td_dataset_table_tbiomed_hor,
#     file_annotated=td_tbiomed_hor,
#     target_file_to_annotate=td_tbiomed_hor_target
# )

# td_task_tbiomed_entity = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=td_dataset_tbiomed_entity,
#     target_file=td_target_tbiomed_entity,
#     table_path=td_dataset_table_tbiomed_entity,
#     file_annotated=td_tbiomed_entity,
#     target_file_to_annotate=td_tbiomed_entity_target
# )


# """ ===============
#     TD TEST TASK 2024 semtab challenge
#     ===============
# """
# test_td_task_tbiodiv_hor = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=test_td_dataset_tbiodiv_hor,
#     target_file=test_td_target_tbiodiv_hor,
#     table_path=test_td_dataset_table_tbiodiv_hor,
#     file_annotated=test_td_tbiodiv_hor,
#     target_file_to_annotate=test_td_target_tbiodiv_hor
# )
# test_td_task_tbiodiv_entity = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=test_td_dataset_tbiodiv_entity,
#     target_file=test_td_target_tbiodiv_entity,
#     table_path=test_td_dataset_table_tbiodiv_entity,
#     file_annotated=test_td_tbiodiv_entity,
#     target_file_to_annotate=test_td_target_tbiodiv_entity
# )

# test_td_task_tbiomed_hor = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=test_td_dataset_tbiomed_hor,
#     target_file=test_td_target_tbiomed_hor,
#     table_path=test_td_dataset_table_tbiomed_hor,
#     file_annotated=test_td_tbiomed_hor,
#     target_file_to_annotate=test_td_target_tbiomed_hor
# )

# test_td_task_tbiomed_entity = TDTask(
#     raw_output_dataset=raw_td,
#     output_dataset=test_td_dataset_tbiomed_entity,
#     target_file=test_td_target_tbiomed_entity,
#     table_path=test_td_dataset_table_tbiomed_entity,
#     file_annotated=test_td_tbiomed_entity,
#     target_file_to_annotate=test_td_target_tbiomed_entity
# )

""" ==================================================="""
""" ==================Round 2================================="""
""" ==================================================="""

"""=================CTA task========================="""
# R2_test_cta_task_wikidata_24 = CTATask(
#     raw_output_dataset=R2_raw_cta,
#     output_dataset=R2_test_cta_dataset_wikidata_24,
#     target_file=R2_test_cta_target_wikidata_24,
#     table_path=R2_test_cta_dataset_table_path_wikidata_24,
#     file_annotated=R2_test_cta_wikidata_24,
#     target_file_to_annotate=R2_test_cta_target_wikidata_24
# )

# R2_cta_task_tbiomed_hor = CTATask(
#     raw_output_dataset=R2_raw_cta,
#     output_dataset=R2_cta_dataset_tbiomed_hor,
#     target_file=R2_cta_target_tbiomed_hor,
#     table_path=R2_cta_dataset_table_tbiomed_hor,
#     file_annotated=R2_cta_tbiomed_hor,
#     target_file_to_annotate=R2_cta_tbiomed_hor_target
# )
# # tbiodiv horizontal
# R2_cta_task_tbiodiv_hor = CTATask(
#     raw_output_dataset=R2_raw_cta,
#     output_dataset=R2_cta_dataset_tbiodiv_hor,
#     target_file=R2_cta_target_tbiodiv_hor,
#     table_path=R2_cta_dataset_table_tbiodiv_hor,
#     file_annotated=R2_cta_tbiodiv_hor,
#     target_file_to_annotate=R2_cta_tbiodiv_hor_target
# )



# """=================TD task========================="""
# # tbiomed entity
# R2_test_td_task_tbiomed_entity = TDTask(
#     raw_output_dataset=R2_raw_td,
#     output_dataset=R2_td_dataset_tbiomed_entity,
#     target_file=R2_td_target_tbiomed_entity,
#     table_path=R2_td_dataset_table_tbiomed_entity,
#     file_annotated=R2_td_tbiomed_entity,
#     target_file_to_annotate=R2_td_tbiomed_entity_target
# )


# # tbiodiv entity
# R2_test_td_task_tbiodiv_entity = TDTask(
#     raw_output_dataset=R2_raw_td,
#     output_dataset=R2_td_dataset_tbiodiv_entity,
#     target_file=R2_td_target_tbiodiv_entity,
#     table_path=R2_td_dataset_table_tbiodiv_entity,
#     file_annotated=R2_td_tbiodiv_entity,
#     target_file_to_annotate=R2_td_tbiodiv_entity_target
# )
# # tbiomed horizontal
# R2_test_td_task_tbiomed_hor = TDTask(
#     raw_output_dataset=R2_raw_td,
#     output_dataset=R2_td_dataset_tbiomed_hor,
#     target_file=R2_td_target_tbiomed_hor,
#     table_path=R2_td_dataset_table_tbiomed_hor,
#     file_annotated=R2_td_tbiomed_hor,
#     target_file_to_annotate=R2_td_tbiomed_hor_target
# )
# # tbiodiv horizontal
# R2_test_td_task_tbiodiv_hor = TDTask(
#     raw_output_dataset=R2_raw_td,
#     output_dataset=R2_td_dataset_tbiodiv_hor,
#     target_file=R2_td_target_tbiodiv_hor,
#     table_path=R2_td_dataset_table_tbiodiv_hor,
#     file_annotated=R2_td_tbiodiv_hor,
#     target_file_to_annotate=R2_td_tbiodiv_hor_target
# )

# """=================CPA task========================="""
# # wikidata R2
# R2_test_cpa_task_wikidata_24 = CPATask(
#     output_dataset=R2_test_cpa_dataset_wikidata_24,
#     target_file_gt=R2_test_cpa_target_wikidata_24,
#     table_path=R2_test_cpa_dataset_table_path_wikidata_24,
#     file_annotated=R2_test_cpa_wikidata_24,
#     target_file=R2_test_cpa_target_wikidata_24
# )

# # tbiomed entity
# R2_test_cpa_task_tbiomed_entity = CPATask(
#     output_dataset=R2_cpa_dataset_tbiomed_entity,
#     target_file_gt=R2_cpa_target_tbiomed_entity,
#     table_path=R2_cpa_dataset_table_tbiomed_entity,
#     file_annotated=R2_cpa_tbiomed_entity,
#     target_file=R2_cpa_tbiomed_entity_target
# )
# # tbiodiv entity
# R2_test_cpa_task_tbiodiv_entity = CPATask(
#     output_dataset=R2_cpa_dataset_tbiodiv_entity,
#     target_file_gt=R2_cpa_target_tbiodiv_entity,
#     table_path=R2_cpa_dataset_table_tbiodiv_entity,
#     file_annotated=R2_cpa_tbiodiv_entity,
#     target_file=R2_cpa_tbiodiv_entity_target
# )
# # tbiomed horizontal
# R2_test_cpa_task_tbiomed_hor = CPATask(
#     output_dataset=R2_cpa_dataset_tbiomed_hor,
#     target_file_gt=R2_cpa_target_tbiomed_hor,
#     table_path=R2_cpa_dataset_table_tbiomed_hor,
#     file_annotated=R2_cpa_tbiomed_hor,
#     target_file=R2_cpa_tbiomed_hor_target
# )
# # tbiodiv horizontal
# R2_test_cpa_task_tbiodiv_hor = CPATask(
#     output_dataset=R2_cpa_dataset_tbiodiv_hor,
#     target_file_gt=R2_cpa_target_tbiodiv_hor,
#     table_path=R2_cpa_dataset_table_tbiodiv_hor,
#     file_annotated=R2_cpa_tbiodiv_hor,
#     target_file=R2_cpa_tbiodiv_hor_target
# )

# """=================RA task========================="""
# R2_ra_task_tbiodiv_hor = RATask(
#     output_dataset=R2_ra_dataset_tbiodiv_hor,
#     target_file_gt=R2_ra_target_tbiodiv_hor,
#     table_path=R2_ra_dataset_table_tbiodiv_hor,
#     file_annotated=R2_ra_tbiodiv_hor,
#     target_file=R2_ra_tbiodiv_hor_target
# )
# R2_ra_task_tbiomed_hor = RATask(
#     output_dataset=R2_ra_dataset_tbiomed_hor,
#     target_file_gt=R2_ra_target_tbiomed_hor,
#     table_path=R2_ra_dataset_table_tbiomed_hor,
#     file_annotated=R2_ra_tbiomed_hor,
#     target_file=R2_ra_tbiomed_hor_target
# )


# """=================CEA task========================="""
# ## Wikidata R2
# R2_test_cea_task_wikidata_24 = CEATask(
#     raw_output_dataset=R2_raw_cea,
#     output_dataset=R2_test_cea_dataset_wikidata_24,
#     target_file=R2_test_cea_target_wikidata_24,
#     table_path=R2_test_cea_dataset_table_path_wikidata_24,
#     file_annotated=R2_test_cea_wikidata_24,
#     target_file_to_annotate=R2_test_cea_target_wikidata_24
# )

# ## tbiomed entity
# R2_test_cea_task_tbiomed_entity = CEATask(
#     raw_output_dataset=R2_raw_cea,
#     output_dataset=R2_cea_dataset_tbiomed_entity,
#     target_file=R2_cea_target_tbiomed_entity,
#     table_path=R2_cea_dataset_table_tbiomed_entity,
#     file_annotated=R2_cea_tbiomed_entity,
#     target_file_to_annotate=R2_cea_tbiomed_entity_target
# )
# ## tbiodiv entity
# R2_test_cea_task_tbiodiv_entity = CEATask(
#     raw_output_dataset=R2_raw_cea,
#     output_dataset=R2_cea_dataset_tbiodiv_entity,
#     target_file=R2_cea_target_tbiodiv_entity,
#     table_path=R2_cea_dataset_table_tbiodiv_entity,
#     file_annotated=R2_cea_tbiodiv_entity,
#     target_file_to_annotate=R2_cea_tbiodiv_entity_target
# )

# ## tbiodiv Horizontal
# R2_test_cea_task_tbiodiv_hor = CEATask(
#     raw_output_dataset=R2_raw_cea,
#     output_dataset=R2_cea_dataset_tbiodiv_hor,
#     target_file=R2_cea_target_tbiodiv_hor,
#     table_path=R2_cea_dataset_table_tbiodiv_hor,
#     file_annotated=R2_cea_tbiodiv_hor,
#     target_file_to_annotate=R2_cea_tbiodiv_hor_target
# )

# ## tbiomed horizontal
# R2_test_cea_task_tbiomed_hor = CEATask(
#     raw_output_dataset=R2_raw_cea,
#     output_dataset=R2_cea_dataset_tbiomed_hor,
#     target_file=R2_cea_target_tbiomed_hor,
#     table_path=R2_cea_dataset_table_tbiomed_hor,
#     file_annotated=R2_cea_tbiomed_hor,
#     target_file_to_annotate=R2_cea_tbiomed_hor_target
# )
