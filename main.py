from object_task import *
from evaluation.cea_evaluator import CEA_Evaluator
from finetuning.utils import compare_csv, csv_to_jsonl, fineTuningJobID, inference, annotateCea, combineJsonFile
from evaluation.cta_evaluator import CTA_Evaluator
from evaluation.ra_evaluator import RA_Evaluator
from evaluation.td_evaluator import TD_Evaluator
from evaluation.cpa_evaluator import CPA_Evaluator


if __name__ == "__main__":
    while True:
        print("************************************************************")
        print("************************************************************")
        print("1. make csv dataset")
        print("2. make json dataset")
        print("3. Combine json datasets")
        print("4. Create FinTuning Job ID in openAI")
        print("5. Make simple inference with model finetuned")
        print("6. Annotate table with model finetuned")
        print("7. Evaluate model with semtab metric")
        print("8. Quit")
        print("************************************************************")
        print("************************************************************")

        choice = input("\nPlease select an option: ")

        if choice == "1":
            """========== CEA Task============"""
            """ LLM track"""
            # cea_task_llm._makeDataset(header=True)
            # cea_task_wikidata_23._makeDataset(header=True, col_before_row=False)
            """ Accuracy Track """
            # cea_task_wikidata_24._makeDataset(header=True, col_before_row=False)
            # cea_task_tbiodiv_hor._makeDataset(header=True, comma_in_cell=True)
            # cea_task_tbiomed_hor._makeDataset(header=True, comma_in_cell=True)
            # cea_task_tbiomed_entity._makeDataset(is_vertical=True, comma_in_cell=True)
            # cea_task_tbiodiv_entity._makeDataset(is_vertical=True, comma_in_cell=True)
            # test_cea_task_llm._makeDataset(
            #     header=True,
            #     is_train=False,
            #     transpose=True
            # )
            # test_cea_task_tbiodiv_hor._makeDataset(header=True, comma_in_cell=True, is_train=False)
            # test_cea_task_tbiomed_hor._makeDataset(header=True, comma_in_cell=True, is_train=False)
            # test_cea_task_tbiodiv_entity._makeDataset(is_vertical=True, comma_in_cell=True, is_train=False)
            # test_cea_task_tbiomed_entity._makeDataset(is_vertical=True, comma_in_cell=True, is_train=False)
            # test_cea_task_wikidata_24._makeDataset(header=True, col_before_row=False, is_train=False, split=639261)
   
            
            """========== CTA Task============"""
            """ Accuracy Track """
            # cta_task_wikidata_23._makeDataset()
            # cta_task_tfood_hor._makeDataset()
            # cta_task_tbiodiv_hor._makeDataset()
            # cta_task_tbiomed_hor._makeDataset()
            # cta_task_wikidata_24._makeDataset()
            
            # test_cta_task_tbiodiv_hor._makeDataset(is_train=False)
            # test_cta_task_tbiomed_hor._makeDataset(is_train=False)
            # R2_test_cta_task_wikidata_24._makeDataset(is_train=False)
            
            """========== CPA Task============"""
            # cpa_task_tbiodiv_entity._makeDataset()
            # cpa_task_tbiomed_entity._makeDataset()
            # cpa_task_wikidata_24._makeDataset(is_entity=False)
            # cpa_task_tbiomed_hor._makeDataset(is_entity=False, is_horizontal=True)
            # cpa_task_tbiodiv_hor._makeDataset(is_entity=False, is_horizontal=True)
            """========== CPA Test Task============"""
            # test_cpa_task_tbiodiv_entity._makeDataset()
            # test_cpa_task_tbiomed_entity._makeDataset()
            # test_cpa_task_wikidata_24._makeDataset(is_entity=False)
            # test_cpa_task_tbiodiv_hor._makeDataset(is_entity=False, is_horizontal=True)
            # test_cpa_task_tbiomed_hor._makeDataset(is_entity=False, is_horizontal=True)
            """========== RA Task============"""
            # ra_task_tbiodiv_hor._makeDataset(header=True)
            # ra_task_tbiomed_hor._makeDataset(header=True)
            # test_ra_task_tbiodiv_hor._makeDataset(header=True, is_train=False)
            # test_ra_task_tbiomed_hor._makeDataset(header=True, is_train=False)
            """========== TD Task============"""
            # td_task_tbiodiv_entity._makeDataset()
            # td_task_tbiodiv_hor._makeDataset(n=4)
            # td_task_tbiomed_entity._makeDataset()
            # td_task_tbiomed_hor._makeDataset(n=4)
            """========== TD Task============"""
            # test_td_task_tbiodiv_entity._makeDataset(is_train=False)
            # test_td_task_tbiodiv_hor._makeDataset(is_train=False)
            # test_td_task_tbiomed_entity._makeDataset(is_train=False)
            # test_td_task_tbiomed_hor._makeDataset(is_train=False)
            print("\n")
        elif choice == "2":
            """ llm 2024 """
            # cea_task_llm._csv_to_jsonl(
            #     csv_path=cea_dataset_llm,
            #     json_path=cea_dataset_json_llm
            # )
            # cea_task_wikidata_24._csv_to_jsonl(
            #     csv_path=cea_dataset_wikidata_24,
            #     json_path=cea_dataset_json_wikidata_24
            # )
            """ Accuracy Track 2024"""
            """ ===RA===="""
            # ra_task_tbiomed_hor._csv_to_jsonl(
            #     csv_path=ra_dataset_tbiomed_hor,
            #     json_path=ra_dataset_json_tbiomed_hor
            # )
            # ra_task_tbiodiv_hor._csv_to_jsonl(
            #     csv_path=ra_dataset_tbiodiv_hor,
            #     json_path=ra_dataset_json_tbiodiv_hor
            # )
            """ ===cta===="""
            # cta_task_wikidata_23._csv_to_jsonl(
            #     csv_path=cta_dataset_wikidata_23, 
            #     json_path=cta_dataset_json_wikidata_23
            # )
            # cta_task_tfood_hor._csv_to_jsonl(
            #     csv_path=cta_dataset_tfood_hor,
            #     json_path=cta_dataset_json_tfood_hor
            # )
            # cta_task_tbiodiv_hor._csv_to_jsonl(
            #     csv_path=cta_dataset_tbiodiv_hor,
            #     json_path=cta_dataset_json_tbiodiv_hor
            # )
            # cta_task_tbiomed_hor._csv_to_jsonl(
            #     csv_path=cta_dataset_tbiomed_hor,
            #     json_path=cta_dataset_json_tbiomed_hor
            # )
            # cta_task_wikidata_24._csv_to_jsonl(
            #     csv_path=cta_dataset_wikidata_24, 
            #     json_path=cta_dataset_json_wikidata_24
            # )
            """ =====TD TASK=========="""
            # td_task_tbiodiv_entity._csv_to_jsonl(
            #     csv_path=td_dataset_tbiodiv_entity, 
            #     json_path=td_dataset_json_tbiodiv_entity
            # )
            # td_task_tbiodiv_hor._csv_to_jsonl(
            #     csv_path=td_dataset_tbiodiv_hor, 
            #     json_path=td_dataset_json_tbiodiv_hor
            # )
            # td_task_tbiomed_entity._csv_to_jsonl(
            #     csv_path=td_dataset_tbiomed_entity, 
            #     json_path=td_dataset_json_tbiomed_entity
            # )
            # td_task_tbiomed_hor._csv_to_jsonl(
            #     csv_path=td_dataset_tbiomed_hor, 
            #     json_path=td_dataset_json_tbiomed_hor
            # )
            print("\n")
        elif choice == "3":
            combineJsonFile(
                path_folder=cea_full_json_path_folder,
                updated_json_path="train_llm1.jsonl"
            )
            # cta_task_tfood_hor._combineJsonFile(
            #     json_path=cta_full_json_path_folder,
            #     json_output_dataset=cta_full_json_train_dataset,
            #     split="train[:100%]"
            # )
            # cta_task_tfood_hor._combineJsonFile(
            #     json_path=cta_full_json_path_folder,
            #     json_output_dataset=cta_full_json_val_dataset,
            #     split="val[:10%]"
            # )
            print("\n")
        elif choice == "4":
            gpttuner.fineTuningJobID(
                training_path="train_table_detection_update.jsonl", 
                validation_file_path="val_table_detection_update.jsonl"
            )
            # gpttuner.fineTuningJobID(
            #     training_path="train_llm1.jsonl",
            #     # validation_file_path="val_llm1.jsonl"
            # )
            print("\n")
        elif choice == "5":
            inference(model=ra_model_finetuned)
            print("\n")
        elif choice == "6":
            """ CEA TASK"""
            """wikidata 2023"""

            # cea_task_wikidata_23._annotate(
            #     comma_in_cell=False,
            #     model=cea_model_finetuned,
            #     col_before_row=False
            # )
            """tfood entity"""
            # cea_task_tfood_entity._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned
            # )
            
            """ LLM 2024 semtab"""
            # cea_task_wikidata_24._annotate(
            #     comma_in_cell=False,
            #     model=cea_model_finetuned_2,
            #     col_before_row=False,
            #     is_symbolic=True
            # )
            # cea_task_tbiodiv_entity._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_3,
            #     is_symbolic=True
            # )
            # cea_task_tbiomed_entity._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_3,
            #     is_symbolic=True
            # )
            # cea_task_tbiodiv_hor._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_3,
            # )
            # cea_task_llm._annotate(
            #     model=cea_model_finetuned,
            #     # is_symbolic=True
            # )
            # test_cea_task_wikidata_24._annotate(
            #     comma_in_cell=False,
            #     model=cea_model_finetuned_4,
            #     col_before_row=False,
            #     # is_llm=False,
            #     # is_symbolic=True,
            #     # is_context=True,
            #     split=79995
            # )
            """ ==========Test CEA========="""
            # test_cea_task_llm._annotate(
            #     model=cea_model_finetuned_4,
            #     is_symbolic=True,
            #     is_context=True,
            #     split=17040
            # )
            # test_cea_task_tbiodiv_entity._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_4,
            #     is_symbolic=True,
            # )
            # test_cea_task_tbiodiv_hor._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_3,
            #     is_symbolic=True,
            #     split=3030
            # )
            # test_cea_task_tbiomed_entity._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_4,
            #     is_symbolic=True
            # )
            # test_cea_task_tbiomed_hor._annotate(
            #     comma_in_cell=True,
            #     model=cea_model_finetuned_4,
            #     is_symbolic=True,
            #     split=325000
            # )
            """ ==========train CTA========="""
            # cta_task_wikidata_24._annotate(model=cta_model_finetuned_1)
            # cta_task_tbiomed_hor._annotate(model=cta_model_finetuned_1)
            # cta_task_tbiodiv_hor._annotate(model=cta_model_finetuned_1)
            """ ==========Test CTA========="""
            # test_cta_task_wikidata_24._annotate(model=cta_model_finetuned_1, is_combine_approach=True, split=36152)
            # test_cta_task_tbiomed_hor._annotate(model=cta_model_finetuned_1, split=44856, is_combine_approach=True)
            # test_cta_task_tbiodiv_hor._annotate(model=cta_model_finetuned_1, is_combine_approach=True, split=34334)
            # R2_test_cta_task_wikidata_24._annotate(model=cta_model_finetuned_1, is_combine_approach=False, split=65000)
            # R2_test_cta_task_wikidata_24._annotate(model=cta_model_finetuned_1, is_combine_approach=True)
            """========== CPA Task============"""
            # cpa_task_tbiodiv_entity._annotate(model=ra_model_finetuned, is_horizontal=True, split=3)
            # cpa_task_tbiomed_entity._annotate()
            # cpa_task_wikidata_24._annotate(split=107)
            # cpa_task_tbiomed_hor._annotate(model=ra_model_finetuned, is_horizontal=True, split=0)
            """========== CPA TEST Task============"""
            # test_cpa_task_wikidata_24._annotate(model=ra_model_finetuned, split=40481)
            # test_cpa_task_tbiodiv_hor._annotate(model=ra_model_finetuned, is_horizontal=True, split=3671)
            # test_cpa_task_tbiomed_hor._annotate(model=ra_model_finetuned, is_horizontal=True, split=7031)
            # test_cpa_task_tbiodiv_entity._annotate(model=ra_model_finetuned, is_horizontal=True, split=9741)
            # test_cpa_task_tbiomed_entity._annotate(model=ra_model_finetuned, is_horizontal=True, split=6719)
            """ ==========Test RA========="""
            # test_ra_task_tbiodiv_hor._annotate(model=ra_model_finetuned, split=492000)
            # test_ra_task_tbiomed_hor._annotate(model=ra_model_finetuned, split=233848)
            # ra_task_tbiodiv_hor._annotate(model=ra_model_finetuned)
            """========== TD Task============"""
            # td_task_tbiodiv_entity._annotate(model=td_model_finetuned)
            # test_td_task_tbiodiv_entity._annotate(model=td_model_finetuned, split=13793)
            # test_td_task_tbiomed_entity._annotate(model=td_model_finetuned, split=9510)
            # test_td_task_tbiodiv_hor._annotate(model=td_model_finetuned, split=12172)
            # test_td_task_tbiomed_hor._annotate(model=td_model_finetuned, split=7180)
            # test_task_a_wordnet._annotate()
            # test_task_a_geoname._annotate(is_entity=True, split=570000)
            # test_task_a_biological._annotate(is_entity=True, is_biological=True, split=0)
            # test_task_a_cellular._annotate(is_entity=True, is_biological=True, split=70000)
            # test_task_a_molecular._annotate(is_entity=True, is_biological=True, split=0)
            print("\n")
        elif choice == "7":
            _client_payload = {}
            """ wikidata """
            # _client_payload["submission_file_path"] = cea_wikidata
            # aicrowd_evaluator = CEA_Evaluator(cea_target_wikidata)  # ground truth
            
            
            """ tfood vertical"""
            # _client_payload["submission_file_path"] = cea_tfood_entity
            # aicrowd_evaluator = CEA_Evaluator(cea_target_tfood_entity)  # ground truth
            
            _client_payload["submission_file_path"] = cea_wikidata_24
            aicrowd_evaluator = CEA_Evaluator(cea_target_wikidata_24)  # ground truth
            
            # _client_payload["submission_file_path"] = cea_tbiodiv_entity
            # aicrowd_evaluator = CEA_Evaluator(cea_target_tbiodiv_entity)  # ground truth
            
            # _client_payload["submission_file_path"] = cea_tbiomed_entity
            # aicrowd_evaluator = CEA_Evaluator(cea_target_tbiomed_entity) 
            
            # _client_payload["submission_file_path"] = cea_tbiodiv_hor
            # aicrowd_evaluator = CEA_Evaluator(cea_target_tbiodiv_hor) 
            
            """ tfood horizontal """
            # _client_payload["submission_file_path"] = cea_wikidata_24
            # aicrowd_evaluator = CEA_Evaluator(cea_target_wikidata_24)  # ground truth
        
            # result = aicrowd_evaluator._evaluate(_client_payload)
            # print(result)
            
            """ ================CTA=========================="""
            # wikidata 24
            _client_payload["submission_file_path"] = cta_wikidata_1_24
            # _client_payload["submission_file_path"] = cta_wikidata_24
            aicrowd_evaluator = CTA_Evaluator(cta_target_wikidata_24)

            # tbiomed hor
            # _client_payload["submission_file_path"] = cta_tbiomed_hor
            # aicrowd_evaluator = CTA_Evaluator(cta_target_tbiomed_hor) 
            
            # tbiodiv hor
            # _client_payload["submission_file_path"] = cta_tbiodiv_hor
            # aicrowd_evaluator = CTA_Evaluator(cta_target_tbiodiv_hor)
            # _client_payload["submission_file_path"] = "tbiomed_cta_test.csv"
            # aicrowd_evaluator = CTA_Evaluator("tbiomed_cta_test_1.csv")  # ground truth
            
            _client_payload["submission_file_path"] = ra_tbiodiv_hor
            aicrowd_evaluator = RA_Evaluator(ra_target_tbiodiv_hor)  # ground truth
            
            _client_payload["submission_file_path"] = td_tbiodiv_entity
            aicrowd_evaluator = TD_Evaluator(td_target_tbiodiv_entity)  # ground truth
            
            _client_payload["submission_file_path"] = cpa_wikidata_24
            aicrowd_evaluator = CPA_Evaluator(cpa_target_wikidata_24)  # ground truth
            
            _client_payload["submission_file_path"] = cpa_tbiomed_hor
            aicrowd_evaluator = CPA_Evaluator(cpa_target_tbiomed_hor)  # ground truth
            
            # Evaluate
            result = aicrowd_evaluator._evaluate(_client_payload)
            # result = aicrowd_evaluator._evaluate_without_ancestor(_client_payload)
            print(result)
            
            print("\n")
        elif choice == "8":
            print("GoodBye !")
            break
        else:
            print("Invalid option. Please select a valid option.\n")