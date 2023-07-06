from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline():
    return Pipeline(  
        [
            node(
                remove_passenger_count,
                "dataset",
                dict(train_remose_passenger="train_remove_passenger", test_remove_passenger="train_remose_passenger"),
                name="node1"
            ),
            # node(
            #     remove_passenger_count, 
            #     inputs=["train_data", "test_data"], 
            #     outputs= ["train_data_inter1", "test_data_inter1"]
            #     ),
            # node(
            #     remove_extrem_values, 
            #     inputs=["train_data_inter1", "test_data_inter1"], 
            #     outputs= ["train_data_inter2", "test_data_inter2"]
            #     ),
            # node(
            #     map_store_and_fwd_flag, 
            #     inputs=["train_data_inter2", "test_data_inter2"], 
            #     outputs= ["train_data_inter3", "test_data_inter3"]
            # ),
            # node(
            #     decompose_date, 
            #     inputs=["train_data_inter3", "test_data_inter3"], 
            #     outputs= ["train_data_inter4", "test_data_inter4"]
            # ),
            # node(
            #     add_is_holidays, 
            #     inputs=["train_data_inter4", "test_data_inter4"], 
            #     outputs= ["train_data_inter5", "test_data_inter5"]
            #     ),
            # node(
            #     compute_distances, 
            #     inputs=["train_data_inter5", "test_data_inter6"], 
            #     outputs= ["train_clean", "test_clean"]
            #     ),
            # node(
            # split_dataset,
            # ["train_clean", "params:test_ratio"],
            # dict(
            #         X_train="X_train",
            #         y_train="y_train",
            #         X_test="X_test",
            #         y_test="y_test"
            #     )
            # )
    ])
                