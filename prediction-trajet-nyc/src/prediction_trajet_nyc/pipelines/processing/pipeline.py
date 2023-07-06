from kedro.pipeline import Pipeline, node

from .nodes import *

def create_pipeline():
    return Pipeline(  
        [
            node(
                remove_passenger_count,
                ["train_data", "test_data"],
                dict(train_remove_passenger="train_remove_passenger", test_remove_passenger="test_remove_passenger")
            ),
            node(
                remove_extrem_values, 
                "train_remove_passenger", 
                "train_remove_values",
                ),

            node(
                map_store_and_fwd_flag, 
                ["train_remove_values","test_remove_passenger"],
                dict(train_map_store="train_map_store",test_map_sotre="test_map_sotre")
                ),
            node(
                decompose_date, 
                ["train_map_store", "test_map_sotre"], 
                dict(train_decompose_date="train_decompose_date", test_decompose_date="test_decompose_date")
            ),
            node(
                add_is_holidays, 
                ["train_decompose_date", "test_decompose_date"], 
                dict(train_is_holiday="train_is_holiday", test_is_holiday="test_is_holiday")
            ),
            node(
                compute_distances, 
                ["train_is_holiday", "test_is_holiday"], 
                dict(train_compute_distances="train_compute_distances", test_compute_distances="test_compute_distances")
                ),
            node(
            split_dataset,
            ["train_compute_distances", "params:test_ratio"],
            dict(
                    X_train="X_train",
                    y_train="y_train",
                    X_test="X_test",
                    y_test="y_test"
                )
            )
    ])
                