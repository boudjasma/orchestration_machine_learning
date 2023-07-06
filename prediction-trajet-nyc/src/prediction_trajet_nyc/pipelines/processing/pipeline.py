from kedro.pipeline import node

def create_pipeline():
    return Pipeline(
        [
            node(
                remove_passenger_count, 
                inputs='input_data', 
                outputs='clean_data'
                ),
            node(
                remove_extremvalues, 
                inputs='clean_data', 
                outputs='formatted_data'
                ),
            node(
                format_flags, 
                inputs='formatted_data', 
                outputs='formatted_data'
            ),
            node(
                format_data, 
                inputs='formatted_data', 
                outputs='formatted_data'
            ),
            node(
                is_holidays, 
                inputs='formatted_data', 
                outputs='formatted_data'
                ),
            node(
                compute_distances, 
                inputs='formatted_data', 
                outputs='processed_data'
                ),
            # Add other nodes and connections here
        
    ])
