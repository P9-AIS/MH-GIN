import pickle, os, csv

def save_processed_traj_data(dataset_identifier, process_data_path, traj_data,
                              min_time_interval, max_time_interval,
                              min_traj_len, max_traj_len):
    processed_data_path = f"{process_data_path}/{dataset_identifier}_with_t_{min_time_interval}_{max_time_interval}_l_{min_traj_len}_{max_traj_len}.pkl"
    print(f"Saving processed trajectory data to {processed_data_path}")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(traj_data, f)
    print("Processed trajectory data saved successfully.")


def load_processed_traj_data(dataset_identifier, process_data_path, 
                            min_time_interval, max_time_interval,
                            min_traj_len, max_traj_len):
    processed_data_path = f"{process_data_path}/{dataset_identifier}_with_t_{min_time_interval}_{max_time_interval}_l_{min_traj_len}_{max_traj_len}.pkl"
    print(f"Loading processed trajectory data from {processed_data_path}")
    with open(processed_data_path, 'rb') as f:
        return pickle.load(f)


def save_processed_traj_status(dataset_identifier, process_data_path, traj_status,
                              min_time_interval, max_time_interval,
                              min_traj_len, max_traj_len):
    processed_data_path = f"{process_data_path}/{dataset_identifier}_with_t_{min_time_interval}_{max_time_interval}_l_{min_traj_len}_{max_traj_len}_status.pkl"
    print(f"Saving processed trajectory status to {processed_data_path}")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(traj_status, f)
    print("Processed trajectory status saved successfully.")


def load_processed_traj_status(dataset_identifier, process_data_path,
                              min_time_interval, max_time_interval,
                              min_traj_len, max_traj_len):
    processed_data_path = f"{process_data_path}/{dataset_identifier}_with_t_{min_time_interval}_{max_time_interval}_l_{min_traj_len}_{max_traj_len}_status.pkl"
    print(f"Loading processed trajectory status from {processed_data_path}")
    with open(processed_data_path, 'rb') as f:
        return pickle.load(f)


def save_index(index_identifier, index_data_path, index_data):
    index_path = f"{index_data_path}/{index_identifier}.pkl"
    if not os.path.exists(f'{index_data_path}'):
        os.makedirs(f'{index_data_path}')
    with open(index_path, 'wb') as f:
        pickle.dump(index_data, f)
    print(f"Index saved to {index_path}")


def load_index(index_identifier, index_data_path):
    index_path = f"{index_data_path}/{index_identifier}.pkl"
    with open(index_path, 'rb') as f:
        return pickle.load(f)


def save_index_statistics(index_identifier, result_data_path, build_time, build_memory):
    output_csv = f"{result_data_path}/IndexStatistics.csv"
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Index Identifier", "Build Time (s)", "Build Memory (MB)"])
        csv_writer.writerow([index_identifier, build_time, build_memory])


def save_dataset_statistics(dataset_identifier, output_csv, data_size_mb, trajectory_number, recordNum):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Dataset Identifier", "Data Size (Mb)", "Trajectory Number", "Record Number",
                                 "Avg Record Number per Trajectory"])
        csv_writer.writerow([dataset_identifier, f"{data_size_mb:.2f}", trajectory_number, recordNum,
                             f"{recordNum / trajectory_number:.2f}"])
