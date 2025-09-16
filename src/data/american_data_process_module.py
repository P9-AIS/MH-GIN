import argparse
import sys, os, pandas as pd
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
print(sys.path)
import src
from src.pipeline.experiment import Experiment
from src.data.data_filter import american_ais_data_filter
from src.data.data_utils import download_ais_dataset, analyze_traj_data, encode_traj_data, split_trajectory2
from src.pipeline.SaveLoadModule import save_processed_traj_data, load_processed_traj_data, save_dataset_statistics, \
    load_processed_traj_status, save_processed_traj_status
from src.utils.file_util import find_project_root

project_root = find_project_root(current_path)

raw_data_path = os.path.join(project_root, 'Data', 'RawData')
process_data_path = os.path.join(project_root, 'Data', 'ProcessedData')
data_result_path = os.path.join(project_root, 'Result')

time_col_name = "BaseDateTime"
Lat_col_name = "LAT"
Lon_col_name = "LON"


# time_formulation = "%d-%m-%YT%H:%M:%S"

def data_filter_draught(df, draught_sort_num=2):
    df['Second'] = pd.to_datetime(df[time_col_name], unit='s')
    df.sort_values(by=['MMSI', 'Second'], inplace=True)
    # filter
    df = df.groupby('MMSI').filter(lambda group: filter_by_draught(group, sort_num=draught_sort_num))
    return df


def filter_by_draught(group, sort_num=2):
    unique_draughts = group['Draft'].nunique()
    return unique_draughts >= sort_num


def process_dataframe(df, chunk_size=2, draught_count_diffs=2000):
    traj_data = {}
    grouped = df.groupby('MMSI')

    for mmsi, group in grouped:
        unique_draughts = []
        seen = set()
        for draught in group['Draft']:
            if draught not in seen:
                unique_draughts.append(draught)
                seen.add(draught)

        draught_chunks = [unique_draughts[i:i + chunk_size] for i in range(0, len(unique_draughts), chunk_size)]

        valid_group_count = 0
        for chunk in draught_chunks:
            if len(chunk) < chunk_size:
                continue

            draught_counts = group[group['Draft'].isin(chunk)].groupby('Draft').size()

            if len(draught_counts) < chunk_size:
                continue

            # [0, 2000]
            # count_diffs = abs(draught_counts.values.max() - draught_counts.values.min())
            # if count_diffs > draught_count_diffs:
            #     continue

            valid_group = group[group['Draft'].isin(chunk)]

            mmsi_id = f"{mmsi}_{valid_group_count}"
            traj_data[mmsi_id] = {
                "vessel_type": valid_group['VesselType'].tolist(),
                "vessel_width": valid_group['Width'].tolist(),
                "vessel_length": valid_group['Length'].tolist(),
                "vessel_draught": valid_group['Draft'].tolist(),
                "destination": valid_group['Destination'].tolist(),
                "cargo_type": valid_group['Cargo'].tolist(),
                "navi_status": valid_group['Status'].tolist(),
                "sog": valid_group['SOG'].tolist(),
                "rot": valid_group['ROT'].tolist(),
                "cog": valid_group['COG'].tolist(),
                "heading_angle": valid_group['Heading'].tolist(),
                "latitude": valid_group[Lat_col_name].tolist(),
                "longitude": valid_group[Lon_col_name].tolist(),
                "timestamp": valid_group[time_col_name].tolist()
            }
            valid_group_count += 1

    return traj_data


def load_ais_csv_dataset(args):
    #region Step 0: Check if processed data exists
    # Generate a unique identifier for the processed dataset
    if len(args.datasets) == 1:
        dataset_identifier = args.datasets[0]
    elif len(args.datasets) > 1:
        dataset_identifier = args.datasets[0] + args.datasets[-1]
    dataset_identifier = f"{dataset_identifier}_{args.datascalability}"
    output_file = os.path.join(process_data_path, f"{dataset_identifier}.csv")
    #endregion

    file_name_list = []
    for dataset in args.datasets:
        #region Step 1: DataSet Selections (dataset list and scalability)
        dataset_start, dataset_end = dataset.split("@")[0], int(dataset.split("@")[1])
        print(dataset_start, dataset_end)
        file_names = [dataset_start[:-2] + str(i).zfill(2) for i in
                      range(int(dataset_start.split(dataset_start[-3])[-1]), int(dataset_end) + 1)]
        file_name_list.extend(file_names)
    print(file_name_list)

    csv_file_list = download_ais_dataset(file_name_list, raw_data_path)
    print("begin load aisdk_dataset")

    df_list = []
    data_size_mb = 0
    for csv_file in csv_file_list:
        print(f"Reading file: {csv_file}")
        df = pd.read_csv(csv_file)
        # data filter
        df = american_ais_data_filter(df)
        df_list.append(df)
        data_size_mb += os.path.getsize(csv_file) / (1024 * 1024)
    df = pd.concat(df_list, ignore_index=True)
    print("end load aisdk_dataset")
    os.makedirs(process_data_path, exist_ok=True)
    df.to_csv(output_file, index=False)

def process_ais_multi_csv_dataset(args):
    #region Step 0: Check if processed data exists
    # Generate a unique identifier for the processed dataset
    if len(args.datasets) == 1:
        dataset_identifier = args.datasets[0]
    elif len(args.datasets) > 1:
        dataset_identifier = args.datasets[0] + args.datasets[-1]
    dataset_identifier = f"{dataset_identifier}_{args.datascalability}"
    max_time_interval = int(float(args.max_time_interval))
    max_traj_len = int(float(args.max_traj_len))
    if args.max_time_interval == '1e9':
        args.max_time_interval = 'inf'
    else:
        args.max_time_interval = int(args.max_time_interval)
    if args.max_traj_len == '1e9':
        args.max_traj_len = 'inf'
    else:
        args.max_traj_len = int(args.max_traj_len)
    processed_data_path = f"{process_data_path}/{dataset_identifier}_with_t_{args.min_time_interval}_{args.max_time_interval}_l_{args.min_traj_len}_{args.max_traj_len}.pkl"

    if args.data_load_cache and os.path.exists(processed_data_path):
        return (
            load_processed_traj_data(dataset_identifier, process_data_path,
                                     min_time_interval=args.min_time_interval,
                                     max_time_interval=args.max_time_interval,
                                     min_traj_len=args.min_traj_len,
                                     max_traj_len=args.max_traj_len),
            load_processed_traj_status(dataset_identifier,
                                       process_data_path,
                                       min_time_interval=args.min_time_interval,
                                       max_time_interval=args.max_time_interval,
                                       min_traj_len=args.min_traj_len,
                                       max_traj_len=args.max_traj_len)
        )
    #endregion

    # region Step 1: DataSet Selections (dataset list and scalability)
    file_name_list = []
    if args.data_download:
        for dataset in args.datasets:
            dataset_start, dataset_end = dataset.split("@")[0], int(dataset.split("@")[1])
            print(dataset_start, dataset_end)
            file_names = [dataset_start[:-2] + str(i).zfill(2) for i in
                          range(int(dataset_start.split(dataset_start[-3])[-1]), int(dataset_end) + 1)]
            file_name_list.extend(file_names)
        print(file_name_list)
        csv_file_list = download_ais_dataset(file_name_list, raw_data_path)
    else:
        for dataset_name in args.datasets:
            file_name_list.append(f"{dataset_name}_{args.datascalability}.csv")
        print(file_name_list)

        csv_file_list = []
        for file_name in file_name_list:
            csv_file_list.append(os.path.join(process_data_path, file_name))

    print("begin load ais_us_dataset")
    df_list = []
    data_size_mb = 0
    for csv_file in csv_file_list:
        print(f"Reading file: {csv_file}")
        df = pd.read_csv(csv_file)
        # data filter
        df = american_ais_data_filter(df)
        df = df[df["MMSI"] != 0]
        df_list.append(df)
        data_size_mb += os.path.getsize(csv_file) / (1024 * 1024)
    df = pd.concat(df_list, ignore_index=True)
    print("end load ais_us_dataset")

    # Calculate Data Size (Mb)
    datascalability = args.datascalability
    df = df.head(int(len(df) * datascalability))
    data_size_mb *= datascalability
    # endregion

    # Step 2 Data Filter and Process
    #region Step 2.1 Trajectory Generation according to connection_ratio
    # Convert timestamps to UNIX timestamp during data loading
    df[time_col_name] = pd.to_datetime(df[time_col_name]).astype('int64') / 10 ** 9
    df[time_col_name] = df[time_col_name].astype(int)
    #endregion

    #region Step 2.2 Filter invalid data
    df = data_filter_draught(df, draught_sort_num=2)
    traj_data = process_dataframe(df, chunk_size=2, draught_count_diffs=2000)
    #endregion

    # Step 3: Stacstic Result of datasets
    #region Step 3.1: Print result of datasets
    traj_data = split_trajectory2(traj_data, min_len=int(args.min_traj_len), max_len=max_traj_len,
                                  min_time_threshold=int(args.min_time_interval),
                                  max_time_threshold=max_time_interval)

    stats = analyze_traj_data(traj_data)
    traj_data = encode_traj_data(traj_data, stats)

    trajectory_number = len(traj_data)
    point_nums = stats['point_count']
    print(f"Data Size (Mb): {data_size_mb:.2f}")
    print(f"Record point Number: {point_nums}")
    print(f"Trajectory (MMSI) number: {trajectory_number}")

    # Generate the result in the requested format
    result_string = f"{str(file_name_list)} & {data_size_mb:.2f} & {trajectory_number} & {point_nums} \\\\"
    print(result_string)
    # endregion

    # region Step 3.2: Write result of datasets
    # Write the result to the specified CSV file
    os.makedirs(data_result_path, exist_ok=True)
    save_dataset_statistics(dataset_identifier, args.dataset_statistics_csv_path, data_size_mb, trajectory_number,
                            point_nums)
    #endregion

    #region Step 4: Save processed data
    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    save_processed_traj_data(dataset_identifier, process_data_path, traj_data,
                             min_time_interval=args.min_time_interval,
                             max_time_interval=args.max_time_interval,
                             min_traj_len=args.min_traj_len,
                             max_traj_len=args.max_traj_len)
    save_processed_traj_status(dataset_identifier, process_data_path, stats,
                               min_time_interval=args.min_time_interval,
                               max_time_interval=args.max_time_interval,
                               min_traj_len=args.min_traj_len,
                               max_traj_len=args.max_traj_len)
    print(f"Processed data saved to {processed_data_path}")
    #endregion

    return traj_data, stats


def hyperparameter_DataProcess(parser):
    parser.add_argument("--config", type=str,
                            default=f'{project_root}/config/config-US.yaml',
                            help='Path to configuration file')
    # Dataset Process
    # AIS_2023_12_11@31 
    # default=['AIS_2024_03_01@31', 'AIS_2024_04_01@30', 'AIS_2024_05_01@31']
    parser.add_argument("--datasets", type=list, default=['AIS_2024_04_01@30'])

    parser.add_argument("--datascalability", type=float, default=1)
    parser.add_argument("--data_download", type=bool, default=True)
    parser.add_argument("--data_load_cache", type=bool, default=False)

    parser.add_argument("--min_time_interval", type=int, default=0)
    parser.add_argument("--max_time_interval", type=str, default='1e9')
    parser.add_argument("--min_traj_len", type=int, default=0)
    parser.add_argument("--max_traj_len", type=str, default='1e9')
    parser.add_argument("--dataset_statistics_csv_path", type=str, default=f"{data_result_path}/DatasetStatistics.csv")
    return parser


def main(args):
    traj_data, stats = process_ais_multi_csv_dataset(args)
    print(f"Record point Number: {stats['point_count']}")
    print(f"Trajectory (MMSI) number: {stats['total_mmsi_count']}")
    print(f"Average trajectory length: {stats['avg_traj_length']:.2f}")
    print(f"Min/Max trajectory length: {stats['min_traj_length']}/{stats['max_traj_length']}")
    print(f"Average time interval: {stats['avg_time_interval']:.2f}s")
    print(f"Vessels with two draughts: {stats['vessel_draught_two_values_count']}")

if __name__ == '__main__':
    parser = hyperparameter_DataProcess(argparse.ArgumentParser(description="MTS-HGNN Data Processing"))
    exp = Experiment(run_fn=main, parser=parser,
                     config_path=src.config['config_dir'])
    exp.deal_data()
