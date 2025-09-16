import os, requests, zipfile
from tqdm import tqdm
import pandas as pd
import numpy as np

# Download raw data of AISDK and AISUS
def download_ais_dataset(file_name_list, raw_data_path):
    csv_file_path_list = []
    for file_name in file_name_list:
        # Step 1: Set base information about raw dataset
        global time_col_name, Lat_col_name, Lon_col_name, time_formulation
        if "aisdk" in file_name:
            download_url = "http://aisdata.ais.dk/2024/"
            csv_file_name = f"{file_name}.csv"
            if ("2006" in file_name):
                csv_file_name = file_name[:5] + "_" + file_name[5:].replace("-", "") + ".csv"
            print(csv_file_name)
            time_col_name, Lat_col_name, Lon_col_name = "# Timestamp", "Latitude", "Longitude"
            time_formulation = "%d/%m/%Y %H:%M:%S"
        else: # https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html
            download_url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/" + file_name[4:8] + "/"
            csv_file_name = f"{file_name}.csv"
            time_col_name, Lat_col_name, Lon_col_name = "BaseDateTime", "LAT", "LON"
            time_formulation = "%Y-%m-%dT%H:%M:%S"

        # Step 2: Check if CSV file already exists
        csv_file_path = os.path.join(raw_data_path, csv_file_name)

        if os.path.exists(csv_file_path):
            print(f"CSV file '{csv_file_path}' already exists. No download needed.")
            csv_file_path_list.append(csv_file_path)
            continue

        # Step 3: Download and unzip if CSV doesn't exist
        def attempt_download(url, zip_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_length = int(response.headers.get('content-length'))
                with open(zip_path, 'wb') as file, tqdm(
                    desc=zip_path,
                    total=total_length,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = file.write(chunk)
                        bar.update(size)
                print(f"ZIP file downloaded successfully as {zip_path}")
                return True
            except requests.exceptions.HTTPError:
                return False

        # First attempt
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        zip_path = os.path.join(raw_data_path, f"{file_name}.zip")
        url = download_url + file_name + ".zip"
        if not attempt_download(url, zip_path):
            # Second attempt
            url = download_url + file_name[:-3] + ".zip"
            zip_path = os.path.join(raw_data_path, f"{file_name[:-3]}.zip")
            if not attempt_download(url, zip_path):
                print(f"Error: Unable to download the file for {file_name}. The file may not exist.")
                return None

        def unzip_file(zip_path, extract_to):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    print(f"File '{zip_path}' has been unzipped.")
            except zipfile.BadZipFile:
                print(f"Error: The file '{zip_path}' is not a valid ZIP file.")
            except Exception as e:
                print(f"Error unzipping the file '{zip_path}': {e}")
        # Unzip the file
        unzip_file(zip_path, raw_data_path)

        # Check if CSV file now exists after unzipping
        if not os.path.exists(csv_file_path):
            print(f"Error: CSV file '{csv_file_path}' not found after unzipping.")
            return None

        csv_file_path_list.append(csv_file_path)
    print(csv_file_path_list)
    return csv_file_path_list


def analyze_traj_data(traj_data):
    import numpy as np
    stats = {
        "total_mmsi_count": 0,
        "point_count": 0,
        "avg_traj_length": None,
        "min_traj_length": float('inf'), 
        "max_traj_length": float('-inf'),
        "avg_time_interval": None,  
        "vessel_draught_two_values_count": 0, 
        "latitude_lengths": {},
       
        "vessel_type_unique": {},
        "destination_unique": {},
        "cargo_type_unique": {},
        "navi_status_unique": {},
        
        "vessel_width_mean": None,
        "vessel_width_std": None,
        "vessel_length_mean": None,
        "vessel_length_std": None,
        "vessel_draught_mean": None,
        "vessel_draught_std": None,
        "sog_mean": None,
        "sog_std": None,
        "rot_mean": None,
        "rot_std": None,
        "cog_mean": None,
        "cog_std": None,
        "heading_angle_mean": None,
        "heading_angle_std": None,
        "latitude_mean": None,  
        "latitude_std": None,  
        "longitude_mean": None,  
        "longitude_std": None, 
       
        "vessel_width_min": float('inf'),
        "vessel_width_max": float('-inf'),
        "vessel_length_min": float('inf'),
        "vessel_length_max": float('-inf'),
        "vessel_draught_min": float('inf'),
        "vessel_draught_max": float('-inf'),
        "sog_min": float('inf'),
        "sog_max": float('-inf'),
        "rot_min": float('inf'),
        "rot_max": float('-inf'),
        "cog_min": float('inf'),
        "cog_max": float('-inf'),
        "heading_angle_min": float('inf'),
        "heading_angle_max": float('-inf'),
        "latitude_min": float('inf'),
        "latitude_max": float('-inf'),
        "longitude_min": float('inf'),
        "longitude_max": float('-inf'),
        
    }

    # Initialize lists to accumulate values for global mean and std computation
    vessel_width_all = []
    vessel_length_all = []
    vessel_draught_all = []
    sog_all = []
    rot_all = []
    cog_all = []
    heading_angle_all = []
    latitude_all = [] 
    longitude_all = []  
    traj_lengths = []  
    time_intervals = []

    global_vessel_types = set()
    global_destinations = set()
    global_cargo_types = set()
    global_navi_statuses = set()

    # Process each trajectory in traj_data
    for mmsi_id, data in traj_data.items():
        stats["total_mmsi_count"] += 1
        latitude_len = len(data["latitude"])
        stats["latitude_lengths"][mmsi_id] = latitude_len
        stats["point_count"] += latitude_len

        traj_lengths.append(latitude_len)
        stats["min_traj_length"] = min(stats["min_traj_length"], latitude_len)
        stats["max_traj_length"] = max(stats["max_traj_length"], latitude_len)

        global_vessel_types.update(data["vessel_type"])
        global_destinations.update(data["destination"])
        global_cargo_types.update(data["cargo_type"])
        global_navi_statuses.update(data["navi_status"])

        stats["vessel_width_min"] = min(stats["vessel_width_min"], min(data["vessel_width"]))
        stats["vessel_width_max"] = max(stats["vessel_width_max"], max(data["vessel_width"]))
        stats["vessel_length_min"] = min(stats["vessel_length_min"], min(data["vessel_length"]))
        stats["vessel_length_max"] = max(stats["vessel_length_max"], max(data["vessel_length"]))
        stats["vessel_draught_min"] = min(stats["vessel_draught_min"], min(data["vessel_draught"]))
        stats["vessel_draught_max"] = max(stats["vessel_draught_max"], max(data["vessel_draught"]))
        stats["sog_min"] = min(stats["sog_min"], min(data["sog"]))
        stats["sog_max"] = max(stats["sog_max"], max(data["sog"]))
        stats["rot_min"] = min(stats["rot_min"], min(data["rot"]))
        stats["rot_max"] = max(stats["rot_max"], max(data["rot"]))
        stats["cog_min"] = min(stats["cog_min"], min(data["cog"]))
        stats["cog_max"] = max(stats["cog_max"], max(data["cog"]))
        stats["heading_angle_min"] = min(stats["heading_angle_min"], min(data["heading_angle"]))
        stats["heading_angle_max"] = max(stats["heading_angle_max"], max(data["heading_angle"]))
        stats["latitude_min"] = min(stats["latitude_min"], min(data["latitude"]))
        stats["latitude_max"] = max(stats["latitude_max"], max(data["latitude"]))
        stats["longitude_min"] = min(stats["longitude_min"], min(data["longitude"]))
        stats["longitude_max"] = max(stats["longitude_max"], max(data["longitude"]))

        # Accumulate values for global mean and std calculation
        vessel_width_all.extend(data["vessel_width"])
        vessel_length_all.extend(data["vessel_length"])
        vessel_draught_all.extend(data["vessel_draught"])
        sog_all.extend(data["sog"])
        rot_all.extend(data["rot"])
        cog_all.extend(data["cog"])
        heading_angle_all.extend(data["heading_angle"])
        latitude_all.extend(data["latitude"])
        longitude_all.extend(data["longitude"])

        if len(data["timestamp"]) > 1:
            time_interval = data["timestamp"][-1] - data["timestamp"][0]
            time_intervals.append(time_interval)

        unique_draught_values = set(data["vessel_draught"])
        if len(unique_draught_values) >= 2:
            stats["vessel_draught_two_values_count"] += 1


    # Compute global mean and standard deviation for the required attributes
    stats["vessel_width_mean"] = float(np.mean(vessel_width_all)) if vessel_width_all else None
    stats["vessel_width_std"] = float(np.std(vessel_width_all)) if vessel_width_all else None
    stats["vessel_length_mean"] = float(np.mean(vessel_length_all)) if vessel_length_all else None
    stats["vessel_length_std"] = float(np.std(vessel_length_all)) if vessel_length_all else None
    stats["vessel_draught_mean"] = float(np.mean(vessel_draught_all)) if vessel_draught_all else None
    stats["vessel_draught_std"] = float(np.std(vessel_draught_all)) if vessel_draught_all else None
    stats["sog_mean"] = float(np.mean(sog_all)) if sog_all else None
    stats["sog_std"] = float(np.std(sog_all)) if sog_all else None
    stats["rot_mean"] = float(np.mean(rot_all)) if rot_all else None
    stats["rot_std"] = float(np.std(rot_all)) if rot_all else None
    stats["cog_mean"] = float(np.mean(cog_all)) if cog_all else None
    stats["cog_std"] = float(np.std(cog_all)) if cog_all else None
    stats["heading_angle_mean"] = float(np.mean(heading_angle_all)) if heading_angle_all else None
    stats["heading_angle_std"] = float(np.std(heading_angle_all)) if heading_angle_all else None
    stats["latitude_mean"] = float(np.mean(latitude_all)) if latitude_all else None
    stats["latitude_std"] = float(np.std(latitude_all)) if latitude_all else None
    stats["longitude_mean"] = float(np.mean(longitude_all)) if longitude_all else None
    stats["longitude_std"] = float(np.std(longitude_all)) if longitude_all else None

    stats["avg_traj_length"] = float(np.mean(traj_lengths)) if traj_lengths else None

    stats["avg_time_interval"] = float(np.mean(time_intervals)) if time_intervals else None

    stats["vessel_type_unique"] = {val: i for i, val in enumerate(sorted(global_vessel_types))}
    stats["destination_unique"] = {val: i for i, val in enumerate(sorted(global_destinations))}
    stats["cargo_type_unique"] = {val: i for i, val in enumerate(sorted(global_cargo_types))}
    stats["navi_status_unique"] = {val: i for i, val in enumerate(sorted(global_navi_statuses))}

    return stats

def encode_traj_data(traj_data, stats):
    encoded_traj_data = {}
    for mmsi_id, data in traj_data.items():
        encoded_data = {}
        encoded_data["vessel_type"] = [stats["vessel_type_unique"][val] for val in data["vessel_type"]]
        encoded_data["destination"] = [stats["destination_unique"][val] for val in data["destination"]]
        encoded_data["cargo_type"] = [stats["cargo_type_unique"][val] for val in data["cargo_type"]]
        encoded_data["navi_status"] = [stats["navi_status_unique"][val] for val in data["navi_status"]]

        encoded_data["latitude"] = data["latitude"]
        encoded_data["longitude"] = data["longitude"]
        encoded_data["vessel_width"] = data["vessel_width"]
        encoded_data["vessel_length"] = data["vessel_length"]
        encoded_data["vessel_draught"] = data["vessel_draught"]
        encoded_data["sog"] = data["sog"]
        encoded_data["rot"] = data["rot"]
        encoded_data["cog"] = data["cog"]
        encoded_data["heading_angle"] = data["heading_angle"]
        encoded_data["timestamp"] = data["timestamp"]

        encoded_traj_data[mmsi_id] = encoded_data

    return encoded_traj_data

def sample_trajectory(traj_data, max_len=4000):
    """
    Process and sample trajectories based on "vessel_draught" categories.
    
    Args:
        traj_data (dict): Dictionary containing trajectory data.
        max_len (int): Maximum number of points allowed per trajectory.
    
    Returns:
        dict: A new dictionary with sampled trajectories.
    """
    sampled_traj_data = {}

    for mmsi_id, trajectory in traj_data.items():
        # Extract the "vessel_draught" field
        vessel_draught = trajectory["vessel_draught"]
        
        # Check if the number of points exceeds max_len
        if len(vessel_draught) > max_len:
            # Get unique draught categories and their counts
            # unique_draughts, draught_counts = np.unique(vessel_draught, return_counts=True)
            
            # Find the indices of the first occurrence of each unique value
            _, indices = np.unique(vessel_draught, return_index=True)
            
            # Sort the indices to preserve the original order
            sorted_indices = np.sort(indices)
            
            # Extract the unique draughts in the original order
            unique_draughts = np.array(vessel_draught)[sorted_indices]
            
            # Count the occurrences of each unique draught
            draught_counts = np.array([vessel_draught.count(d) for d in unique_draughts])
                    
            # for draught_count in draught_counts:
            #     if draught_count < 0:
            #         continue
            
            # Calculate the total number of points
            total_points = sum(draught_counts)
            
            # Calculate the sampling ratio for each category
            ratios = draught_counts / total_points
            
            # Calculate the number of samples for each category
            sample_counts = (ratios * max_len).astype(int)
            
            # Ensure the total number of sampled points equals max_len
            diff = max_len - sum(sample_counts)
            if diff > 0:
                # Distribute the remaining points to the largest categories
                sample_counts[np.argmax(sample_counts)] += diff
            
            for count in sample_counts:
                if count < 0:
                    continue
            
            # Initialize sampled trajectory
            sampled_trajectory = {key: [] for key in trajectory.keys()}
            
            # Sample points for each draught category
            for draught, count in zip(unique_draughts, sample_counts):
                

                # Find indices where "vessel_draught" matches the current category
                indices = [i for i, d in enumerate(vessel_draught) if d == draught]
                
                # Calculate the step size for uniform sampling
                step_size = len(indices) / count
                
                # Perform uniform sampling along the trajectory
                sampled_indices = [indices[int(i * step_size)] for i in range(count)]
                
                # Append sampled data to the new trajectory
                for key in trajectory.keys():
                    sampled_trajectory[key].extend([trajectory[key][i] for i in sampled_indices])
            
            # Store the sampled trajectory
            sampled_traj_data[mmsi_id] = sampled_trajectory
        else:
            # If the trajectory is already within max_len, keep it unchanged
            sampled_traj_data[mmsi_id] = trajectory
    
    return sampled_traj_data

def split_trajectory(traj_dict, min_len=100, max_len=500, time_threshold=5):
    """
    Split trajectories based on time intervals and maximum length
    
    Args:
        traj_dict (dict): Original trajectory dictionary
        max_len (int): Maximum segment length
        time_threshold (int): Time interval threshold in minutes
    
    Returns:
        dict: Processed trajectory dictionary with segmented data
    """
    split_results = {}
    
    for mmsi, data in traj_dict.items():
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') 
        # Ensure chronological order by timestamp
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        # Calculate time differences between consecutive points
        time_diff = df['timestamp'].diff().dt.total_seconds() / 60  # Convert to minutes
        # Identify indices where time gap exceeds threshold
        split_indices = df.index[time_diff > time_threshold].tolist()
        
        # Create split ranges with start/end indices
        split_indices = [0] + split_indices + [len(df)]
        split_ranges = [(split_indices[i], split_indices[i+1]) 
                       for i in range(len(split_indices)-1)]
        
        # Split into initial segments based on time gaps
        segments = []
        for start, end in split_ranges:
            # Use .loc to explicitly get a copy of the data
            segment = df.loc[start:end-1].copy()
            if len(segment) >= min_len:
                segments.append(segment)
        
        # Further split segments exceeding max length
        final_segments = []
        for seg in segments:
            if len(seg) > max_len:
                num_chunks = (len(seg)-1) // max_len + 1
                for i in range(num_chunks):
                    start_idx = i * max_len
                    end_idx = min((i+1)*max_len, len(seg))
                    # Use .iloc with copy to avoid view
                    chunk = seg.iloc[start_idx:end_idx].copy()
                    if len(chunk) >= min_len:
                        final_segments.append(chunk)
            else:
                if len(seg) >= min_len:
                    # Explicit copy of the segment
                    final_segments.append(seg.copy())
        
        # Convert segments back to dictionary format
        for idx, seg_df in enumerate(final_segments):
            mmsi_id = f"{mmsi}_{idx}"
            # Use .loc for safe assignment
            seg_df['timestamp'] = seg_df['timestamp'].astype('int64') // 10**9
            # seg_df['timestamp'] = pd.to_datetime(seg_df['timestamp'], unit='s')
            seg_dict = seg_df.to_dict(orient='list')
            split_results[mmsi_id] = seg_dict
    return split_results


def split_trajectory2(traj_dict, min_len=100, max_len=500, min_time_threshold=60, max_time_threshold=300):
    """
    Split trajectories based on time intervals and maximum length
    
    Args:
        traj_dict (dict): Original trajectory dictionary
        min_len (int): Minimum segment length
        max_len (int): Maximum segment length
    
    Returns:
        dict: Processed trajectory dictionary with segmented data
    """
    split_results = {}
    
    for mmsi, data in traj_dict.items():
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # Ensure timestamps are in datetime format
        df = df.sort_values(by='timestamp').reset_index(drop=True)  # Sort by timestamp
        
        segments = []
        i = 0
        n = len(df)
        
        while i < n:
            current_segment = []
            current_time = df.loc[i, 'timestamp']
            current_segment.append(i)  # Start with first point
            
            j = i + 1
            last_valid_j = i  # Track last valid position
            
            while j < n:
                next_time = df.loc[j, 'timestamp']
                time_diff = (next_time - current_time).total_seconds() # Time difference in seconds

                if len(current_segment) >= max_len:
                    break

                 # Check termination conditions
                if time_diff > max_time_threshold:
                    break
                
                # Check if point meets time condition
                if min_time_threshold <= time_diff <= max_time_threshold:
                    current_segment.append(j)
                    current_time = next_time
                    last_valid_j = j  # Update last valid position
                
                j += 1
            
            # Create segment from collected valid indices
            if len(current_segment) >= min_len:
                segment = df.iloc[current_segment].copy()
                segments.append(segment)
            
            # Move to next starting point after last valid position
            i = last_valid_j + 1 if last_valid_j >= i else j
        
        # Convert segments back to dictionary format
        for idx, seg_df in enumerate(segments):
            mmsi_id = f"{mmsi}_{idx}"
            seg_df['timestamp'] = seg_df['timestamp'].astype('int64') // 10**9  # Convert timestamp to seconds
            seg_dict = seg_df.to_dict(orient='list')
            split_results[mmsi_id] = seg_dict
    return split_results