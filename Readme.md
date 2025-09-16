# MH-GIN: Multi-scale Heterogeneous Graph-based Imputation Network for AIS Data
This is the official implementation of the paper "MH-GIN: Multi-scale Heterogeneous Graph-based Imputation Network for AIS Data".

## Abstract
Location-tracking data from the Automatic Identification System, much of which is publicly available, plays a key role in a range of maritime safety and monitoring applications. However, the data suffers from missing values that hamper downstream applications. Imputing the missing values is challenging because the values of different attributes are updated at diverse rates, resulting in the occurrence of multiscale dependencies among attributes. Existing imputation methods that assume similar update rates across attributes are unable to capture and exploit such dependencies, limiting their imputation accuracy. We propose MH-GIN, a Multi-scale Heterogeneous Graph-based Imputation Network that aims improve imputation accuracy by capturing multi-scale dependencies. Specifically, MH-GIN first extracts multi-scale temporal features for each attribute while preserving their intrinsic heterogeneous characteristics. Then, it constructs a multi-scale heterogeneous graph to explicitly model dependencies between heterogeneous attributes to enable more accurate imputation of missing values through graph propagation. Experimental results on two real-world datasets find that MH-GIN is capable of a 21\%--106\% reduction in imputation errors compared to state-of-the-art methods, while maintaining computational efficiency.

## Environment Setting
```bash
bash environment_install.sh
```

## Code Structure
```
.
├── Data/              # Directory for automatically storing datasets
├── config/            # Configuration of model structure and train parameters
├── src/
│   ├── data/          # Data processing
│   ├── logging/       # log operate util
│   ├── pipeline/      # model train predict and save pipleine
│   ├── models/        # Implementation of our method
│   ├── utils/         # Implementation of common functions
│   ├── main.py        # Main program entry point
│   └── environment_install.sh  # Shell script for installing required Python packages
└── README.md          # Main documentation file with project overview and instructions
```

## Dataset Preparation
The datasets can be automatically downloaded from http://aisdata.ais.dk/2024/
or https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/index.html based on the dataset hyperparameter.
The dataset parameter follows the format:

- For AISDK dataset: aisdk-YYYY-MM-DD1@DD2 (e.g. aisdk-2023-01-01@02)
- For AISUS dataset: AIS_YYYY_MM_DD1@DD2 (e.g. AIS_2023_12_11@31)
where YYYY-MM-DD1@DD2 specifies the date range of the dataset.

You should use the data processing tools we provided to obtain the MT-HGNN dataset. Taking the processing of the U.S. dataset as an example, you should complete the following two steps.

+ Step 1: Update the dataset settings in the Configuration File (./config/config-US.yaml).
    - datasets: A list of months for the AIS data to be processed.
    - min_time_interval: The minimum time interval between any two consecutive points in each trajectory.
    - max_time_interval: The maximum time interval between any two consecutive points in each trajectory.
    - min_traj_len: The minimum length of a trajectory.
    - max_traj_len: The maximum length of a trajectory.

+ Step 2: Run the data process script.
    ```python
    python src/data/american_data_process_module.py --config config-US.yaml
    ```
  
## Training
we train MH-HGNN across different datasets and take US as an example:
  ```python
   python src/main.py --config config-US.yaml
  ```