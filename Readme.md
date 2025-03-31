# MH-GIN: Multi-scale Heterogeneous Graph-based Imputation Network for AIS Data
This is the official implementation of the paper "MH-GIN: Multi-scale Heterogeneous Graph-based Imputation Network for AIS Data".

## Abstract
Automatic Identification System (AIS) data plays a vital role in maritime monitoring but frequently suffers from missing values. However, imputing these missing values is challenging due to multi-scale dependencies among attributes arising from their diverse update rates. Existing imputation methods typically assume uniform update rates across all variables, thus failing to effectively capture such multi-scale dependencies and consequently limiting imputation accuracy.

To address this challenge, we propose MH-GIN, a Multi-scale Heterogeneous Graph-based Imputation Network. Specifically, MH-GIN first extracts multi-scale temporal features for each attribute while preserving their intrinsic heterogeneous characteristics. Then, it constructs a multi-scale heterogeneous graph to explicitly model dependencies between heterogeneous attributes, facilitating effective imputation of missing values through graph propagation. Experimental results on two real-world datasets demonstrate that MH-GIN achieves 21.51%–106.52% lower error rates compared to state-of-the-art methods, while maintaining computational efficiency.

## Environment Setting
```bash
bash environment_install.sh
```

## Code Structure
```
.
├── Data/               # Directory for automatically storing datasets
├── Result/             # Directory for automatically storing experimental results
│   └── log/           # The log of train model
├── src/
│   ├── config/        # Configuration of model structure and train parameters
│   ├── data/          # Data processing
│   ├── models/        # Implementation of our method
│   ├── utils/         # Implementation of common functions
│   ├── main.py        # Main program entry point
│   └── environment_install.sh  # Shell script for installing required Python packages
└── README.md          # Main documentation file with project overview and instructions
```

## Dataset Preparation
The datasets can be automatically downloaded from http://web.ais.dk/aisdata/ 
or https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2023/index.html based on the dataset hyperparameter.
The dataset parameter follows the format:

- For AISDK dataset: aisdk-YYYY-MM-DD1@DD2 (e.g. aisdk-2023-01-01@02)
- For AISUS dataset: AIS_YYYY_MM_DD1@DD2 (e.g. AIS_2023_12_11@31)
- 
where YYYY-MM-DD1@DD2 specifies the date range of the dataset.
