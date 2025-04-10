import numpy as np

# define attributes
attributes = ["vessel_type", "vessel_width", "vessel_length", "vessel_draught", "destination", "cargo_type",
              "navi_status", "sog", "rot", "cog", "heading_angle", "latitude", "longitude", "timestamp"]

def preprocess_traj_data_v1(traj_data, stats, max_len, ratio):
    padding_values = {
        "vessel_type": len(stats["vessel_type_unique"]) ,
        "destination": len(stats["destination_unique"]) ,
        "cargo_type": len(stats["cargo_type_unique"]) ,
        "navi_status": len(stats["navi_status_unique"]),
        "latitude": 91.0,
        "longitude": 181.0,
        "default": -1.0
    }

    data = []
    labels = []
    masks = []
    padding_masks = []

    for traj_id, traj in traj_data.items():
        # Step 1: Extract and pad each attribute
        attr_data = [np.array(traj[attr], dtype=np.float32) for attr in attributes]
        assert all(len(attr_data[0]) == len(attr) for attr in attr_data), "Inconsistent trajectory data lengths"

        for i, attr in enumerate(attributes):
            if len(attr_data[i]) > max_len:
                attr_data[i] = attr_data[i][:max_len]
            else:
                pad_value = padding_values.get(attr, padding_values["default"])
                pad_length = max_len - len(attr_data[i])
                attr_data[i] = np.pad(
                    attr_data[i], (0, pad_length), mode="constant", constant_values=pad_value
                )

        labels.append(np.stack(attr_data, axis=-1))  # [maxlen, n]
        
        # Step 2: Generate original mask
        ori_masks = []
        for i, attr in enumerate(attributes):
            ori_mask_value = []
            for at_data in attr_data[i]:
                value = (at_data != padding_values.get(attr, padding_values["default"]))
                ori_mask_value.append(value)
            ori_masks.append(ori_mask_value)
        # [n, maxlen]
        ori_masks = np.array(ori_masks)
        padding_masks.append(ori_masks)
       
        # mask = ori_masks.copy()
        mask = np.ones((len(attributes), max_len), dtype=bool)  # [n, maxlen]
        for i, attr in enumerate(attributes):
            valid_indices = np.where(attr_data[i] != padding_values.get(attr, padding_values["default"]))[0]
            if len(valid_indices) > 0:
                if attr in ["latitude", "longitude"]:
                    # Ensure synchronized missing positions for latitude and longitude
                    if i == attributes.index("latitude"):
                        # Randomly select missing indices for latitude and longitude together
                        mask_indices = np.random.choice(
                            valid_indices, size=int(ratio * len(valid_indices)), replace=False
                        )
                        attr_data[i][mask_indices] = padding_values.get(attr, padding_values["default"])
                        attr_data[attributes.index("longitude")][mask_indices] = (
                            padding_values.get("longitude", padding_values["default"]))
                        mask[i, mask_indices] = False
                        mask[attributes.index("longitude"), mask_indices] = False
                else:
                    # Handle other attributes independently
                    mask_indices = np.random.choice(
                        valid_indices, size=int(ratio * len(valid_indices)), replace=False
                    )
                    attr_data[i][mask_indices] = padding_values.get(attr, padding_values["default"])
                    mask[i, mask_indices] = False

        traj_features = np.stack(attr_data, axis=-1)  # [maxlen, num_attributes]
        data.append(traj_features)
        masks.append(mask)

    # change NumPy
    data = np.array(data, dtype=np.float32)  # [B, maxlen, num_attributes]
    labels = np.array(labels, dtype=np.float32)  # [B, maxlen, num_attributes]
    masks = np.array(masks, dtype=bool)  # [B, num_attributes, maxlen]
    masks = np.transpose(masks, axes=(0, 2, 1)) # [B, maxlen, num_attributes]
    padding_masks = np.array(padding_masks, dtype=bool)  # [B, num_attributes, maxlen]
    padding_masks = np.transpose(padding_masks, axes=(0, 2, 1)) # [B, maxlen, num_attributes]

    return data, labels, masks, padding_masks

def preprocess_traj_data_v2(traj_data, stats, max_len, ratio):
    padding_values = {
        "vessel_type": len(stats["vessel_type_unique"]),
        "destination": len(stats["destination_unique"]),
        "cargo_type": len(stats["cargo_type_unique"]),
        "navi_status": len(stats["navi_status_unique"]),
        "latitude": 91.0,
        "longitude": 181.0,
        "default": -1.0
    }

    data = []
    labels = []
    masks = []
    padding_masks = []
    total = 0

    for traj_id, traj in traj_data.items():
        # Step 1: Extract and pad each attribute
        attr_data = [np.array(traj[attr], dtype=np.float32) for attr in attributes]
        assert all(len(attr_data[0]) == len(attr) for attr in attr_data), "Inconsistent trajectory data lengths"

        for i, attr in enumerate(attributes):
            if len(attr_data[i]) > max_len:
                attr_data[i] = attr_data[i][:max_len]
            else:
                pad_value = padding_values.get(attr, padding_values["default"])
                pad_length = max_len - len(attr_data[i])
                attr_data[i] = np.pad(
                    attr_data[i], (0, pad_length), mode="constant", constant_values=pad_value
                )

        labels.append(np.stack(attr_data, axis=-1))  # [maxlen, n]
        
        # Step 2: Generate original mask
        ori_masks = []
        for i, attr in enumerate(attributes):
            ori_mask_value = []
            for at_data in attr_data[i]:
                value = (at_data != padding_values.get(attr, padding_values["default"]))
                ori_mask_value.append(value)
            ori_masks.append(ori_mask_value)
        ori_masks = np.array(ori_masks)
        padding_masks.append(ori_masks)
       
        # Initialize mask
        mask = np.ones((len(attributes), max_len), dtype=bool)  # [n, maxlen]
        
        # Determine if full mask for vessel_type group
        mask_full = np.random.rand() < ratio

        for i, attr in enumerate(attributes):
            valid_indices = np.where(attr_data[i] != padding_values.get(attr, padding_values["default"]))[0]
            if len(valid_indices) == 0:
                continue

            if attr in ["vessel_type", "vessel_width", "vessel_length"]:
                if np.random.rand() < ratio:
                    mask[i, valid_indices] = False
                    attr_data[i][valid_indices] = padding_values.get(attr, padding_values["default"])

            elif attr in ["vessel_draught", "destination", "cargo_type", "navi_status"]:
                num_to_mask = int(ratio * len(valid_indices))
                if num_to_mask > 0:
                    start_idx = np.random.choice(len(valid_indices) - num_to_mask + 1)
                    actual_start = valid_indices[start_idx]
                    actual_end = valid_indices[start_idx + num_to_mask - 1] + 1
                    mask[i, actual_start:actual_end] = False
                    attr_data[i][actual_start:actual_end] = padding_values.get(attr, padding_values["default"])
                    total += 1

            elif attr in ["latitude", "longitude"]:
                if i == attributes.index("latitude"):
                    mask_indices = np.random.choice(valid_indices, size=int(ratio * len(valid_indices)), replace=False)
                    attr_data[i][mask_indices] = padding_values.get(attr, padding_values["default"])
                    longitude_idx = attributes.index("longitude")
                    attr_data[longitude_idx][mask_indices] = padding_values.get("longitude", padding_values["default"])
                    mask[i, mask_indices] = False
                    mask[longitude_idx, mask_indices] = False
            else:
                # Handle other attributes independently
                mask_indices = np.random.choice(valid_indices, size=int(ratio * len(valid_indices)), replace=False)
                attr_data[i][mask_indices] = padding_values.get(attr, padding_values["default"])
                mask[i, mask_indices] = False

        traj_features = np.stack(attr_data, axis=-1)  # [maxlen, num_attributes]
        data.append(traj_features)
        masks.append(mask)

    # Convert to NumPy arrays
    data = np.array(data, dtype=np.float32)  # [B, maxlen, num_attributes]
    labels = np.array(labels, dtype=np.float32)  # [B, maxlen, num_attributes]
    masks = np.array(masks, dtype=bool)  # [B, num_attributes, maxlen]
    masks = np.transpose(masks, axes=(0, 2, 1))  # [B, maxlen, num_attributes]
    padding_masks = np.array(padding_masks, dtype=bool)  # [B, num_attributes, maxlen]
    padding_masks = np.transpose(padding_masks, axes=(0, 2, 1))  # [B, maxlen, num_attributes]

    return data, labels, masks, padding_masks


