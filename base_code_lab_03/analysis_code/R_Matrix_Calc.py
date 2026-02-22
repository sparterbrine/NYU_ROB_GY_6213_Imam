import pandas as pd
import numpy as np
import json

def calculate_and_save_r_matrix(file_path, output_json):
    # 1. Load the data
    df = pd.read_csv(file_path)
    
    # 2. Define the state vector columns
    state_cols = ['x', 'y', 'z', 'rotx (rad)', 'roty', 'rotz']
    
    # 3. Group by ground truth positions to isolate noise
    groups = df.groupby(['Absolute x (cm)', 'Abs y', 'Abs z'])
    
    group_covariances = []
    
    for _, group in groups:
        g_data = group[state_cols].copy()
        
        # Normalize rotx (rad) around the first value to prevent wrap-around jumps
        if not g_data.empty:
            first_rotx = g_data['rotx (rad)'].iloc[0]
            g_data['rotx (rad)'] = g_data['rotx (rad)'].apply(
                lambda x: x - 2*np.pi if (x - first_rotx) > np.pi 
                else (x + 2*np.pi if (x - first_rotx) < -np.pi else x)
            )
        
        # Calculate covariance if the group has multiple samples and non-zero variance
        if len(g_data) > 1:
            cov = g_data.cov()
            if not np.allclose(cov, 0):
                group_covariances.append(cov)
                
    if not group_covariances:
        R_matrix_df = df[state_cols].cov()
    else:
        # 4. Average the covariance matrices from all static points
        R_matrix_df = pd.concat(group_covariances).groupby(level=0).mean()
        R_matrix_df = R_matrix_df.reindex(index=state_cols, columns=state_cols)

    # 5. Format for JSON export
    r_data = {
        "metadata": {
            "source": file_path,
            "description": "Measurement Noise Covariance Matrix (R) for EKF"
        },
        "columns": list(R_matrix_df.columns),
        "matrix": R_matrix_df.values.tolist(),
        "diagonal": np.diag(R_matrix_df).tolist()
    }
    
    with open(output_json, 'w') as f:
        json.dump(r_data, f, indent=4)
    
    print(f"R matrix successfully saved to {output_json}")
    return R_matrix_df

# Run the calculation
r_matrix = calculate_and_save_r_matrix('R_Data.csv', 'R_matrix.json')