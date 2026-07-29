import os
import numpy as np
import pandas as pd

def generate_microbiome_data(n_samples=250, seed=100):
    """
    Generates synthetic microbiome abundance data representing succession over PMI.
    """
    np.random.seed(seed)
    
    # Ischemic time in minutes (0 to 1440 mins - matching GTEx range)
    ischemic_time = np.random.uniform(0, 1440, n_samples)
    # Ambient temperature (typical room or morgue temperature in C)
    temp = np.random.uniform(10, 25, n_samples)
    adh = (ischemic_time / 60.0) * temp # Accumulated Degree Hours
    
    # Taxa succession dynamics:
    # 1. Pseudomonas (aerobic, decreases quickly)
    # 2. Lactobacillus (peaks mid-decomp)
    # 3. Bacteroides (anaerobic, increases steadily)
    # 4. Clostridium (anaerobic, peaks late)
    
    pseudo = 100.0 * np.exp(-0.002 * adh) + 2.0
    lacto = 40.0 * (adh / 500.0) * np.exp(-adh / 500.0) + 1.0
    bacteroides = 5.0 + 95.0 / (1.0 + np.exp(-0.0015 * (adh - 400)))
    clostridium = 0.5 + 150.0 / (1.0 + np.exp(-0.0025 * (adh - 800)))
    
    abundances = np.column_stack([pseudo, lacto, bacteroides, clostridium])
    abundances += np.random.lognormal(mean=0, sigma=0.25, size=abundances.shape)
    abundances = np.clip(abundances, 0.01, None)
    
    relative_abundances = abundances / abundances.sum(axis=1, keepdims=True)
    shannon_div = -np.sum(relative_abundances * np.log(relative_abundances + 1e-9), axis=1)
    
    df = pd.DataFrame({
        'ischemic_time_min': ischemic_time,
        'temperature': temp,
        'adh': adh,
        'taxon_Pseudomonas': relative_abundances[:, 0],
        'taxon_Lactobacillus': relative_abundances[:, 1],
        'taxon_Bacteroides': relative_abundances[:, 2],
        'taxon_Clostridium': relative_abundances[:, 3],
        'shannon_diversity': shannon_div
    })
    return df

def generate_matched_validation_data(n_samples=80, seed=999):
    """
    Generates a matched validation dataset containing both RNA expression features
    and Microbiome features for the same individuals to evaluate meta-model fusion.
    """
    np.random.seed(seed)
    ischemic_time = np.random.uniform(0, 1440, n_samples)
    temp = np.random.uniform(10, 25, n_samples)
    adh = (ischemic_time / 60.0) * temp
    
    # 1. Simulating RNA features
    # Housekeepers (stable) vs Labile (degrading)
    # Baseline log expressions
    GAPDH = np.clip(10.0 - 0.00005 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    ACTB = np.clip(11.0 - 0.00008 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    CYC1 = np.clip(8.0 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    RPL13A = np.clip(9.0 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    EIF4A2 = np.clip(8.5 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    B2M = np.clip(12.0 - 0.00006 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    TOP1 = np.clip(7.5 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples), 1.0, 15.0)
    
    PPIA = np.clip(7.0 - 0.0008 * adh + np.random.normal(0, 0.25, n_samples), 1.0, 15.0)
    
    # Demographics and covariates
    age = np.random.uniform(20, 70, n_samples)
    sex = np.random.choice([1, 2], n_samples)
    hardy = np.random.choice([1, 2, 3, 4], n_samples)
    rin = np.clip(8.5 - 0.002 * adh + np.random.normal(0, 0.4, n_samples), 1.0, 10.0)
    
    # 2. Microbiome features
    pseudo = 100.0 * np.exp(-0.002 * adh) + 2.0
    lacto = 40.0 * (adh / 500.0) * np.exp(-adh / 500.0) + 1.0
    bacteroides = 5.0 + 95.0 / (1.0 + np.exp(-0.0015 * (adh - 400)))
    clostridium = 0.5 + 150.0 / (1.0 + np.exp(-0.0025 * (adh - 800)))
    
    abundances = np.column_stack([pseudo, lacto, bacteroides, clostridium])
    abundances += np.random.lognormal(mean=0, sigma=0.25, size=abundances.shape)
    abundances = np.clip(abundances, 0.01, None)
    relative_abundances = abundances / abundances.sum(axis=1, keepdims=True)
    shannon_div = -np.sum(relative_abundances * np.log(relative_abundances + 1e-9), axis=1)
    
    df = pd.DataFrame({
        'SAMPID': [f'GTEX-VAL-{i:03d}-SM-VAL1' for i in range(n_samples)],
        'ischemic_time_min': ischemic_time,
        'temperature': temp,
        'rin': rin,
        'SEX': sex,
        'AGE': [f'{int(a//10)*10}-{int(a//10)*10+9}' for a in age],
        'DTHHRDY': hardy,
        # Gene Expressions (counts, unlogged)
        'GAPDH': np.expm1(GAPDH),
        'ACTB': np.expm1(ACTB),
        'CYC1': np.expm1(CYC1),
        'RPL13A': np.expm1(RPL13A),
        'EIF4A2': np.expm1(EIF4A2),
        'B2M': np.expm1(B2M),
        'TOP1': np.expm1(TOP1),
        'PPIA': np.expm1(PPIA),
        # Extra required reference genes
        'RPS29': np.expm1(8.0 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples)),
        'RPS18': np.expm1(9.0 - 0.00005 * adh + np.random.normal(0, 0.15, n_samples)),
        'UBC': np.expm1(11.0 - 0.00007 * adh + np.random.normal(0, 0.15, n_samples)),
        'SDHA': np.expm1(7.0 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples)),
        'YWHAZ': np.expm1(8.5 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples)),
        'RPS10': np.expm1(8.0 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'TBP': np.expm1(6.0 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'HPRT1': np.expm1(6.5 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'PGK1': np.expm1(9.0 - 0.00005 * adh + np.random.normal(0, 0.15, n_samples)),
        'POLR2A': np.expm1(7.0 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'RPLP0': np.expm1(10.0 - 0.00005 * adh + np.random.normal(0, 0.15, n_samples)),
        'TUBB': np.expm1(9.5 - 0.00006 * adh + np.random.normal(0, 0.15, n_samples)),
        'ALAS1': np.expm1(6.0 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples)),
        'IPO8': np.expm1(6.5 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'PUM1': np.expm1(7.5 - 0.00003 * adh + np.random.normal(0, 0.15, n_samples)),
        'HMBS': np.expm1(6.0 - 0.00004 * adh + np.random.normal(0, 0.15, n_samples)),
        # Microbiome relative abundances
        'taxon_Pseudomonas': relative_abundances[:, 0],
        'taxon_Lactobacillus': relative_abundances[:, 1],
        'taxon_Bacteroides': relative_abundances[:, 2],
        'taxon_Clostridium': relative_abundances[:, 3],
        'shannon_diversity': shannon_div
    })
    return df

if __name__ == '__main__':
    os.makedirs('data/raw/microbiome', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("Generating synthetic microbiome succession data...")
    micro_df = generate_microbiome_data()
    micro_df.to_csv('data/raw/microbiome/microbiome_data.csv', index=False)
    
    print("Generating synthetic matched validation data...")
    val_df = generate_matched_validation_data()
    val_df.to_csv('data/processed/matched_validation_data.csv', index=False)
    print("Data simulation complete!")
