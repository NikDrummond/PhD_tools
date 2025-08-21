import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def optimize_dataframe(df):
    """ Optimize DataFrame memory for larger datasets. """

    categorical_cols = ['neuron_id','neuron_type', 'subtype', 'edge_type', 'neuron_group', 'subgroup']
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    df['segment_length'] = df['segment_length'].astype('float32')

    return df

def create_summary_table(df):
    """Quicjk descriptives summary table"""
    
    summary = df.groupby(['neuron_type', 'subtype','edge_type']).agg(
        n_neurons = ('neuron_id', 'nunique'),
        n_segments = ('segment_length', 'count'),
        mean_length = ('segment_length','mean'),
        median_length = ('segment_length','median'),
        std_length = ('segment_length', 'std'),
        min_length = ('segment_length', 'min'),
        max_length = ('segment_length', 'max'),
        q25 = ('segment_length', lambda x: x.quantile(0.25)),
        q75 = ('segment_length', lambda x: x.quantile(0.75)),

    ).round(2)

    summary['segments_per_neuron'] = (summary['n_segments'] / summary['n_segments']).round(1)
    summary['cv'] = (summary['std_length'] / summary['mean_length']).round(3)

    return summary

def prepare_parallel_groups(df):
    """Creat grouped DataFrames for parallel processing"""

    grouped = df.groupby(['neuron_id', 'edge_type'])

    # create metadata for each group
    group_info = []
    for (neuron_id, edge_type), group_df in grouped:
        info = {
            'neuron_id':neuron_id,
            'edge_type':edge_type,
            'neuron_type':group_df['neuron_type'].iloc[0],
            'subtype':group_df['subtype'].iloc[0],
            'n_segments': len(group_df),
            'data_index': group_df.index.tolist()
        }
        group_info.append(info)

    group_metadata = pd.DataFrame(group_info)

    return group_metadata


def explore_hierarchical_structure(df):
    """
    Initial exploration of the hierarchical structure in the data.
    """
    print("=" * 60)
    print("HIERARCHICAL DATA STRUCTURE EXPLORATION")
    print("=" * 60)
    
    # Basic counts
    print("\n1. BASIC COUNTS")
    print("-" * 40)
    print(f"Total segments: {len(df):,}")
    print(f"Unique neurons: {df['neuron_id'].nunique():,}")
    print(f"Unique neuron-edge combinations: {df.groupby(['neuron_id', 'edge_type']).ngroups:,}")
    
    # Neurons per group
    print("\n2. NEURONS PER GROUP")
    print("-" * 40)
    neuron_counts = df.groupby(['neuron_type', 'subtype'])['neuron_id'].nunique().unstack()
    print(neuron_counts)
    print(f"\nTotal per neuron_type:")
    print(df.groupby('neuron_type')['neuron_id'].nunique())
    
    # Segments per neuron
    print("\n3. SEGMENTS PER NEURON-EDGE COMBINATION")
    print("-" * 40)
    segments_per_neuron_edge = df.groupby(['neuron_id', 'edge_type']).size()
    print(segments_per_neuron_edge.describe())
    
    # Check for any neurons with unusual segment counts
    low_count = segments_per_neuron_edge < 50
    high_count = segments_per_neuron_edge > 150
    
    if low_count.any():
        print(f"\n⚠ Warning: {low_count.sum()} neuron-edge combinations have <50 segments")
    if high_count.any():
        print(f"⚠ Warning: {high_count.sum()} neuron-edge combinations have >150 segments")
    
    return segments_per_neuron_edge

def calculate_icc_per_subgroup(df):
    """
    Calculate ICC for each of the 16 subgroups to quantify neuron-level clustering.
    Uses one-way random effects model.
    """
    from scipy import stats
    
    print("\n" + "=" * 60)
    print("INTRACLASS CORRELATION COEFFICIENT (ICC) ANALYSIS")
    print("=" * 60)
    
    icc_results = []
    
    for (neuron_type, subtype, edge_type), group_df in df.groupby(['neuron_type', 'subtype', 'edge_type']):
        # Get neuron-level means and variances
        neuron_stats = group_df.groupby('neuron_id')['segment_length'].agg(['mean', 'var', 'count'])
        
        # Calculate between and within neuron variance components
        grand_mean = group_df['segment_length'].mean()
        
        # Between-neuron variance (variance of neuron means)
        n_neurons = len(neuron_stats)
        mean_group_size = neuron_stats['count'].mean()
        
        # Between-group sum of squares
        ss_between = np.sum(neuron_stats['count'] * (neuron_stats['mean'] - grand_mean)**2)
        ms_between = ss_between / (n_neurons - 1)
        
        # Within-group sum of squares
        ss_within = np.sum((neuron_stats['count'] - 1) * neuron_stats['var'])
        df_within = group_df.shape[0] - n_neurons
        ms_within = ss_within / df_within
        
        # Calculate ICC using variance components
        var_between = (ms_between - ms_within) / mean_group_size
        var_within = ms_within
        
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
        
        # F-statistic for testing if ICC > 0
        f_stat = ms_between / ms_within if ms_within > 0 else np.inf
        p_value = 1 - stats.f.cdf(f_stat, n_neurons - 1, df_within)
        
        result = {
            'neuron_type': neuron_type,
            'subtype': subtype,
            'edge_type': edge_type,
            'subgroup': f"{neuron_type}_{subtype}_{edge_type}",
            'n_neurons': n_neurons,
            'n_segments': len(group_df),
            'var_between': var_between,
            'var_within': var_within,
            'icc': icc,
            'f_stat': f_stat,
            'p_value': p_value
        }
        icc_results.append(result)
    
    icc_df = pd.DataFrame(icc_results)
    icc_df = icc_df.sort_values('icc', ascending=False)
    
    print("\nICC BY SUBGROUP:")
    print("-" * 40)
    print(icc_df[['subgroup', 'icc', 'p_value', 'n_neurons']].to_string(index=False))
    
    print("\n\nICC INTERPRETATION:")
    print("-" * 40)
    high_icc = icc_df['icc'] > 0.5
    moderate_icc = (icc_df['icc'] > 0.1) & (icc_df['icc'] <= 0.5)
    low_icc = icc_df['icc'] <= 0.1
    
    print(f"High ICC (>0.5): {high_icc.sum()} subgroups - Strong neuron-level clustering")
    print(f"Moderate ICC (0.1-0.5): {moderate_icc.sum()} subgroups - Moderate clustering")
    print(f"Low ICC (≤0.1): {low_icc.sum()} subgroups - Weak clustering")
    
    if icc_df['icc'].mean() > 0.1:
        print(f"\n⚠ Mean ICC = {icc_df['icc'].mean():.3f} - Hierarchical modeling is necessary!")
    
    return icc_df

def visualize_hierarchical_structure(df, icc_results):
    """
    Create visualizations to understand the hierarchical structure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ICC Heatmap
    ax = axes[0, 0]
    icc_pivot = icc_results.pivot_table(
        index=['neuron_type', 'subtype'], 
        columns='edge_type', 
        values='icc'
    )
    sns.heatmap(icc_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'ICC'})
    ax.set_title('Intraclass Correlation Coefficient by Subgroup')
    
    # 2. Distribution of segment lengths (sample 4 subgroups)
    ax = axes[0, 1]
    sample_subgroups = icc_results.nlargest(2, 'icc')['subgroup'].tolist() + \
                      icc_results.nsmallest(2, 'icc')['subgroup'].tolist()
    
    for subgroup in sample_subgroups[:4]:
        nt, st, et = subgroup.split('_')
        subset = df[(df['neuron_type']==nt) & (df['subtype']==st) & (df['edge_type']==et)]
        subset_sample = subset.sample(min(1000, len(subset)))
        ax.hist(subset_sample['segment_length'], alpha=0.5, label=subgroup, bins=30, density=True)
    
    ax.set_xlabel('Segment Length')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Segment Lengths (Sample Subgroups)')
    ax.legend()
    
    # 3. Neuron-level means distribution
    ax = axes[1, 0]
    neuron_means = df.groupby(['neuron_id', 'neuron_type', 'subtype', 'edge_type'])['segment_length'].mean().reset_index()
    
    for edge in ['internal', 'external']:
        subset = neuron_means[neuron_means['edge_type'] == edge]
        ax.hist(subset['segment_length'], alpha=0.5, label=edge, bins=50, density=True)
    
    ax.set_xlabel('Mean Segment Length per Neuron')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Neuron-Level Mean Lengths')
    ax.legend()
    
    # 4. Coefficient of Variation by neuron
    ax = axes[1, 1]
    neuron_cv = df.groupby(['neuron_id', 'edge_type'])['segment_length'].agg(['mean', 'std'])
    neuron_cv['cv'] = neuron_cv['std'] / neuron_cv['mean']
    neuron_cv = neuron_cv.reset_index()
    
    bp = ax.boxplot([neuron_cv[neuron_cv['edge_type']=='internal']['cv'],
                      neuron_cv[neuron_cv['edge_type']=='external']['cv']],
                     labels=['Internal', 'External'])
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Within-Neuron Variability (CV) by Edge Type')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return neuron_means

def create_variance_report(df, icc_results):
    """
    Create a comprehensive variance decomposition report.
    """
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION SUMMARY REPORT")
    print("=" * 60)
    
    # Overall statistics
    print("\n1. OVERALL LENGTH STATISTICS")
    print("-" * 40)
    print(f"Mean ± SD: {df['segment_length'].mean():.2f} ± {df['segment_length'].std():.2f}")
    print(f"Median [IQR]: {df['segment_length'].median():.2f} [{df['segment_length'].quantile(0.25):.2f}, {df['segment_length'].quantile(0.75):.2f}]")
    print(f"Range: [{df['segment_length'].min():.2f}, {df['segment_length'].max():.2f}]")
    print(f"Skewness: {df['segment_length'].skew():.3f}")
    print(f"Kurtosis: {df['segment_length'].kurtosis():.3f}")
    
    # Edge type comparison
    print("\n2. EDGE TYPE COMPARISON")
    print("-" * 40)
    edge_stats = df.groupby('edge_type')['segment_length'].agg(['mean', 'std', 'median'])
    print(edge_stats)
    
    # Neuron type comparison
    print("\n3. NEURON TYPE COMPARISON")
    print("-" * 40)
    neuron_type_stats = df.groupby('neuron_type')['segment_length'].agg(['mean', 'std', 'median'])
    print(neuron_type_stats)
    
    # Variance partitioning summary
    print("\n4. VARIANCE PARTITIONING")
    print("-" * 40)
    mean_icc = icc_results['icc'].mean()
    print(f"Average ICC across all subgroups: {mean_icc:.3f}")
    print(f"→ {mean_icc*100:.1f}% of variance is between neurons")
    print(f"→ {(1-mean_icc)*100:.1f}% of variance is within neurons")
    
    # Recommendations
    print("\n5. MODELING RECOMMENDATIONS")
    print("-" * 40)
    
    if mean_icc > 0.5:
        print("✓ HIGH neuron-level clustering detected (ICC > 0.5)")
        print("  → Use full hierarchical modeling approach")
        print("  → Consider neuron-specific parameter estimates")
    elif mean_icc > 0.1:
        print("✓ MODERATE neuron-level clustering detected (0.1 < ICC < 0.5)")
        print("  → Two-stage hierarchical approach recommended")
        print("  → Fit distributions to each neuron, then analyze parameter distributions")
    else:
        print("✓ LOW neuron-level clustering detected (ICC < 0.1)")
        print("  → Could use pooled approach, but hierarchical still recommended")
        print("  → Neuron effects are minimal but should be verified")
    
    return edge_stats, neuron_type_stats



# Define all distribution fitting functions
class DistributionFitter:
    """
    Class to handle fitting of all 9 candidate distributions.
    """
    
    @staticmethod
    def fit_lognormal(data):
        """Fit log-normal distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.lognorm.fit(data, floc=0)
        loglik = np.sum(stats.lognorm.logpdf(data, *params))
        return params, loglik, {'s': params[0], 'scale': params[2]}
    
    @staticmethod
    def fit_gamma(data):
        """Fit gamma distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.gamma.fit(data, floc=0)
        loglik = np.sum(stats.gamma.logpdf(data, *params))
        return params, loglik, {'a': params[0], 'scale': params[2]}
    
    @staticmethod
    def fit_weibull(data):
        """Fit Weibull distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.weibull_min.fit(data, floc=0)
        loglik = np.sum(stats.weibull_min.logpdf(data, *params))
        return params, loglik, {'c': params[0], 'scale': params[2]}
    
    @staticmethod
    def fit_pareto(data):
        """Fit Pareto distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        # Pareto requires data > scale parameter
        min_val = np.min(data) * 0.99
        params = stats.pareto.fit(data, floc=min_val)
        loglik = np.sum(stats.pareto.logpdf(data, *params))
        return params, loglik, {'b': params[0], 'scale': params[2]}
    
    @staticmethod
    def fit_exponential(data):
        """Fit exponential distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.expon.fit(data, floc=0)
        loglik = np.sum(stats.expon.logpdf(data, *params))
        return params, loglik, {'scale': params[1]}
    
    @staticmethod
    def fit_loglogistic(data):
        """Fit log-logistic (Fisk) distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.fisk.fit(data, floc=0)
        loglik = np.sum(stats.fisk.logpdf(data, *params))
        return params, loglik, {'c': params[0], 'scale': params[2]}
    
    @staticmethod
    def fit_burr12(data):
        """Fit Burr Type XII distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.burr12.fit(data, floc=0)
        loglik = np.sum(stats.burr12.logpdf(data, *params))
        return params, loglik, {'c': params[0], 'd': params[1], 'scale': params[3]}
    
    @staticmethod
    def fit_betaprime(data):
        """Fit Beta Prime distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        params = stats.betaprime.fit(data, floc=0)
        loglik = np.sum(stats.betaprime.logpdf(data, *params))
        return params, loglik, {'a': params[0], 'b': params[1], 'scale': params[3]}
    
    @staticmethod
    def fit_expweibull(data):
        """Fit Exponentiated Weibull distribution."""
        if np.any(data <= 0):
            return None, -np.inf, {}
        
        # Using exponweib from scipy
        params = stats.exponweib.fit(data, floc=0)
        loglik = np.sum(stats.exponweib.logpdf(data, *params))
        return params, loglik, {'a': params[0], 'c': params[1], 'scale': params[3]}

def fit_all_distributions(segment_data):
    """
    Fit all 9 distributions to a single neuron-edge combination.
    """
    fitter = DistributionFitter()
    
    distributions = {
        'lognormal': fitter.fit_lognormal,
        'gamma': fitter.fit_gamma,
        'weibull': fitter.fit_weibull,
        'pareto': fitter.fit_pareto,
        'exponential': fitter.fit_exponential,
        'loglogistic': fitter.fit_loglogistic,
        'burr12': fitter.fit_burr12,
        'betaprime': fitter.fit_betaprime,
        'expweibull': fitter.fit_expweibull
    }
    
    results = {}
    for dist_name, fit_func in distributions.items():
        try:
            params, loglik, param_dict = fit_func(segment_data)
            n = len(segment_data)
            k = len(param_dict)  # number of parameters
            
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik
            
            # Perform KS test
            if params is not None:
                dist_obj = getattr(stats, dist_name.replace('lognormal', 'lognorm').replace('loglogistic', 'fisk').replace('expweibull', 'exponweib').replace('betaprime', 'betaprime').replace('burr12', 'burr12'))
                ks_stat, ks_pval = stats.kstest(segment_data, lambda x: dist_obj.cdf(x, *params))
            else:
                ks_stat, ks_pval = np.nan, np.nan
            
            results[dist_name] = {
                'params': params,
                'param_dict': param_dict,
                'loglik': loglik,
                'aic': aic,
                'bic': bic,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
                'n_params': k
            }
        except Exception as e:
            results[dist_name] = {
                'params': None,
                'param_dict': {},
                'loglik': -np.inf,
                'aic': np.inf,
                'bic': np.inf,
                'ks_stat': np.nan,
                'ks_pval': np.nan,
                'n_params': 0,
                'error': str(e)
            }
    
    return results

def process_single_neuron_edge(args):
    """
    Process a single neuron-edge combination.
    """
    neuron_id, edge_type, neuron_type, subtype, segment_lengths = args
    
    # Fit all distributions
    fit_results = fit_all_distributions(segment_lengths)
    
    # Find best distribution by AIC
    best_dist_aic = min(fit_results.keys(), key=lambda x: fit_results[x]['aic'])
    best_dist_bic = min(fit_results.keys(), key=lambda x: fit_results[x]['bic'])
    
    # Calculate AIC and BIC differences from best
    min_aic = fit_results[best_dist_aic]['aic']
    min_bic = fit_results[best_dist_bic]['bic']
    
    for dist_name in fit_results:
        fit_results[dist_name]['delta_aic'] = fit_results[dist_name]['aic'] - min_aic
        fit_results[dist_name]['delta_bic'] = fit_results[dist_name]['bic'] - min_bic
    
    # Calculate Akaike weights
    delta_aics = np.array([fit_results[d]['delta_aic'] for d in fit_results])
    # Handle infinite AICs
    finite_mask = np.isfinite(delta_aics)
    akaike_weights = np.zeros(len(delta_aics))
    if finite_mask.any():
        finite_deltas = delta_aics[finite_mask]
        finite_weights = np.exp(-finite_deltas / 2)
        finite_weights /= finite_weights.sum()
        akaike_weights[finite_mask] = finite_weights
    
    for i, dist_name in enumerate(fit_results.keys()):
        fit_results[dist_name]['akaike_weight'] = akaike_weights[i]
    
    return {
        'neuron_id': neuron_id,
        'edge_type': edge_type,
        'neuron_type': neuron_type,
        'subtype': subtype,
        'n_segments': len(segment_lengths),
        'mean_length': np.mean(segment_lengths),
        'std_length': np.std(segment_lengths),
        'best_dist_aic': best_dist_aic,
        'best_dist_bic': best_dist_bic,
        'fit_results': fit_results
    }

def parallel_fit_distributions(df, n_cores=None):
    """
    Improved parallel fitting with better progress tracking.
    """
    if n_cores is None:
        n_cores = mp.cpu_count() - 1
    
    print(f"\n{'='*60}")
    print(f"PARALLEL DISTRIBUTION FITTING")
    print(f"{'='*60}")
    print(f"Using {n_cores} cores")
    
    # Prepare data for parallel processing
    grouped = df.groupby(['neuron_id', 'edge_type', 'neuron_type', 'subtype'])
    
    args_list = []
    for (neuron_id, edge_type, neuron_type, subtype), group_df in grouped:
        segment_lengths = group_df['segment_length'].values
        args_list.append((neuron_id, edge_type, neuron_type, subtype, segment_lengths))
    
    total_tasks = len(args_list)
    print(f"Fitting distributions to {total_tasks:,} neuron-edge combinations...")
    print(f"Estimated time: {total_tasks * 0.1 / n_cores / 60:.1f} to {total_tasks * 0.2 / n_cores / 60:.1f} minutes")
    
    # Run parallel fitting with chunksize for better performance
    start_time = time.time()
    chunksize = max(1, total_tasks // (n_cores * 10))  # Optimize chunk size
    
    results = []
    with mp.Pool(n_cores) as pool:
        # Use imap_unordered for better performance and progress tracking
        with tqdm(total=total_tasks, desc="Fitting distributions") as pbar:
            for result in pool.imap_unordered(process_single_neuron_edge, args_list, chunksize=chunksize):
                results.append(result)
                pbar.update(1)
                
                # Print periodic updates
                if pbar.n % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = pbar.n / elapsed
                    remaining = (total_tasks - pbar.n) / rate
                    print(f"  Processed {pbar.n}/{total_tasks} - Rate: {rate:.1f} fits/sec - ETA: {remaining/60:.1f} min")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Average time per neuron-edge: {elapsed_time/total_tasks:.3f} seconds")
    
    return results

def summarize_fit_results(fit_results):
    """
    Create summary tables from the fitting results.
    """
    print(f"\n{'='*60}")
    print("DISTRIBUTION FITTING SUMMARY")
    print(f"{'='*60}")
    
    # Convert results to DataFrame for easier analysis
    summary_data = []
    detailed_params = []
    
    for result in fit_results:
        # Summary row
        summary_row = {
            'neuron_id': result['neuron_id'],
            'edge_type': result['edge_type'],
            'neuron_type': result['neuron_type'],
            'subtype': result['subtype'],
            'subgroup': f"{result['neuron_type']}_{result['subtype']}_{result['edge_type']}",
            'n_segments': result['n_segments'],
            'mean_length': result['mean_length'],
            'best_dist_aic': result['best_dist_aic'],
            'best_dist_bic': result['best_dist_bic']
        }
        
        # Add AIC values and weights for each distribution
        for dist_name in result['fit_results']:
            summary_row[f'aic_{dist_name}'] = result['fit_results'][dist_name]['aic']
            summary_row[f'weight_{dist_name}'] = result['fit_results'][dist_name]['akaike_weight']
            summary_row[f'ks_pval_{dist_name}'] = result['fit_results'][dist_name]['ks_pval']
            
            # Store parameters separately
            param_row = {
                'neuron_id': result['neuron_id'],
                'edge_type': result['edge_type'],
                'distribution': dist_name,
                'params': result['fit_results'][dist_name]['param_dict']
            }
            detailed_params.append(param_row)
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    params_df = pd.DataFrame(detailed_params)
    
    # Analyze best distributions by subgroup
    print("\n1. BEST DISTRIBUTION BY SUBGROUP (% of neurons)")
    print("-" * 50)
    
    for subgroup in sorted(summary_df['subgroup'].unique()):
        subgroup_data = summary_df[summary_df['subgroup'] == subgroup]
        n_neurons = len(subgroup_data)
        
        print(f"\n{subgroup} (n={n_neurons} neurons):")
        
        # Count best distributions by AIC
        aic_counts = subgroup_data['best_dist_aic'].value_counts()
        for dist, count in aic_counts.head(3).items():
            pct = (count / n_neurons) * 100
            print(f"  {dist:12s}: {pct:5.1f}% ({count} neurons)")
    
    # Overall distribution preferences
    print("\n2. OVERALL DISTRIBUTION RANKINGS")
    print("-" * 50)
    
    dist_names = ['lognormal', 'gamma', 'weibull', 'pareto', 'exponential', 
                  'loglogistic', 'burr12', 'betaprime', 'expweibull']
    
    overall_best_counts = summary_df['best_dist_aic'].value_counts()
    print("\nBy AIC (across all neuron-edge combinations):")
    for dist, count in overall_best_counts.items():
        pct = (count / len(summary_df)) * 100
        print(f"  {dist:12s}: {pct:5.1f}% ({count}/{len(summary_df)})")
    
    return summary_df, params_df

def visualize_distribution_preferences(summary_df):
    """
    Create visualizations of distribution preferences across subgroups.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Heatmap of best distribution by subgroup
    ax = axes[0, 0]
    
    # Create proportion matrix
    prop_matrix = summary_df.groupby(['subgroup', 'best_dist_aic']).size().unstack(fill_value=0)
    prop_matrix = prop_matrix.div(prop_matrix.sum(axis=1), axis=0) * 100
    
    sns.heatmap(prop_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': '% of neurons'})
    ax.set_title('Best Distribution (by AIC) Across Subgroups')
    ax.set_xlabel('Distribution')
    ax.set_ylabel('Subgroup')
    
    # 2. Distribution of Akaike weights for top distributions
    ax = axes[0, 1]
    
    # Get average weights for each distribution
    weight_cols = [col for col in summary_df.columns if col.startswith('weight_')]
    avg_weights = summary_df[weight_cols].mean()
    avg_weights = avg_weights.sort_values(ascending=False)
    avg_weights.index = [col.replace('weight_', '') for col in avg_weights.index]
    
    ax.bar(range(len(avg_weights)), avg_weights.values)
    ax.set_xticks(range(len(avg_weights)))
    ax.set_xticklabels(avg_weights.index, rotation=45, ha='right')
    ax.set_ylabel('Average Akaike Weight')
    ax.set_title('Average Model Weights Across All Fits')
    ax.grid(True, alpha=0.3)
    
    # 3. AIC differences from best model
    ax = axes[1, 0]
    
    # Calculate average delta AIC for each distribution
    delta_aic_data = []
    for dist in ['lognormal', 'gamma', 'weibull', 'exponential', 'loglogistic']:
        aic_col = f'aic_{dist}'
        if aic_col in summary_df.columns:
            # Calculate delta from best for each row
            best_aic = summary_df[[f'aic_{d}' for d in ['lognormal', 'gamma', 'weibull', 'exponential', 'loglogistic'] 
                                   if f'aic_{d}' in summary_df.columns]].min(axis=1)
            delta = summary_df[aic_col] - best_aic
            # Remove infinite values
            delta_finite = delta[np.isfinite(delta)]
            if len(delta_finite) > 0:
                delta_aic_data.append({
                    'distribution': dist,
                    'mean_delta': delta_finite.mean(),
                    'median_delta': delta_finite.median()
                })
    
    delta_df = pd.DataFrame(delta_aic_data)
    if not delta_df.empty:
        delta_df = delta_df.sort_values('mean_delta')
        ax.bar(range(len(delta_df)), delta_df['mean_delta'].values)
        ax.set_xticks(range(len(delta_df)))
        ax.set_xticklabels(delta_df['distribution'].values, rotation=45, ha='right')
        ax.set_ylabel('Mean ΔAIC from Best')
        ax.set_title('Average AIC Difference from Best Model')
        ax.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='ΔAIC=2')
        ax.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='ΔAIC=4')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Edge type comparison
    ax = axes[1, 1]
    
    edge_comparison = summary_df.groupby(['edge_type', 'best_dist_aic']).size().unstack(fill_value=0)
    edge_comparison = edge_comparison.div(edge_comparison.sum(axis=1), axis=0) * 100
    
    edge_comparison.T.plot(kind='bar', ax=ax)
    ax.set_xlabel('Distribution')
    ax.set_ylabel('% of neurons')
    ax.set_title('Distribution Preference by Edge Type')
    ax.legend(title='Edge Type')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_fitting_results(fit_results, summary_df, params_df):
    """
    Save the fitting results to files for later use.
    """
    import pickle
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save raw results as pickle (preserves all information)
    with open('distribution_fit_results.pkl', 'wb') as f:
        pickle.dump(fit_results, f)
    print("✓ Saved raw fitting results to 'distribution_fit_results.pkl'")
    
    # Save summary DataFrame
    summary_df.to_csv('distribution_fit_summary.csv', index=False)
    print("✓ Saved summary to 'distribution_fit_summary.csv'")
    
    # Save parameters DataFrame
    params_df.to_csv('distribution_parameters.csv', index=False)
    print("✓ Saved parameters to 'distribution_parameters.csv'")
    
    # Create a report of key findings
    report = []
    report.append("DISTRIBUTION FITTING REPORT")
    report.append("=" * 60)
    report.append(f"Total neuron-edge combinations analyzed: {len(fit_results):,}")
    report.append(f"Distributions tested: 9")
    report.append("")
    report.append("TOP 3 DISTRIBUTIONS BY FREQUENCY:")
    top_dists = summary_df['best_dist_aic'].value_counts().head(3)
    for i, (dist, count) in enumerate(top_dists.items(), 1):
        pct = (count / len(summary_df)) * 100
        report.append(f"{i}. {dist}: {pct:.1f}% of neurons")
    
    report_text = "\n".join(report)
    with open('distribution_fit_report.txt', 'w') as f:
        f.write(report_text)
    print("✓ Saved report to 'distribution_fit_report.txt'")
    
    print("\n" + report_text)
    
    return report_text


