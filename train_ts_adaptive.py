"""
Training Script: Time-Series Adaptive Line Strategy
Entry point for training with information Sharpe + tracking constraints
"""
import yaml
import argparse
from gpquant_ext.dataloader import load_panel
from gpquant_ext.trainer import TimeSeriesTrainer


def load_config(config_path='config_ts_adaptive.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return get_default_config()


def get_default_config():
    """Get default configuration"""
    return {
        # Data
        'data_path': 'daily_am_pm_data.csv',

        # Walk-forward
        'train_months': 12,
        'val_months': 1,
        'embargo': 10,

        # Strategy parameters
        'k_tanh': 1.0,
        'z_L': 40,
        'cost_bps': 0.0005,
        'vol_window': 60,
        'ann_K': 504,

        # Fitness constraints
        'cap_mdd': 0.05,
        'lambda_mdd': 3.0,
        'gamma_excess': 2.0,
        'delta_complexity': 0.001,
        'epsilon_turnover': 0.1,
        'hard_constraints': False,

        # GP configuration
        'gp_config': {
            'population_size': 200,
            'generations': 30,
            'tournament_size': 20,
            'stopping_criteria': 2.0,
            'p_crossover': 0.7,
            'p_subtree_mutate': 0.15,
            'p_hoist_mutate': 0.1,
            'p_point_mutate': 0.05,
            'init_depth': (3, 6),
            'init_method': 'half and half',
            'const_range': (1, 10),
            'ts_const_range': (2, 60),
            'build_preference': [0.75, 0.75],
            'parsimony_coefficient': 0.001,
        },

        # Output
        'results_path': 'results_ts_adaptive.csv',
    }


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Time-Series Adaptive Line Strategy')
    parser.add_argument('--config', type=str, default='config_ts_adaptive.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to data CSV (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (overrides config)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.data:
        config['data_path'] = args.data
    if args.output:
        config['results_path'] = args.output

    print("="*80)
    print("TIME-SERIES ADAPTIVE LINE STRATEGY")
    print("="*80)
    print("\nConfiguration loaded:")
    for key, value in config.items():
        if key != 'gp_config':
            print(f"  {key}: {value}")

    print("\nGP Configuration:")
    for key, value in config['gp_config'].items():
        print(f"  {key}: {value}")

    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)

    panel = load_panel(config['data_path'])

    # Initialize trainer
    trainer = TimeSeriesTrainer(config)

    # Train with walk-forward
    results = trainer.train_walkforward(panel)

    # Save results
    if results:
        trainer.save_results(config['results_path'])

    print("\n" + "="*80)
    print("Training Complete")
    print("="*80)


if __name__ == "__main__":
    main()
