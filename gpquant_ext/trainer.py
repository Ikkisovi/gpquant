"""
Trainer: Main training loop integrating GPQuant with time-series strategy
"""
import numpy as np
import pandas as pd
from gpquant.SymbolicRegressor import SymbolicRegressor

from .ops_ts import register_ts_ops, get_ts_whitelist
from .features import make_features
from .strategy_ts import map_f_to_positions
from .fitness import fitness_info_sharpe_constrained, evaluate_window_metrics, print_metrics
from .benchmark import equal_weight_buy_hold
from .dataloader import DataSplitter


class TimeSeriesTrainer:
    """
    Trainer for time-series adaptive line strategy

    Integrates:
        - Feature engineering
        - GPQuant symbolic regression
        - Fixed strategy template (line-deviation mapping)
        - Information Sharpe fitness with tracking constraints
        - Walk-forward validation
    """

    def __init__(self, config):
        """
        Initialize trainer

        Args:
            config: dict with configuration parameters
        """
        self.config = config
        self.results = []
        self.best_formulas = []

        # Register time-series operators
        register_ts_ops()

        print("\n" + "="*80)
        print("Time-Series Adaptive Line Trainer Initialized")
        print("="*80)
        self._print_config()

    def _print_config(self):
        """Print configuration"""
        print("\nConfiguration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def train_fold(self, train_panel, val_panel, fold_idx):
        """
        Train on a single fold

        Args:
            train_panel: Training data DataFrame
            val_panel: Validation data DataFrame
            fold_idx: Fold index

        Returns:
            dict: Results for this fold
        """
        print(f"\n{'='*80}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*80}")

        # ====================================================================
        # 1. Feature Engineering
        # ====================================================================

        print("\n[1/5] Feature Engineering...")
        X_train, rets_train, prices_train, meta_train = make_features(train_panel)
        X_val, rets_val, prices_val, meta_val = make_features(val_panel)

        # ====================================================================
        # 2. Benchmark (Equal-Weight Buy-and-Hold)
        # ====================================================================

        print("\n[2/5] Calculating Benchmark...")
        bench_train = equal_weight_buy_hold(rets_train)
        bench_val = equal_weight_buy_hold(rets_val)

        print(f"  Train benchmark: {len(bench_train)} periods")
        print(f"  Val benchmark: {len(bench_val)} periods")

        # ====================================================================
        # 3. GPQuant Search for Adaptive Line f(t)
        # ====================================================================

        print("\n[3/5] Genetic Programming Search...")

        # Create custom fitness function
        def custom_fitness_function(program, X, y, sample_weight):
            """
            Custom fitness for GPQuant

            Args:
                program: Symbolic tree program
                X: Feature matrix
                y: Target (not used, we use returns directly)
                sample_weight: Sample weights (not used)

            Returns:
                float: Fitness score
            """
            try:
                # Execute formula on features to get f(t)
                factor_values = program.execute(X)

                # Convert to DataFrame matching prices structure
                factor_df = pd.Series(factor_values, index=X.index).to_frame('factor')

                # Map to positions using fixed strategy
                w, net_returns, metrics = map_f_to_positions(
                    factor_df,
                    prices_train,
                    rets_train,
                    k=self.config.get('k_tanh', 1.0),
                    z_L=self.config.get('z_L', 40),
                    cost_bps=self.config.get('cost_bps', 0.0005),
                    vol_window=self.config.get('vol_window', 60),
                )

                # Calculate fitness with constraints
                fitness = fitness_info_sharpe_constrained(
                    net_returns,
                    bench_train,
                    ann_factor=self.config.get('ann_K', 504),
                    cap_mdd_rel=self.config.get('cap_mdd', 0.05),
                    lambda_mdd=self.config.get('lambda_mdd', 3.0),
                    gamma_excess=self.config.get('gamma_excess', 2.0),
                    delta_complexity=self.config.get('delta_complexity', 0.001),
                    epsilon_turnover=self.config.get('epsilon_turnover', 0.1),
                    formula_complexity=program.length_,
                    turnover_series=metrics['turnover'],
                    hard_constraints=self.config.get('hard_constraints', False),
                )

                return fitness

            except Exception as e:
                print(f"    Error in fitness evaluation: {e}")
                return -999.0

        # Initialize SymbolicRegressor
        gp_config = self.config.get('gp_config', {})

        sr = SymbolicRegressor(
            population_size=gp_config.get('population_size', 200),
            generations=gp_config.get('generations', 30),
            tournament_size=gp_config.get('tournament_size', 20),
            stopping_criteria=gp_config.get('stopping_criteria', 2.0),
            p_crossover=gp_config.get('p_crossover', 0.7),
            p_subtree_mutate=gp_config.get('p_subtree_mutate', 0.15),
            p_hoist_mutate=gp_config.get('p_hoist_mutate', 0.1),
            p_point_mutate=gp_config.get('p_point_mutate', 0.05),
            init_depth=gp_config.get('init_depth', (3, 6)),
            init_method=gp_config.get('init_method', 'half and half'),
            function_set=get_ts_whitelist(),  # Only time-series safe operators
            variable_set=list(X_train.columns),
            const_range=gp_config.get('const_range', (1, 10)),
            ts_const_range=gp_config.get('ts_const_range', (2, 60)),
            build_preference=gp_config.get('build_preference', [0.75, 0.75]),
            metric='custom',  # Use our custom fitness
            parsimony_coefficient=gp_config.get('parsimony_coefficient', 0.001),
        )

        # Override fitness function
        sr.metric = type('obj', (object,), {
            'sign': 1,  # Higher is better
            'function': custom_fitness_function
        })()

        # Fit
        print(f"  Population: {sr.population_size}, Generations: {sr.generations}")
        sr.fit(X_train, rets_train['ret1'])  # y is dummy, not used

        # Best formula
        best_formula = sr.best_estimator
        best_fitness_train = sr.best_fitness

        print(f"\n✓ Best formula found:")
        print(f"    {best_formula}")
        print(f"    Train fitness: {best_fitness_train:.4f}")

        # ====================================================================
        # 4. Validation
        # ====================================================================

        print("\n[4/5] Validation...")

        # Execute on validation set
        factor_val = best_formula.execute(X_val)
        factor_val_df = pd.Series(factor_val, index=X_val.index).to_frame('factor')

        # Map to positions
        w_val, net_val, metrics_val = map_f_to_positions(
            factor_val_df,
            prices_val,
            rets_val,
            k=self.config.get('k_tanh', 1.0),
            z_L=self.config.get('z_L', 40),
            cost_bps=self.config.get('cost_bps', 0.0005),
            vol_window=self.config.get('vol_window', 60),
        )

        # Evaluate metrics
        val_metrics = evaluate_window_metrics(
            net_val,
            bench_val,
            ann_factor=self.config.get('ann_K', 504)
        )

        # ====================================================================
        # 5. Report
        # ====================================================================

        print("\n[5/5] Results Summary")
        print_metrics(val_metrics, window_name=f"Validation - Fold {fold_idx}")

        # Store results
        fold_result = {
            'fold': fold_idx,
            'formula': str(best_formula),
            'formula_complexity': best_formula.length_,
            'train_fitness': best_fitness_train,
            **val_metrics,
            'mean_turnover': metrics_val['mean_turnover'],
        }

        return fold_result

    def train_walkforward(self, panel):
        """
        Train with walk-forward validation

        Args:
            panel: Full panel DataFrame

        Returns:
            list: Results for all folds
        """
        print("\n" + "="*80)
        print("Walk-Forward Training")
        print("="*80)

        # Create data splitter
        splitter = DataSplitter(
            panel,
            train_months=self.config.get('train_months', 12),
            val_months=self.config.get('val_months', 1),
            embargo_periods=self.config.get('embargo', 10),
        )

        n_folds = splitter.count_folds()
        print(f"\nTotal folds: {n_folds}")

        # Train each fold
        self.results = []

        for fold_idx, (train_df, val_df) in enumerate(splitter.get_splits()):
            try:
                fold_result = self.train_fold(train_df, val_df, fold_idx)
                self.results.append(fold_result)
            except Exception as e:
                print(f"\n✗ Error in fold {fold_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ====================================================================
        # Aggregate Results
        # ====================================================================

        print("\n" + "="*80)
        print("Walk-Forward Summary")
        print("="*80)

        if len(self.results) == 0:
            print("✗ No successful folds")
            return self.results

        results_df = pd.DataFrame(self.results)

        print(f"\n{results_df.to_string(index=False)}")

        # Aggregate statistics
        print(f"\n{'='*80}")
        print("Aggregate Statistics (Validation)")
        print(f"{'='*80}")

        print(f"\nInformation Sharpe:")
        print(f"  Mean: {results_df['information_sharpe'].mean():>8.4f}")
        print(f"  Std:  {results_df['information_sharpe'].std():>8.4f}")
        print(f"  Min:  {results_df['information_sharpe'].min():>8.4f}")
        print(f"  Max:  {results_df['information_sharpe'].max():>8.4f}")

        print(f"\nConstraint Satisfaction:")
        pct_mdd = results_df['meets_mdd_constraint'].mean() * 100
        pct_excess = results_df['meets_excess_constraint'].mean() * 100
        pct_all = results_df['meets_all_constraints'].mean() * 100

        print(f"  MDD Constraint:     {pct_mdd:>6.1f}% of folds")
        print(f"  Excess Constraint:  {pct_excess:>6.1f}% of folds")
        print(f"  All Constraints:    {pct_all:>6.1f}% of folds")

        print(f"\nComplexity:")
        print(f"  Mean formula length: {results_df['formula_complexity'].mean():.1f}")

        print(f"\nTurnover:")
        print(f"  Mean: {results_df['mean_turnover'].mean():.4f}")

        return self.results

    def save_results(self, path):
        """Save results to CSV"""
        if len(self.results) == 0:
            print("No results to save")
            return

        results_df = pd.DataFrame(self.results)
        results_df.to_csv(path, index=False)
        print(f"\n✓ Results saved to {path}")
