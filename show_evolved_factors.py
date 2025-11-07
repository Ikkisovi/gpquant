"""
展示GPQuant训练过程中进化出的因子

这个脚本会运行一个小型训练，并详细展示每一代最佳因子的演变
"""
import pandas as pd
import numpy as np
from gpquant.SymbolicRegressor import SymbolicRegressor
from data_processor import (
    load_daily_am_pm_data,
    prepare_market_data,
    get_all_symbols,
)


def show_evolution():
    """展示因子进化过程"""
    print("="*80)
    print("GPQuant 因子进化演示")
    print("="*80)

    # 加载数据
    print("\n1. 加载数据...")
    df = load_daily_am_pm_data("daily_am_pm_data.csv")
    symbols = get_all_symbols(df)
    symbol = symbols[0]

    market_df = prepare_market_data(df, symbol, slippage=0.001)

    # 使用较小的数据集以便快速演示
    train_df = market_df.iloc[:200].copy()

    print(f"   训练数据: {len(train_df)} 个时间点")
    print(f"   可用变量: O, H, L, C, V, vwap")

    # 配置参数
    print("\n2. 配置遗传规划参数...")
    transformer_kwargs = {
        "init_cash": 10000,
        "charge_ratio": 0.0002,
        "d_center": 15,
        "d_band": 15,
        "band_width": 2.0,
        "factor_threshold": 0.3,
        "price": train_df["C"].values,
    }

    print("   可用函数库:")
    print("   - 基础数学: add, sub, mul, div, sqrt, log, abs, sign...")
    print("   - 时间序列: ts_mean, ts_std, ts_delta, ts_rank...")
    print("   - 技术指标: ts_MFI, ts_ATR, ts_ADX, ts_CCI...")
    print("   - 条件逻辑: if_then_else, if_cond_then_else...")
    print(f"   总共: 68个函数可组合!")

    # 初始化GP
    print("\n3. 初始化种群...")
    sr = SymbolicRegressor(
        population_size=50,  # 小种群便于展示
        tournament_size=8,
        generations=10,
        stopping_criteria=1.5,
        p_crossover=0.65,
        p_subtree_mutate=0.2,
        p_hoist_mutate=0.1,
        p_point_mutate=0.05,
        init_depth=(3, 5),
        init_method="half and half",
        function_set=[],  # 使用所有函数
        variable_set=["O", "H", "L", "C", "V", "vwap"],
        const_range=(1, 10),
        ts_const_range=(2, 20),
        build_preference=[0.75, 0.75],
        metric="sharpe ratio",
        transformer="three_line",
        transformer_kwargs=transformer_kwargs,
        parsimony_coefficient=0.005,
    )

    print(f"   生成了 {sr.population_size} 个随机因子公式...")

    # 自定义训练循环以展示更多细节
    print("\n4. 开始进化...")
    print("="*80)

    sr._SymbolicRegressor__build()  # 构建初始种群

    all_factors = []  # 记录所有代的因子

    for generation in range(sr.generations):
        # 评估当前代
        fitness = np.array([tree.fitness(train_df, train_df["C"]) for tree in sr.trees])
        sr.fitness = fitness

        # 找到最佳个体
        best_idx = np.nanargmax(sr.metric.sign * fitness)
        best_tree = sr.trees[best_idx]
        best_fitness = sr.metric.sign * fitness[best_idx]

        # 记录
        factor_formula = str(best_tree)
        all_factors.append({
            'generation': generation + 1,
            'formula': factor_formula,
            'fitness': best_fitness,
            'complexity': len(factor_formula)
        })

        # 显示本代信息
        print(f"\n第 {generation + 1:2d} 代:")
        print(f"  最佳因子: {factor_formula}")
        print(f"  Sharpe:   {best_fitness:.4f}")
        print(f"  复杂度:   {len(factor_formula)} 字符")

        # 显示种群多样性
        unique_formulas = len(set([str(tree) for tree in sr.trees]))
        print(f"  种群多样性: {unique_formulas}/{sr.population_size} 个不同公式")

        # 显示fitness分布
        valid_fitness = fitness[~np.isnan(fitness)]
        if len(valid_fitness) > 0:
            print(f"  Fitness范围: [{np.min(valid_fitness):.4f}, {np.max(valid_fitness):.4f}]")

        # 检查是否达到停止条件
        if sr.metric.sign * (best_fitness - sr.stopping_criteria) > 0:
            print(f"\n达到目标 Sharpe {sr.stopping_criteria}，停止训练！")
            break

        # 进化到下一代
        if generation < sr.generations - 1:
            sr._SymbolicRegressor__evolve()

    # 总结
    print("\n" + "="*80)
    print("5. 进化总结")
    print("="*80)

    factors_df = pd.DataFrame(all_factors)

    print("\n进化轨迹:")
    print(factors_df.to_string(index=False))

    print("\n最终最佳因子:")
    print(f"  公式: {factors_df.iloc[-1]['formula']}")
    print(f"  Sharpe: {factors_df.iloc[-1]['fitness']:.4f}")

    # 分析因子组成
    final_formula = factors_df.iloc[-1]['formula']
    print("\n因子分析:")

    # 检测使用的函数
    used_functions = []
    function_keywords = [
        'ts_mean', 'ts_std', 'ts_delta', 'ts_rank', 'ts_max', 'ts_min',
        'ts_corr', 'ts_ema', 'ts_MFI', 'ts_ATR', 'ts_CCI', 'ts_ADX',
        'div', 'mul', 'add', 'sub', 'sqrt', 'log', 'abs', 'sign'
    ]

    for func in function_keywords:
        if func in final_formula:
            used_functions.append(func)

    if used_functions:
        print(f"  使用的函数: {', '.join(used_functions)}")
    else:
        print(f"  使用的函数: 无（简单变量或常数）")

    # 检测使用的变量
    used_vars = []
    for var in ['O', 'H', 'L', 'C', 'V', 'vwap']:
        if var in final_formula:
            used_vars.append(var)

    if used_vars:
        print(f"  使用的变量: {', '.join(used_vars)}")

    # 性能改进
    if len(factors_df) > 1:
        initial_sharpe = factors_df.iloc[0]['fitness']
        final_sharpe = factors_df.iloc[-1]['fitness']
        improvement = final_sharpe - initial_sharpe
        print(f"\n性能改进:")
        print(f"  初始 Sharpe: {initial_sharpe:.4f}")
        print(f"  最终 Sharpe: {final_sharpe:.4f}")
        print(f"  改进幅度:    {improvement:.4f} ({improvement/abs(initial_sharpe)*100:.1f}%)")

    print("\n" + "="*80)
    print("展示完成！")
    print("\n这就是遗传规划如何自动创造和优化量化因子的过程。")
    print("每一代都在前一代的基础上，通过交叉、变异等操作探索新的因子组合。")
    print("="*80)


if __name__ == "__main__":
    show_evolution()
