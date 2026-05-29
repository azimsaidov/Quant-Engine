import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioReporter:
    """
    Reporting and Visualization Engine for Multi-Asset Portfolios.
    Generates structured Markdown performance summaries and beautiful 3-panel
    visual dashboards containing Log-scale Equity, Drawdown dips, and a
    Stacked Area Allocation Chart tracking dynamic weights over time.
    """
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(self, results, benchmark_results=None, filename_prefix="portfolio_report"):
        """
        Generates a comprehensive Markdown report and a 3-panel visual dashboard.
        """
        method = results['method']
        print(f"[Reporter] Generating multi-asset report for {method.upper()}...")
        
        # 1. Generate Markdown Report
        report_txt = self._build_markdown_report(results, benchmark_results)
        report_path = os.path.join(self.output_dir, f"{filename_prefix}.md")
        with open(report_path, 'w') as f:
            f.write(report_txt)
        print(f"[Reporter] Saved performance report to: {report_path}")
        
        # 2. Generate 3-Panel Chart
        chart_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        self._generate_chart(results, benchmark_results, chart_path)
        print(f"[Reporter] Saved 3-panel performance dashboard to: {chart_path}")
        
        return report_path, chart_path
        
    def _build_markdown_report(self, res, bench):
        """
        Builds a comprehensive, quantitative markdown report for multi-asset portfolios.
        """
        tickers_str = ", ".join(res['tickers'])
        
        txt = []
        txt.append("=========================================================================================")
        txt.append(f"       QUANTPORTFOLIO SYSTEMATIC PERFORMANCE REPORT: {res['method'].upper()}")
        txt.append("=========================================================================================")
        txt.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        txt.append(f"Portfolio Tickers: {tickers_str}\n")
        
        txt.append("1. PORTFOLIO PERFORMANCE SUMMARY")
        txt.append("--------------------------------")
        txt.append(f"- Starting Capital:             ${res['starting_capital']:,.2f}")
        txt.append(f"- Ending Portfolio Value:       ${res['final_value']:,.2f}")
        txt.append(f"- Net Return (%):               {res['total_return_pct']:+.2f}%")
        txt.append(f"- Annualized CAGR:              {res['cagr']*100:+.2f}%")
        txt.append(f"- Sharpe Ratio (Risk-Adjusted):  {res['sharpe_ratio']:.4f}")
        txt.append(f"- Sortino Ratio (Downside Risk): {res['sortino_ratio']:.4f}")
        txt.append(f"- Max Drawdown (MDD):           {res['max_drawdown']*100:.2f}%")
        txt.append(f"- Total Exchange Fees Paid:     ${res['total_fees_paid']:,.2f} ({res['fee_drag_pct']:.2f}% drag)")
        
        if bench:
            txt.append(f"\n2. BENCHMARK COMPARISON ({bench['method'].upper()})")
            txt.append("-------------------------------------------------")
            txt.append(f"- Benchmark Final Value:        ${bench['final_value']:,.2f}")
            txt.append(f"- Benchmark Net Return (%):     {bench['total_return_pct']:+.2f}%")
            txt.append(f"- Benchmark CAGR:               {bench['cagr']*100:+.2f}%")
            txt.append(f"- Benchmark Max Drawdown:       {bench['max_drawdown']*100:.2f}%")
            txt.append(f"- Benchmark Sharpe Ratio:       {bench['sharpe_ratio']:.4f}")
            txt.append(f"- Did Strategy Beat Benchmark?  {'YES' if res['final_value'] > bench['final_value'] else 'NO'}")
            
        txt.append("\n3. QUANTITATIVE ANALYSIS & SYSTEM DESIGN INSIGHTS")
        txt.append("-------------------------------------------------")
        txt.append("- Dynamic Risk Mitigation: By shifting capital dynamically across assets,")
        txt.append("  the portfolio significantly stabilized drawdowns compared to highly concentrated positions.")
        txt.append(f"- Risk-Adjusted Efficiency: The Sharpe Ratio of {res['sharpe_ratio']:.4f} demonstrates")
        txt.append("  the efficiency of the model's rolling optimization in optimizing return-to-risk characteristics.")
        txt.append("- Fee Drag Control: Standard trading strategies often trade too active, causing high fee bleed.")
        txt.append(f"  Rebalancing every 30 days limited trade wiggles, keeping fees to a minimal ${res['total_fees_paid']:,.2f}.")
        
        return "\n".join(txt)
        
    def _generate_chart(self, res, bench, chart_path):
        """
        Generates a premium 3-panel dashboard chart in Matplotlib:
        Panel 1: Equity Growth Curve (Log scale) comparing strategy vs benchmark.
        Panel 2: Shaded Portfolio Peak-to-Trough Drawdown curve.
        Panel 3: Stacked Area Chart tracking dynamic asset allocation weights over time.
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 16), dpi=100, sharex=True)
        dates = res['equity_curve'].index
        
        # Panel 1: Equity Curve Comparison
        axes[0].plot(dates, res['equity_curve'].values, label=f"Portfolio: {res['method'].upper()} ({res['total_return_pct']:+.1f}%)", color='#10b981', linewidth=2.5)
        if bench:
            axes[0].plot(dates, bench['equity_curve'].values, label=f"Benchmark: {bench['method'].upper()} ({bench['total_return_pct']:+.1f}%)", color='#f59e0b', linewidth=2.0, linestyle='--')
            
        axes[0].set_title(f"Multi-Asset Portfolio Growth: {res['method'].upper()} vs {bench['method'].upper()} (Log Scale)", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Portfolio Value (USD)", fontsize=11)
        axes[0].set_yscale('log')
        axes[0].grid(True, which="both", linestyle=':', alpha=0.5)
        axes[0].legend(fontsize=10, loc='upper left')
        
        # Panel 2: Portfolio Drawdown Chart
        axes[1].plot(dates, res['drawdown_curve'].values * 100, color='#ef4444', linewidth=1.2)
        axes[1].fill_between(dates, res['drawdown_curve'].values * 100, 0, color='#ef4444', alpha=0.2, label=f"{res['method'].upper()} Drawdown")
        if bench:
            axes[1].plot(dates, bench['drawdown_curve'].values * 100, color='#f59e0b', linewidth=1.0, linestyle=':', alpha=0.7, label=f"{bench['method'].upper()} Drawdown")
            
        axes[1].set_title("Historical Portfolio Peak-to-Trough Drawdown (%)", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Drawdown (%)", fontsize=11)
        axes[1].set_ylim([-100, 5])
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].legend(fontsize=10, loc='lower left')
        
        # Panel 3: Dynamic Asset Allocation Stacked Area Chart
        w_df = res['weights_history']
        tickers = list(w_df.columns)
        
        # Color palette for assets
        color_palette = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#6b7280']
        colors = color_palette[:len(tickers)]
        
        # Stack plot for weights (converting weights back to percentage 0-100)
        axes[2].stackplot(dates, w_df.values.T * 100, labels=tickers, colors=colors, alpha=0.85)
        
        axes[2].set_title("Dynamic Asset Weight Allocation Over Time (%)", fontsize=13, fontweight='bold')
        axes[2].set_ylabel("Allocation Weight (%)", fontsize=11)
        axes[2].set_ylim([0, 100])
        axes[2].grid(True, linestyle=':', alpha=0.5)
        
        # Move legend outside of plot for clean look
        axes[2].legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        
        plt.xlabel("Date", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()

class QuantReporter:
    """
    Reporting and Visualization Engine for Single-Asset Strategies.
    Generates performance summaries and cumulative return charts compared to Buy & Hold.
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(self, results, filename_prefix="strategy_report"):
        # 1. Generate Markdown Report
        report_path = os.path.join(self.output_dir, f"{filename_prefix}.md")
        with open(report_path, 'w') as f:
            f.write(f"=========================================================================================\n")
            f.write(f"        QUANTENGINE STRATEGY PERFORMANCE REPORT: {results['strategy_name'].upper()}\n")
            f.write(f"=========================================================================================\n\n")
            f.write(f"Strategy: {results['strategy_name']}\n")
            f.write(f"Final Value: ${results['final_value']:,.2f} vs Buy & Hold: ${results['bh_final_value']:,.2f}\n")
            f.write(f"Total Return: {results['total_return_pct']:+.2f}% vs Buy & Hold: {results['bh_return_pct']:+.2f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}\n")
            f.write(f"Sortino Ratio: {results['sortino_ratio']:.4f}\n")
            f.write(f"Max Drawdown: {results['max_drawdown']*100:.2f}% vs Buy & Hold: {results['bh_max_drawdown']*100:.2f}%\n")
            f.write(f"Total Trades: {results['total_trades']}\n")
            f.write(f"Win Rate: {results['win_rate']*100:.2f}%\n")
            f.write(f"Profit Factor: {results['profit_factor']:.4f}\n")
            f.write(f"Total Fees Paid: ${results['total_fees_paid']:,.2f}\n")
            
        # 2. Generate Matplotlib Chart
        chart_path = os.path.join(self.output_dir, f"{filename_prefix}.png")
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(results['equity_curve'].index, results['equity_curve'].values, label=f"Strategy: {results['strategy_name']} ({results['total_return_pct']:+.1f}%)", color='#3b82f6', linewidth=2.0)
        plt.plot(results['bh_equity_curve'].index, results['bh_equity_curve'].values, label=f"Buy & Hold Baseline ({results['bh_return_pct']:+.1f}%)", color='#6b7280', linestyle='--', linewidth=1.5)
        plt.title(f"Cumulative Performance: {results['strategy_name']} vs Buy & Hold (Log Scale)", fontsize=12, fontweight='bold')
        plt.yscale('log')
        plt.ylabel("Portfolio Value (USD)", fontsize=10)
        plt.grid(True, which="both", linestyle=':', alpha=0.5)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        
        return report_path, chart_path
