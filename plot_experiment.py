#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import List
import argparse
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.figure import Figure
import colorcet as cc
import seaborn as sns

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rliable.plot_utils import _annotate_and_decorate_axis
import json
import collections

from marl_eval.plotting_tools.plotting import (
    aggregate_scores,
    performance_profiles,
    probability_of_improvement,
    sample_efficiency_curves,
)

from marl_eval.utils.data_processing_utils import (
    create_matrices_for_rliable,
    data_process_pipeline,
)

from matplotlib import pyplot as plt

from marl_eval.utils.data_processing_utils import (
    lower_case_inputs,
    get_and_aggregate_data_single_task
)


def metric_to_label(metric):
    m = metric.replace('agents_', '')
    if m == 'ext_return':
        return "Average Return"
    elif m == 'average_zapped_others':
        return "Other agents zapped"
    elif m == 'ext_return_equality':
        return "Equality"
    elif m == 'last_tsr_range':
        return '$e^{avg}$'
    elif m == 'average_age':
        return 'Average Age ($\hat{ \\tau }$)'
    elif m == 'proportion_own_coins':
        return 'Proportion of Own Coins'

    return m.replace('_', ' ').title()


def load_and_merge_json_dicts(
    json_input_files: List[str], json_output_file: Optional[str] = None
) -> Dict:
    """Loads and merges json dictionaries to form the ``marl-eval`` input dictionary .

    Args:
       json_input_files (list of str): a list containing the absolute paths to the json files
       json_output_file (str, optional): if specified, the merged dictionary will be also written
            to the file in this absolute path

    Returns:
        the dict obtained by merging all the json files

    """

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    dicts = []
    for file in json_input_files:
        with open(file, "r") as f:
            dicts.append(json.load(f))
    full_dict = {}
    for single_dict in dicts:
        update(full_dict, single_dict)

    if json_output_file is not None:
        with open(json_output_file, "w+") as f:
            json.dump(full_dict, f, indent=4)

    return full_dict

def bootstrap_confidence_interval(dataset, confidence=0.95, iterations=10000, sample_size=1.0, statistic=np.mean):
    """
    Bootstrap the confidence intervals for a given sample of a population
    and a statistic.

    Args:
        dataset: A list of values, each a sample from an unknown population
        confidence: The confidence value (a float between 0 and 1.0)
        iterations: The number of iterations of resampling to perform
        sample_size: The sample size for each of the resampled (0 to 1.0
                     for 0 to 100% of the original data size)
        statistic: The statistic to use. This must be a function that accepts
                   a list of values and returns a single value.

    Returns:
        Returns the upper and lower values of the confidence interval.
    """
    dataset = np.array(dataset)  # Convert to numpy array if not already
    n_size = int(len(dataset) * sample_size)
    stats = np.zeros(iterations)

    for i in range(iterations):
        # Sample (with replacement) using numpy
        sample = np.random.choice(dataset, size=n_size, replace=True)
        # Calculate user-defined statistic and store it
        stats[i] = statistic(sample)

    stats = np.sort(stats)
    # Compute percentiles for the confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (confidence + (1 - confidence) / 2) * 100
    lval, uval = np.percentile(stats, [lower_percentile, upper_percentile])

    return lval, uval


def get_algorithm_styles(algorithms: List[str], palette="colorblind") -> Dict[str, Dict[str, str]]:
    """Returns a mapping of colors, linestyles and legends for each algorithm"""
    color_map = {}
    linestyle_map = {}
    legend_map = {algo: algo.upper() for algo in algorithms}
    unmatched = []
    for algo in algorithms:
        # Color
        if "ippo" in algo:
            color_map[algo] = "green"
        else:
            unmatched.append(algo)

        linestyle_map[algo] = 'solid'

    if len(set(color_map.values())) == 1:
        colors = sns.color_palette(palette, len(algorithms))
        for algo, color in zip(algorithms, colors):
            color_map[algo] = color

    return color_map, linestyle_map, legend_map


def aggregate_data_single_task_with_conf_intervals(
    processed_data: Dict[str, Any],
    metric_name: str,
    metrics_to_normalize: List[str],
    task_name: str,
    environment_name: str,
    bounds: str
) -> Dict[str, Any]:
    """Compute the 95% boostrapped CI over all independent \
        experiment runs at each evaluation step for a given \
        environment and task.

    Args:
        processed_data: Dictionary containing processed data.
        metric_name: Name of metric to aggregate.
        metrics_to_normalize: List of metrics to normalize.
        task_name: Name of task to aggregate.
        environment_name: Name of environment to aggregate.
    """

    mean_ci_lp_up = get_and_aggregate_data_single_task(processed_data, metric_name, metrics_to_normalize, task_name, environment_name)

    if metric_name in metrics_to_normalize:
        metric_to_find = f"mean_norm_{metric_name}"
    else:
        metric_to_find = f"mean_{metric_name}"

    # Get the data for the given metric and environment
    task_data = processed_data[environment_name][task_name]

    # Get the algorithm names, number of runs and total steps
    algorithms = list(task_data.keys())
    runs = list(task_data[algorithms[0]].keys())
    steps = list(task_data[algorithms[0]][runs[0]].keys())

    # Remove absolute metric from steps.
    steps = [step for step in steps if "absolute" not in step.lower()]

    for step in steps:
        # Loop over each algorithm
        for algorithm in algorithms:
            # Get the data for the given algorithm
            algorithm_data = task_data[algorithm]
            # Compute the 95% boostrapped CI for the given algorithm over all seeds at a given step
            run_total = []
            for run in runs:
                run_total.append(algorithm_data[run][step][metric_to_find])

            if "lp" not in mean_ci_lp_up[algorithm].keys() or "up" not in mean_ci_lp_up[algorithm].keys():
                mean_ci_lp_up[algorithm]["lp"] = []
                mean_ci_lp_up[algorithm]["up"] = []

            if len(run_total) > 1 and bounds == 'boostrapped CI':
                lp, up = bootstrap_confidence_interval(run_total)
                mean_ci_lp_up[algorithm]["lp"].append(lp)
                mean_ci_lp_up[algorithm]["up"].append(up)
            else:
                mean_ci_lp_up[algorithm]["lp"].append(np.min(run_total))
                mean_ci_lp_up[algorithm]["up"].append(np.max(run_total))

    return mean_ci_lp_up


def plot_single_task_curve(
    aggregated_data: Dict[str, Any],
    algorithms: list,
    colors: Optional[Dict] = None,
    color_palette: str = "colorblind",
    linestyles: Optional[Dict] = None,
    figsize: tuple = (7, 5),
    xlabel: str = "Number of Frames (in millions)",
    ylabel: str = "Aggregate Human Normalized Score",
    ax: Optional[Axes] = None,
    labelsize: str = "xx-large",
    ticklabelsize: str = "xx-large",
    legends: Optional[Dict] = None,
    run_times: Optional[Dict] = None,
    **kwargs: Any,
) -> Figure:

    extra_info = aggregated_data.pop("extra")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if algorithms is None:
        algorithms = list(aggregated_data.keys())
    
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
        
    if linestyles is None:
        linestyles = {algorithm: "solid" for algorithm in algorithms}

    marker = kwargs.pop("marker", "o")
    linewidth = kwargs.pop("linewidth", 2)

    last_values = {}
    for algorithm in sorted(algorithms):
        print(f"\tPlotting {algorithm}")
        x_axis_len = len(aggregated_data[algorithm]["mean"])

        # Set x-axis values to match evaluation interval steps.
        x_axis_values = np.arange(x_axis_len) * extra_info["evaluation_interval"]

        if run_times is not None:
            x_axis_values = np.linspace(0, run_times[algorithm] / 60, x_axis_len)

        metric_values = np.array(aggregated_data[algorithm]["mean"])
        confidence_interval = np.array(aggregated_data[algorithm]["ci"])
        lower, upper = (
            np.array(aggregated_data[algorithm]["lp"]),
            np.array(aggregated_data[algorithm]["up"]),
        )

        print(f"\tLast Value: {metric_values[-1]:.3f} ({lower[-1]:.3f}, {upper[-1]:.3f})")
        last_values[algorithm] = metric_values[-1]

        if legends is not None:
            algorithm_name = legends[algorithm]
        else:
            algorithm_name = algorithm

        algorithm_name = algorithm_name.replace(' Red', '').replace(' Blue', '')

        ax.plot(
            x_axis_values,
            metric_values,
            color=colors[algorithm],
            marker=marker,
            linewidth=linewidth + 1 if 'fair' in algorithm.lower() else linewidth,
            label=algorithm_name,
            linestyle=linestyles[algorithm]
        )
        ax.fill_between(
            x_axis_values, y1=lower, y2=upper, color=colors[algorithm], alpha=0.1
        )

    if 'iql w/ fair&localia - strong (+1.5 per apple)' in algorithms and 'iql w/ fair&localia - weak (+0.5 per apple)' in algorithms:
        print(f"\t Difference from iql w/ fair&localia - strong (+1.5 per apple) to iql w/ fair&localia - weak (+0.5 per apple): {last_values['iql w/ fair&localia - strong (+1.5 per apple)'] - last_values['iql w/ fair&localia - weak (+0.5 per apple)']}")
        print(f"\t Ratio of iql w/ fair&localia - strong (+1.5 per apple) to iql w/ fair&localia - weak (+0.5 per apple): {last_values['iql w/ fair&localia - strong (+1.5 per apple)'] / last_values['iql w/ fair&localia - weak (+0.5 per apple)']}\n")
    elif 'iql w/ fair&localia - regular' in algorithms and '-iql w/ fair&localia - strong (wider zap beam)' in algorithms:
        print(f"\t Difference from -iql w/ fair&localia - strong (wider zap beam) to iql w/ fair&localia - regular: {last_values['-iql w/ fair&localia - strong (wider zap beam)'] - last_values['iql w/ fair&localia - regular']}")
        print(f"\t Ratio of -iql w/ fair&localia - strong (wider zap beam) to iql w/ fair&localia - regular: {last_values['-iql w/ fair&localia - strong (wider zap beam)'] / last_values['iql w/ fair&localia - regular']}\n")

    return _annotate_and_decorate_axis(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        labelsize=labelsize,
        ticklabelsize=ticklabelsize,
        **kwargs,
    )

def plot_single(
    processed_data: Dict[str, Dict[str, Any]],
    environment_name: str,
    task_name: str,
    metric_name: str,
    metrics_to_normalize: List[str],
    xlabel: str = "Timesteps",
    run_times: Optional[Dict[str, float]] = None,
    color_palette=None,
    bounds="boostrapped CI"
) -> Figure:
    """Produces aggregated plot for a single task in an environment.

    Args:
        processed_data: Dictionary containing processed data.
        environment_name: Name of environment to produce plots for.
        task_name: Name of task to produce plots for.
        metric_name: Name of metric to produce plots for.
        metrics_to_normalize: List of metrics that are normalised.
        xlabel: Label for x-axis.
        run_times: Dictionary that maps each algorithm to the number of seconds it
            took to run. If None, then environment steps will be displayed.
    """

    metric_name, task_name, environment_name, metrics_to_normalize = lower_case_inputs(
        metric_name, task_name, environment_name, metrics_to_normalize
    )

    task_mean_ci_min_max_data = aggregate_data_single_task_with_conf_intervals( # aggregate with the addition of boostrapped confidence intervals
        processed_data=processed_data,
        environment_name=environment_name,
        metric_name=metric_name,
        task_name=task_name,
        metrics_to_normalize=metrics_to_normalize,
        bounds=bounds
    )

    if metric_name in metrics_to_normalize:
        ylabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        ylabel = " ".join(metric_name.split("_")).capitalize()

    # Upper case all algorithm names
    upper_algo_dict = {
        (algo.lower() if algo != "extra" else algo): value
        for algo, value in task_mean_ci_min_max_data.items()
    }
    task_mean_ci_min_max_data = upper_algo_dict
    algorithms = sorted(list(task_mean_ci_min_max_data.keys()))
    algorithms.remove("extra")

    if run_times is not None:
        run_times = {algo.lower(): value for algo, value in run_times.items()}
        xlabel = "Time (Minutes)"

    color_map, linestyle_map, legend_map = get_algorithm_styles(algorithms)

    fig = plot_single_task_curve(
        task_mean_ci_min_max_data,
        algorithms=algorithms,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=algorithms,
        figsize=(15, 8),
        colors=color_map,
        color_palette=color_palette,
        legends=legend_map,
        linestyles=linestyle_map,
        run_times=run_times,
        marker="",
    )

    return fig


# Function to find keys under 'absolute_metrics' in the JSON structure
def find_absolute_metrics_keys(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == 'absolute_metrics' and isinstance(value, dict):
                return list(value.keys())
            else:
                result = find_absolute_metrics_keys(value)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_absolute_metrics_keys(item)
            if result:
                return result
    return None


def replace_absolute_metrics_with_last_avg(data):
    """
    Replaces absolute_metrics with the average of the last `num_last_evals` evaluation episodes.

    Args:
        data (dict): The raw dictionary containing experiment results.
        num_last_evals (int): Number of last evaluation episodes to average over.

    Returns:
        dict: Updated data dictionary with modified absolute_metrics.
    """
    for env in data.keys():
        for task in data[env].keys():
            for alg in data[env][task].keys():
                for seed in data[env][task][alg].keys():
                    last_values = {}  # Store last few evaluation values for averaging

                    # Iterate through all steps to find the last ones
                    step_keys = sorted([s for s in data[env][task][alg][seed].keys() if "step" in s],
                                       key=lambda x: int(x.split("_")[-1]))

                    last_step = step_keys[-1]

                    for metric_name, values in data[env][task][alg][seed][last_step].items():
                        if metric_name == "step_count":
                            continue  # Skip step counter

                        if metric_name not in last_values:
                            last_values[metric_name] = []

                        last_values[metric_name].extend(values)  # Collect all last N values

                    # Compute averages and store in absolute_metrics
                    if "absolute_metrics" not in data[env][task][alg][seed]:
                        data[env][task][alg][seed]["absolute_metrics"] = {}

                    for metric_name, values in last_values.items():
                        if values:
                            data[env][task][alg][seed]["absolute_metrics"][metric_name] = [np.mean(values)]

    return data

def remove_uncommon_keys(data):
    print("Checking for uncommon keys...")
    task_key = list(data["socialjax"].keys())[0]  # Extract the task key

    metric_sets = []
    for alg, alg_data in data["socialjax"][task_key].items():
        seed_key = list(alg_data.keys())[0]  # Extract the seed key
        metric_sets.append(set(alg_data[seed_key]['absolute_metrics'].keys()))

    if metric_sets:
        common_metrics_set = set.intersection(*metric_sets)

        for alg, alg_data in data["socialjax"][task_key].items():
            seed_key = list(alg_data.keys())[0]  # Extract the seed key

            keys_to_remove = set(alg_data[seed_key]["absolute_metrics"].keys()) - common_metrics_set
            for step, values in alg_data[seed_key].items():
                for key in keys_to_remove:
                    if step == "absolute_metrics":
                        print(f"\tRemoving {key} from {alg}")
                    del values[key]


def plot_experiments(json_files, metric_to_plot, env_name, bounds):
    if len(json_files) == 0:
        return
    raw_dict = load_and_merge_json_dicts(experiment_json_files)

    remove_uncommon_keys(raw_dict)
    absolute_metrics = find_absolute_metrics_keys(raw_dict)

    if metric_to_plot is None:
        metrics_to_plot = absolute_metrics
    else:
        if metric_to_plot not in absolute_metrics:
            return
        metrics_to_plot = [metric_to_plot]

    for metric in sorted(metrics_to_plot):
        processed_data = data_process_pipeline(raw_dict, metrics_to_normalize=[])
        (
            environment_comparison_matrix,
            sample_efficiency_matrix,
        ) = create_matrices_for_rliable(processed_data, environment_name=env_name, metrics_to_normalize=[]) # makes changes on the processed_data

        for task_name in raw_dict[env_name].keys():
            fig, ax = plt.subplots(figsize=(15, 8))  # Create a figure and axis

            print(f"Plotting {metric}")
            plot_single( # get_and_aggregate_data_single_task method (called within) expects every algorithm to have the same run names (ex: seed_0, seed_1)
                processed_data=processed_data,
                environment_name=env_name,
                task_name=task_name,
                metric_name=metric,
                metrics_to_normalize=[],
                color_palette=cc.glasbey_category10,
                bounds=bounds
            )

            if args['ymin'] and args['ymax']:
                plt.ylim(float(args['ymin']), float(args['ymax']))
            plt.xlabel("Timesteps", fontsize=24)
            plt.ylabel(metric_to_label(metric), fontsize=24)
            plt.xticks(fontsize=22)
            plt.yticks(fontsize=22)
            plt.gca().xaxis.get_offset_text().set_fontsize(14)
            pdf_file_name = f"{directory}/result_{metric}_{task_name}.pdf"
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.21), ncol=2, fontsize=22)
            plt.savefig(pdf_file_name, bbox_inches='tight')
            plt.close()
            try:
                subprocess.run(["pdfcrop", pdf_file_name])
                os.remove(pdf_file_name)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--folder", default=None, help="folder containing experiment results")
    parser.add_argument("-m", "--metric", default=None, help="metric to plot")
    parser.add_argument("-ymin", "--ymin", default=None, help="minimum value of y axis")
    parser.add_argument("-ymax", "--ymax", default=None, help="maximum value of y axis")
    parser.add_argument("-b", "--bounds", default='boostrapped CI', help="what to draw as error bounds")
    args = vars(parser.parse_args())

    metric_to_plot = args['metric']
    env_name = "socialjax"
    directory = Path(args['folder']).absolute()

    # Regular metrics
    experiment_json_files = []
    for file_path in directory.rglob('*.json'):
        experiment_json_files.append(file_path)

    plot_experiments(experiment_json_files, metric_to_plot, env_name, args['bounds'])
