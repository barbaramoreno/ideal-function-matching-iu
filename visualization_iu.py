""" Visualization module: this module creates the bokeh plots for the project, it
generates 3 interactive plots that can be open in the browser, which are:
- training data vs their best matching ideal functions
- test data classification showing which points got mapped where
- deviations chart showing how far each test point was from its ideal
Reference from Programming with Python – CSEMDSPWP01 Course Book. IU:
-Unit 3 "Data Visualization"
"""
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Legend
from bokeh.palettes import Category10
import pandas as pd
import numpy as np

def plot_training_vs_ideal(train_df, ideal_df, matched_functions, filename="plots/training_vs_ideal.html"):
    """
    This plots shows each training function next to its best matching ideal function
    and it creates a 2x2 grid so it can show all 4 comparisons at once
    Args:
        train_df: training data [x, y1, y2, y3, y4]
        ideal_df: ideal functions data
        matched_functions: dict {train_col: ideal_col}
        filename: where to the html file is saved
    """
    output_file(filename, title="Training Data vs Ideal Functions")

    x = train_df['x'].values
    fig_list = []
    colors = ['#7b2d8e', '#ff6f00', '#00897b', '#c62828']

    for i, (train_col, ideal_col) in enumerate(matched_functions.items()):
        p = figure(
            title=f"{train_col} vs {ideal_col}",
            x_axis_label="x",
            y_axis_label="y",
            width=500,
            height=400
        )

        # training data shown as scatter points
        p.scatter(
            x, train_df[train_col].values,
            size=4,
            color=colors[i],
            alpha=0.6,
            legend_label=f"Training {train_col}"
        )

        # ideal function shown as a line on top
        p.line(
            x, ideal_df[ideal_col].values,
            line_width=2,
            color="black",
            legend_label=f"Ideal {ideal_col}"
        )

        # interactive legend to hide/show each series
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        fig_list.append(p)

    # 2x2 grid
    grid = gridplot(
        [[fig_list[0], fig_list[1]],
         [fig_list[2], fig_list[3]]],
        merge_tools=True
    )

    save(grid)
    print(f"  Plot saved: {filename}")

def plot_test_results(test_df, ideal_df, matched_functions, classification_df, filename="plots/test_results.html"):
    """
    This plot is from test data points showing which ideal function they got assigned to
    the mapped points and got the color of their ideal function
    The unmapped points show as gray X marks
    Args:
        test_df: original test data [x, y]
        ideal_df: ideal functions data
        matched_functions: dict {train_col: ideal_col}
        classification_df: classification results [x, y, delta_y, ideal_func]
        filename: where to the html file is saved
    """
    output_file(filename, title="Test Data Classification")

    p = figure(
        title="Test Data Mapped to Ideal Functions",
        x_axis_label="x",
        y_axis_label="y",
        width=900,
        height=600
    )

    x_ideal = ideal_df['x'].values
    colors = ['#7b2d8e', '#ff6f00', '#00897b', '#c62828']
    chosen_ideals = list(matched_functions.values())

    # the 4 chosen ideal functions as lines
    for i, ideal_col in enumerate(chosen_ideals):
        p.line(
            x_ideal,
            ideal_df[ideal_col].values,
            line_width=2,
            color=colors[i],
            legend_label=f"Ideal {ideal_col}",
            alpha=0.7
        )

    # plot mapped test points, colored by their assigned function
    if not classification_df.empty:
        for i, ideal_col in enumerate(chosen_ideals):
            filtered_data = classification_df[classification_df['ideal_func'] == ideal_col]
            if not filtered_data.empty:
                p.scatter(
                    filtered_data['x'].values,
                    filtered_data['y'].values,
                    size=8,
                    color=colors[i],
                    marker="circle",
                    legend_label=f"Test -> {ideal_col}",
                    alpha=0.8
                )

    # unmapped points
    if not classification_df.empty:
        matched_pairs = classification_df[['x', 'y']].apply(tuple, axis=1).tolist()
        all_indices = test_df[['x', 'y']].apply(tuple, axis=1).tolist()
        unmapped_mask = [t not in matched_pairs for t in all_indices]
        unmatched_pts = test_df[unmapped_mask]
    else:
        unmatched_pts = test_df

    if not unmatched_pts.empty:
        p.scatter(
            unmatched_pts['x'].values,
            unmatched_pts['y'].values,
            size=6,
            color="gray",
            marker="x",
            legend_label="Not mapped",
            alpha=0.5
        )

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    save(p)
    print(f"  Plot saved: {filename}")

def plot_deviations(classification_df, filename="plots/deviations.html"):
    """
    This plot shows the deviations (delta_y) for each mapped test point
    and shows how far each point was from its assigned ideal function
    and its grouped by color per ideal function
        Args:
        classification_df: classification results
        filename: where to save the html file
    """
    output_file(filename, title="Deviations from Mapped Test Points")

    if classification_df.empty:
        print("  No results to plot deviations.")
        return

    p = figure(
        title="Deviation (Delta Y) per Test Point",
        x_axis_label="x",
        y_axis_label="|Delta Y|",
        width=900,
        height=400
    )

    assigned_funcs = classification_df['ideal_func'].unique()
    colors = ['#7b2d8e', '#ff6f00', '#00897b', '#c62828']

    for i, func in enumerate(assigned_funcs):
        filtered_data = classification_df[classification_df['ideal_func'] == func]
        p.scatter(
            filtered_data['x'].values,
            filtered_data['delta_y'].values,
            size=7,
            color=colors[i],
            legend_label=f"{func}",
            alpha=0.7
        )

    p.legend.location = "top_right"
    p.legend.click_policy = "hide"

    save(p)
    print(f"  Plot saved: {filename}")

def create_all_plots(train_df, ideal_df, test_df, matched_functions, classification_df):
    """
    This generates all the visualization plots for the project
    and creates a plots/ folder that saves the 3 html files
    Args:
        train_df: training data
        ideal_df: ideal functions data
        test_df: test data
        matched_functions: dict {train_col: ideal_col}
        classification_df: test classification results
    """
    import os
    os.makedirs("plots", exist_ok=True)

    print("\n  Generating Plots")
    plot_training_vs_ideal(train_df, ideal_df, matched_functions)
    plot_test_results(test_df, ideal_df, matched_functions, classification_df)
    plot_deviations(classification_df)
    print("  All plots generated\n")
