""" Script with functions to make visualizations"""
import pandas as pd
import plotly.express as px


def make_scatter(df: pd.DataFrame, x: str, y: str) -> int:
    """
    Make scatter plot from a dataframe and the columns x and y
    :param df: pandas dataframe with data
    :param x: column to plot over xaxis
    :param y: column to plot over yaxis
    :return: 1 if success
    """
    fig = px.scatter(x=df[x], y=df[y])

    fig.update_layout(
        title=f'{x} Vs {y}',
        xaxis_title=x,
        yaxis_title=y,
    )
    fig.write_html(f'../res/1_{x}_{y}.html')
    return 1

def print_distribution(df: pd.DataFrame, col: str) -> int:
    """
    Plot a distribution of a feature which is part of a dataframe
    :param df: dataframe with the data
    :param col: column to plot the distribution
    :return: 1 if success
    """
    fig = px.histogram(df, x=col, y=col,
                       marginal="box")
    fig.write_html(f'../res/2_distribution_{col}.html')
    return 1

def print_correlation_matrix(df: pd.DataFrame) -> int:
    """
    Print the heathmap with the correlation values
    :param df: dataframe with all the data
    :return: 1 if success
    """
    fig = px.imshow(df.corr().round(2), text_auto=True)
    fig.write_html('../res/3_corr_matrix.html')

    return 1

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean data from a dataframe
    :param df: dataframe to be cleaned
    :return: Cleaned dataframe
    """
    return df
