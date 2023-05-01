import plotly.express as px


def make_scatter(df, x, y):

    fig = px.scatter(x=df[x], y=df[y])

    fig.update_layout(
        title=f'{x} Vs {y}',
        xaxis_title=x,
        yaxis_title=y,
    )
    fig.write_html(f'../res/1_{x}_{y}.html')
    return 1

def print_distribution(df, col):
    fig = px.histogram(df, x=col, y=col,
                       marginal="box")
    fig.write_html(f'../res/2_distribution_{col}.html')
    return 1

def print_correlation_matrix(df):

    fig = px.imshow(df.corr().round(2), text_auto=True)
    fig.write_html('../res/3_corr_matrix.html')

    return 1

def clean_data(df):

    return df