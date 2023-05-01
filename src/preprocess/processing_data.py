import pandas as pd
import preprocess.painter as pt

# from sklearn.preprocessing import OneHotEncoder

def get_data(file):

    df = pd.read_csv(f'../dat/{file}')
    return df

def check_quality(df):
    print("Missing values in data set : ")
    print(df.isna().sum())
    print("\n")
    return 1

def clean_data(df):

    return df

def describe_data_scatter(df):

    for col in df.columns:
        pt.make_scatter(df, 'cost', col)
        pt.print_distribution(df, col)


    return 1

def make_correlation(df):
    pt.print_correlation_matrix(df)
    return 1

def apply_one_hot_encoder(df, to_encode):

    enc = OneHotEncoder()
    enc_data = pd.DataFrame(enc.fit_transform(
        df[to_encode]).toarray())
    df = df.join(enc_data)
    df.drop(to_encode, axis=1, inplace=True)

    return df

def discard_columns(df, columns):

    return df.drop(columns, axis = 1)