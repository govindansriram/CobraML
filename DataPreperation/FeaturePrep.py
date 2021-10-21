import pandas as pd

"""
Type list categorical, int, float, string
"""


def consistent_col(df: pd.DataFrame,
                   column: str,
                   dtype: str) -> pd.DataFrame:
    dtype = dtype.lower()

    if dtype == 'category':
        df = df.copy()[~df[column].isna()]
        df[column] = df[column].apply(lambda x: str(x) if not isinstance(x, str) else x).astype(dtype)

    elif dtype == "string":
        df[column] = df[column].apply(lambda x: "n/a" if x is None else str(x) if not isinstance(x, str) else x)

    elif dtype == "int":
        df = df[df[column].apply(lambda x: isinstance(x, int))]

    else:
        df[column] = df[column].apply(lambda x: float(x) if isinstance(x, int) else x)
        df = df.copy()[~df[column].isna()]
        df = df[df[column].apply(lambda x: isinstance(x, float))]

    return df


def convert_columns(df: pd.DataFrame,
                    type_dict: dict[str, str],
                    drop_column_list: list[str]):
    df = df.drop(columns=drop_column_list)

    for column in list(df.columns):
        df = consistent_col(df,
                            column,
                            type_dict[column])

    return df


if __name__ == '__main__':
    # dataframe = pd.read_csv(filepath_or_buffer="D:/DataSets/Amazon/archive/kindle_reviews.csv")

    dataframe = pd.DataFrame(data={"apple": ["Tim", 1, 2, None, 4, 5, ],
                                   "orange": ["water", "water", None, "right", 10, 20],
                                   "banana": ["hey you!", None, "how you been?", "awesome!", "see you soon.", 10]})

    print(convert_columns(dataframe,
                          {"apple": "float",
                           "orange": "category",
                           "banana": "string"},
                          []))
