import pandas as pd


def consistent_col(df: pd.DataFrame,
                   column: str) -> pd.DataFrame:
    number_count = 0
    string_count = 0
    not_none_count = 0

    unique = set()

    for i in df[column]:

        if isinstance(i, int) or isinstance(i, float):
            number_count += 1
            unique.add(str(float(i)).lower())
            not_none_count += 1

        else:
            if i is not None:
                string_count += 1
                unique.add(str(i).lower())
                not_none_count += 1

    if number_count > string_count:
        dtype = "float"
    else:
        if len(unique) / not_none_count >= 0.5:
            dtype = "string"
        else:
            dtype = "category"

    if dtype == 'category':
        df = df.copy()[~df[column].isna()]
        df[column] = df[column].apply(lambda x: str(x) if not isinstance(x, str) else x).astype(dtype)

    elif dtype == "string":
        df[column] = df[column].apply(lambda x: "n/a" if x is None else str(x) if not isinstance(x, str) else x)

    else:
        df[column] = df[column].apply(lambda x: float(x) if isinstance(x, int) else x)
        df = df.copy()[~df[column].isna()]
        df = df[df[column].apply(lambda x: isinstance(x, float))]

    return df.reset_index(drop=True)


def convert_columns(df: pd.DataFrame):
    for column in list(df.columns):
        df = consistent_col(df,
                            column)

    return df


def get_codes(series: pd.Series):

    key_df = pd.concat([series, series.cat.codes], axis=1)
    key_dict = key_df.to_dict('split')
    key_dict.pop("index")

    full_dict = {}
    for data in key_dict["data"]:
        full_dict.update({data[0]: data[1]})

    return full_dict


if __name__ == '__main__':
    dataframe = pd.read_csv(filepath_or_buffer="D:/DataSets/Amazon/archive/kindle_reviews.csv")

    dataframe = dataframe.drop(columns=["Unnamed: 0",
                                        "helpful",
                                        "reviewTime"])

    conv_df = convert_columns(dataframe)

    column_list = list(conv_df.select_dtypes('category').columns)

    for column in column_list:
        print(len(get_codes(conv_df[column])))

    # conv_df = convert_categories(conv_df)
