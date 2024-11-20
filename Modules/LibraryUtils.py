import pandas as pd
from itertools import product

def get_bb_dict_from_smiles(pool_id: str, smiles: list) -> dict:
    """
    Creates a dictionary from a list of SMILES strings with unique keys.

    Args:
        pool_id (str): A prefix for the keys in the dictionary.
        smiles (list): A list of SMILES strings.

    Returns:
        dict: A dictionary with keys formatted as "<pool_id><index>" (zero-padded) 
              and values as SMILES strings.
    """
    return {f"{pool_id}{i:03}": value for i, value in enumerate(smiles, start=1)}

def get_bb_combos(bb_dicts: list):
    """
    Generate all combinations of keys from a list of building block dictionaries.

    Args:
        bb_dicts (list): List of dictionaries. The dictionaries need to be keyed by bb ID.

    Returns:
        list: List of tuples representing all combinations of keys (bb IDs).
    """
    if not bb_dicts:
        return []  # Return an empty list if input is empty

    # Extract the keys from all dictionaries
    all_ids = [list(bb.keys()) for bb in bb_dicts]

    # Generate the Cartesian product of all keys
    bb_combos = list(product(*all_ids))

    return bb_combos

def concatenate_tuples(tuples_list: list[tuple]) -> list[str]:
    """
    Concatenates each tuple in a list into a single string with "_" as the delimiter.

    Args:
        tuples_list (list[tuple]): A list of tuples to be concatenated.

    Returns:
        list[str]: A list of concatenated strings.
    """
    return ["_".join(map(str, tpl)) for tpl in tuples_list]



def tuples_to_dataframe(tuples_list: list[tuple], column_names: list[str] = None) -> pd.DataFrame:
    """
    Converts a list of tuples into a pandas DataFrame.

    Args:
        tuples_list (list[tuple]): A list of tuples to be converted.
        column_names (list[str], optional): Column names for the DataFrame. 
                                            If not provided, defaults to numbered columns.

    Returns:
        pd.DataFrame: A DataFrame representing the tuples.
    """
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(tuples_list)

    # Assign column names if provided
    if column_names:
        if len(column_names) != df.shape[1]:
            raise ValueError("Number of column names must match the number of tuple elements.")
        df.columns = column_names

    return df

def map_column_with_dict(df: pd.DataFrame, existing_col: str, new_col: str, mapping_dict: dict ) -> pd.DataFrame:
    """
    Maps values in an existing DataFrame column to a new column using a dictionary.

    Args:
        df (pd.DataFrame): The input DataFrame.
        existing_col (str): The name of the column to map values from.
        mapping_dict (dict): A dictionary with keys as existing column values and values as the mapped values.
        new_col (str): The name of the new column to be created.

    Returns:
        pd.DataFrame: The updated DataFrame with the new column.
    """
    df[new_col] = df[existing_col].map(mapping_dict)
    return df

def merge_columns(df: pd.DataFrame, columns: list[str], merge_col: str, delimiter: str = ".") -> pd.DataFrame:
    """
    Merges a list of columns in a DataFrame into a new column using a specified delimiter.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to merge.
        new_col (str): Name of the new column to create.
        delimiter (str, optional): The delimiter to use for merging. Defaults to ".".

    Returns:
        pd.DataFrame: The updated DataFrame with the new column.
    """
    df[merge_col] = df[columns].astype(str).agg(delimiter.join, axis=1)
    return df 

def library_df_from_bb_pools(poolA, poolB, poolC):
    bbA_dict = get_bb_dict_from_smiles("A", poolA)
    bbB_dict = get_bb_dict_from_smiles("B", poolB)
    bbC_dict = get_bb_dict_from_smiles("C", poolC)

    bb_combo_tuples = get_bb_combos([bbA_dict, bbB_dict, bbC_dict])

    library_IDs = concatenate_tuples(bb_combo_tuples)

    library_df = tuples_to_dataframe(bb_combo_tuples, ['bbA_ID', 'bbB_ID', 'bbC_ID'])

    library_df.index = library_IDs

    map_column_with_dict(library_df, 'bbA_ID', 'bbA_SMILES', bbA_dict)
    map_column_with_dict(library_df, 'bbB_ID', 'bbB_SMILES', bbB_dict)
    map_column_with_dict(library_df, 'bbC_ID', 'bbC_SMILES', bbC_dict)

    merge_columns(library_df, ['bbA_SMILES', 'bbB_SMILES', 'bbC_SMILES'], 'union_SMILES')

    return library_df