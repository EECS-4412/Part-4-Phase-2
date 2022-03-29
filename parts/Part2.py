from clients.SqliteClient import SqlClient
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

def part2():
    sql_client = SqlClient()
    query = """
        Get some data.
    """
    df = pd.read_sql_query(query, sql_client.conn)
    KMeans(n_cluster = 3, random_state=0, metric='euclidean').fit(df)
    KMeans(n_cluster = 4, random_state=0, metric='euclidean').fit(df)
    KMeans(n_cluster = 5, random_state=0, metric='euclidean').fit(df)

    # 🎉
