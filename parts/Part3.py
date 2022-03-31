from clients.SqliteClient import SqlClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def part3():
    sql_client = SqlClient()
    query = """
    SELECT
         pa.DISPLAY_FIRST_LAST as name, (strftime('%Y', 'now') - strftime('%Y', pa.BIRTHDATE)) - (strftime('%m-%d', 'now') < strftime('%m-%d', pa.BIRTHDATE)) AS 'age', pa.PTS as points, MAX(ps.value) as salary
    FROM
        PLAYER_ATTRIBUTES pa
    LEFT JOIN
        (
            SELECT *
            FROM Player_Salary ps
        ) ps
    ON
        pa.DISPLAY_FIRST_LAST = ps.namePlayer
    WHERE slugSeason = '2020-21'
    GROUP BY
        pa.DISPLAY_FIRST_LAST
    """
    df = pd.read_sql_query(query, sql_client.conn)
    cols = ['age', 'points', 'salary']
    df[cols] = standardize(df[cols])
    x = df[cols]

    estimators = [
        ("k_means_3", KMeans(n_clusters=3)),
        ("k_means_4", KMeans(n_clusters=4)),
        ("k_means_`5`", KMeans(n_clusters=5)),
    ]

    fignum = 1
    for name, est in estimators:
        print("Running: " + name + "...")

        label = est.fit_predict(x)

        labels = est.labels_
        # TODO give actual label for each k-means in k_means_5
        df['label'] = labels

        u_labels = np.unique(label)
        inputs = []
        outputs = []
        for index, row in df.iterrows():
            inputs.append((row['age'], row['points'], row['salary']))
            outputs.append(row['label'])


        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.25, random_state=0)
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        y_pred_train = gnb.fit(X_train, y_train).predict(X_train)

        scores = cross_val_score(gnb, inputs, outputs, cv=5) 

        print('%0.3f accuracy with a standard deviation of %0.3f' % (scores.mean(), scores.std()))

        print(f'Number of mislabeled points out of a total {len(X_train)} points (on training) : {(y_train != y_pred_train).sum()}')
        print(f'Number of mislabeled points out of a total {len(X_test)} points (on testing) : {(y_test != y_pred).sum()}')

def standardize(col):
    return StandardScaler().fit_transform(col)
