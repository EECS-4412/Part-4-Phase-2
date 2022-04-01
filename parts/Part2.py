from clients.SqliteClient import SqlClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

#  age points salary


def part2():
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
    df.to_csv('./figures/215494925-215528797-215659501—T2Org.csv')

    cols = ['age', 'points', 'salary']
    df[cols] = standardize(df[cols])
    df.to_csv('./figures/215494925-215528797-215659501—T2Mod.csv')
    x = df[cols]

    estimators = [
        ("k_means_3", KMeans(n_clusters=3)),
        ("k_means_4", KMeans(n_clusters=4)),
        ("k_means_5", KMeans(n_clusters=5)),
    ]

    fignum = 1
    for name, est in estimators:
        print("Running: " + name + "...")

        label = est.fit_predict(x)

        labels = est.labels_
        # TODO give actual label for each k-means in k_means_5
        df['label'] = labels
        df.to_csv('./figures/215494925-215528797-215659501T2Class.csv')

        fig = plt.figure(fignum, figsize=(12, 9))
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=134)

        u_labels = np.unique(label)
        for i in u_labels:
            ax.scatter(x[label == i]['age'], x[label == i]['points'],
                       x[label == i]['salary'], label=i)
        ax.legend()
        ax.set_xlabel("Age")
        ax.set_ylabel("Points")
        ax.set_zlabel("Salary")
        ax.set_title(name)
        ax.dist = 12
        fignum = fignum + 1
        ax.figure.savefig('./figures/' + name + '.png')

        def animate(i):
            nonlocal ax
            ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=30, azim=130+i)
            for i in u_labels:
                ax.scatter(x[label == i]['age'], x[label == i]['points'],
                           x[label == i]['salary'], label=i)
            ax.legend()
            ax.set_xlabel("Age")
            ax.set_ylabel("Points")
            ax.set_zlabel("Salary")
            ax.set_title(name)
            ax.dist = 12

        ani = FuncAnimation(fig, animate, frames=20,
                            interval=200, repeat=False)
        ani.save('./figures/' + name + '.gif')

        print('\nSum of Squared Errors\t' + str(est.inertia_) +
              '\nCluster Centres:\n' + str(est.cluster_centers_) +
              '\n')


def standardize(col):
    return StandardScaler().fit_transform(col)


def discretize(row):
    def map_age(val):
        val = int(val)
        if val < 21:
            return 1
        if val < 25:
            return 2
        if val < 30:
            return 3
        if val < 35:
            return 4
        else:
            return 5

    def map_points(val):
        val = int(val)
        if val <= 3:
            return 1
        if val <= 9:
            return 2
        if val <= 12:
            return 3
        else:
            return 4

    def map_salary(val):
        if val < 1_000_000:
            return 1
        if val < 2_000_000:
            return 2
        if val < 3_000_000:
            return 3
        if val < 4_000_000:
            return 4
        if val < 5_000_000:
            return 5
        if val < 6_000_000:
            return 6
        if val < 10_000_000:
            return 7
        if val < 15_000_000:
            return 8
        if val < 25_000_000:
            return 9
        if val < 35_000_000:
            return 10
        else:
            return 11

    return [
        row[0],
        map_age(row.age),
        map_points(row.points),
        map_salary(row.salary),
    ]
