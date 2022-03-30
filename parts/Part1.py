from clients.SqliteClient import SqlClient
import pandas as pd
from collections import Counter
from efficient_apriori import apriori
import json


def part1():
    sql_client = SqlClient()
    query = """
    SELECT
        pa.DISPLAY_FIRST_LAST as name, pa.HEIGHT as height, pa.PTS as points, MAX(ps.value) as salary
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
    # df[['name', 'height', 'weight', 'salary']] = df.apply(discretize, axis=1, result_type='expand')
    df[['name', 'height', 'points', 'salary']] = df.apply(
        discretize, axis=1, result_type='expand')
    print('Head of our data:')
    print(df.head(20))
    freq = find_freq(df)
    _, top_5 = association(df, freq)
    for rule in top_5:
        count_1 = counts(rule, df)
        support_1 = rule.support
        support_2 = support(count_1)
        conf_1 = rule.confidence
        conf_2 = confidence(count_1)
        print(support_1, support_2)
        print(conf_1, conf_2)


def association(df, freq):
    top_10 = [set(x[0]) for x in freq]
    tuples = [tuple(row[1:]) for row in df.values.tolist()]
    itemsets, rules = apriori(tuples, min_support=0.01, min_confidence=0.01)
    association_results = list(rules)
    # Print out every rule with 2 items on the left hand side,
    # 1 item on the right hand side, sorted by lift
    rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(
        rule.rhs) == 1 and set([*rule.lhs, *rule.rhs]) in top_10, rules)
    print('Associations:')
    rules_rhs = sorted(rules_rhs, key=lambda rule: -rule.confidence)
    print("5 most high conf")
    for rule in rules_rhs[:5]:
        print(rule)  # Prints the rule and its confidence, support, lift, ...
    print("5 most low conf")
    for rule in rules_rhs[-5:]:
        print(rule)  # Prints the rule and its confidence, support, lift, ...
    top_5 = rules_rhs[:5]
    return association_results, top_5


def counts(rule, df):
    lhs = list(rule.lhs)
    rhs = list(rule.rhs)
    supports = {
        '11': 0,
        '01': 0,
        '10': 0,
        '00': 0
    }
    print(lhs)
    print(rhs)
    for item in df.iterrows():
        item = list(item[1])
        if all([x in item for x in lhs]) and all([x in item for x in rhs]):
            supports['11'] += 1
        elif all([x in item for x in lhs]) and not all([x in item for x in rhs]):
            supports['10'] += 1
        elif not all([x in item for x in lhs]) and all([x in item for x in rhs]):
            supports['01'] += 1
        if not all([x in item for x in lhs]) and not all([x in item for x in rhs]):
            supports['00'] += 1

    #supports = {key: value/len(df.index) for key, value in supports.items()}
    return supports


def support(val):
    total = sum([x for x in val.values()])
    return val['11']/total


def confidence(val):
    total = sum([x for x in val.values()])
    return (val['11'])/(val['11'] + val['10'])


def lift(rule):
    return rule.lift


def find_freq(df):
    ret = Counter([tuple(x[1:]) for x in df.to_numpy()]).most_common(10)
    print('Frequencies:')
    print(*ret, sep='\n')
    return ret


def discretize(row):
    def map_height(val):
        val = int(val)
        if val < 70:
            return 'h-1'
        if val < 75:
            return 'h-2'
        if val < 80:
            return 'h-3'
        if val < 85:
            return 'h-4'
        else:
            return 'h-5'

    def map_points(val):
        val = int(val)
        if val <= 3:
            return 'p-1'
        if val <= 9:
            return 'p-2'
        if val <= 12:
            return 'p-3'
        else:
            return 'p-4'

    # def map_weight(val):
    #     val = int(val)
    #     if val < 150:
    #         return 'w-1'
    #     if val < 180:
    #         return 'w-2'
    #     if val < 210:
    #         return 'w-3'
    #     if val < 240:
    #         return 'w-4'
    #     if val < 270:
    #         return 'w-5'
    #     else: # 300 +
    #         return 'w-6'

    def map_salary(val):
        if val < 1_000_000:
            return 's-1'
        if val < 2_000_000:
            return 's-2'
        if val < 3_000_000:
            return 's-3'
        if val < 4_000_000:
            return 's-4'
        if val < 5_000_000:
            return 's-5'
        if val < 6_000_000:
            return 's-6'
        if val < 10_000_000:
            return 's-7'
        if val < 15_000_000:
            return 's-8'
        if val < 25_000_000:
            return 's-9'
        if val < 35_000_000:
            return 's-10'
        if val < 50_000_000:
            return 's-11'

    return [
        row[0],
        map_height(row.height),
        # map_weight(row.weight),
        map_points(row.points),
        map_salary(row.salary),
    ]
