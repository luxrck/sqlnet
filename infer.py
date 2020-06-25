from model import SQLNet


def generate_sql(table, q, sqli, headers):
    COND = {0: ">", 1: "<", 2: "==", 3: "!="}
    CONN = {0: "AND", 1: "OR"}
    AGG = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}


    sql = []
    for b,y in enumerate(sqli):
        sn, sca, wn, wconn, wco, wv, wvm = y
        conn = CONN[wconn]
        sql_tmpl = f"SELECT {','.join([f'{agg}({col})' for col,agg in sca])} "   \
                f"FROM {table} "     \
                f"WHERE {f' {conn} '.join(f'(\'{}\' {} \'{}\')')}"