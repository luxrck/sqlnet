import json
import random
from collections import OrderedDict

from collections import OrderedDict
from urllib.parse import unquote

from sanic import Sanic, response

from jinja2 import Environment, FileSystemLoader, select_autoescape

from xsql import infer, xsql_init


env = Environment(
    loader=FileSystemLoader('tmpl'),
    autoescape=select_autoescape(['html', 'xml'])
)


app = Sanic()


cros_headers = {
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PATCH, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Origin, Content-Type, X-Auth-Token",
}


tables = OrderedDict(json.load(open("data/nl2sql/nl2sql.tables.json")))
table_ids = [line.strip() for line in open("data/nl2sql/nl2sql.tables.id").readlines()]


@app.route("/api/qa")
async def sql_qa(request):
    qargs = request.get_args(keep_blank_values=True)
    question = unquote(qargs.get("q", ""))
    table_idx = qargs.get("table_idx", "")

    if not question or not table_idx:
        return json({}, headers=cros_headers)

    try:
        table_idx = int(table_idx)
        table_id = table_ids[table_idx]
    except:
        table_id = table_idx
    
    q = {"question": question, "table_id": table_id}

    # import pdb; pdb.set_trace()
    try:
        results = infer([q], tables)
    except Exception as e:
        print(e.with_traceback())
        results = []
    # results = [{"sql": "test", "sels": ["col1", "col2", "col3"], "data": [["0.23", "2019年", "安徽省滁州市"]]}]
    if not results:
        results = {}
    else:
        results = results[0]
    return response.json(results, headers=cros_headers)


@app.route("/table/<table_idx>")
async def sql_table(request, table_idx):
    try:
        table_idx = int(table_idx)
        table_idx %= len(table_ids)
        table_id = table_ids[table_idx]
    except:
        table_id = table_idx
        table_idx = table_ids.index(table_id)

    table = tables[table_id]

    headers = table["header"]
    rows = table["rows"]
    examples = [ex["question"] for ex in table["example"]]

    template = env.get_template("index")
    print(examples)

    return response.html(template.render(headers=headers, rows=rows, table_idx=table_idx, examples=examples))


# reverse proxy router: /nl2sql/ -> [self.ip:port]
@app.route("/")
async def index(request):
    table_idx = random.randint(0, 3000)
    return response.redirect(f"/nl2sql/table/{table_idx}")


if __name__ == "__main__":
    # load_tables("train", tables)
    # load_tables("dev", tables)
    
    xsql_init()
    
    app.run(host="0.0.0.0", port=9600)
