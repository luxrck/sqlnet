import sys
import json
import traceback

import sqlite3
from fuzzywuzzy import fuzz

import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from transformers import BertModel, BertTokenizer

from model import SQLNet

from dataset import *
from train import *

#torch.cuda.set_device(0)


def find_val(rows, c_idx, val):
    top_score = 0
    y_val = val
    for i, row in enumerate(rows):
        g_val = str(row[c_idx])
        score = fuzz.ratio(val, g_val)
        if score > top_score:
            top_score = score
            y_val = g_val
    return y_val

def infer(q, tables):
    global app, db, bert_tokenizer
    app.model.eval()
    sqls = []
    with torch.no_grad():
        inputs = generate_inputs(bert_tokenizer, q, tables)
        y_predict = app.model(inputs)
        sqli = app.model.predict(y_predict, h_length=inputs[-2], records=q, tables=tables)
        print("sqli:", sqli)
        sqls = app.model.sql(sqli, q, tables)

        for b,sql in enumerate(sqls):
            table_id = q[b]["table_id"]
            header_types = tables[table_id]["types"]
            select_all = f'SELECT * FROM "{table_id}";'
            # print(sql)
            resp = db.execute(sql["sql"]).fetchall()
            if not resp:
                rows = db.execute(select_all).fetchall()
                conds = sql["conds"]
                conds_s = []
                for cond in conds:
                    col_, op_, val_, c_idx = cond
                    col_type_ = header_types[c_idx]
                    if cond[1] in ("==", "=", "like") and col_type_ == "text":
                        # cond[3]: header_idx
                        # cond[2]: w_val
                        # cond[1]: w_op
                        # cond[0]: w_col
                        val_ = find_val(rows, c_idx, val_)
                        cond[2] = val_
                    cond = f'("{col_}" {op_} "{val_}")'
                    conds_s.append(cond)
                sql_matched = sql["tmpl"](sql["sel"], sql["table_id"], sql["conn"], conds_s)
                sql["sql"] = sql_matched
                resp = db.execute(sql_matched).fetchall()
            sql["data"] = resp
            print(f"sql[{b}]:", sql)
    return sqls


def xsql_init():
    bs = 16
    path_wikisql = "data/nl2sql"
    bert_type = "hfl/chinese-bert-wwm-ext"
    dropout_p = 0.1
    device = "cuda"
    
    global app, db, bert_tokenizer
    # train_data, train_table, dev_data, dev_table, train_iter, dev_iter = get_data(path_wikisql, bs=bs)

    db = sqlite3.connect("data/nl2sql/nl2sql.sqlite3")

    print("Load bert....")
    bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
    print("Load bert....[OK]")
    model = SQLNet(bert, dropout_p=dropout_p).to(device)

    app = App(model=model)
    app.extend(Checkpoint())

    app = app.fastforward(to=".+\.37\.pt")   \
       .set_optimizer(optim.AdamW, lr=1e-6) \
       .to("auto")  \
       .save_every(iters=-1)    \
       .build() \
       .eval()


if __name__ == "__main__":
    bs = 16
    path_wikisql = "data/nl2sql"
    bert_type = "hfl/chinese-bert-wwm-ext"
    dropout_p = 0.1
    device = "cuda"
    
    train_data, train_table, dev_data, dev_table, train_iter, dev_iter = get_data(path_wikisql, bs=bs)

    db = sqlite3.connect("data/nl2sql/nl2sql.sqlite3")

    print("Load bert....")
    bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
    print("Load bert....[OK]")
    model = SQLNet(bert, dropout_p=dropout_p).to(device)

    app = App(model=model)
    app.extend(Checkpoint())


    @app.on("train")
    def sql_train(e):
        e.model.zero_grad()
        inputs, labels, col_sel, col_whr, col_match, gold = None, None, None, None, None, None
        try:
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_samples(bert_tokenizer, e.batch, train_table, device=e.device)
        except Exception as e:
            print(e)
            return {
                "loss": -1,
                }
        
        gw_val = labels[-2]
        y_predict = model(inputs, gw_val=gw_val)
        loss, labels = model.loss(y_predict, labels)
        # sql_i = model.predict(y_predict, labels, q_length=inputs[3], h_length=inputs[4])
        detail = model.detail(y_predict, labels, col_sel=col_sel, col_whr=col_whr, col_match=col_match, q_length=inputs[3], records=e.batch, tables=train_table)

        loss.backward()

        s_num, s_col, s_agg, w_num, w_conn_op, w_col, w_op, w_val_s, w_val_e = detail
        w_val = w_val_s[0]
        w_val_all = w_val_s[1]
        w_val_match = w_val_e

        return {
            "loss": loss.item(),
            "info": {
                "s_num": s_num,
                "s_col": s_col,
                "s_agg": s_agg,
                "w_num": w_num,
                "w_con": w_conn_op,
                "w_col": w_col,
                "w_op": w_op,
                "w_val": w_val,
                "w_val_all": w_val_all,
                "w_val_match": w_val_match,
            }
        }

        # return loss.item()


    @app.on("evaluate")
    def sql_eval(e):
        model = e.model
        inputs, labels, col_sel, col_whr, col_match, gold = None, None, None, None, None, None
        try:
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_samples(bert_tokenizer, e.batch, train_table, device=e.device)
        except Exception as e:
            print(e)
            # traceback.print_tb()
            sys.exit()

        gw_val = labels[-2]
        gw_val = None
        y_predict = model(inputs, gw_val=gw_val)
        # loss, labels = model.loss(y_predict, labels)
        sql_i = model.predict(y_predict, h_length=inputs[4], records=e.batch, tables=train_table)
        # import pdb; pdb.set_trace()
        # sql = model.sql(sql_i, e.batch, train_table)

        # b = len(sql_i)
        # for i in range(b):
        #     y_ = sql_i[i]
        #     g_ = gold[i]
        #     sn, sca, wn, wconn, wco, wv, wvm = y_
        #     gsn, gsca, gwn, gwconn, gwco, gwv, gwvm = g_
        #     if gsca.issubset(sca) and wconn == gwconn and wco == gwco and wvm == gwvm:
        #         continue
        #     else:
        #         e.batch[i]["y"] = y_
        #         e.batch[i]["g"] = g_
        #         out = {
        #             "q": e.batch[i]["question"],
        #             "t": e.batch[i]["table_id"],
        #             "y": [sca, wco, wv, wvm],
        #             "g": [gsca, gwco, gwv, gwvm],
        #             "h": dev_table[e.batch[i]["table_id"]]["header"],
        #         }
        #         print(out)

        return sql_i, gold


    y,g = app.fastforward(to="last")   \
       .set_optimizer(optim.AdamW, lr=1e-6) \
       .to("auto")  \
       .save_every(epochs=-1)    \
       .build() \
       .eval(train_iter)
    #    .run(train_iter, max_iters=10000, train=True)
    
    lsn = 0
    lsca = 0
    lwn = 0
    lwconn = 0
    lwco = 0
    lwvm = 0

    lx = 0
    cnt = 0
    for i in range(len(y)):
        b = len(y[i])
        for bi in range(b):
            y_ = y[i][bi]
            g_ = g[i][bi]
            sn, sca, wn, wconn, wco, wv, wvm = y_
            gsn, gsca, gwn, gwconn, gwco, gwv, gwvm = g_
            if gsca == sca and wconn == gwconn and wco == gwco and wvm == gwvm:
                lx += 1
            if sn == gsn:
                lsn += 1
            if sca == gsca:
                lsca += 1
            if wn == gwn:
                lwn += 1
            if wconn == gwconn:
                lwconn += 1
            if wco == gwco:
                lwco += 1
            if wvm == gwvm:
                lwvm += 1
            cnt += 1
    print(lx / cnt, lsn / cnt, lsca / cnt, lwn / cnt, lwconn / cnt, lwco / cnt, lwvm / cnt)
    # app.model.eval()
    # with torch.no_grad():
    #     table_id = None
    #     while True:
    #         text = input("> ")
    #         if 'use' in text:
    #             table_id = text.split(' ')[1]
    #         else:
    #             q = {"question": text, "table_id": table_id}
    #             import pdb; pdb.set_trace()
    #             try:
    #                 print(infer([q], train_table))
    #             except Exception as e:
    #                 print(e)

    # 0.87451564828614 0.9943368107302534 0.9630402384500745 0.9797317436661699 0.9937406855439642 0.9374068554396423 0.9126676602086438