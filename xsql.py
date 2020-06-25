import sys
import json
import traceback

import sqlite3

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

torch.cuda.set_device(0)



def infer(q, tables):
    global app, db, bert_tokenizer
    app.model.eval()
    sqls = []
    with torch.no_grad():
        inputs = generate_inputs(bert_tokenizer, q, tables)
        y_predict = app.model(inputs)
        sqli = app.model.predict(y_predict, h_length=inputs[-2])
        sqls = app.model.sql(sqli, q, tables)

        for b,sql in enumerate(sqls):
            resp = db.execute(sql["sql"]).fetchall()
            sql["data"] = resp
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

    app = app.fastforward(to="last")   \
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
        loss, labels = model.loss(y_predict, gw_val=gw_val)
        # sql_i = model.predict(y_predict, labels, q_length=inputs[3], h_length=inputs[4])
        detail = model.detail(y_predict, labels, col_sel=col_sel, col_whr=col_whr, col_match=col_match, q_length=inputs[3], records=t, table=train_table)

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
        sql_i = model.predict(y_predict, h_length=inputs[4])
        import pdb; pdb.set_trace()
        # sql = model.sql(sql_i, e.batch, train_table)

        return sql_i, gold


    y,g = app.fastforward(to="last")   \
       .set_optimizer(optim.AdamW, lr=1e-6) \
       .to("auto")  \
       .save_every(iters=-1)    \
       .build() \
       .eval(train_iter)
    #    .run(train_iter, max_iters=10000, train=False)
    #    .eval(train_iter)
    
    lx = 0
    cnt = 0
    for i in range(len(y)):
        b = len(y[i])
        for bi in range(b):
            y_ = y[i][bi]
            g_ = g[i][bi]
            sn, sca, wn, wconn, wco, wv, wvm = y_
            gsn, gsca, gwn, gwconn, gwco, gwv, gwvm = g_
            if sn == gsn and sca == gsca and wn == gwn and wconn == gwconn and wco == gwco and wvm == gwvm:
                lx += 1
            cnt += 1
    print(lx / cnt)
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