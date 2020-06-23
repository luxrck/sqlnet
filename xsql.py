import sys
import json
import traceback

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


if __name__ == "__main__":
    path_wikisql = "data/nl2sql"
    bert_type = "hfl/chinese-bert-wwm-ext"
    dropout_p = 0.1
    bs = 16
    device = "cuda"

    train_data, train_table, dev_data, dev_table, train_iter, dev_iter = get_data(path_wikisql, bs=bs)

    print("Load bert....")
    bert = BertModel.from_pretrained(bert_type, output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_type)
    print("Load bert....[OK]")
    model = SQLNet(bert, dropout_p=dropout_p).to(device)

    app = Trainer(App(model=model))
    app.extend(Checkpoint())


    @app.on("train")
    def sql_train(e):
        e.model.zero_grad()
        inputs, labels, col_sel, col_whr, col_match, gold = None, None, None, None, None, None
        try:
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_inputs(bert_tokenizer, e.batch, train_table, device=e.device)
        except Exception as e:
            print(e)
            return {
                "loss": -1,
                }
            
        y_predict = model(inputs, labels)
        loss, labels = model.loss(y_predict, labels)
        sql_i = model.predict(y_predict, labels, q_length=inputs[3], h_length=inputs[4])
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
        inputs, labels, col_sel, col_whr, col_match, gold = None, None, None, None, None, None
        try:
            inputs, labels, (col_sel, col_whr, col_match), gold = generate_inputs(bert_tokenizer, e.batch, dev_table, device=e.device)
        except Exception as e:
            # print(e)
            traceback.print_tb()
            sys.exit()

        y_predict = model(inputs, labels)
        # loss, labels = model.loss(y_predict, labels)
        sql_i = model.predict(y_predict, q_length=inputs[3], h_length=inputs[4])
        
        return sql_i, gold


    y, g = app.fastforward(to="last")   \
       .set_optimizer(optim.AdamW, lr=1e-6) \
       .to("auto")  \
       .save_every(iters=-1)  \
       .run(train_iter, max_iters=10000, train=False)   \
       .eval(dev_iter)
    
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