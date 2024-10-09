import dgl
import torch
import psycopg2

if __name__ == "__main__":
	print(dgl.__version__)
	print(torch.__version__)
	psycopg2.connect(dbname='indexselection_tpcds___1', user='postgres', password='password', host='127.0.0.1', port='5432')
