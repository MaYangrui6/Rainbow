from sql import Sql,_global
import networkx as nx
import matplotlib.pyplot as plt
from core import database, Sql, Plan, load
import os
print(os.getcwd())
if __name__ == "__main__":
    # print(1)
    database.setup(dbname='indexselection_tpcds___1', user='postgres', password='password', host='127.0.0.1', port='5432', cache=False)

    # 创建SQL实例
    sql_query = """select  
   w_state
  ,i_item_id
  ,sum(case when (cast(d_date as date) < cast ('1998-03-21' as date)) 
 		then cs_sales_price - coalesce(cr_refunded_cash,0) else 0 end) as sales_before
  ,sum(case when (cast(d_date as date) >= cast ('1998-03-21' as date)) 
 		then cs_sales_price - coalesce(cr_refunded_cash,0) else 0 end) as sales_after
 from
   catalog_sales left outer join catalog_returns on
       (cs_order_number = cr_order_number 
        and cs_item_sk = cr_item_sk)
  ,warehouse 
  ,item
  ,date_dim
 where
     i_current_price between 0.99 and 1.49
 and i_item_sk          = cs_item_sk
 and cs_warehouse_sk    = w_warehouse_sk 
 and cs_sold_date_sk    = d_date_sk
 and d_date between (cast ('1998-03-21' as date) - interval '30 days')
                and (cast ('1998-03-21' as date) + interval '30 days') 
 group by
    w_state,i_item_id
 order by w_state,i_item_id
limit 100;

-- end query 26 in stream 0 using template query40.tpl

"""
    sql_instance = Sql(sql_query)

    # 构建异构图
    g, data_dict, node_indexes,edge_list = sql_instance.to_hetero_graph_dgl()

    print('edge_list:',edge_list)
    # 打印图特征表示
    print("Graph Nodes:", g.nodes())
    print("Graph Edges:", g.edges())
    print("Node Features:", g.ndata)
    print("Edge Features:", g.edata)

    # 打印数据字典和节点索引
    print("Data Dictionary:", data_dict)
    print("Node Indexes:", node_indexes)

    nx_graph = g.to_networkx()

    # 使用 spring 布局绘制图形
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.savefig('./graph_output.png', format='png', dpi=300)
    plt.show()
    print('finish')
    print(nx_graph.nodes(data=True))