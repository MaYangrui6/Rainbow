import torch
from math import log


class Config:
    def __init__(self, ):
        self.datafile = 'JOBqueries.workload'
        self.schemaFile = "schema.sql"
        self.database = 'indexselection_tpcds___10'
        self.user = 'postgres'
        self.password = "password"
        self.userName = self.user
        self.usegpu = True
        self.head_num = 10
        self.input_size = 47
        self.sql_size = 20
        self.hidden_size = 47 * 2  # self.input_size 的整数倍
        self.batch_size = 64
        self.ip = "127.0.0.1"
        self.port = 5432
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpudevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_weight = 0.00  # for au, 0:disable,0.01:enable
        self.max_column = 100
        self.max_alias_num = 40
        self.cost_test_for_debug = False
        self.max_hint_num = 20
        self.max_time_out = 120 * 1000
        self.threshold = log(3) / log(self.max_time_out)
        self.leading_length = 2
        self.try_hint_num = 3
        self.mem_size = 2000
        self.mcts_v = 1.1
        self.mcts_input_size = self.max_alias_num * self.max_alias_num + self.max_column
        self.searchFactor = 4
        self.U_factor = 0.0
        self.log_file = 'log_c3_h64_s4_t3.txt'
        self.latency_file = 'latency_record.txt'
        self.modelpath = 'model/'
        self.offset = 20