# GPL License
# Copyright (C) 2022, UESTC
# All Rights Reserved 
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference: 
from configs.configs import TaskDispatcher
from UDL.AutoDL.trainer import main
from common.builder import build_model, derainSession

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='derain', mode='entrypoint', arch='Uformer_KAv13')
    cfg.eval = True
    cfg.datasets = {"val": 'Rain100L'}
    print(TaskDispatcher._task.keys())
    main(cfg, build_model, derainSession)
