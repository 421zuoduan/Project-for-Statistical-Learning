# The code framework is derived from other people's code, see the README for details
from configs.configs import TaskDispatcher
from UDL.AutoDL.trainer import main
from common.builder import build_model, derainSession

if __name__ == '__main__':
    cfg = TaskDispatcher.new(task='derain', mode='entrypoint', arch='Uformer_KAv13')
    print(TaskDispatcher._task.keys())
    main(cfg, build_model, derainSession)
