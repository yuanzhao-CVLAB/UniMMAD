import datetime
import os
import shutil
from tqdm import tqdm
from itertools import chain
from accelerate.logging import get_logger
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List

from myutils.loss_log import LossLog
from data.mmad_datasets import get_dataset,WeightedClassSampler
from mymodels.unidomain_ts_multimodal import UniMMAD
from eval import evalute
logger = get_logger(__name__)



def print_param_sum(parameters, model):
    total_params = 0
    # 遍历优化器的所有参数组
    for param in parameters:
        # 累加参数的总元素数
        if param.dtype in [torch.float32, torch.float64, torch.float16]:
            total_params += param.numel()
            param.requires_grad = True
        else:
            print("*"*10,"not supported dtype ",param.dtype,"*"*10)



    # 计算总参数量（以兆为单位）
    return  total_params  / 1e6
def denormlize_img(img):
    img = 255 * img.cpu().mul(torch.tensor([0.485, 0.456, 0.406])[:, None, None]).add_(
        torch.tensor([0.229, 0.224, 0.225])[:, None, None])
    img = F.interpolate(img, size=256, mode="bilinear")
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img_pil = img.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    return img_pil
def train_one_epoch(args,accelerator,epoch,iters,trainer,optimizer,avg_loss_log,training_dataset_loader):
    tbar = tqdm(training_dataset_loader, disable=not accelerator.is_local_main_process,ncols=150)
    log = ""
    for i, sample in enumerate(tbar):
        iters+=i
        loss,loss_str = trainer.train_step(sample,output_dir= args["output_dir"],path = sample["path"],epoch = epoch)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        avg_loss_log.update(loss_str)
        log = 'Epoch:%d, %s lr: %.6f' % (epoch, avg_loss_log, optimizer.param_groups[0]['lr'],)
        tbar.set_description(log)
    logger.info(("*"*10)+log)
    return iters
def filter_params(model,model_name,args,model_parameters):
    filtered_params = {}
    total_params = 0
    for name, param in model.named_parameters():
        filtered_params[name] = param
    if len(filtered_params) > 0:
        total_params += print_param_sum(filtered_params.values(), model_name)
        model_parameters.append(list(filtered_params.values()))
    return total_params

def load_parameters(args,names, models):

    model_parameters = []
    assert len(names) == len(models)
    for model_name, model in zip(names, models):
        total_params = 0
        if isinstance(model, List):
            for p in model:
                total_params+= filter_params(p, model_name, args, model_parameters)
        else:
            total_params += filter_params(model, model_name, args, model_parameters)
        logger.info(f'{model_name} Total parameters: {total_params:.2f}M')
    return model_parameters

def train(args,accelerator):

    training_dataset,_ = get_dataset(args)
    sampler = WeightedClassSampler(training_dataset)
    training_dataset_loader = DataLoader(training_dataset,sampler = sampler, batch_size=args['batchsize'], shuffle=False, num_workers=args["num_workers"], pin_memory=True, drop_last=True)

    trainer =UniMMAD(args)
    file_path = f'{trainer.__module__.replace(".",os.path.sep)}.py'
    shutil.copyfile(file_path,f'{args["output_dir"]}/logs/{args["data_type"]}_{datetime.datetime.now().strftime("%m_%d_%H_%M")}.py')

    (names,*models) =trainer.get_models()
    avg_loss_log = LossLog()

    model_parameters  = load_parameters(args,names, models)
    logger.info(f'model_parameters: {sum([sum(p.numel() for p in parameters) for parameters in model_parameters] )/1e6}M')
    optimizer_model = optim.Adam([{"params":chain(*model_parameters ),"lr":args['lr'],"momentum":0.9,},                                ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args['EPOCHS'])

    tqdm_epoch = range(int(args["resume"].split(os.sep)[-1].split("_")[-1])+1 if args["resume"]!="none" else 0, args['EPOCHS'])
    (*models,trainer,training_dataset_loader) = accelerator.prepare(*models,trainer,training_dataset_loader)


    if args["resume"]!="none":
        accelerator.load_state(args["resume"], strict=True)
        logger.info("load chekpoints....:"+args["resume"])
    iters = 0
    if args["mode"]=="eval":
        evalute(trainer,args, accelerator)
        return

    for epoch in tqdm_epoch:
        sampler.set_epoch_samples()
        avg_loss_log.reset()
        iters = train_one_epoch(args,accelerator,epoch,iters,trainer,optimizer_model,avg_loss_log,training_dataset_loader)
        if (epoch + 1) % args["save_checkpoint_epoch"]== 0 :
            output_dir = f"{args['output_dir']}/checkpoints/epoch_{epoch}"
            logger.info(f"save checkpoint:{output_dir}")
            accelerator.save_state(output_dir=output_dir)

        if (epoch + 1) % args["print_bar_step"] == 0 or (epoch + 1)==args["EPOCHS"]:
            evalute(trainer,args, accelerator)
        scheduler.step()
