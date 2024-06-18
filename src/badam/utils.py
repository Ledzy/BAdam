from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging


logger = logging.get_logger(__name__)

class BAdamZeRO3Callback(TrainerCallback):
    """ Handler for setup BAdam's training process with ZeRO-3. """
    def __init__(self, *args, **kwargs):
        self.init_loss_scale = kwargs.get("init_loss_scale", 12)
        
    def on_train_begin(self, *args, **kwargs):
        
        optimizer = kwargs["optimizer"] # DeepSpeedOptimizerWrapper
        
        # Create the BlockOptimizer's reference to DeepSpeedZeroOptimizer_Stage3
        ds_optim = optimizer.optimizer # DeepSpeedZeroOptimizer_Stage3
        badam_optim = ds_optim.optimizer # BlockOptimizer
        badam_optim.ds_optimizer = ds_optim
        
        # adjust the loss scale when it is not specified in the configuration file
        if not hasattr(ds_optim, "dynamic_loss_args"):
            ds_optim.cur_scale = 2**self.init_loss_scale
            logger.info(f"Reducing initial loss scale to {ds_optim.cur_scale} for avoiding unnecessary attempts.")
    