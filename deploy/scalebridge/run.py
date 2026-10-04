import os
import sys
import time
from pathlib import Path

# Make `scalebridge` and `third_party` importable when running this script
# directly from the deploy/ directory without installing anything.
_DEPLOY_ROOT = Path(__file__).resolve().parents[1]
if str(_DEPLOY_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from scalebridge.utils.logging import logger

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="base"
)
def main(cfg: DictConfig) -> None:

    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, 'run.log')
    logger.info(f'Hydra output directory: {hydra_log_path}')

    agent = instantiate(cfg.agent)
    metadata_dict = agent.get_meta_data()
    
    env = instantiate(cfg.env, metadata_dict=metadata_dict)
    
    print_cnt = 0
    dt = env.dt
    obs_dict = env.reset()

    for _ in range(20):
        agent.get_action(obs_dict)

    while True:
        timer = time.time()
        
        agent.before_step(obs_dict)

        tgt_dof_pos, action = agent.get_action(obs_dict)
        
        obs_dict = env.step(tgt_dof_pos, action)

        if (print_cnt < 5):
            logger.info(f'[Loop] Step {print_cnt}')
            logger.info(f'[Loop] The policy runs at a frequency of {1/(time.time() - timer)}HZ')
            logger.info(f'[Loop] One step takes {time.time()- timer}')
            print_cnt += 1
        
        agent.after_step(obs_dict, action)

        time_until_next_step = dt - (time.time() - timer)
        if time_until_next_step > 0:
            try:
                time.sleep(time_until_next_step)
            except:
                exit()
        else:
            logger.warning(f"[Loop] The policy layer suffers from excessive frame drop: {time_until_next_step}")    


if __name__=="__main__":
    main()