### Preliminaries
# imports
from baselines.common import set_global_seeds, tf_util as U
from pettingzoo.sisl import multiwalker_v9
from baselines import bench
import os.path as osp
import gym, logging
import os
from baselines import logger
import sys
import warnings
# pathfinding
dirname_rel = os.path.dirname(__file__)
splitted = dirname_rel.split("/")
dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted)-3])+"/")
sys.path.append('../../../')
sys.path.append(dirname_rel)
###

### The training function to be called
def train(seed, num_options, app, saves ,wsaves, epoch, dc, num, render ,timesteps_per_batch, comm_weight, explo_iters,\
          begintime, stoptime, deviation, entro_iters, final_entro_iters, pol_ov_op_ent, final_pol_ov_op_ent, optim_epochs, optim_stepsize,\
          optim_batchsize, clip_param, gamma, lam, neurons, num_hid_layers, std, position_noise, angle_noise,\
          forward_reward, terminate_reward, fall_reward, max_cycles):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    # Launch environment
    if render == True:
        render = "human"
    else:
        render = None
    env = multiwalker_v9.env(render_mode=render,
                             n_walkers=num,
                             position_noise=position_noise,
                             angle_noise=angle_noise,
                             forward_reward=forward_reward,
                             terminate_reward=terminate_reward,
                             fall_reward=-fall_reward,
                             shared_reward=False,
                             terminate_on_fall=True,
                             remove_on_fall=True,
                             max_cycles=max_cycles)
    env.reset(seed=seed)
    # Define policy function
    def policy_fn(name, q_space, ac_space, pi_space, mu_space, num):
        return mlp_policy.MlpPolicy(name=name,
                                    hid_size=neurons,
                                    num_hid_layers=num_hid_layers,
                                    num_options=num_options,
                                    dc=dc,
                                    q_space=q_space,
                                    ac_space=ac_space,
                                    pi_space=pi_space,
                                    mu_space=mu_space,
                                    num=num,
                                    gaussian_fixed_var=False)
    # setting up warnings
    gym.logger.setLevel(logging.WARN)
    # start learning
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=300000000,
                        timesteps_per_batch=timesteps_per_batch,
                        comm_weight=comm_weight,
                        explo_iters=explo_iters,
                        begintime=begintime,
                        stoptime=stoptime,
                        deviation=deviation,
                        entro_iters=entro_iters,
                        final_entro_iters=final_entro_iters,
                        pol_ov_op_ent=pol_ov_op_ent,
                        final_pol_ov_op_ent=final_pol_ov_op_ent,
                        entcoeff=0.1,
                        optim_epochs=optim_epochs,
                        optim_stepsize=optim_stepsize,
                        optim_batchsize=optim_batchsize,
                        clip_param=clip_param,
                        gamma=gamma,
                        lam=lam,
                        schedule='constant',
                        num_options=num_options,
                        app=app,
                        saves=saves,
                        wsaves=wsaves,
                        epoch=epoch,
                        seed=seed,
                        dc=dc,
                        num=num)
    # end learning
    env.close()
    print('Learning procedure finished')
### Calling function for terminal
def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--app', help='Append to folder name', type=str, default='')
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1)
    parser.add_argument('--dc', type=float, default=0.1)
    parser.add_argument('--num', help='Number of Agents', type=int, default=2)
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    parser.add_argument('--comm_weight', type=float, default=0.25) # parameters for trajectory generation
    parser.add_argument('--explo_iters', type=float, default=900)
    parser.add_argument('--begintime', type=float, default=1500)
    parser.add_argument('--stoptime', type=float, default=2500)
    parser.add_argument('--deviation', type=float, default=0.25)
    parser.add_argument('--entro_iters', type=float, default=800)
    parser.add_argument('--final_entro_iters', type=float, default=1500)
    parser.add_argument('--pol_ov_op_ent', type=float, default=0.1)
    parser.add_argument('--final_pol_ov_op_ent', type=float, default=0.02)
    parser.add_argument('--timesteps_per_batch', type=int, default=2048) # parameters for optimization process
    parser.add_argument('--optim_epochs', type=float, default=10)
    parser.add_argument('--optim_stepsize', type=float, default=3e-5)
    parser.add_argument('--optim_batchsize', type=int, default=32)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--neurons', type=float, default=96) # parameters for dnn
    parser.add_argument('--num_hid_layers', type=float, default=3)
    parser.add_argument('--std', dest='std', action='store_true', default=False) # DISABLED
    parser.add_argument('--position_noise', type=float, default=0) # parameters for multiwalker
    parser.add_argument('--angle_noise', type=float, default=0)
    parser.add_argument('--forward_reward', type=float, default=5.0)
    parser.add_argument('--terminate_reward', type=float, default=-100.0)
    parser.add_argument('--fall_reward', type=float, default=-10.0)
    parser.add_argument('--max_cycles', type=float, default=400.0)
    args = parser.parse_args()
    train(seed=args.seed,
          num_options=2,
          app=args.app,
          saves=args.saves,
          wsaves=args.wsaves,
          epoch=args.epoch,
          dc=args.dc,
          num=args.num,
          render=args.render,
          timesteps_per_batch=args.timesteps_per_batch,
          comm_weight=args.comm_weight,
          explo_iters=args.explo_iters,
          begintime=args.begintime,
          stoptime=args.stoptime,
          deviation=args.deviation,
          entro_iters=args.entro_iters,
          final_entro_iters=args.final_entro_iters,
          pol_ov_op_ent=args.pol_ov_op_ent,
          final_pol_ov_op_ent=args.final_pol_ov_op_ent,
          optim_epochs=args.optim_epochs,
          optim_stepsize=args.optim_stepsize,
          optim_batchsize=args.optim_batchsize,
          clip_param=args.clip_param,
          gamma=args.gamma,
          lam=args.lam,
          neurons=args.neurons,
          std=args.std,
          num_hid_layers=args.num_hid_layers,
          position_noise=args.position_noise,
          angle_noise=args.angle_noise,
          forward_reward=args.forward_reward,
          terminate_reward=args.terminate_reward,
          fall_reward=args.fall_reward,
          max_cycles=args.max_cycles)

if __name__ == '__main__':
    main()
###
