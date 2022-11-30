import psutil
import copy
import sys
from baselines.common import Dataset, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
from gym import spaces
import os
import shutil

def traj_segment_generator(agents, env, horizon, num, comm_weight, explo_iters, begintime, stoptime,
                           deviation, stochastic, num_options, seed):

    #| Index: [start, end) | Description                                                  |   Values    |
    #|_____________________|______________________________________________________________|_____________|
    #|          01         | Walker X position relative to package (0 = left, 1 =right)   | [-inf, inf] |
    #|          02         | Walker Y position relative to package                        | [-inf, inf] |
    #|          03         | Package angle                                                | [-inf, inf] |
    #|          04         | X Velocity                                                   |   [-1, 1]   |
    #|          05         | Y Velocity                                                   |   [-1, 1]   |
    #|          06         | Hull angle                                                   |  [0, 2*pi]  |
    #|          07         | Hull angular velocity                                        | [-inf, inf] |
    #|          08         | Hip joint 1 angle                                            | [-inf, inf] |
    #|          09         | Hip joint 1 speed                                            | [-inf, inf] |
    #|          10         | Knee joint 1 angle                                           | [-inf, inf] |
    #|          11         | Knee joint 1 speed                                           | [-inf, inf] |
    #|          12         | Leg 1 ground contact flag                                    |    {0, 1}   |
    #|          13         | Hip joint 2 angle                                            | [-inf, inf] |
    #|          14         | Hip joint 2 speed                                            | [-inf, inf] |
    #|          15         | Knee joint 2 angle                                           | [-inf, inf] |
    #|          16         | Knee joint 2 speed                                           | [-inf, inf] |
    #|          17         | Leg 2 ground contact flag                                    |    {0, 1}   |
    #|          +          | Distances to eachother                                       | [-inf, inf] |

    ### Initialize basic data
    begin = begintime
    stop = stoptime
    start=time.time()
    t = 0
    comm_weight = comm_weight
    timer = 0.1
    glob_count = 0
    glob_count_thresh = -1
    # Initialize current states
    print('Initializing running arrays...')
    states = lineup([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]+(num)*copy.copy([0,0,0,0]), num)
    options = lineup(1, num) # Causing initial communication
    acts = lineup([1,1,1,1], num)
    obs = []
    for i in range(num):
        # adding the package informations and stats
        obs.append(states[i][:3] + states[i][:17])
        # adding the communication and timers
        for j in range(num):
            if j == i:
                continue
            else:
                obs[i] += states[i][17 + j * 4:17 + (j + 1) * 4] + [0]
        # adding the real distances
        for j in range(num):
            if j == i:
                continue
            else:
                obs[i] += states[i][17 + j * 4:17 + (j + 1) * 4]
    rews = lineup([0], num)
    news = lineup(True, num)
    term = lineup(True, num)
    term_p = lineup([], num)
    vpreds = lineup([], num)
    logstds = lineup([], num)
    ep_rets = lineup([], num)
    ep_lens = lineup([], num)
    cur_ep_ret = lineup(0, num)
    cur_ep_len = lineup(0, num)
    curr_opt_duration = lineup(0, num)
    print('...Done')
    # Initialize history lists
    print('Initializing history arrays...')
    optpol_p_hist = lineup([], num)
    value_val_hist = lineup([], num)
    obs_hist = []
    rews_hist = []
    realrews_hist = []
    vpreds_hist = []
    news_hist = []
    opts_hist = []
    acts_hist = []
    opt_duration_hist = []
    logstds_hist = []
    for agent in env.agents:
        obs_hist.append([obs[i] for _ in range(horizon)])
        rews_hist.append(np.zeros(horizon, 'float32'))
        realrews_hist.append(np.zeros(horizon, 'float32'))
        vpreds_hist.append(np.zeros(horizon, 'float32'))
        news_hist.append(np.zeros(horizon, 'int32'))
        opts_hist.append(np.zeros(horizon, 'int32'))
        acts_hist.append([acts[i] for _ in range(horizon)])
        opt_duration_hist.append([[] for _ in range(num_options)])
        logstds_hist.append([[] for _ in range(num_options)])
    print('...Done')
    ###

    ### Initial step
    print('Performing initial step...')
    # perform initial action
    action = [1, 1, 1, 1]
    for i in range(num):
        env.step(action)
    # Gather initial information
    for i in range(num):
        states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]], \
                                        env.terminations[env.agents[i]]
    # save correct obs
    for i in range(num):
        obs[i][0:3] = states[i][0:3] # Saving package_old, only in initial step
        obs[i][3:20] = states[i][:17] # Overwriting package_new and stats
    # perform initial communication
    for i in range(num):
        # Receive communication
        for j in range(num):
            if j == i:
                continue
            else:
                if j <= i:
                    k = j
                else:
                    k = j-1
            if options[j] == 1:
                obs[i][20 + (k * 5): 20 + (k+1) * 5] = np.concatenate([states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
            else:
                # Timer goes up
                obs[i][20 + 4 + k * 5] += timer
        # Save the actual distances
        for j in range(num):
            if j == i:
                continue
            else:
                if j <= i:
                    k = j
                else:
                    k = j-1
            obs[i][20+(num-1)*5+k*4 : 20+(num-1)*5+(k+1)*4]=states[i][17+j*4:17+(j+1)*4]
    ###

    ### Begin sampling loop
    while True:

        ### Apply action
        for i in range(num):
            # Receive actions and vpreds
            acts[i], vpreds[i], feats , logstds[i] = agents[i].act(stochastic, obs[i], options[i])
            if t > 0 and t % horizon == 0:
                print(acts[i])
            logstds_hist[i][options[i]].append(copy.copy(logstds[i]))
            # add noise for encouraging exploration
            for k in range(4):
                acts[i][k] = noise(deviation, copy.copy(acts[i][k]), explo_iters)
            for k in range(4):
                acts[i][k] = renoise(deviation, copy.copy(acts[i][k]), begin, stop)
        ###

        ### Check if horizon is reached and yield segs
        if t > 0 and t % horizon == 0:

            # Update the missing ep_rets
            for i in range(num):
                print(acts[i])
                ep_rets[i].append(cur_ep_ret[i])
                cur_ep_ret[i] = 0
                ep_lens[i].append(cur_ep_len[i])
                cur_ep_len[i]=0
            # Create new segs
            segs = []
            for i in range(num):
                segs.append({"ob": np.array(obs_hist[i]), "rew": np.array(rews_hist[i]), "realrew": realrews_hist[i],
                             "vpred" : np.array(vpreds_hist[i]), "new" : news_hist[i], "ac" : np.array(acts_hist[i]), "opts" : opts_hist[i],
                             "nextvpred": vpreds[i] * (1 - news[i]), "ep_rets" : ep_rets[i],"ep_lens" : ep_lens[i],
                             'term_p': term_p[i], 'value_val': value_val_hist[i], "opt_dur": opt_duration_hist[i],
                             "optpol_p": optpol_p_hist[i],"logstds": logstds_hist[i]})
            print('...Done, duration: ', str(int(time.time() - start)), 'seconds')
            yield segs
            # Restart process
            start=time.time()
            ep_rets = lineup([], num)
            ep_lens = lineup([], num)
            term_p = lineup([], num)
            value_val_hist = lineup([], num)
            opt_duration_hist = lineup([[] for _ in range(num_options)], num)
            logstds_hist = lineup([[] for _ in range(num_options)], num)
            curr_opt_duration = lineup(0., num)
        ###

        ### Save generated data
        j = t % horizon
        for i in range(num):
            obs_hist[i][j] = copy.copy(obs[i])
            vpreds_hist[i][j] = copy.copy(vpreds[i])
            news_hist[i][j] = copy.copy(news[i])
            acts_hist[i][j] = copy.copy(acts[i])
            opts_hist[i][j] = copy.copy(options[i])
        ###

        ### Apply step function
        # Apply the gathered actions to the environment
        for i in range(num):
            if np.isnan(np.min(acts[i])) == True:
                acts[i] = [0, 0, 0, 0]
                print('WARNING: NAN DETECTED')
            env.step(copy.copy(acts[i])) # Careful: Without this "copy" operation acts is modified
        # Receive the new states, rews and news
        for i in range(num):
            states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]],\
                                          env.terminations[env.agents[i]]
        # Update package_old in case there was communication in this step
        for i in range(num):
            if options[i] == 1:
                obs[i][0:3] = copy.copy(states[i][0:3])
        # combine reward and do logging
        for i in range(num):
            rews[i] = copy.copy(rews[i])*1.0
            if options[i] == 1:
                rews[i] = rews[i] - comm_weight
            rews[i] = rews[i]/10 if num_options >1 else rews[i] # To stabilize learning.
            cur_ep_ret[i] += rews[i]*10 if num_options > 1 else rews[i]
            cur_ep_len[i] += 1
            rews_hist[i][j] = rews[i]
            realrews_hist[i][j] = rews[i]
            curr_opt_duration[i] += 1
        ###

        ### Update the observation with new values
        # save correct obs
        for i in range(num):
            obs[i][3:20] = copy.copy(states[i][:17])  # Updating package_new and stats
        # Calculate option
        for i in range(num):
            options[i] = agents[i].get_option(copy.copy(obs[i])) # Might be unnecessary since no manipulation happens
            opt_duration_hist[i][options[i]].append(curr_opt_duration[i])
            curr_opt_duration[i] = 0.
        # Receive communication
        for i in range(num):
            for j in range(num):
                if j == i:
                    continue
                else:
                    if j <= i:
                        k = j
                    else:
                        k = j - 1
                if options[j] == 1:
                    obs[i][20 + (k * 5): 20 + (k + 1) * 5] = np.concatenate(
                        [states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
                else:
                    obs[i][20 + 4 + k * 5] += timer # Timer goes up
            # Save the actual distances for vpred
            for j in range(num):
                if j == i:
                    continue
                else:
                    if j <= i:
                        k = j
                    else:
                        k = j - 1
                obs[i][20 + (num - 1) * 5 + k * 4: 20 + (num - 1) * 5 + (k + 1) * 4] = states[i][17 + j * 4:17 + (j + 1) * 4]
        # Only for logging
        for i in range(num):
            t_p = []
            v_val = []
            for oopt in range(num_options):
                v_val.append(agents[i].get_vpred([obs[i]],[oopt])[0][0])
                t_p.append(agents[i].get_tpred([obs[i]],[oopt])[0][0])
            term_p[i].append(t_p)
            optpol_p_hist[i].append(agents[i]._get_op([obs[i]])[0][0])
            value_val_hist[i].append(v_val)
            term[i] = agents[i].get_term([obs[i]],[options[i]])[0][0]
        ###

        # Check if termination of episode happens
        if any(news):
            # if new rollout starts -> reset last action and start anew
            for i in range(num):
                news[i] = True
                ep_rets[i].append(cur_ep_ret[i])
                cur_ep_ret[i] = 0
                ep_lens[i].append(cur_ep_len[i])
                cur_ep_len[i] = 0
            env.reset(seed=seed)
            # perform initial action
            for i in range(num):
                env.step(action)
            # Gather initial information
            for i in range(num):
                states[i], rews[i], news[i] = env.observe(env.agents[i]), env.rewards[env.agents[i]], \
                                              env.terminations[env.agents[i]]
                options[i] = 1
            # save correct obs
            for i in range(num):
                obs[i][3:20] = states[i][:17]  # Overwriting package_new and stats
                obs[i][0:3] = states[i][0:3]  # Saving package_old, only in initial step
            # Perform initial communication
            for i in range(num):
                # Receive communication
                for j in range(num):
                    if j == i:
                        continue
                    else:
                        if j <= i:
                            k = j
                        else:
                            k = j - 1
                    if options[j] == 1:
                        obs[i][20 + (k * 5): 20 + (k + 1) * 5] = np.concatenate([states[i][17 + (j * 4):17 + (j + 1) * 4], [0]])
                    else:  # Just a relic, never going to happen
                        # Timer goes up
                        obs[i][20 + 4 + k * 5] += 1
                # Save the actual distances
                for j in range(num):
                    if j == i:
                        continue
                    else:
                        if j <= i:
                            k = j
                        else:
                            k = j - 1
                    obs[i][20 + (num - 1) * 5 + k * 4: 20 + (num - 1) * 5 + (k + 1) * 4] = states[i][17 + j * 4:17 + (j + 1) * 4]
            ###
        t += 1
        ###
    ###

def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          comm_weight, explo_iters, begintime, stoptime, deviation, # factors for traj generation
          entro_iters, final_entro_iters, pol_ov_op_ent, final_pol_ov_op_ent, # entropy parameters
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          num_options=1,
          app='',
          saves=False,
          wsaves=False,
          epoch=-1,
          seed=1,
          dc=0,
          num):

    ### Fundamental definitions
    evals = False
    optim_batchsize_ideal = optim_batchsize
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # encrypter necessary so tf1 does not mess up everything
    encrypter = ['abcd', 'efgh', 'ijkl', 'mnop', 'qrst', 'uvwx', 'bcde', 'fghi', 'jklm', 'nopq', 'rstu', 'vwxa', 
                 'cdef', 'ghij', 'klmn', 'opqr', 'stuv', 'wxab', 'defg', 'hijk', 'lmno', 'pqrs', 'tuvw', 'xabc']
    encrypter_old = ['dcba', 'hgfe', 'lkji', 'ponm', 'tsrq', 'xwvu', 'edcb', 'ihgf', 'mlkj', 'qpon', 'utsr', 'axwv',
                     'fedc', 'jihg', 'nmkl', 'rqpo', 'vuts', 'baxw', 'gfed', 'kjih', 'onml', 'srqp', 'wvut', 'cbax']
    encrypter_lrmult = ['abcd_lrmult', 'efgh_lrmult', 'ijkl_lrmult', 'mnop_lrmult', 'qrst_lrmult', 'uvwx_lrmult',
                        'bcde_lrmult', 'fghi_lrmult', 'jklm_lrmult', 'nopq_lrmult', 'rstu_lrmult', 'vwxa_lrmult',
                        'cdef_lrmult', 'ghij_lrmult', 'klmn_lrmult', 'opqr_lrmult', 'stuv_lrmult', 'wxab_lrmult',
                        'defg_lrmult', 'hijk_lrmult', 'lmno_lrmult', 'pqrs_lrmult', 'tuvw_lrmult', 'xabc_lrmult']
    gamename = 'MultiWalker_'
    gamename += 'seed' + str(seed) + '_'
    gamename += app
    version_name = 'HEM' # This variable: "version name, defines the name of the training"
    dirname = '{}_{}_saves/'.format(version_name,gamename)

    dirname_rel = os.path.dirname(__file__) # Retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
    splitted = dirname_rel.split("/")
    dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted) - 3]) + "/")
    dirname = dirname_rel + "train_results/" + dirname
    # if saving -> create the necessary directories
    if wsaves:
        first=True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False
        files = ['pposgd_simple.py','mlp_policy.py','run_mujoco.py'] # copy also the original files into the folder where the training results are stored
        if epoch >=0:
            first = False
        else:
            first = True
        for i in range(len(files)):
            src = os.path.join(dirname_rel,'baselines/baselines/ppo1/') + files[i]
            #dest = os.path.join('/home/nfunk/results_NEW/ppo1/') + dirname
            dest = dirname + "src_code/"
            if (first):
                os.makedirs(dest)
                first = False
            shutil.copy2(src,dest)
    ###

    ### Setup all tensorflow functions and dependencies
    # Setup all functions
    ac_space = spaces.Box(low=np.array([-1,-1,-1,-1]),high=np.array([1,1,1,1]), shape=(4,),dtype=np.float32)
    # add the dimension in the observation space
    q_space_shape = (17+(num-1)*(5+4),)
    pi_space_shape = (17+(num-1)*(5),)
    mu_space_shape = (6,)
    ac_space_shape = (4,)
    # create lists to store agents with policies and variables in
    agents = []
    oldagents = []
    lrmults = []
    clip_params = []
    atargs = []
    rets = []
    pol_ov_op_ents = []
    obs = []
    ops = []
    term_adv = [] ## NO NAME CHANGE
    acs = []
    kloldnew = [] ## NO NAME CHANGE
    ents = []
    meankl = [] ## NO NAME CHANGE
    meanents = []
    pol_entpen = [] ## NO NAME CHANGE
    ratios = []
    atarg_clips = []
    surr1 = []
    surr2 = []
    pol_surr = []
    vf_losses = []
    total_losses = []
    losses = []
    log_pi = [] ## NO NAME CHANGE
    old_log_pi = []
    entropies = [] ## NOT ENTS
    ratio_pol_ov_op = [] ## NO NAME CHANGE
    term_adv_clip = []
    surr1_pol_ov_op = [] ## NO NAME CHANGE
    surr2_pol_ov_op = [] ## NO NAME CHANGE
    pol_surr_pol_ov_op = [] ## NO NAME CHANGE
    op_loss = []
    for i in range(num):
        agents.append(policy_func(encrypter[i], q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Construct network for new policy
        oldagents.append(policy_func(encrypter_old[i], q_space_shape, ac_space, pi_space_shape, mu_space_shape, num)) # Network for old policy
        lrmults.append(tf.placeholder(name=encrypter_lrmult[i], dtype=tf.float32, shape=[])) # Learning rate multiplier, updated with schedule
        clip_params.append(clip_param * lrmults[i]) # Annealed clipping parameter epsilon for PPO
        atargs.append(tf.placeholder(dtype=tf.float32, shape=[None])) # Target advantage function
        rets.append(tf.placeholder(dtype=tf.float32, shape=[None])) # Empirical return
        pol_ov_op_ents.append(tf.placeholder(dtype=tf.float32, shape=None)) # Entropy coefficient for policy over options
        obs.append(U.get_placeholder_cached(name="ob")) # Observations of each agent
        ops.append(U.get_placeholder_cached(name="option")) # Option chosen by each agent
        term_adv.append(U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])) # Advantage of termination by each agent #### NO, adv of pol_ov_op ####
        acs.append(agents[i].pdtype.sample_placeholder([None]))
        kloldnew.append(oldagents[i].pd.kl(agents[i].pd)) # Output datatype
        ents.append(agents[i].pd.entropy()) # Entropy-scheduling values for each agent
        meankl.append(U.mean(kloldnew[i])) # Mean kl for each agent
        meanents.append(U.mean(ents[i])) # Mean entropies for each agent
        pol_entpen.append((-entcoeff) * meanents[i])
        ratios.append(tf.exp(agents[i].pd.logp(acs[i]) - oldagents[i].pd.logp(acs[i]))) # Probability of choosing action under new policy vs old policy (PPO)
        atarg_clips.append(atargs[i]) # Advantage of choosing the action#
        surr1.append(ratios[i] * atarg_clips[i]) # Surrogates from conservative policy iteration
        surr2.append(U.clip(ratios[i], 1.0 - clip_params[i], 1.0 + clip_params[i]) * atarg_clips[i])
        pol_surr.append(- U.mean(tf.minimum(surr1[i], surr2[i]))) # PPO's pessimistic surrogate (L^CLIP)
        vf_losses.append(U.mean(tf.square(agents[i].vpred - rets[i]))) # Losses on the Q-function
        total_losses.append(pol_surr[i] + vf_losses[i]) # The total losses
        log_pi.append(tf.log(tf.clip_by_value(agents[i].op_pi, 1e-5, 1.0))) # calculate logarithm of propability of policy over options
        old_log_pi = tf.log(tf.clip_by_value(oldagents[i].op_pi, 1e-5, 1.0)) # calculate logarithm of propability of policy over options old parameter
        entropies.append(-tf.reduce_sum(agents[i].op_pi * log_pi[i], reduction_indices=1)) # calculate entropy of policy over options
        ratio_pol_ov_op.append(tf.exp(tf.transpose(log_pi[i])[ops[i][0]] - tf.transpose(old_log_pi[i])[ops[i][0]])) # calculate the ppo update for the policy over options
        term_adv_clip.append(term_adv[i])
        surr1_pol_ov_op.append(ratio_pol_ov_op[i] * term_adv_clip[i]) # surrogate from conservative policy iteration
        surr2_pol_ov_op.append(U.clip(ratio_pol_ov_op[i], 1.0 - clip_params[i], 1.0 + clip_params[i]) * term_adv_clip[i])
        pol_surr_pol_ov_op.append(- U.mean(tf.minimum(surr1_pol_ov_op[i], surr2_pol_ov_op[i]))) # PPO's pessimistic surrogate (L^CLIP)
        op_loss.append(pol_surr_pol_ov_op[i] - pol_ov_op_ents[i] * tf.reduce_sum(entropies[i]))
        losses.append([pol_surr[i], pol_entpen[i], vf_losses[i], meankl[i], meanents[i], op_loss[i]])
        total_losses[i] += op_loss[i] # add loss of policy over options to total loss
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "op_loss"]
    # Setup optimizers and initialize session
    var_lists = []
    term_lists = []
    lossandgrads = []
    adams = []
    assign_old_eq_new = []
    compute_losses = []
    for i in range(num):
        var_lists.append(agents[i].get_trainable_variables())
        term_lists.append(var_lists[i][6:8])
        # define function that we will later do gradient descent on
        lossandgrads.append(U.function([obs[i], acs[i], atargs[i], rets[i], lrmults[i] ,ops[i], term_adv[i],
                            pol_ov_op_ents[i]], losses[i] + [U.flatgrad(total_losses[i], var_lists[i])]))
        adams.append(MpiAdam(var_lists[i], epsilon=adam_epsilon))
        # define function that will assign the current parameters to the old policy
        assign_old_eq_new.append(U.function([], [], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldagents[i].get_variables(), agents[i].get_variables())]))
        compute_losses.append(U.function([obs[i], acs[i], atargs[i], rets[i], lrmults[i], ops[i]], losses[i]))
    U.initialize()
    for i in range(num):
        adams[i].sync()
    ###

    ### Prepare logs and savers
    # initialize "savers" which will store the results
    saver = tf.train.Saver(max_to_keep=10000)
    saver_best = tf.train.Saver(max_to_keep=20)
    # Define the names of the .csv files for individual agents that are going to be stored
    results = []
    results_best_model = []
    for i in range(num):
        results.append(0)
        results_best_model.append(0)
        if saves:
            if epoch>=0:
                results[i] = open(dirname + version_name + '_' + gamename +'_' +'agent_' \
                                  + str(i+1) +'_epoch_' + str(epoch) + '_results.csv','w')
                results_best_model[i] = open(dirname + version_name + '_' + gamename +'_' \
                                             +'agent_'+str(i+1)+'epoch_'+str(epoch)+'_bestmodel.csv','w')
            else:
                results[i] = open(dirname + version_name + '_' + gamename + '_' \
                                  + 'agent_' + str(i + 1) + '_results.csv', 'w')
                results_best_model[i] = open(dirname + version_name + '_' + gamename + '_' \
                                             + 'agent_' + str(i + 1) + '_bestmodel.csv', 'w')
            out = 'epoch,average reward,real reward,comm savings, average comm savings'
            for opt in range(num_options): out += ',option {} dur'.format(opt)
            for opt in range(num_options): out += ',option {} std'.format(opt)
            #for opt in range(num_options): out += ',option {} term'.format(opt)
            for opt in range(num_options): out += ',option {} adv'.format(opt)
            for opt in range(num_options): out += ',option {} pol_surr'.format(opt)
            for opt in range(num_options): out += ',option {} pol_entpen'.format(opt)
            for opt in range(num_options): out += ',option {} vf_loss'.format(opt)
            for opt in range(num_options): out += ',option {} meankl'.format(opt)
            for opt in range(num_options): out += ',option {} meanents'.format(opt)
            for opt in range(num_options): out += ',option {} op_loss'.format(opt)
            #[pol_surr[i], pol_entpen[i], vf_losses[i], meankl[i], meanents[i], op_loss[i]]
            out+='\n'
            results[i].write(out)
            # results.write('epoch,avg_reward,option 1 dur, option 2 dur, option 1 term, option 2 term\n')
            results[i].flush()
    # define a csv file for the group performance
    if saves:
        if epoch>=0:
            group_results = open(dirname + version_name + '_' + gamename +'_' +'epoch_'
                                 +str(epoch)+'_group_results.csv','w')
        else:
            group_results = open(dirname + version_name + '_' + gamename + '_' + 'group_results.csv', 'w')
        out = 'epoch,average reward,sum of average reward,real reward,average comm savings, real comm savings'
        out += '\n'
        group_results.write(out)
        group_results.flush()
    # speciality: if running the training with epoch argument -> a model is loaded
    if epoch >= 0:
        dirname = '{}_{}_saves/'.format(version_name, gamename, num_options)
        dirname_rel = os.path.dirname(
            __file__)  # Retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
        splitted = dirname_rel.split("/")
        dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted) - 3]) + "/")
        dirname = dirname_rel + "train_results/" + dirname # so bis hier hin sind wir dann in dem Folder indem unsere Ergebnisse gespeichert werden
        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###

    ### Start training process
    episodes_so_far = 0 # Can't be retrieved if using the epoch argument
    timesteps_so_far = 0
    if epoch >= 0:
        timesteps_so_far += timesteps_per_batch * epoch
    global iters_so_far
    iters_so_far = 0
    if epoch >= 0:
        iters_so_far += int(epoch)
    des_pol_op_ent = pol_ov_op_ent # define policy over options entropy scheduling
    if epoch>entro_iters:
        if epoch>=final_entro_iters:
            des_pol_op_ent=final_pol_ov_op_ent
        else:
            des_pol_op_ent=pol_ov_op_ent+(final_pol_ov_op_ent-pol_ov_op_ent)/(final_entro_iters-entro_iters)*(iters_so_far-entro_iters)
        des_pol_op_ent=des_pol_op_ent/final_pol_ov_op_ent # warning, possible conflict with exact iter of schedule
    max_val = -100000 # define max_val, this will be updated to always store the best model
    tstart = time.time()
    allrew = max_val-1
    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    seg_gen = traj_segment_generator(agents=agents, env=env, horizon=timesteps_per_batch, num=num, comm_weight=comm_weight, explo_iters=explo_iters, begintime=begintime,
                                     stoptime=stoptime, deviation=deviation, stochastic=True, num_options=num_options, seed=seed)
    segs = []
    datas = []
    savbuffer = lineup(deque(maxlen=100), num)
    comm_savings = lineup(0, num)
    lenbuffer = lineup(deque(maxlen=100), num)
    rewbuffer = lineup(deque(maxlen=100), num)
    realrew = lineup([], num)
    for i in range(num):
        datas.append([0 for _ in range(num_options)])
        segs.append([])
    ###

    ### Start training loop
    while True:
        losssaver=lineup([[], []], num)
        # Some error collecting
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break
        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        # adapt the entropy to the desired value
        if (iters_so_far+1)>entro_iters:
            if iters_so_far>final_entro_iters:
                des_pol_op_ent=final_pol_ov_op_ent
            else:
                des_pol_op_ent=pol_ov_op_ent+(final_pol_ov_op_ent-pol_ov_op_ent)/(final_entro_iters-entro_iters)*(iters_so_far-entro_iters)
        # every 50 epochs save the best model
        if iters_so_far % 50 == 0 and wsaves:
            print('Saving weights...')
            if epoch > 0:
                filename = dirname + '{}_epoch_{}_reload.ckpt'.format(gamename,iters_so_far)
                save_path = saver.save(U.get_session(),filename)
            else:
                filename = dirname + '{}_epoch_{}.ckpt'.format(gamename, iters_so_far)
                save_path = saver.save(U.get_session(), filename)
            print('...Done')
        # adaptively save best model -> if current reward is highest, save the model
        if (allrew>max_val) and wsaves:
            print('Saving best weights...')
            max_val = allrew
            for i in range(num):
                results_best_model[i].write('epoch: '+str(iters_so_far) + 'rew: ' + str(np.mean(rewbuffer[i])) + '\n')
                results_best_model[i].flush()
                filename = dirname + 'agent_' + str(i+1) + '_best.ckpt'.format(gamename,iters_so_far)
            save_path = saver_best.save(U.get_session(),filename)
            if epoch > 0:
                filename = dirname + '{}_epoch_{}_reload.ckpt'.format(gamename,iters_so_far)
                save_path = saver_best.save(U.get_session(),filename)
            else:
                filename = dirname + '{}_epoch_{}.ckpt'.format(gamename, iters_so_far)
                save_path = saver_best.save(U.get_session(), filename)
            print('...Done')

        ### Starting the training iteration
        logger.log("*********** Iteration %i *************" % iters_so_far)
        # Sample (s,a)-Transitions
        print('Sampling trajectories...')
        segs = seg_gen.__next__()
        # Evaluation sequence for one iteration
        segment=[]
        if wsaves:
            print("Optimizing...")
            start_opt = time.time()
        # splitting by agent i
        lrlocal = lineup([],num)
        listoflrpairs = lineup([],num)
        for i in range(num):
            #print('-----------Agent', str(i + 1), '------------')
            add_vtarg_and_adv(segs[i], gamma, lam) # Calculate A(s,a,o) using GAE
            # calculate information for logging
            opt_d = []
            std = []
            for j in range(num_options):
                dur = np.mean(segs[i]['opt_dur'][j]) if len(segs[i]['opt_dur'][j]) > 0 else 0.
                opt_d.append(dur)
                logstd = np.mean(segs[i]['logstds'][j]) if len(segs[i]['logstds'][j]) > 0 else 0.
                std.append(np.exp(logstd))
            print("mean std of agent ", str(i+1), ":", std)
            obs[i], acs[i], ops[i], atargs[i], tdlamret = segs[i]["ob"], segs[i]["ac"], segs[i]["opts"], segs[i]["adv"],\
                                                          segs[i]["tdlamret"]
            #vpredbefore = segs[i]["vpred"] # predicted value function before udpate
            atargs[i] = (atargs[i] - atargs[i].mean()) / atargs[i].std() # standardized advantage function estimate
            if hasattr(agents[i], "ob_rms"): agents[i].ob_rms.update(obs[i]) # update running mean/std for policy
            if hasattr(agents[i], "ob_rms_only"): agents[i].ob_rms_only.update(obs[i])
            assign_old_eq_new[i]() # set old parameter values to new parameter values
            # minimum batch size:
            min_batch=160
            t_advs = [[] for _ in range(num_options)]
            # select all the samples concerning one of the options
            for opt in range(num_options):
                indices = np.where(ops[i]==opt)[0]
                #print("batch size:",indices.size, "for opt ", opt)
                opt_d[opt] = indices.size
                if not indices.size:
                    t_advs[opt].append(0.)
                    continue
                # This part is only necessary when we use options. We proceed to these verifications in order not to discard any collected trajectories.
                if datas[i][opt] != 0:
                    if (indices.size < min_batch and datas[i][opt].n > min_batch):
                        datas[i][opt] = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices], atarg=atargs[i][indices],
                                                     vtarg=tdlamret[indices]), shuffle=not agents[i].recurrent)
                        t_advs[opt].append(0.)
                        continue
                    elif indices.size + datas[i][opt].n < min_batch:
                        # pdb.set_trace()
                        oldmap = datas[i][opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],obs[i][indices]))
                        cat_ac = np.concatenate((oldmap['ac'],acs[i][indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atargs[i][indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[i][opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                                shuffle=not agents[i].recurrent)
                        t_advs[opt].append(0.)
                        continue
                    elif (indices.size + datas[i][opt].n > min_batch and datas[i][opt].n < min_batch)\
                            or (indices.size > min_batch and datas[i][opt].n < min_batch):
                        oldmap = datas[i][opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],obs[i][indices]))
                        cat_ac = np.concatenate((oldmap['ac'],acs[i][indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atargs[i][indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[i][opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg),
                                                    shuffle=not agents[i].recurrent)
                    if (indices.size > min_batch and datas[i][opt].n > min_batch):
                        datas[i][opt] = d = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices],
                                                         atarg=atargs[i][indices], vtarg=tdlamret[indices]),
                                                    shuffle=not agents[i].recurrent)
                elif datas[i][opt] == 0:
                    datas[i][opt] = d = Dataset(dict(ob=obs[i][indices], ac=acs[i][indices], atarg=atargs[i][indices],
                                                     vtarg=tdlamret[indices]), shuffle=not agents[i].recurrent)
                # define the batchsize of the optimizer:
                optim_batchsize = optim_batchsize or obs[i].shape[0]
                #print("optim epochs:", optim_epochs)
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):

                        # Calculate advantage for using specific option here
                        tadv,nodc_adv = agents[i].get_opt_adv(batch["ob"],[opt])
                        tadv = tadv if num_options > 1 else np.zeros_like(tadv)
                        t_advs[opt].append(nodc_adv)
                        # calculate the gradient
                        *newlosses, grads = lossandgrads[i](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"],
                                                            cur_lrmult, [opt], tadv,des_pol_op_ent)
                        # perform gradient update
                        if wsaves:
                            adams[i].update(grads, optim_stepsize * cur_lrmult)
                        losses.append(newlosses)
                    # at the end of batch, save all the mean losses

                    tempsaver=[0,0,0,0,0,0]
                    for n in range(len(losses[0])):
                        k=n+1
                        tempsaver[n]=np.mean(np.array(losses)[:,n:k])
                    losssaver[i][opt].append(tempsaver.copy())
            ###

            # do logging:
            lrlocal[i] = copy.copy((segs[i]["ep_lens"], segs[i]["ep_rets"]))  # local values
            listoflrpairs[i] = (MPI.COMM_WORLD.allgather(lrlocal[i]))  # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs[i]))

            lrlocal[i] = copy.copy((segs[i]["ep_lens"], segs[i]["ep_rets"])) # local values
            listoflrpairs[i] = (MPI.COMM_WORLD.allgather(lrlocal[i])) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs[i]))
            lenbuffer[i].extend(lens)
            rewbuffer[i].extend(rews)
            for k in range(num):
                realrew[i]=np.mean(rews)
            comm_savings[i] = len(np.where(ops[i]==0)[0])/timesteps_per_batch
            savbuffer[i].extend([comm_savings[i]])
            # The last started episode pulls down the average and is therefore discarded in statistics
            logger.record_tabular("EpLenMean of Agent " + str(i+1) + ": ", np.mean(lenbuffer[i][:,:-1]))
            logger.record_tabular("EpRewMean of Agent " + str(i + 1) + ": ", np.mean(rewbuffer[i]))
            logger.record_tabular("Comm_sav of Agent " + str(i + 1) + ": ", comm_savings[i])

            ### Agent Book keeping
            if saves:
                out = "{},{},{},{},{}"
                for _ in range(num_options): out += ",{},{},{},{},{},{},{},{},{}"
                out += "\n"
                info = [iters_so_far, np.mean(rewbuffer[i]), realrew[i], np.mean(savbuffer[i]), comm_savings[i]]
                for j in range(num_options): info.append(opt_d[j])
                for j in range(num_options): info.append(std[j])
                for j in range(num_options): info.append(np.mean(t_advs[j]))
                for n in range(len(losses[0])):
                    k=n+1
                    for j in range(num_options):
                        if np.array(losssaver[i][j]).ndim == 1:
                            info.append(losssaver[i][j])
                        elif np.array(losssaver[i][j]).ndim == 0:
                            info.append(0)
                            print('WARNING: 0 dim in losses, only one option chosen')
                        else:
                            info.append(np.mean(np.array(losssaver[i][j])[:,n:k]))

                results[i].write(out.format(*info))
                results[i].flush()
            ###
        ###
        if wsaves:
            print('...Done, duration: ', str(int(time.time() - start_opt)), 'seconds')
        ### Group Book keeping
        allrew=0
        sumrew=0
        for i in range(num):
            sumrew += np.mean(rewbuffer[i])
        allrew = sumrew/num
        realrealrew = np.mean(realrew)
        avg_comm_save = 0
        for i in range(num):
            avg_comm_save += comm_savings[i]
        avg_comm_save = avg_comm_save/num
        avg_group_comm_save = 0
        for i in range(num):
            avg_group_comm_save += np.mean(savbuffer[i])
        avg_group_comm_save = avg_group_comm_save/num
        if saves:
            out = "{},{},{},{},{},{}"
            out += "\n"
            info = [iters_so_far, allrew, sumrew, realrealrew, avg_group_comm_save,avg_comm_save]
            group_results.write(out.format(*info))
            group_results.flush()
        ###

        ### Final logging
        iters_so_far += 1
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        ###

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def appender(rando, limit):
    out = []
    for i in range(limit):
        out.append(rando[i][:-1]) # The last unfinished episode pulls down the average and is therefore discarded
    return np.mean(out)

def sumer(rando, limit):
    out = 0
    for i in range(limit):
        out += np.mean(rando[i][:-1]) # The last unfinished episode pulls down the average and is therefore discarded
    return out

def lineup(func, count):
    x = []
    for i in range(count):
        x.append(copy.copy(func))
    return x

def render(env, times):
    env.render()
    time.sleep(times)

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1] # False = 0, True = 1, nonterminal if next step is not new
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def sorting(count, i):
    x = []
    for j in range(count):
        x.append(j)
    x.remove(i)
    return x

def noise(deviation, x, ending):
    factor = deviation * (ending-iters_so_far)/ending
    if factor <= 0.0:
        factor = 0.0
    noise = np.random.normal(0.0, factor)
    x += noise
    x = clip(x, -1.0, 1.0)
    return x

def renoise(deviation, x, begin, end):
    # looks like a tent
    # only on from begin
    if iters_so_far >= begin:
        # catch cases where we are above end
        if iters_so_far >= end:
            renfactor = 0.0
        # within tent do
        else:
            # first half
            if iters_so_far <= (begin+((end-begin)/2)):
                renfactor = deviation*2*((iters_so_far-begin)/(end-begin))
            # second half:
            else:
                renfactor = deviation*(1-((2*iters_so_far-(end+begin))/(end-begin)))
    else:
        renfactor = 0.0
    # add noise and return x
    noises = np.random.normal(0.0, renfactor)
    x += noises
    x = clip(x, -1.0, 1.0)
    #print(factor, 'renoise')
    return x

def clip(value, low, high):
    if value >= high:
        value = high
    elif value <= low:
        value = low
    return value
