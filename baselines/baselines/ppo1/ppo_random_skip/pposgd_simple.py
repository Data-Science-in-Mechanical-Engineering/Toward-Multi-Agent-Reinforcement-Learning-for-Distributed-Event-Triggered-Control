# Copyright (c) 2020 Max Planck Gesellschaft

from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from gym import spaces
import os
import shutil
import copy

def traj_segment_generator(agents, env, horizon, num, comm_weight, explo_iters, deviation, stochastic, num_options, seed):

    ### Initialize basic data
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
    value_val_hist = lineup([], num)
    obs_hist = []
    rews_hist = []
    realrews_hist = []
    vpreds_hist = []
    news_hist = []
    opts_hist = []
    acts_hist = []
    logstds_hist = []
    for agent in env.agents:
        obs_hist.append([obs[i] for _ in range(horizon)])
        rews_hist.append(np.zeros(horizon, 'float32'))
        realrews_hist.append(np.zeros(horizon, 'float32'))
        vpreds_hist.append(np.zeros(horizon, 'float32'))
        news_hist.append(np.zeros(horizon, 'int32'))
        opts_hist.append(np.zeros(horizon, 'int32'))
        acts_hist.append([acts[i] for _ in range(horizon)])
        logstds_hist.append([])
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
            acts[i], vpreds[i], feats , logstds[i] = agents[i].act(stochastic, obs[i])
            logstds_hist[i].append(copy.copy(logstds[i]))
            # using no noise since we use original implementation
            for k in range(4):
                acts[i][k] = noise(deviation, copy.copy(acts[i][k]), explo_iters)
        ###

        ### Check if horizon is reached and yield segs
        if t > 0 and t % horizon == 0:
            # Update the missing ep_rets
            for i in range(num):
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
                             'value_val': value_val_hist[i],"logstds": logstds_hist[i]})
            print('...Done, duration: ', str(int(time.time() - start)), 'seconds')
            yield segs
            # Restart process
            start=time.time()
            ep_rets = lineup([], num)
            ep_lens = lineup([], num)
            value_val_hist = lineup([], num)
            logstds_hist = lineup([], num)
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
            options[i] = copy.copy(randomoption(50)) # The included value corresponds to the savings!
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
                v_val.append(agents[i].get_vpred([obs[i]])[0][0])
            value_val_hist[i].append(v_val)
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
                    else:  
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
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          comm_weight, explo_iters, begintime, stoptime, deviation,  # factors for traj generation
          entro_iters, final_entro_iters, pol_ov_op_ent, final_pol_ov_op_ent,  # entropy parameters
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

    num_options = 1 # Hacky solution -> enables to use same logging!
    optim_batchsize_ideal = optim_batchsize
    np.random.seed(seed)
    tf.set_random_seed(seed)
    encrypter = ['abcd', 'efgh', 'ijkl', 'mnop', 'qrst', 'uvwx', 'bcde', 'fghi', 'jklm', 'nopq', 'rstu', 'vwxa',
                 'cdef', 'ghij', 'klmn', 'opqr', 'stuv', 'wxab', 'defg', 'hijk', 'lmno', 'pqrs', 'tuvw', 'xabc']
    encrypter_old = ['dcba', 'hgfe', 'lkji', 'ponm', 'tsrq', 'xwvu', 'edcb', 'ihgf', 'mlkj', 'qpon', 'utsr', 'axwv',
                     'fedc', 'jihg', 'nmkl', 'rqpo', 'vuts', 'baxw', 'gfed', 'kjih', 'onml', 'srqp', 'wvut', 'cbax']
    encrypter_lrmult = ['abcd_lrmult', 'efgh_lrmult', 'ijkl_lrmult', 'mnop_lrmult', 'qrst_lrmult', 'uvwx_lrmult',
                        'bcde_lrmult', 'fghi_lrmult', 'jklm_lrmult', 'nopq_lrmult', 'rstu_lrmult', 'vwxa_lrmult',
                        'cdef_lrmult', 'ghij_lrmult', 'klmn_lrmult', 'opqr_lrmult', 'stuv_lrmult', 'wxab_lrmult',
                        'defg_lrmult', 'hijk_lrmult', 'lmno_lrmult', 'pqrs_lrmult', 'tuvw_lrmult', 'xabc_lrmult']

    ### Book-keeping
    gamename = 'MultiWalker_'
    gamename += 'seed' + str(seed) + '_'
    gamename += app
    version_name = 'PPO_random'  # This variable: "version name, defines the name of the training"
    dirname = '{}_{}_saves/'.format(version_name,gamename)
    
    # retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
    dirname_rel = os.path.dirname(__file__)
    splitted = dirname_rel.split("/")
    dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted)-3])+"/")
    dirname = dirname_rel + "train_results/" + dirname
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



    # Setup losses and stuff
    # ----------------------------------------
    ac_space = spaces.Box(low=np.array([-1,-1,-1,-1]),high=np.array([1,1,1,1]), shape=(4,),dtype=np.float32)
    # add the dimension in the observation space
    q_space_shape = (17+(num-1)*(5+4),)
    pi_space_shape = (17+(num-1)*(5),)
    mu_space_shape = (6,)
    ac_space_shape = (4,)

    # create lists to store agents with policies and variables in
    agents = []
    oldagents = []
    atargs = []
    rets = []
    lrmults = []

    obs = []
    acs = []
    kloldnew = []
    ents = []
    meankl = []
    meanents = []
    pol_entpen = []

    ratios = []
    surr1 = []
    surr2 = []
    pol_surr = []
    vf_losses = []
    total_losses = []
    losses = []
    for i in range(num):
        agents.append(policy_func(encrypter[i], q_space_shape, ac_space, pi_space_shape, mu_space_shape,
                                  num))  # Construct network for new policy
        oldagents.append(policy_func(encrypter_old[i], q_space_shape, ac_space, pi_space_shape, mu_space_shape,
                                     num))  # Network for old policy
        atargs.append(tf.placeholder(dtype=tf.float32, shape=[None]))  # Target advantage function
        rets.append(tf.placeholder(dtype=tf.float32, shape=[None]))  # Empirical return
        lrmults.append(tf.placeholder(name=encrypter_lrmult[i], dtype=tf.float32, shape=[]))
        obs.append(U.get_placeholder_cached(name="ob"))  # Observations of each agent
        acs.append(agents[i].pdtype.sample_placeholder([None]))
        kloldnew.append(oldagents[i].pd.kl(agents[i].pd))
        ents.append(agents[i].pd.entropy())
        meankl.append(U.mean(kloldnew[i]))
        meanents.append(U.mean(ents[i]))
        pol_entpen.append((-entcoeff) * meanents[i])
        ratios.append(tf.exp(agents[i].pd.logp(acs[i]) - oldagents[i].pd.logp(acs[i])))
        surr1.append(ratios[i] * atargs[i])
        surr2.append(U.clip(ratios[i], 1.0 - clip_param, 1.0 + clip_param) * atargs[i])
        pol_surr.append(- U.mean(tf.minimum(surr1[i], surr2[i])))  # PPO's pessimistic surrogate (L^CLIP)
        vf_losses.append(U.mean(tf.square(agents[i].vpred - rets[i])))  # Losses on the Q-function
        total_losses.append(pol_surr[i] + pol_entpen[i]+ vf_losses[i])  # The total losses
        losses.append([pol_surr[i], pol_entpen[i], vf_losses[i], meankl[i], meanents[i]])
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_lists = []
    lossandgrads = []
    adams = []
    assign_old_eq_new = []
    compute_losses = []
    for i in range(num):
        var_lists.append(agents[i].get_trainable_variables())
        lossandgrads.append(U.function([obs[i], acs[i], atargs[i], rets[i], lrmults[i]], losses[i] + [U.flatgrad(total_losses[i], var_lists[i])]))
        adams.append(MpiAdam(var_lists[i], epsilon=adam_epsilon))
        assign_old_eq_new.append(U.function([], [], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldagents[i].get_variables(), agents[i].get_variables())]))
        compute_losses.append(U.function([obs[i], acs[i], atargs[i], rets[i], lrmults[i]], losses[i]))

    U.initialize()
    for i in range(num):
        adams[i].sync()

    ### Prepare logs
    saver = tf.train.Saver(max_to_keep=10000)
    saver_best = tf.train.Saver(max_to_keep=1)

    ### More book-kepping
    results=[]
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
            out = 'epoch,average reward,real reward,comm savings'
            out+='\n'
            results[i].write(out)
            results[i].flush()
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

    if epoch >= 0:
        dirname = '{}_{}_saves/'.format(version_name, gamename, num_options)
        dirname_rel = os.path.dirname(
            __file__)  # Retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
        splitted = dirname_rel.split("/")
        dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted) - 3]) + "/")
        dirname = dirname_rel + "train_results/" + dirname 
        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###
    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0 # Can't be retrieved if using the epoch argument
    timesteps_so_far = 0
    if epoch >= 0:
        timesteps_so_far += timesteps_per_batch * epoch
    global iters_so_far
    iters_so_far = 0
    if epoch >= 0:
        iters_so_far += int(epoch)
    max_val = -100000
    tstart = time.time()
    allrew =max_val -1
    max_val = -100000
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"
    seg_gen = traj_segment_generator(agents=agents, env=env, horizon=timesteps_per_batch, num=num,
                                     comm_weight=comm_weight, explo_iters=explo_iters, deviation=deviation,
                                     stochastic=True, num_options=num_options, seed=seed)
    tstart = time.time()

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

    while True:
        losssaver=lineup([[], []], num)
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

        logger.log("********** Iteration %i ************"%iters_so_far)

        if iters_so_far % 50 == 0 and wsaves:
            print('Saving weights...')
            if epoch > 0:
                filename = dirname + '{}_epoch_{}_reload.ckpt'.format(gamename,iters_so_far)
                save_path = saver.save(U.get_session(),filename)
            else:
                filename = dirname + '{}_epoch_{}.ckpt'.format(gamename, iters_so_far)
                save_path = saver.save(U.get_session(), filename)
            print('...Done')
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
        segs = seg_gen.__next__()
        segment = []
        if wsaves:
            print("Optimizing...")
            start_opt = time.time()
        # splitting by agent i
        lrlocal = lineup([],num)
        listoflrpairs = lineup([],num)
        # Splitting by agent
        for i in range(num):
            add_vtarg_and_adv(segs[i], gamma, lam)
            std = []
            for j in range(num_options):
                logstd = np.mean(segs[i]['logstds']) if len(segs[i]['logstds']) > 0 else 0.
                std.append(np.exp(logstd))
            print("mean std of agent ", str(i + 1), ":", std)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            obs[i], acs[i], atargs[i], tdlamret = segs[i]["ob"], segs[i]["ac"], segs[i]["adv"], segs[i]["tdlamret"]
            vpredbefore = segs[i]["vpred"] # predicted value function before udpate
            atargs[i] = (atargs[i] - atargs[i].mean()) / atargs[i].std()

            d = Dataset(dict(ob=obs[i], ac=acs[i], atarg=atargs[i], vtarg=tdlamret), shuffle=not agents[i].recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(agents[i], "ob_rms"): agents[i].ob_rms.update(obs[i]) # update running mean/std for policy ## hasattr = has attribute, True when it has, False if not

            assign_old_eq_new[i]() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrads[i](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adams[i].update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
            tempsaver = [0, 0, 0, 0, 0, 0]
            losses = []
            lens=segs[i]['ep_lens']
            rews=segs[i]['ep_rets']
            lenbuffer[i].extend(lens)
            rewbuffer[i].extend(rews)
            for k in range(num):
                realrew[i]=np.mean(rews)
            comm_savings[i] = len(np.where(segs[i]["opts"] == 0)[0]) / timesteps_per_batch
            savbuffer[i].extend([comm_savings[i]])
            logger.record_tabular("EpLenMean Agent " + str(i+1), np.mean(lenbuffer[i]))
            logger.record_tabular("EpRewMean Agent " + str(i+1), np.mean(rewbuffer[i]))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        avg_group_comm_save = 0
        for i in range(num):
            avg_group_comm_save += np.mean(savbuffer[i])
        avg_group_comm_save = avg_group_comm_save/num
        ### Book keeping
        if saves:
            out = "{},{},{},{}"
            out+="\n"
            info = [iters_so_far, np.mean(rewbuffer[i]), realrew[i], np.mean(savbuffer[i])]
            results[i].write(out.format(*info))
            results[i].flush()
        ###
        allrew = 0
        sumrew = 0
        for i in range(num):
            sumrew += np.mean(rewbuffer[i])
        allrew = sumrew / num
        realrealrew = np.mean(realrew)
        avg_comm_save = 0
        for i in range(num):
            avg_comm_save += comm_savings[i]
        avg_comm_save = avg_comm_save / num
        avg_group_comm_save = 0
        for i in range(num):
            avg_group_comm_save += np.mean(savbuffer[i])
        avg_group_comm_save = avg_group_comm_save / num
        if saves:
            out = "{},{},{},{},{},{}"
            out += "\n"
            info = [iters_so_far, allrew, sumrew,realrealrew, avg_group_comm_save, avg_comm_save]
            group_results.write(out.format(*info))
            group_results.flush()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

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

def randomoption(prop):
    n = np.random.randint(1,100)
    if n > prop:
        return 1
    else:
        return 0

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
