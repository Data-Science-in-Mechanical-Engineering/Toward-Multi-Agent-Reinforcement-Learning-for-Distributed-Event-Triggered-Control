from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

# This file represents the parametrization of the NN policies
def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options,size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]
    else:
        return ret

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, q_space, ac_space, hid_size, num_hid_layers, pi_space, mu_space, num, gaussian_fixed_var=True,
              num_options=2, dc=0):
        ### Determine all the basic values
        # determine the dimensions of the state space and observation space
        self.ac_space_dim = ac_space.shape[0]
        self.q_space_dim = q_space[0]
        self.pi_space_dim = pi_space[0]
        self.mu_space_dim = mu_space[0]
        self.num=int(num)
        self.dc = dc
        self.last_action = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.last_action_init = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space) # GaussianDiag
        sequence_length = None
        ###

        ### Define options and observations
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list((20+9*(self.num-1),))) #should be same
        option =  U.get_placeholder(name="option", dtype=tf.int32, shape=[None]) #should be same
        # create filters
        with tf.variable_scope("obfilter_q"):
            self.ob_rms_q = RunningMeanStd(shape=q_space)
        with tf.variable_scope("obfilter_pi"):
            self.ob_rms_pi = RunningMeanStd(shape=pi_space)
        with tf.variable_scope("obfilter_mu"):
            self.ob_rms_mu = RunningMeanStd(shape=mu_space)
        obz_q = tf.clip_by_value((ob[:,3:(self.q_space_dim+3)] - self.ob_rms_q.mean) / self.ob_rms_q.std, -10.0, 10.0)
        obz_pi = tf.clip_by_value((ob[:,3:(self.pi_space_dim+3)] - self.ob_rms_pi.mean) / self.ob_rms_pi.std, -10.0, 10.0)
        obz_mu = tf.clip_by_value((ob[:,:self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)
        ###

        ### Define the Q-function network here
        last_out0 = obz_q # for option 0
        last_out1 = obz_q # for option 1
        for i in range(num_hid_layers):
            last_out0 = tf.nn.tanh(U.dense(last_out0, hid_size, "vffc0%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out1 = tf.nn.tanh(U.dense(last_out1, hid_size, "vffc1%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        last_out0 = U.dense(last_out0, 1, "vfff0", weight_init=U.normc_initializer(1.0))
        last_out1 = U.dense(last_out1, 1, "vfff1", weight_init=U.normc_initializer(1.0))
        self.vpred = U.switch(option[0], last_out1, last_out0)[:,0]
        ###

        ### Define the policy over options
        # define input and structure
        last_out0 = obz_mu # for option 0, idea: if we don't communicate, we focus on contr rew
        last_out1 = obz_mu # for option 1, idea: if we communicate we focus on network
        for i in range(num_hid_layers):
            last_out0 = tf.nn.tanh(U.dense(last_out0, hid_size, "oppi0%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out1 = tf.nn.tanh(U.dense(last_out1, hid_size, "oppi1%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        last_out0 = U.dense(last_out0, 1, "oppif0", weight_init=U.normc_initializer(1.0))
        last_out1 = U.dense(last_out1, 1, "oppif1", weight_init=U.normc_initializer(1.0))
        last_out = tf.concat([last_out0, last_out1], 1)
        # return the probabilities for executing the options
        self.op_pi = tf.nn.softmax(last_out)
        # define tpred
        self.tpred = tf.nn.sigmoid(dense3D2(tf.stop_gradient(last_out), 1, "termhead", option, num_options=num_options,
                                            weight_init=U.normc_initializer(1.0)))[:,0]
        # we always terminate
        termination_sample = tf.constant([True])
        ###

        ### Implementing intra-option-policy
        # define input
        last_out = obz_pi
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        # Insert original option independent version from ppo1:
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2],
                                     initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            mean = U.dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.constant(-0.7, name="logstd", shape=[1, pdtype.param_shape()[0] // 2])
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
            #pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []
        # designing ac
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        ac = tf.clip_by_value(ac,-1.0,1.0)
        ###

        ### Defining the extractable values
        self.last_action = tf.stop_gradient(ac)
        self._act = U.function([stochastic, ob, option], [ac, self.vpred, last_out, logstd])
        self._get_v = U.function([ob, option], [self.vpred])
        self.get_term = U.function([ob, option], [termination_sample])
        self.get_tpred = U.function([ob, option], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])        
        self._get_op = U.function([ob], [self.op_pi])
        ###

    def act(self, stochastic, ob, option):
        # this function returns the action and the Q-function prediction (here denoted as vpred)
        ac1, vpred1, feats, logstd =  self._act(stochastic, [ob], [option])
        return ac1[0], vpred1[0], feats[0], logstd[0]


    def get_option(self,ob):
        # this function returns which option is being used
        op_prob = self._get_op([ob])[0][0]
        return np.random.choice(range(len(op_prob)), p=op_prob)


    def get_term_adv(self, ob, curr_opt):
        # this function calculates the terminal advantage -> not needed as we always terminate
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        op_prob = self._get_op(ob)[0].transpose()
        return (vals[curr_opt[0]] - np.sum((op_prob*vals),axis=0) + self.dc),  ( vals[curr_opt[0]]
                                                                                 - np.sum((op_prob*vals),axis=0) )


    def get_opt_adv(self, ob, curr_opt):
        # this function returns the advantage over the options using the maximum Q-value as baseline
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        # choose max value as reference:
        vals_max = np.amax(vals,axis=0)

        return ((vals[curr_opt[0]] - vals_max+ self.dc ),  (vals[curr_opt[0]] - vals_max) )

    def get_opt_adv_oldpi(self, ob, curr_opt, oldpi):
        # this function returns the advantage over the options using the weighted average of the Q-values of the old parameters
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        # choose max value as reference:
        vals_max = np.amax(vals,axis=0)

        return ((vals[curr_opt[0]] - vals_max+ self.dc ),  (vals[curr_opt[0]] - vals_max) )

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def reset_last_act(self):
        self.last_action = self.last_action_init
        return self.last_action

