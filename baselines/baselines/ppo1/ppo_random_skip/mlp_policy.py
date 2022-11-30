from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, q_space, ac_space, hid_size, num_hid_layers, pi_space, mu_space, num, gaussian_fixed_var=True,
              num_options=2, dc=0):
        self.ac_space_dim = ac_space.shape[0]
        self.q_space_dim = q_space[0]
        self.pi_space_dim = pi_space[0]
        self.mu_space_dim = mu_space[0]
        self.num = int(num)
        self.dc = dc
        self.last_action = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.last_action_init = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space)  # GaussianDiag
        sequence_length = None
        ob = U.get_placeholder(name="ob", dtype=tf.float32,
                               shape=[sequence_length] + list((20 + 9 * (self.num - 1),)))  # should be same
        option = U.get_placeholder(name="option", dtype=tf.int32, shape=[None])  # should be same
        # create filters
        with tf.variable_scope("obfilter_q"):
            self.ob_rms_q = RunningMeanStd(shape=q_space)
        with tf.variable_scope("obfilter_pi"):
            self.ob_rms_pi = RunningMeanStd(shape=pi_space)
        with tf.variable_scope("obfilter_mu"):
            self.ob_rms_mu = RunningMeanStd(shape=mu_space)
        obz_q = tf.clip_by_value((ob[:, 3:(self.q_space_dim + 3)] - self.ob_rms_q.mean) / self.ob_rms_q.std, -10.0,
                                 10.0)
        obz_pi = tf.clip_by_value((ob[:, 3:(self.pi_space_dim + 3)] - self.ob_rms_pi.mean) / self.ob_rms_pi.std, -10.0,
                                  10.0)
        obz_mu = tf.clip_by_value((ob[:, :self.mu_space_dim] - self.ob_rms_mu.mean) / self.ob_rms_mu.std, -10.0, 10.0)


        self.pdtype = pdtype = make_pdtype(ac_space)

        last_out = obz_q
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]
        
        last_out = obz_pi
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))            
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            mean = U.dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.constant(-0.9, name="logstd", shape=[1, pdtype.param_shape()[0] // 2])
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        ac = tf.clip_by_value(ac, -1.0, 1.0)

        self.last_action = tf.stop_gradient(ac)
        self._act = U.function([stochastic, ob], [ac, self.vpred, last_out, logstd])
        self.get_vpred = U.function([ob], [self.vpred])
        self._get_v = U.function([ob, option], [self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1, feats, logstd =  self._act(stochastic, [ob])
        return ac1[0], vpred1[0], feats[0], logstd[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

