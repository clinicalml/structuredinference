"""
Parse command line and store result in params
Model : DKF
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for variational autoencoder")
parser = argparse.ArgumentParser()
parser.add_argument('-dset','--dataset', action='store',default = '', help='Dataset', type=str)

#Recognition Model
parser.add_argument('-rl','--rnn_layers', action='store',default = 1, help='Number of layers in the RNN', type=int, choices=[1,2])
parser.add_argument('-rs','--rnn_size', action='store',default = 600, help='Hidden unit size in q model/RNN', type=int)
parser.add_argument('-rd','--rnn_dropout', action='store',default = 0.1, help='Dropout after each RNN output layer', type=float)
parser.add_argument('-vm','--var_model', action='store',default = 'R', help='Variational Model', type=str, choices=['L','LR','R'])
parser.add_argument('-infm','--inference_model', action='store',default = 'structured', help='Inference Model', type=str, choices=['mean_field','structured'])
parser.add_argument('-ql','--q_mlp_layers', action='store',default = 1, help='#Layers in Recognition Model', type=int)
parser.add_argument('-useprior','--use_generative_prior', action='store_true', help='Use genertative prior in inference network')

#Generative model
parser.add_argument('-ds','--dim_stochastic', action='store',default = 100, help='Stochastic dimensions', type=int)
parser.add_argument('-dh','--dim_hidden', action='store', default = 200, help='Hidden dimensions in DKF', type=int)
parser.add_argument('-tl','--transition_layers', action='store', default = 2, help='Layers in transition fxn', type=int)
parser.add_argument('-ttype','--transition_type', action='store', default = 'simple_gated', help='Layers in transition fxn', type=str, choices=['mlp','simple_gated'])
parser.add_argument('-previnp','--use_prev_input', action='store_true', help='Use previous input in transition')
parser.add_argument('-usenade','--use_nade', action='store_true', help='Use NADE ')

parser.add_argument('-el','--emission_layers', action='store',default = 2, help='Layers in emission fxn', type=int)
parser.add_argument('-etype','--emission_type', action='store',default = 'mlp', help='Type of emission fxn', type=str, choices=['mlp','conditional','res'])

#Weights and Nonlinearity
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)
parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of initialization for weights', type=str, choices=['uniform','normal','xavier','he','orthogonal'])
parser.add_argument('-nl','--nonlinearity', action='store',default = 'relu', help='Nonlinarity',type=str, choices=['relu','tanh','softplus','maxout','elu'])
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-mstride','--maxout_stride', action='store',default = 4, help='Stride for maxout',type=int)
parser.add_argument('-fg','--forget_bias', action='store',default = -5., help='Bias for forget gates', type=float)

parser.add_argument('-vonly','--validate_only', action='store_true', help='Only build fxn for validation')

#Optimization
parser.add_argument('-lr','--lr', action='store',default = 8e-4, help='Learning rate', type=float)
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam','rmsprop'])
parser.add_argument('-bs','--batch_size', action='store',default = 20, help='Batch Size',type=int)
parser.add_argument('-ar','--anneal_rate', action='store',default = 10., help='Number of param. updates before anneal=1',type=float)
parser.add_argument('-repK','--replicate_K', action='store',default = None, help='Number of samples used for the variational bound. Created by replicating the batch',type=int)
parser.add_argument('-shuf','--shuffle', action='store_true',help='Shuffle during training')
parser.add_argument('-covexp','--cov_explicit', action='store_true',help='Explicitly parameterize covariance')
parser.add_argument('-nt','--ntrain', action='store',type=int,default=5000,help='number of training')

#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.05, help='Amount of regularization',type=float)
parser.add_argument('-rspec','--reg_spec', action='store',default = '_', help='String to match parameters (Default is generative model)',type=str)

#Save/load
parser.add_argument('-debug','--debug', action='store_true',help='Debug')
parser.add_argument('-uid','--unique_id', action='store',default = 'uid',help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 2000, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadFile', action='store',default = './NOSUCHFILE', help='Reload from saved model',type=str)
parser.add_argument('-params','--paramFile', action='store',default = './NOSUCHFILE', help='Reload parameters from saved model',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 25, help='Frequency of saving',type=int)
params = vars(parser.parse_args())

hmap       = OrderedDict() 
hmap['lr']='lr'
hmap['var_model']='vm'
hmap['inference_model']='inf'
hmap['dim_hidden']='dh'
hmap['dim_stochastic']='ds'
hmap['nonlinearity']='nl'
hmap['batch_size']='bs'
hmap['epochs']='ep'
hmap['rnn_size']='rs'
hmap['transition_type']='ttype'
hmap['emission_type']='etype'
hmap['use_prev_input']='previnp'
hmap['anneal_rate']='ar'
hmap['reg_value']='rv'
hmap['use_nade']='nade'
hmap['ntrain']='nt'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.4e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'

params['expt_name'] = params['unique_id']
params['unique_id'] = combined[:-1]+'-'+params['unique_id']
params['unique_id'] = 'DKF_'+params['unique_id'].replace('.','_')
"""
import cPickle as pickle
with open('default.pkl','wb') as f:
    pickle.dump(params,f)
"""
