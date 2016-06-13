import sys,os,h5py,glob
sys.path.append('../')
from baselines.filters import runFilter
import numpy as np
from utils.misc import getPYDIR
from datasets.synthp import params_synthetic

def runBaselines(DIR):
    DATADIR = getPYDIR()+'/datasets/synthetic'
    assert os.path.exists(DATADIR),DATADIR+' not found. must have this to run baselines'
    if not os.path.exists(DIR):
        os.mkdir(DIR)

    for f in glob.glob(DATADIR+'/*.h5'):
        dataset = os.path.basename(f).replace('.h5','')
        print 'Reading from: ',f,' Saving to: ',DIR+'/'+dataset+'-baseline.h5'

        filterType = params_synthetic[dataset]['baseline']
        h5fout  = h5py.File(DIR+'/'+dataset+'-baseline.h5',mode='w')
        h5f     = h5py.File(f,mode='r')

        print 'Running filter: ',filterType,' on train'
        X       = h5f['train'].value
        mus,cov,ll = runFilter(X, params_synthetic, dataset, filterType)
        h5fout.create_dataset('train_mu',data = mus)
        h5fout.create_dataset('train_cov',data = cov)
        h5fout.create_dataset('train_ll',data = np.array([ll]))
        rmse       = np.sqrt(np.square(mus-h5f['train_z'].value.squeeze()).mean())
        h5fout.create_dataset('train_rmse',data = np.array([rmse]))

        print 'Running filter: ',filterType,' on valid'
        X       = h5f['valid'].value
        mus,cov,ll = runFilter(X, params_synthetic, dataset, filterType)
        h5fout.create_dataset('valid_mu',data = mus)
        h5fout.create_dataset('valid_cov',data = cov)
        h5fout.create_dataset('valid_ll',data = np.array([ll]))
        rmse       = np.sqrt(np.square(mus-h5f['valid_z'].value.squeeze()).mean())
        h5fout.create_dataset('valid_rmse',data = np.array([rmse]))

        print 'Running filter: ',filterType,' on test'
        X       = h5f['test'].value
        mus,cov,ll = runFilter(X, params_synthetic, dataset, filterType)
        h5fout.create_dataset('test_mu',data = mus)
        h5fout.create_dataset('test_cov',data = cov)
        h5fout.create_dataset('test_ll',data = np.array([ll]))
        rmse       = np.sqrt(np.square(mus-h5f['test_z'].value.squeeze()).mean())
        h5fout.create_dataset('test_rmse',data = np.array([rmse]))

        h5f.close()
        h5fout.close()
if __name__=='__main__':
    runBaselines('./baselines')
