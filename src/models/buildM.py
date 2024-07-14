from .convNet import SimpleConvNet
import os
import warnings 

def build_model (model, model_config):
    if model == 'convNet': 
        i_c = model_config.get('in_channels', None)
        o_c = model_config.get('out_channels', None)
        if i_c is None or o_c is None:
            m = SimpleConvNet()
        else:
            warnings.warn('Creating model: In and Out channels are not provided. Using default values!')
            m = SimpleConvNet(i_c, o_c)
            
        lp = model_config.get('load_path', None)
        
        if lp is not None:
            assert os.path.exists(lp), f'Loading weights: Path {lp} does not exist!'
            m.load_weights(lp)
        
        return m
    else:
        raise ValueError('Undefined model!') 