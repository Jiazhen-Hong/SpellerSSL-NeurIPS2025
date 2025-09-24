

import os, sys
import numpy as np
import mne
from collections import Counter
import scipy
import torch
# from ...models.dev.visualize_sinc_filters import visulize_filter
# from ...models.dev.visualize_sinc_filters import plot_hist

ch_names =  [   'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5',
                'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',  'CP5', 'CP3',
                'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2',
                'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7',  'F5',  'F3',
                'F1',  'Fz',  'F2',  'F4',  'F6',  'F8',  'FT7', 'FT8',
                'T7',  'T8',  'T9',  'T10', 'TP7', 'TP8', 'P7',  'P5',
                'P3',  'P1',  'Pz',  'P2',  'P4',  'P6',  'P8',  'PO7',
                'PO3', 'POz', 'PO4', 'PO8', 'O1',  'Oz',  'O2',  'Iz',  'stim']
eeg_ch_n = 64
sfreq = 240
ch_types = ['eeg']*eeg_ch_n + ['stim']
event_dict = {'non': 1, 'target': 2}
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
info.set_montage('standard_1020')


def code_transfer(test_data_code, label):
    test_data_code = np.squeeze(test_data_code, axis = 0)
    new_code = test_data_code[test_data_code != 0]
    new_code = new_code.reshape(len(label), int(new_code.shape[0]/len(label)))
    stimul_code = new_code[:, ::24]
    return stimul_code

def data_transfer(test_data, label):    
    a, b, c, d = test_data.shape
    test_data = test_data.reshape(len(label), 180, b, c, d)
    test_data = torch.tensor(test_data, dtype = torch.float32)
    return test_data

def get_data_IIb(mode:str, path: str):
    '''
    load BCI competition II P300 data
    mode: str:  'train' or 'test'
        load which data
    path: str
        original data path
    Returns
	-------
    '''
    if mode == 'train':
        label = 'train_label.txt'
        s = 1
        e = 11
    elif mode == 'test':
        label = 'test_label.txt'
        s = 1
        e = 8
    first = True
    for i in range(s, e+1):
        if mode == 'train':
            if i <=5:
                name = 'AAS010R0'+ str(i)+'.mat'
            else:
                name = 'AAS011R0'+ str(i-5)+'.mat'
        elif mode == 'test':
            name = 'AAS012R0' + str(i) + '.mat'
        print(name)
        file = os.path.join(path,name)
        x = scipy.io.loadmat(file)
        Signal = x['signal'] 
        Flashing = x['Flashing']
        StimulusCode = x['StimulusCode']
        Signal = Signal.transpose(1,0)  # 0~1
        Flashing = Flashing.transpose(1,0)
        StimulusCode = StimulusCode.transpose(1,0)  
        if first:
            tSignal = Signal
            tFlashing = Flashing
            tStimulusCode = StimulusCode
            first = False
        else:
            tSignal       = np.concatenate((tSignal,Signal), axis = -1)
            tFlashing     = np.concatenate((tFlashing,Flashing), axis = -1)
            tStimulusCode = np.concatenate((tStimulusCode,StimulusCode), axis = -1)
    with open(os.path.join(path, label), "r") as f:  
        Target = f.read()  
    intense_id = np.where(tFlashing[0] == 1)[0]
    block_s_list = [intense_id[0]]
    for i in range(len(intense_id)-1):
        if (intense_id[i+1] - intense_id[i]) > (2.0*240):
            block_s = intense_id[i+1]
            block_s_list.append(block_s)
    block_s_list.append(len(tFlashing[0]))
    block_n = len(block_s_list) - 1
    chara_map = {'A':(7,1), 'B':(7,2), 'C':(7,3), 'D':(7,4), 'E':(7,5), 'F':(7,6),
                    'G':(8,1), 'H':(8,2), 'I':(8,3), 'J':(8,4), 'K':(8,5), 'L':(8,6),
                    'M':(9,1), 'N':(9,2), 'O':(9,3), 'P':(9,4), 'Q':(9,5), 'R':(9,6),
                    'S':(10,1), 'T':(10,2), 'U':(10,3), 'V':(10,4), 'W':(10,5), 'X':(10,6),
                    'Y':(11,1), 'Z':(11,2), '1':(11,3), '2':(11,4), '3':(11,5), '4':(11,6),
                    '5':(12,1), '6':(12,2), '7':(12,3), '8':(12,4), '9':(12,5), '_':(12,6) }

    t_trigger = np.zeros(tFlashing.shape[1])    
    enable_in = True
    for k in range(block_n):
        s = block_s_list[k]
        e = block_s_list[k+1]
        
        signal = tSignal[:, s:e ]
        StimulusCode = tStimulusCode[:,s:e]
        Flashing = tFlashing[:,s:e]
        trigger = np.zeros(Flashing.shape[1])
        print(chara_map[Target[k]])
        a, b = chara_map[Target[k]] # k [0, 41]  
        index_a  = np.where( StimulusCode[0] == a )
        index_b  = np.where( StimulusCode[0] == b )
        id_ = np.where(Flashing[0] == 1)  #
        p300_id = np.union1d(index_a, index_b) #
        non_id = np.setdiff1d(id_, p300_id)  
        
        trigger[non_id] = 1
        trigger[p300_id] = 2
        t_trigger[s:e] = trigger
    non_id  = np.where(t_trigger == 1)[0]
    p300_id = np.where(t_trigger == 2)[0]
    a = np.setdiff1d(  non_id  ,  non_id[::24] )
    b = np.setdiff1d(  p300_id , p300_id[::24] )
    c = np.union1d(a,b)
    t_trigger[c] = 0
    print(Counter(t_trigger))
    print(Counter(t_trigger)[1]/Counter(t_trigger)[2])
    # x1 = np.arange(t_trigger.shape[0])
    # plt.scatter(x1, t_trigger)
    x = np.concatenate((tSignal, t_trigger[np.newaxis, :]), axis = 0)

    return mne.io.RawArray(x, info), tStimulusCode, Target







def get_data_III(subject:str, mode, path):
    '''
    load BCI competition III P300 data
    subject: str: 'A' or 'B'
    mode: str:  'train' or 'test'
        load which data
    path: str
        original data path
    Returns
	-------
    '''
    if subject not in['A', 'B']:
        raise ValueError("wrong subject")
    
    if mode not in ['train', 'test']:
        raise ValueError("wrong mode")
        
    if mode == 'train':
        name = 'Subject_'+ subject +'_Train.mat'

    elif mode == 'test':
        name  = 'Subject_' + subject +'_Test.mat'
        label = 'Subject_'+ subject +'_Test_label.txt'
        
    file = os.path.join(path, name)
    x = scipy.io.loadmat(file)
    Signal = x['Signal'] 
    
    Flashing = x['Flashing']
    StimulusCode = x['StimulusCode'] 

    StimulusCode = np.expand_dims(StimulusCode, axis = -1)
    Flashing     = np.expand_dims(Flashing, axis = -1)

    Signal = Signal.transpose(0, 2, 1)
    Flashing = Flashing.transpose(0, 2, 1)
    StimulusCode = StimulusCode.transpose(0, 2, 1)
    
    if mode == 'train':
        Target = str(x['TargetChar'][0])
    
    elif mode == 'test':
        with open(os.path.join(path, label), "r") as f:  
            Target = f.read()  

    chara_map = {'A':(7,1), 'B':(7,2), 'C':(7,3), 'D':(7,4), 'E':(7,5), 'F':(7,6),
                    'G':(8,1), 'H':(8,2), 'I':(8,3), 'J':(8,4), 'K':(8,5), 'L':(8,6),
                    'M':(9,1), 'N':(9,2), 'O':(9,3), 'P':(9,4), 'Q':(9,5), 'R':(9,6),
                    'S':(10,1), 'T':(10,2), 'U':(10,3), 'V':(10,4), 'W':(10,5), 'X':(10,6),
                    'Y':(11,1), 'Z':(11,2), '1':(11,3), '2':(11,4), '3':(11,5), '4':(11,6),
                    '5':(12,1), '6':(12,2), '7':(12,3), '8':(12,4), '9':(12,5), '_':(12,6) }
    if mode == 'train':
        char_num = 85
    elif mode == 'test':
        char_num = 100

    stimulate = np.zeros([char_num, 1, 7794])    
    for k in range(0, char_num):  
        a, b = chara_map[Target[k]]
        
        index_a  = np.where( StimulusCode[k][0] == a )
        index_b  = np.where( StimulusCode[k][0] == b )
        
        Flashing[k][0][index_a] = 2
        Flashing[k][0][index_b] = 2  
        non_index    =  np.where(Flashing[k][0] == 1 )[0][::24]
        target_index =  np.where(Flashing[k][0] == 2 )[0][::24]
        
        stimulate[k][0][non_index]    = 1
        stimulate[k][0][target_index] = 2
    a,b,c = Signal.shape
    x = np.zeros([a, b+1, c])
    x[:, 0:64, :] = Signal
    x[:, -1,   :] = np.squeeze ( stimulate, axis = 1)
    
    x = x.transpose(1, 0, 2)
    x = x.reshape(65, -1)

    return mne.io.RawArray(x, info), StimulusCode.transpose(1, 0, 2), Target


def preprocess(epoch_len: float,detrend, baseline, my_filter, downsample: int, raw, plot = False):
    '''
    Preprocessing
    my_filter: ['mne', l_freq, h_freq] or ['my', filter] or [None] 
    Returns
    ----
    data_x
    data_y
    '''
    raw = raw.pick_types(eeg=True, stim=True, ecg = False, eog=False)
    
    if my_filter[0] == 'mne':         # ['mne', l_freq, h_freq]
        l_freq = my_filter[1]
        h_freq = my_filter[2]
        raw.filter(l_freq, h_freq)
    
    events = mne.find_events(raw, stim_channel='stim', initial_event = True)
    epochs = mne.Epochs(raw, events, event_id = event_dict, tmin = 0, tmax = epoch_len, 
                baseline = baseline, detrend = detrend, picks = ['eeg'], preload = True, verbose = None)
        
    if plot: # plot average epochs  
        plot_picks = ['eeg']
        evo_kwargs = dict(picks=plot_picks, spatial_colors=True,
                            verbose='error')  # ignore warnings about spatial colors
        
        for key in event_dict.keys():    
            fig = epochs[key].average(picks=plot_picks).plot(**evo_kwargs)
            fig.suptitle('detrend: {}, baseline: {}, {} epochs average '.format(detrend, baseline, key))
            mne.viz.tight_layout()
    '''
    remove the last time sample point, filter data, downsample
    '''
    if my_filter[0] == None or my_filter[0] == 'mne':
        data_x = epochs.get_data()[:,:,:-1][:,:,::downsample]
    
    elif my_filter[0] == 'my':         # ['my', filter]
        filter = my_filter[1]
        data_x = filter.apply (epochs.get_data()[:,:,:-1]) [:,:,::downsample]        
    
    data_y = epochs.events[:, -1]
    
    return [data_x, data_y-1]

