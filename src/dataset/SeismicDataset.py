import numpy as np
import segyio 
import os
from torch.utils.data import Dataset
import torch 
from torch.nn.utils.rnn import pad_sequence

class SeismicDataset(Dataset):
    """
    A dataset pipeline for dealing with SEGY files
    Args:
        segy_dir (str): The directory containing the SEGY files
        structured (bool): Indicate if data is structured or unstructured 
        mode (str): The mode for loading the data. ['traces', 'inline', 'xline', 'time'] 
        transform (callable, optional): Optional transform to be applied to the data 
    """
    def __init__(self, segy_dir, structured =True, mode='traces', transform=None):
        self.segy_dir = segy_dir
        self.mode = mode
        
        self.transform = transform if isinstance(transform, list) else [transform] if transform else []
        self.structured = structured
        
        self._check_mode()
        
        self.file_list = self._get_segy_files()     
        self.data = self._open_segy_files()
        self.index_map = self._create_index_map()
        
    def _get_segy_files(self):         
        files = [os.path.join(self.segy_dir, f) for f in os.listdir(self.segy_dir) if f.endswith('.sgy') or f.endswith('.segy')]            
        
        assert len(files) > 0,'No SEGY files found in the specified directory'
        
        # print (f"Found {len(files)} SEGY files in {self.segy_dir}")
        
        return files
        
    def __getitem__(self, index):
        try:
            f_idx, p_idx = self.index_map[index]
            d_point = self._load_data(self.data[f_idx], p_idx)
         
            d_point = self._preprocess_data(d_point)
            
            for t in self.transform:
                if t is not None:
                    d_point = t(d_point)
         
            return d_point
        except Exception as e:
            print(f"Error accessing data at index {index}: {str(e)}")
            return None
        
    def _open_segy_files(self):
        return [segyio.open(f, "r", ignore_geometry=not self.structured) for f in self.file_list]
    
    def _load_data(self, file, index):
        if self.mode == 'traces':
            return file.trace[index]
        elif self.mode == 'time':
            return file.depth_slice[index]
        elif self.mode == 'inline':
            return file.iline[index]
        elif self.mode == 'xline':
            return file.xline[index]
    
    def __len__(self):
        return len(self.index_map)
    
    def _get_nitems(self, file):
        if self.mode == 'traces': 
            return file.tracecount
        elif self.mode == 'inline':
            return len(file.iline)
        elif self.mode == 'xline':
            return len(file.xline)
        elif self.mode == 'time':
            return len(file.depth_slice)
        else: 
            raise ValueError('Invalid mode')
    
    def _create_index_map(self):
        index_mapping = []
        for f_idx, file in enumerate(self.data):
            n_items = self._get_nitems(file)
            # print (f"Found {n_items},{self.mode} in {self.file_list[f_idx]}")
            for i_idx in range(n_items):
                if self.mode == 'inline':
                    index_mapping.append((f_idx, file.ilines[i_idx]))
                elif self.mode == 'xline':
                    index_mapping.append((f_idx, file.xlines[i_idx]))
                else:
                    index_mapping.append((f_idx, i_idx))
        
        return index_mapping
    
    def _check_mode(self):
        valid_modes = ['traces', 'inline', 'xline', 'time']
        if self.mode not in valid_modes:
            raise ValueError('Invalid mode. Must be one of ', valid_modes)
        if not self.structured and self.mode != 'traces':
            raise ValueError('Invalid mode. Mode must be \'traces\' for unstructured data')
    
    def __del__(self):
        if hasattr(self, 'data'):
            for file in self.data:
                file.close()
                
    @staticmethod
    def padded_collate(batch):
        """
        Custom collate function to handle variable length sequences - left padding with zeros
        """
        batch = [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item.float() for item in batch]
        
        if len(batch[0].shape) == 2:
            batch.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            max_h, max_w = max(item.shape[0] for item in batch), max(item.shape[1] for item in batch)
            
            padded_batch = torch.zeros(len(batch), max_h, max_w)
            for i, item in enumerate(batch):
                h, w = item.shape
                padded_batch[i, :h, :w] = item
            
            original_shapes = torch.tensor([item.shape for item in batch])
            
            return padded_batch, original_shapes
        else:
            batch.sort(key=lambda x: len(x), reverse=True)
            lengths = [len(item) for item in batch]
            padded_seqs = pad_sequence(batch, batch_first=True)
            return padded_seqs, torch.tensor(lengths)
        
        
    def get_metadata(self):
        metadata = []
        print("Metadeta: Not implemented yet")
        # @TODO: segy metadata extraction
        return metadata
        
    def _preprocess_data(self, data):
        """
            @TODO: Preprocess the data 
        """

        return data