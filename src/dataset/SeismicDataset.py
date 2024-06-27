import numpy as np
import segyio 
import os
from torch.utils.data import Dataset
import torch 
from torch.nn.utils.rnn import pad_sequence
import json
from typing import Dict, Any


class SeismicDataset(Dataset):
    """
    A dataset pipeline for dealing with SEGY files
    Args:
        segy_dir (str): The directory containing the SEGY files
        structured (bool): Indicate if data is structured or unstructured 
        mode (str): The mode for loading the data. ['traces', 'inline', 'xline', 'time', 'cube'] 
        transform (callable, optional): Optional transform to be applied to the data 
        cache_size (int): The size of the cache for storing loaded data (if non-positive no caching is done)
    """
    def __init__(self,
                 data_src: str = None,
                 structured: bool =True,
                 mode: str ='traces',
                 transform: str=None,
                 cache_size: int = -1):
        
        if (data_src is None):
            return 
        
        self.data_src:str = data_src
        self.mode: str = mode
        
        self.transform = transform if isinstance(transform, list) else [transform] if transform else []
        self.structured: bool = structured
        self.cache_size: int = cache_size
        self.from_np: bool = False
        
        self._check_mode()
        
        self.file_list = self._get_data_files()
        self.data = self._initialize_data()
        self.sgy_index_map = self._create_sgy_index_map()
        self.np_index_map = []
        self.data_cache = {}
    
    def _check_mode(self) -> None:
        valid_modes = ['traces', 'inline', 'xline', 'time', 'cube']
        if self.mode not in valid_modes:
            raise ValueError('Invalid mode. Must be one of ', valid_modes)
        if not self.structured and self.mode != 'traces':
            raise ValueError('Invalid mode. Mode must be \'traces\' for unstructured data')

    def _get_data_files(self) -> list[str]:
        files = []
        
        if os.path.isdir(self.data_src):
            files = [os.path.join(self.data_src, f) for f in os.listdir(self.data_src) if f.endswith(('.sgy', '.segy'))]
        elif os.path.isfile(self.data_src):
            files = [self.data_src]
        else:
            raise ValueError("Invalid data source")
        
        if len(files) == 0:
            raise ValueError("No supported files found in the directory")
        
        return files
    
    def _initialize_data(self) -> list[Any]:
        data = []
        for file in self.file_list:
            if file.endswith(('.sgy', '.segy')):
                dt = segyio.open(file, "r", ignore_geometry=not self.structured)
                data.append(dt)                
            else:
                raise ValueError("Invalid file format")
        return data
    
    def _create_sgy_index_map(self) -> list[tuple]:
        sgy_index_mapping = []
        for f_idx, file in enumerate(self.data):
            n_items = self._get_nitems(file)
            # print (f"Found {n_items},{self.mode} in {self.file_list[f_idx]}")
            for i_idx in range(n_items):
                if self.mode == 'inline':
                    sgy_index_mapping.append((f_idx, file.ilines[i_idx]))
                elif self.mode == 'xline':
                    sgy_index_mapping.append((f_idx, file.xlines[i_idx]))
                else:
                    # Note: just a single item for the 'cube' mode -- i_idx = 0
                    sgy_index_mapping.append((f_idx, i_idx))
                    
        return sgy_index_mapping
    
    def _get_nitems(self, file) -> int:
        if self.mode == 'traces': 
            return file.tracecount
        elif self.mode == 'inline':
            return len(file.iline)
        elif self.mode == 'xline':
            return len(file.xline)
        elif self.mode == 'time':
            return len(file.depth_slice)
        elif self.mode == 'cube':
            return 1
        else:
            raise ValueError('Invalid mode')
    
    def __getitem__(self, index) -> np.ndarray:
        try:
            if self.from_np:
                f_idx, _ = self.sgy_index_map[index]
                offset = self.np_index_map[f_idx][0]
                p_idx = index - offset
                
                key = (f_idx, p_idx)
                if key in self.data_cache:
                    self.data_cache[key] = self.data_cache.pop(key)
                    d_point = self.data_cache[key]
                else:
                    if self.mode == 'cube':
                        d_point = self.data[f_idx]
                    else:
                        d_point = self.data[f_idx][p_idx]
                    self._update_cache(key, d_point)
                    
                return d_point
            else:
                f_idx, p_idx = self.sgy_index_map[index]
                
                key = (f_idx, p_idx)
                if key in self.data_cache:
                    self.data_cache[key] = self.data_cache.pop(key)
                    d_point = self.data_cache[key]
                else:
                    d_point = self._load_process_data(self.data[f_idx], p_idx)
                    self._update_cache(key, d_point)
                    
                return d_point
        except Exception as e:
            print(f"Error accessing data at index {index}: {str(e)}")
            return None
    
    def _load_process_data(self, file, index) -> np.ndarray:
        d_point = self._load_data(file, index)
        d_point = self._preprocess_data(d_point)
        for t in self.transform:
                if t is not None:
                    d_point = t(d_point)
                    
        return d_point
    
    def _load_data(self, file, index):
        if isinstance(file, np.ndarray):
            pass 
        else:
            if self.mode == 'traces':
                return file.trace[index]
            elif self.mode == 'time':
                return file.depth_slice[index]
            elif self.mode == 'inline':
                return file.iline[index]
            elif self.mode == 'xline':
                return file.xline[index]
            elif self.mode == 'cube':
                """
                infer the shape of the cube and return the data as a 3D ndarray
                """
                d = (len(file.ilines) * len(file.xlines))
                ntrace = file.tracecount
                gen = file.gather[:,:,:]
                shape = (ntrace//d, file.trace[0].shape[0])
                return self._stack_generator(gen, d, shape) 
            else: 
                raise ValueError('Invalid mode')

    def _stack_generator(self, 
                         gen,
                         num_items: int,
                         shape: tuple) -> np.ndarray:
        result = np.empty((num_items, *shape), dtype=np.float32)
        for i in range(num_items):
            result[i] = next(gen)
        
        return result

    def _update_cache(self, 
                      key  : tuple,
                      value: np.ndarray):
        if self.cache_size < 1:
            return
        
        if len(self.data_cache) > self.cache_size:
            self.data_cache.popitem()
        self.data_cache[key] = value
    
    def __len__(self) -> int:
        return len(self.sgy_index_map)
    
    def __del__(self):
        if hasattr(self, 'data'):
            for file in self.data:
                if self.from_np:
                    del file
                else:
                    file.close()
                    
      
    def _preprocess_data(self,
                         data: np.ndarray) -> np.ndarray:
        """
            @TODO: Preprocess the data 
        """
        return data
                
    @staticmethod
    def padded_collate(batch):
        """
        Custom collate function to handle variable length sequences - left padding with zeros
        """
        batch = [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item.float() for item in batch]
        
        if len(batch[0].shape) == 3:
            batch.sort (key=lambda x: x.shape[0] * x.shape[1] * x.shape[2], reverse=True)
            max_d, max_h, max_w = max(item.shape[0] for item in batch), max(item.shape[1] for item in batch), max(item.shape[2] for item in batch)
            
            padded_batch = torch.zeros(len(batch), max_d, max_h, max_w)
            
            for i, item in enumerate(batch):
                d, h, w = item.shape
                padded_batch[i, :d, :h, :w] = item
            
            original_shapes = torch.tensor([item.shape for item in batch])
            
            return padded_batch, original_shapes
            
        elif len(batch[0].shape) == 2:
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
        
    def _create_np_index_map(self):
        self.np_index_map = []
        offset = 0
        for f_idx, file in enumerate(self.data):
            n_items = self._get_nitems(file)
            if f_idx == 0:
                offset = 0
            else:
                offset = self.np_index_map[f_idx-1][0] + self.np_index_map[f_idx-1][1]
            self.np_index_map.append((offset, n_items))
       
    def save_dataset(self,
                     path: str):
        """
        Save the dataset to a npy file
        """
        os.makedirs(path, exist_ok=True)
        
        np.save(os.path.join(path, 'sgy_index_map.npy'), self.sgy_index_map)
        
        self._create_np_index_map()
        print (type(self.np_index_map), type(self.np_index_map[0]), type(self.sgy_index_map))        
        np.save(os.path.join(path, 'np_index_map.npy'), self.np_index_map)
        
        expected_shapes = [self._get_sgy_file_shape(file) for file in self.data]
        # Save metadata
        metadata = {
            'mode': self.mode,
            'structured': self.structured,
            'cache_size': self.cache_size,
            'file_list': self.file_list,
            'shape': expected_shapes
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        # Save processed data
        for f_idx, file in enumerate(self.data):
            file_path = os.path.join(path, f'data_{f_idx}.npy')
            mmap_file = np.memmap(file_path, dtype='float32', mode='w+', shape=expected_shapes[f_idx])
            
            for i in range(self._get_nitems(file)):
                idx = i
                if self.mode == 'inline':
                    idx = file.ilines[i]
                elif self.mode == 'xline':
                    idx = file.xlines[i]
                
                data_point = self._load_process_data(file, idx)
                if self.mode == 'cube':
                    mmap_file[:] = data_point
                else:
                    mmap_file[i] = data_point
                
                assert isinstance(data_point, np.ndarray)
                # print (f"Saving data point {data_point.shape} from file {f_idx}")
            
            mmap_file.flush()
            del mmap_file  # Close the memmap file

        print(f"Dataset saved to {path}")
    
    def load_dataset(self,
                     path: str):
        """
        Load the dataset from a npy files (created by this class dataset)
        """
        self.sgy_index_map = np.load(os.path.join(path, 'sgy_index_map.npy'))
        
        self.np_index_map = np.load(os.path.join(path, 'np_index_map.npy'))
        
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.mode = metadata['mode']
        self.structured = metadata['structured']
        self.cache_size = metadata['cache_size']
        self.file_list = metadata['file_list']
        self.from_np = True
        expected_shapes = metadata['shape']
            
        # load processed data
        self.data = []
        for f_idx, file in enumerate(self.file_list):
            file_path = os.path.join(path, f'data_{f_idx}.npy')
            
            mmap_file = np.memmap(file_path, dtype='float32', mode='r', shape=expected_shapes[f_idx])
            self.data.append(mmap_file)
        
        print(f"Dataset loaded from {path}")

    @classmethod
    def from_path(cls, path: str):
        """
        Class method to create and return a new instance of DatasetLoader
        from a given npy path.
        """
        instance = cls()
        instance.load_dataset(path)
        return instance
    
    def _get_sgy_file_shape(self, file: Any) -> tuple:
        """
        Get the shape of the file based on the current mode
        """
        if self.mode == 'traces':
            return (file.tracecount, len(file.trace[0]))
        elif self.mode == 'inline':
            return (len(file.ilines), file.iline.shape[0], file.iline.shape[1])
        elif self.mode == 'xline':
            return (len(file.xlines), file.xline.shape[0], file.xline.shape[1])
        elif self.mode == 'time':
            return (len(file.depth_slice), file.depth_slice.shape[0], file.depth_slice.shape[1])
        elif self.mode == 'cube':
            d = (len(file.ilines) * len(file.xlines))
            ntrace = file.tracecount
            shape = (ntrace//d, file.trace[0].shape[0])
            return (d, shape[0], shape[1])
        else:
            raise ValueError('Invalid mode')