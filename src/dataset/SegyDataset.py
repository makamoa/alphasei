import numpy as np
import segyio 
import os
from torch.utils.data import Dataset
import torch 
from torch.nn.utils.rnn import pad_sequence
import json
from typing import Dict, Any, Iterator, List
import tqdm

class SegyDataset(Dataset):
    """
    A dataset pipeline for dealing with SEGY files
    Args:
        data_src (str): The directory containing the SEGY files
        structured (bool : True): Indicate if data is structured or unstructured 
        mode (str : 'traces'): The mode for loading the data. ['traces', 'iline', 'xline', 'time', 'cube'] 
        stack_cube (bool : true): Indicate if the cube data should be stacked if it is not -> taking the mean over the offset axis
        transform (callable, optional): Optional transform to be applied to the data on a single item (based on the mode)
        cache_size (int : -1): The size of the cache for storing loaded data (if non-positive no caching is done)
    """
    def __init__(self,
                 data_src: str = None,
                 structured: bool =True,
                 mode: str ='traces',
                 stack_cube: bool = True,
                 transform: Any=None,
                 cache_size: int = -1,
                 mmap: bool = False,):
        
        if (data_src is None):
            return 
        
        self.data_src:str = data_src
        self.mode: str = mode
        
        self.transform = transform if isinstance(transform, list) else [transform] if transform else []
        self.structured: bool = structured
        self.cache_size: int = cache_size
        self.from_np: bool = False
        self.stack_cube: bool = stack_cube
        self.mmap: bool = mmap
        
        self._check_mode()
        
        self.file_list = self._get_data_files()
        self.data = self._initialize_data()
        self.sgy_index_map = self._create_sgy_index_map()
        self.np_index_map = []
        self.data_cache = {}
    
    def _check_mode(self) -> None:
        valid_modes = ['traces', 'iline', 'xline', 'time', 'cube']
        if self.mode not in valid_modes:
            raise ValueError('Invalid mode. Must be one of ', valid_modes)
        if (not self.structured) and (self.mode != 'traces'):
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
                if self.mmap:
                    success = dt.mmap()
                    if success:
                        print(f"File {file} successfully memory mapped")
                    else:
                        print(f"Memory mapping failed for {file}, falling back to standard I/O")
                data.append(dt)                
            else:
                raise ValueError("Invalid file format")
        return data
    
    def _create_sgy_index_map(self) -> np.ndarray:
        dtype = [('file_idx', int), ('item_idx', int)]
        
        if self.mode == 'traces' or self.mode == 'time':
            return np.fromiter(
                ((f_idx, i_idx) for f_idx, file in enumerate(self.data) 
                for i_idx in range(self._get_nitems(file))),
                dtype=dtype
            )
        
        elif self.mode in ['iline', 'xline']:
            return np.fromiter(
                ((f_idx, int(getattr(file, self.mode + 's')[i_idx])) 
                for f_idx, file in enumerate(self.data)
                for i_idx in range(self._get_nitems(file))),
                dtype=dtype
            )
        
        elif self.mode == 'cube':
            return np.array([(f_idx, 0) for f_idx in range(len(self.data))], dtype=dtype)
    
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
       
    def _get_nitems(self, file) -> int:
        if self.mode == 'traces': 
            return len(file.trace)
        elif self.mode == 'iline':
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
                    d_point = self._load_process_data(self.data[f_idx], p_idx)
                    # if self.mode == 'cube':
                    #     d_point = self.data[f_idx]
                    # else:
                    #     d_point = self.data[f_idx][p_idx]
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
        if self.from_np:
            d_point = file
            if self.mode != 'cube':
                d_point = d_point[index]
        else:
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
            elif self.mode == 'iline':
                return file.iline[index]
            elif self.mode == 'xline':
                return file.xline[index]
            elif self.mode == 'cube':
                return segyio.tools.cube(file)
            else: 
                raise ValueError('Invalid mode')

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
                 
    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]
        
    def _preprocess_data(self,
                         data: np.ndarray) -> np.ndarray:
        """
            @TODO: Preprocess the data 
        """
        
        if self.mode == 'cube':
            if self.stack_cube and len(data.shape) == 4:
                stacked = np.mean (data, axis=2)
                print (f"The cube data was stacked to shape {stacked.shape}, from {data.shape}")
                return stacked
        return data
                             
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
                     path: str, 
                     chunk_size: int = 0):
        """
        Save the dataset to npy files
        - Folder contains: 
        - metadata.json: metadata about the dataset
        - data_{idx}.npy: the processed data based on the mode 
            - Each file is separately saved
            - The structure of the data is based on the mode -
        """
        os.makedirs(path, exist_ok=True)
        
        np.save(os.path.join(path, 'sgy_index_map.npy'), self.sgy_index_map)
        
        self._create_np_index_map()
        # print (type(self.np_index_map), type(self.np_index_map[0]), type(self.sgy_index_map))        
        np.save(os.path.join(path, 'np_index_map.npy'), self.np_index_map)
        
        expected_shapes = [self._get_sgy_file_shape(file) for file in self.data]
        # Save metadata
        metadata = {
            'mode': self.mode,
            'structured': self.structured,
            'cache_size': self.cache_size,
            'file_list': self.file_list,
            'shape': expected_shapes,
            'stack_cube': self.stack_cube,
            'data_src': self.data_src, 
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
        # Save processed data
        for f_idx, file in enumerate(self.data):            
                file_path = os.path.join(path, f'data_{f_idx}.npy')
                mmap_file = np.memmap(file_path, dtype='float32', mode='w+', shape=expected_shapes[f_idx])
                
                # use tqdm for progress bar 
                for i in tqdm.tqdm(range(self._get_nitems(file)), desc=f"Saving data from file {f_idx}"):
                # def save_chunk(i, file, mmap_file):
                    idx = i
                    if self.mode == 'iline':
                        idx = file.ilines[i]
                    elif self.mode == 'xline':
                        idx = file.xlines[i]
                    
                    data_point = self._load_process_data(file, idx)
                    if self.mode == 'cube':
                        mmap_file[:] = data_point
                    else:
                        mmap_file[i] = data_point
                    
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
        self.stack_cube = metadata['stack_cube']
        self.data_src = metadata['data_src']
        expected_shapes = metadata['shape']
            
        # load processed data
        self.data = []
        for f_idx, file in enumerate(self.file_list):
            file_path = os.path.join(path, f'data_{f_idx}.npy')
            mmap_file = np.memmap(file_path, dtype='float32', mode='r', shape=expected_shapes[f_idx])
            self.data.append(mmap_file)
        
        print(f"Dataset loaded from {path}")
    
    def _get_sgy_file_shape(self, file: Any) -> tuple:
        """
        Get the shape of the file based on the current mode
        """
        if self.mode == 'traces':
            return (file.tracecount, len(file.trace[0]))
        elif self.mode == 'iline':
            return (len(file.ilines), file.iline.shape[0], file.iline.shape[1])
        elif self.mode == 'xline':
            return (len(file.xlines), file.xline.shape[0], file.xline.shape[1])
        elif self.mode == 'time':
            return (len(file.depth_slice), file.depth_slice.shape[0], file.depth_slice.shape[1])
        elif self.mode == 'cube':
            shape = segyio.tools.cube(file).shape
            if len(shape) == 4 and self.stack_cube:
                return (shape[0], shape[1], shape[3])
            return shape
        else:
            raise ValueError('Invalid mode')
    @classmethod
    def from_path(cls, path: str):
        """
        Class method to create and return a new instance of DatasetLoader
        from a given npy path.
        """
        instance = cls()
        instance.load_dataset(path)
        instance._check_mode()
        instance.transform = [None]
        instance.data_cache = {}
        return instance
    
    def npy_transformations (self, 
                             transform: List[Any]):
        """
        Apply transformations to the loaded numpy data
        """
        transform if isinstance(transform, list) else [transform] if transform else []
        
    @staticmethod
    def padded_collate(batch):
        """
        Custom collate function to handle variable length sequences - padding with zeros
        """
        batch = [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item.float() for item in batch]
        
        if len(batch[0].shape) == 4:
            for item in batch:
                assert (len(item.shape) == 4), f"Mixed pre-stacked and post-stacked data"
            
            batch.sort(key=lambda x: x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3], reverse=True)
            max_d, max_h, max_off, max_w = max(item.shape[0] for item in batch), max(item.shape[1] for item in batch), max(item.shape[2] for item in batch), max(item.shape[3] for item in batch)
            
            padded_batch = torch.zeros(len(batch), max_d, max_h, max_off, max_w)
            
            for i, item in enumerate(batch):
                d, h, off, w = item.shape
                padded_batch[i, :d, :h, :off, :w] = item
            
            original_shapes = torch.tensor([item.shape for item in batch])
            
            return padded_batch, original_shapes
        
        elif len(batch[0].shape) == 3:
            batch.sort (key=lambda x: x.shape[0] * x.shape[1] * x.shape[2], reverse=True)
            max_d, max_h, max_w = max(item.shape[0] for item in batch), max(item.shape[1] for item in batch), max(item.shape[2] for item in batch)
            
            padded_batch = torch.zeros(len(batch), max_d, max_h, max_w)
            
            for i, item in enumerate(batch):
                assert (len (item.shape) == 3), f"Mixed pre-stacked and post-stacked data"
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