import numpy as np
from torch.utils.data import Dataset
import json
from typing import Dict, Any, Iterator, List, Tuple, Union, _TypingEllipsis
import torch
import copy

class NpyDataset(Dataset):
    """
    A dataset class for dealing with npy seismic files 
    Args:
        data_src (str): A list of dictionaries containing the paths to the seismic data and labels 
            - [{'train': cube_path, 'label': labels_path, order : ('x', 'y', 'z')}, ...]
        dt_transformations (List[Callable]): A list of transformations to apply to the data
        lb_transformations (List[Callable]): A list of transformations to apply to the labels
        mode (str): The mode to use for the dataset (default: 'windowed', 'original')
            - 'windowed': Return windowed slices of the data 
            - 'original': Return full slices of the data
        dtype (np.dtype): The datatype to use for the data   (default: np.float32)
        ltype (np.dtype): The datatype to use for the labels (default: np.float32)
        norm: (bool): Whether to normalize the data when loading (default: False)
            - in original mode, the data is normalized slice by slice
            - in windowed mode, the data is normalized window by window
        window_x (int): The width of the windowed slice (default: 128)
        window_y (int): The height of the windowed slice (default: 128)
        stride (int): The stride to use when creating windowed slices (default: 30)
    """
    def __init__(self,
                 paths: List[Any] = [],
                 dt_transformations: list[callable]=[], 
                 lb_transformations: list[callable]=[],
                 dtype: np.dtype = np.float32,
                 ltype: np.dtype = np.float32, 
                 norm: bool = False,
                 mode = 'original', 
                 window_x: int = 128,
                 window_y: int = 128,
                 stride: int = 30,
                 ):
        self.paths = copy.deepcopy(paths)
        
        self.dt_t = []
        if dt_transformations:
            if isinstance(dt_transformations, list):
                self.dt_t = copy.deepcopy(dt_transformations)
            else:
                self.dt_t = [copy.deepcopy(dt_transformations)]
        
        self.lb_t = []
        if lb_transformations:
            if isinstance(lb_transformations, list):
                self.lb_t = copy.deepcopy(lb_transformations)
            else:
                self.lb_t = [copy.deepcopy(lb_transformations)]
        
        self.dtype = dtype
        self.ltype = ltype
        self.mode = mode
        if norm:
            print("Normalizing the data")
            self.dt_t.append(normalize_slice)
        
        self.window_x = None
        self.window_y = None
        self.stride = None
        self.padding = None
        self.raw_data, self.raw_labels, self.orders, self.n_slices_per_file = self._initialize_data() #          
        if self.mode == 'windowed':
            self.window_x = window_x
            self.window_y = window_y 
            self.stride = stride
            self._calculate_window_sizes()
        
    def _initialize_data(self) -> tuple[List[np.ndarray], List[np.ndarray], List[tuple[str, str, str]], List[int]]:
        data = []
        labels = []
        orders = []
        n_slices = []
        
        for path in self.paths:
            _dt = np.load(path['train'], mmap_mode='r')
            _lb = np.load(path['label'], mmap_mode='r')

            data.append(_dt)
            labels.append(_lb)
            orders.append(path['order'])
            n_slices.append(_dt.shape[1] + _dt.shape[2])
            
        return data, labels, orders, n_slices
    
    def __len__(self) -> int:
        if self.mode == 'original':
            return self._slices_len()
        elif self.mode == 'windowed':
            return self._windowed_slices_len()
        else:
            raise ValueError(f"Invalid mode {self.mode}")
    
    def _slices_len(self) -> int: 
        return sum(self.n_slices_per_file)
    
    def _calculate_window_sizes(self):
        self.window_sizes = []
        # print ("Calculating window sizes", end = ": ")
        for i, _dt in enumerate(self.raw_data):
            # go over each file once
            ws = {}
            x_pos = self.orders[i].index('x')
            y_pos = self.orders[i].index('y')
            z_pos = self.orders[i].index('z')
            
            h = _dt.shape[z_pos]
            # slice 1
            w = _dt.shape[y_pos]
            n_h = max(0 , (h - self.window_y) // self.stride + 1)
            n_w = max(0, (w - self.window_x) // self.stride + 1)
            
            ws['s1'] = (n_h, n_w)
            
            # slice 2
            w = _dt.shape[x_pos]
            n_h = max(0, (h - self.window_y) // self.stride + 1)
            n_w = max(0, (w - self.window_x) // self.stride + 1)
            
            ws['s2'] = (n_h, n_w)
            
            # print (ws, end = ", ")
            self.window_sizes.append(copy.deepcopy(ws))   
        # print ("=============")
    
    def _windowed_slices_len(self) -> int:
        return sum([ws['s1'][0] * ws['s1'][1] + ws['s2'][0] * ws['s2'][1] for ws in self.window_sizes])
    
    def __getitem__(self, idx) -> tuple [np.ndarray, np.ndarray]:
        if self.mode == 'original': 
            return self._get_original_slice(idx)
        elif self.mode == 'windowed':
            return self._get_windowed_slice(idx)
        else:
            raise ValueError(f"Invalid mode {self.mode}")       

    def _get_original_slice(self, idx: int) -> tuple [np.ndarray, np.ndarray]:
        """
        Get the original slice of seismic data and labels
        """
        assert idx < len(self), f"Index {idx} out of bounds"
        # find the file that contains the index
        _idx = idx
        _file_idx = 0
        while _idx >= self.n_slices_per_file[_file_idx]:
            _idx -= self.n_slices_per_file[_file_idx]
            _file_idx += 1
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        
        idx -= sum(self.n_slices_per_file[:_file_idx])
        
        # find the 'x' index in _or
        x_pos = _or.index('x')
        # find the 'y' index in _or
        y_pos = _or.index('y')
        
        # assuming the data is transposed to ('z', 'x', 'y') -- done on the fly later on 
        if idx < _dt.shape[x_pos]:
            idx = (slice(None), idx, slice(None))
        else:
            idx = (slice(None), slice(None), idx - _dt.shape[x_pos])
        
        # print (_file_idx, idx, x_pos)
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), idx).astype(self.ltype)
        
        for t in self.dt_t:
            _dt = t(_dt)
        
        for t in self.lb_t:
            _lb = t(_lb)
        
        return _dt, _lb
    
    def _get_windowed_slice(self, idx: int) -> tuple [np.ndarray, np.ndarray]:
        # find the file that contains the index
        _idx = idx
        _file_idx = 0
        while _idx >= self.window_sizes[_file_idx]['s1'][0] * self.window_sizes[_file_idx]['s1'][1] + self.window_sizes[_file_idx]['s2'][0] * self.window_sizes[_file_idx]['s2'][1]:
            f_items = self.window_sizes[_file_idx]['s1'][0] * self.window_sizes[_file_idx]['s1'][1] + self.window_sizes[_file_idx]['s2'][0] * self.window_sizes[_file_idx]['s2'][1]
            # print ("File at", _file_idx, "index at", _idx, "items", f_items)
            _idx -= f_items
            _file_idx += 1
        
        # print ("File at", _file_idx, "index now at", _idx)
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        _ws = self.window_sizes[_file_idx]
        
        x_pos = _or.index('x')
        y_pos = _or.index('y')
        z_pos = _or.index('z')
        
        # find the slice that contains the index
        # s1 moves along the x axis first
        if _idx < _ws['s1'][0] * _ws['s1'][1] * _dt.shape[x_pos]:
            # from an iline
            
            slice_idx = _idx // (_ws['s1'][0] * _ws['s1'][1])
            window_idx = _idx % (_ws['s1'][0] * _ws['s1'][1])
            slice_idx = (slice(None), slice_idx, slice(None))
            # print ("getting an inline at ", slice_idx, "window at", window_idx)
            _ws = _ws['s1']
        else:
            # from a crossline
            slice_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * _dt.shape[x_pos]) // (_ws['s2'][0] * _ws['s2'][1])
            window_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * _dt.shape[x_pos]) % (_ws['s2'][0] * _ws['s2'][1])
            slice_idx = (slice(None), slice(None), slice_idx)
            _ws = _ws['s2']
            # print ("getting a xline at ", slice_idx, "window at", window_idx)
        
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), slice_idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), slice_idx).astype(self.ltype)
        
        # get the window from the slice 
        _dt, _lb = self._get_window(_dt, _lb, window_idx, _ws)
        
        for t in self.dt_t:
            _dt = t(_dt)
            
        for t in self.lb_t:
            _lb = t(_lb)
            
        
        return _dt, _lb
    
    def _get_window(self, dt: np.ndarray, lb: np.ndarray, idx: int, ws) -> tuple [np.ndarray, np.ndarray]:
        # given a slice, get the window with the given index
        # ws : h x w
        row = idx // ws[1]
        col = idx % ws[1]
        
        st_h = row * self.stride
        st_w = col * self.stride
        
        # print ("window at", st_h, st_w, " with shape: ", dt.shape)
        dt = dt[st_h:st_h+self.window_y, st_w:st_w+self.window_x]
        lb = lb[st_h:st_h+self.window_y, st_w:st_w+self.window_x]
        return dt, lb
    
    def __del__(self):
        for file in self.raw_data:
            del file
        pass
                 
    def __iter__(self) -> Iterator[tuple [np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]
           
    def save_dataset(self,
                     path: str, 
                     chunk_size: int = 0):
        """
        Save the dataset - transformed to npy file
        - chunk_size: the number of slices to save in each file (0: save all in one file)
        - path: the path to save the dataset
        """
        raise NotImplementedError("Method not implemented")
        pass
    
    def load_dataset(self,
                     path: str):
        """
        Load the dataset from a given path
        """
        raise NotImplementedError("Method not implemented")
        pass
    
    @classmethod
    def from_path(cls, path: str, new_dtransformations: list[callable]=[], new_lbtransformations: list[callable]=[]):
        """
        Class method to create and return a new instance of DatasetLoader from a given npy path - created by save_dataset function
        - new_dtransformations: list of new data transformations to apply
        - new_lbtransformations: list of new label transformations to apply
        """
        raise NotImplementedError("Method not implemented")
        pass
    
    
    @staticmethod
    def create_windowed_collate_fn(self, norm_flag: bool = True, window_x: int = 128, window_y: int = 128, stride: int = 30):
        """
        Create a collate function that return windowed slices:
            if in windowed mode, the collate function would use a batch of windowed slices as specified by the class parameters, and return them 
             - Here you would get the same number of samples as the batch size
             - norm_falg: normalize each window in the batch
                - Note: if the dataset normalization flag is on then this would result in applying normalization twice on each window
            
            if in original mode, the collate function would use a batch of full slices, window them and return them
            - Here your batch is how many slices to use 
            - The number of samples returned would be the number of windows created using these slices
            - norm_falg: normalize each window in the batch
                - Note: if the dataset normalization flag is on then this would result in applying normalization twice (once on the slice and once on the window)
        """
    
        if self.mode == 'windowed':            
            def windowed_collate(batch):
                # Batch is already windowed, just need to stack and convert to tensors
                data = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                
                # Convert to torch tensors and stack
                data = torch.stack([torch.from_numpy(d) for d in data])
                labels = torch.stack([torch.from_numpy(l) for l in labels])
                
                if norm_flag: 
                    print ("normalizing windows")
                    # normalize each item in the batch (a single window)
                    data = normalize_windows(data)
                return data, labels
            
            return copy.deepcopy(windowed_collate)
        
        elif self.mode == 'original':
            def original_collate(batch):
                # print("Original collate")
                # Process each slice individually
                data_windows_list = []
                label_windows_list = []
                
                # one item at a time as the slices might be in different shapes
                for item in batch:
                    data = torch.from_numpy(item[0])
                    label = torch.from_numpy(item[1])
                    
                    # Window the data and labels
                    data_windows = data.unfold(0, window_y, stride).unfold(1, window_x, stride)
                    label_windows = label.unfold(0, window_y, stride).unfold(1, window_x, stride)
                    
                    # Reshape to have all windows as the first dimension
                    h, w, window_h, window_w = data_windows.shape
                    data_windows = data_windows.reshape(-1, window_h, window_w)
                    label_windows = label_windows.reshape(-1, window_h, window_w)
                    
                    data_windows_list.append(data_windows)
                    label_windows_list.append(label_windows)
                
                data_windows = torch.cat(data_windows_list, dim=0)
                label_windows = torch.cat(label_windows_list, dim=0)
                
                if norm_flag: 
                    print ("normalizing----------")
                    # normalize each item in the batch (a single window)
                    data_windows = normalize_windows(data_windows)
                
                return data_windows, label_windows
            
            return copy.deepcopy(original_collate)
        
        
# helper functions
def get_transposed_slice(
        data: np.ndarray,
        current_order: Tuple[str, str, str],
        transposed_order: Tuple[str, str, str],
        transposed_index: Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]
    ) -> np.ndarray:
        """
        Get a slice from the original 3D data as if it were transposed, without transposing.
        
        Args:
        data: The original 3D numpy array
        current_order: Tuple representing the current order of dimensions (e.g., ('x', 'y', 'z'))
        transposed_order: Tuple representing the desired order of dimensions (e.g., ('z', 'x', 'y'))
        transposed_index: The index or slice in the transposed order, must be a 3-tuple with either integers or slices
        
        Returns:
        np.ndarray: The slice of the data as if it were transposed
        """
        dim_map = tuple(current_order.index(dim) for dim in transposed_order)
        
        original_index = [0, 0, 0]
        original_index[dim_map[0]] = transposed_index[0]
        original_index[dim_map[1]] = transposed_index[1]
        original_index[dim_map[2]] = transposed_index[2]
        
        result = data[tuple(original_index)]
        
        remaining_dims = [(i if i < len(result.shape) else len(result.shape) - 1) for i, idx in enumerate(transposed_index)]
        
        # account for the need to transpose the final result 
        if len(remaining_dims) > 1:
            transpose_map = {dim_map[i]: i for i in remaining_dims}
            transpose_order = tuple(transpose_map[i] for i in range(3) if i in transpose_map)
            result = np.transpose(result, transpose_order)
        
        return result
    
def normalize_windows(windows): 
    # Given a batch of windows, normalize each window individually
    mean = windows.mean(dim=(1, 2), keepdim=True)
    std = windows.std(dim=(1, 2), keepdim=True)
    # print ("Mean:", mean, "Std:", std)
    windows = (windows - mean) / (std + 1e-8)
    return windows

def normalize_slice(slice):
    # print ("----------- Normalizing Slice ------------------", slice.mean(), slice.std())
    # Normalize a single slice
    mean = slice.mean()
    std = slice.std()
    slice =(slice - mean) / (std + 1e-8)
    # print ("Normalized:", slice.mean(), slice.std())
    return slice