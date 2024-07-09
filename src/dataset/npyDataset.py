import numpy as np
from torch.utils.data import Dataset
import json
from typing import Dict, Any, Iterator, List, Tuple, Union, _TypingEllipsis
import torch
import copy
import os

class NpyDataset(Dataset):
    """
    A dataset class for dealing with npy seismic files 
    Data is asummed to be in the format (z, x, y), the order of the dimensions can be specified but would be transposed on the fly
    Args:
        data_src (str): A list of dictionaries containing the paths to the seismic data and labels 
            - [{'train': cube_path, 'label': labels_path, 'order' : ('x', 'y', 'z'), 'range' : {'x': (st, end) , 'y': (st, end)} }, ...]
            - Note: the range is optional and it should be between 0-1, if not provided the full range is used
        lb_transformations (List[Callable]/Callable): Transformations to apply to the labels
        dt_transformations (List[Callable]/Callable): Transformations to apply to the data
        mode (str): The mode to use for the dataset (default: 'windowed', 'slice')
            - 'windowed': Return windowed slices of the data 
            - 'slice'   : Return full slices of the data
            - 'trace'   : Return a single trace of the data
        dtype (np.dtype): The datatype to use for the data   (default: np.float32)
        ltype (np.dtype): The datatype to use for the labels (default: np.float32)
        norm: (bool): Whether to normalize the data when loading (default: False)
            - in slice mode, the data is normalized slice by slice
            - in windowed mode, the data is normalized window by window
            - in trace mode, the data is normalized trace by trace
        Only in windowed mode:
            - window_w (int): The width of the windowed slice (default: 128)  - depth of the slice
            - window_h (int): The height of the windowed slice (default: 128) - width of the slice
            - stride_w (int): The stride to use when creating windowed slices on width (default: 30)
            - stride_h (int): The stride to use when creating windowed slices on width (default: 30)
    """
    def __init__(self,
                 paths: List[Any] = [],
                 dt_transformations: list[callable]=[], 
                 lb_transformations: list[callable]=[],
                 dtype: np.dtype = np.float32,
                 ltype: np.dtype = np.float32, 
                 norm: bool = False,
                 mode: str = 'slice', 
                 window_w: int = 128,
                 window_h: int = 128,
                 stride_w: int = 30,
                 stride_h: int = 30
                 ):
        validmodes = ['slice', 'windowed', 'traces']
        if mode not in validmodes:
            raise ValueError(f"Invalid mode {mode}")
        
        self.paths = copy.deepcopy(paths)
        self.mode = mode
        self.raw_data, self.raw_labels, self.orders, self.ranges, self.n_items_per_file = self._initialize_data() 
        
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
        
        self.dtype = np.dtype(dtype)
        self.ltype = np.dtype(ltype)        
        
        if norm:
            print("Normalizing the data")
            self.dt_t.append(normalize_item)
        
        self.window_w = None
        self.window_h = None
        self.stride_w = None
        self.stride_h = None
        self.padding = None
          
        if self.mode == 'windowed':
            self.window_w = window_w
            self.window_h = window_h 
            self.stride_w = stride_w
            self.stride_h = stride_h
            self._calculate_window_sizes()
        
    def _initialize_data(self) -> tuple[List[np.ndarray], List[np.ndarray], List[tuple[str, str, str]], List[int]]:
        data = []
        labels = []
        orders = []
        ranges = []
        n_items = []
        
        for path in self.paths:
            _dt = np.load(path['train'], mmap_mode='r')
            _lb = np.load(path['label'], mmap_mode='r')
            _or = path['order']
            
            x_pos = _or.index('x')
            y_pos = _or.index('y')
            
            x_st = 0
            x_end = _dt.shape[x_pos]
            y_st = 0
            y_end = _dt.shape[y_pos]
            self.window_sizes =[]
            
            if path.get('range') is not None:
                x_range = path['range']['x']
                if x_range:
                    x_st = (int) (x_range[0] * _dt.shape[x_pos])
                    x_end = (int) (x_range[1] * _dt.shape[x_pos])
                else: 
                    raise ValueError(f"Invalid range {path['range']}")
                
                y_range = path['range']['y']
                if y_range:
                    y_st = (int) (y_range[0] * _dt.shape[y_pos])
                    y_end = (int) (y_range[1] * _dt.shape[y_pos])
                else: 
                    raise ValueError(f"Invalid range {path['range']}")
            
            data.append(_dt)
            labels.append(_lb)
            orders.append(path['order'])
            ranges.append((x_st, x_end, y_st, y_end))
            if self.mode == 'traces':
                n_items.append((x_end - x_st) * (y_end - y_st))
            else:
                n_items.append((x_end - x_st) + (y_end - y_st))
            
        return data, labels, orders, ranges, n_items
    
    def __len__(self) -> int:
        if self.mode == 'slice':
            return self._slices_len()
        elif self.mode == 'windowed':
            return self._windowed_slices_len()
        elif self.mode == 'traces':
            return self._num_traces()
        else:
            raise ValueError(f"Invalid mode {self.mode}")
    
    def _num_traces(self) -> int:
        return sum(self.n_items_per_file)
    
    def _slices_len(self) -> int: 
        return sum(self.n_items_per_file)
    
    def _calculate_window_sizes(self):
        self.window_sizes = []
        # print ("Calculating window sizes", end = ": ")
        for i, _dt in enumerate(self.raw_data):
            # go over each file once
            ws = {}
            
            h = self._get_shape(i, 'z')
            # slice 1
            w = self._get_shape(i, 'y')
            n_h = max(0, (h - self.window_h) // self.stride_h + 1)
            n_w = max(0, (w - self.window_w) // self.stride_w + 1)
            
            ws['s1'] = (n_h, n_w)
            
            # slice 2
            w = self._get_shape(i, 'x')
            n_h = max(0, (h - self.window_h) // self.stride_h + 1)
            n_w = max(0, (w - self.window_w) // self.stride_w + 1)
            
            ws['s2'] = (n_h, n_w)
            
            # print (ws, end = ", ")
            self.window_sizes.append(copy.deepcopy(ws))   
        # print ("=============")
        
    def _get_shape (self, f_idx: int, dim: str) -> int:
        _dt = self.raw_data[f_idx]
        _or = self.orders[f_idx]
        _pos = _or.index(dim)
        if dim == 'z': 
            return _dt.shape[_pos]
        if dim == 'x': 
            _rg = self.ranges[f_idx]
            x_st, x_end, _, _ = _rg
            return x_end - x_st
        if dim == 'y':
            _rg = self.ranges[f_idx]
            _, _, y_st, y_end = _rg
            return y_end - y_st
        
        raise ValueError(f"Invalid dimension {dim}")
    
    def _windowed_slices_len(self) -> int:
        return sum([ws['s1'][0] * ws['s1'][1] *self._get_shape(i, 'x') + ws['s2'][0] * ws['s2'][1] * self._get_shape(i, 'y') for i,  ws in enumerate (self.window_sizes)])
    
    def __getitem__(self, idx) -> tuple [np.ndarray, np.ndarray]:
        if self.mode == 'slice': 
            return self._get_original_slice(idx)
        elif self.mode == 'windowed':
            return self._get_windowed_slice(idx)
        elif self.mode == 'traces':
            return self._get_trace(idx)
        else:
            raise ValueError(f"Invalid mode {self.mode}")       
        
    def _get_trace(self, idx: int) -> tuple [np.ndarray]:
        """
        Get the trace at the given index and its label
        """
        _idx = idx
        _file_idx = 0
        while _idx >= self.n_items_per_file[_file_idx]:
            _idx -= self.n_items_per_file[_file_idx]
            _file_idx += 1
        
        # print ("in file", _file_idx, "index", _idx, "total items", self.n_items_per_file[_file_idx])
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _rg = self.ranges[_file_idx]
        _or = self.orders[_file_idx]
        
        idx -= sum(self.n_items_per_file[:_file_idx])
        
        # print ("searching for which iline in: ", (_rg[3] , " to ",  _rg[2]))
        
        iline_idx = idx // (_rg[3] - _rg[2]) + _rg[2]
        
        # print ("getting iline at", iline_idx)
        
        iline_idx = (slice(None),  iline_idx, slice(None))
        

        # get the iline slice of the data
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), iline_idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), iline_idx).astype(self.ltype)
        # shape: (z, x)
        
        # print (_dt.shape, _lb.shape)
        
        
        # find which trace to get
        trace_idx = idx % (_rg[3] - _rg[2]) + _rg[2]
        
        # print ("getting trace number:", trace_idx)
        
        _dt = _dt[:, trace_idx]
        _lb = _lb[:, trace_idx]
        
        for t in self.dt_t:
            _dt = t(_dt)
        
        for t in self.lb_t:
            _lb = t(_lb)
            
        return _dt, _lb
        
    def _get_original_slice(self, idx: int) -> tuple [np.ndarray, np.ndarray]:
        """
        Get the original slice of seismic data and labels
        """
        assert idx < len(self), f"Index {idx} out of bounds"
        # find the file that contains the index
        _idx = idx
        _file_idx = 0
        while _idx >= self.n_items_per_file[_file_idx]:
            _idx -= self.n_items_per_file[_file_idx]
            _file_idx += 1
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        _rg = self.ranges[_file_idx]
        
        idx -= sum(self.n_items_per_file[:_file_idx])
        
        # find the 'x' index in _or
        x_dim = self._get_shape(_file_idx, 'x')
        
        # assuming the data is transposed to ('z', 'x', 'y') -- done on the fly later on 
        if idx < x_dim:
            # indexing along the x axis
            idx += _rg[0] # add the starting index
            y_slice = slice(_rg[2], _rg[3]) # get the y slice
            idx = (slice(None), idx, y_slice)
        else:
            # indexing along the y axis
            idx += _rg[2] # add the starting index
            x_slice = slice(_rg[0], _rg[1]) # get the x slice
            idx = (slice(None), x_slice, idx - x_dim)
        
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
        while _idx >= self.window_sizes[_file_idx]['s1'][0] * self.window_sizes[_file_idx]['s1'][1] * self._get_shape(_file_idx, 'x') + self.window_sizes[_file_idx]['s2'][0] * self.window_sizes[_file_idx]['s2'][1]* self._get_shape(_file_idx, 'y'):
            f_items = self.window_sizes[_file_idx]['s1'][0] * self.window_sizes[_file_idx]['s1'][1] * self._get_shape(_file_idx, 'x') + self.window_sizes[_file_idx]['s2'][0] * self.window_sizes[_file_idx]['s2'][1]* self._get_shape(_file_idx, 'y')
            # print ("File at", _file_idx, "index at", _idx, "items", f_items)
            _idx -= f_items
            _file_idx += 1
        
        # print ("File at", _file_idx, "index now at", _idx)
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        _rg = self.ranges[_file_idx]
        _ws = self.window_sizes[_file_idx]
        
        x_pos = _or.index('x')
        y_pos = _or.index('y')
        z_pos = _or.index('z')
        
        # find the slice that contains the index
        # s1 moves along the x axis first
        x_dim = self._get_shape(_file_idx, 'x')
        if _idx < _ws['s1'][0] * _ws['s1'][1] * x_dim:
            # along the x axis
            slice_idx = _idx // (_ws['s1'][0] * _ws['s1'][1])
            y_slice = slice(_rg[2], _rg[3]) # get the y slice
            slice_idx = (slice(None), slice_idx, y_slice)
            
            window_idx = _idx % (_ws['s1'][0] * _ws['s1'][1])
            # print ("getting an inline at ", slice_idx, "window at", window_idx)
            _ws = _ws['s1']
        else:
            # from a crossline
            slice_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * x_dim) // (_ws['s2'][0] * _ws['s2'][1])
            x_slice = slice(_rg[0], _rg[1]) # get the x slice
            slice_idx = (slice(None), x_slice, slice_idx)
            
            window_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * x_dim) % (_ws['s2'][0] * _ws['s2'][1])
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
        
        st_h = row * self.stride_h
        st_w = col * self.stride_w
        
        # print ("window at", st_h, st_w, " with shape: ", dt.shape)
        dt = dt[st_h:st_h+self.window_h, st_w:st_w+self.window_w]
        lb = lb[st_h:st_h+self.window_h, st_w:st_w+self.window_w]
        return dt, lb
    
    def __del__(self):
        for file in self.raw_data:
            del file
        pass
                 
    def __iter__(self) -> Iterator[tuple [np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]
    
    def get_config(self) -> Dict[str, Any]:
        # compose all the metadata of the dataset to a dictionary
        config = {
            'paths': self.paths,
            'dt_transformations': [t.__name__ for t in self.dt_t],
            'lb_transformations': [t.__name__ for t in self.lb_t],
            'dtype': str(self.dtype),
            'ltype': str(self.ltype),
            'norm': self.norm if hasattr(self, 'norm') else False,
            'mode': self.mode,
            'window_w': self.window_w,
            'window_h': self.window_h,
            'stride_w': self.stride_w,
            'stride_h': self.stride_h
        }
        return config
           
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
    
    def save_config(self, path: str) -> str:
        """
        Save the metadata of the dataset to a given path or a json file
        
        Returns:
        str: The path to the saved file
        """
        # check if the path is a directory or json file
        if path.endswith('.json'):
            with open(path, 'w') as f:
                mt = self.get_config()
                print (mt)
                json.dump(mt, f)
                return path
        else: 
            # make sure the path exists and is a directory
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(self.get_metadata(), f)              
            return os.path.join(path, 'metadata.json')
    
    @classmethod
    def from_config(cls, path: str) -> 'NpyDataset':
        """
        Load the metadata of the dataset from a given path
        """
        with open(path, 'r') as f:
            config = json.load(f)
            config['dtype'] = np.dtype(config['dtype'])
            config['ltype'] = np.dtype(config['ltype'])
            
            # Recreate transformation functions (assuming they are defined in the global scope)
            config['dt_transformations'] = [globals()[name] for name in config['dt_transformations']]
            config['lb_transformations'] = [globals()[name] for name in config['lb_transformations']]
        
            return cls(**config)        
    
    @staticmethod
    def create_windowed_collate_fn(self, norm_flag: bool = True, window_w: int = 128, window_h: int = 128, stride_w: int = 30, stride_h: int = 30):
        """
        Create a collate function that return windowed slices:
            if in windowed mode, the collate function would use a batch of windowed slices (windowed dataset) as specified by the class parameters, not the function, and return them 
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
        
        elif self.mode == 'slice':
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
                    data_windows = data.unfold(0, window_h, stride_h).unfold(1, window_w, stride_w)
                    label_windows = label.unfold(0, window_h, stride_h).unfold(1, window_w, stride_w)
                    
                    # Reshape to have all windows as the first dimension
                    h, w, window_h, window_w = data_windows.shape
                    data_windows = data_windows.reshape(-1, window_h, window_w)
                    label_windows = label_windows.reshape(-1, window_h, window_w)
                    
                    # print (data_windows.shape, label_windows.shape)
                    
                    data_windows_list.append(data_windows)
                    label_windows_list.append(label_windows)
                
                data_windows = torch.cat(data_windows_list, dim=0)
                label_windows = torch.cat(label_windows_list, dim=0)
                
                if norm_flag: 
                    # print ("normalizing----------")
                    # normalize each item in the batch (a single window)
                    data_windows = normalize_windows(data_windows)
                
                return data_windows, label_windows
            
            return copy.deepcopy(original_collate)
            
        elif self.mode == 'traces':
            def traces_collate(batch):
                # Process each slice individually
                # window the traces by the given window_x 
                # window_y is redundant as it is always 1
                
                data_windows_list = []
                label_windows_list = []
                
                # one item at a time as the traces might be of different lengths
                for item in batch:
                    data = torch.from_numpy(item[0]) # shape: (z, )
                    label = torch.from_numpy(item[1]) # shape: (z, )
                    
                    # window over the z axis
                    data_windows = data.unfold(0, window_w, stride_w)
                    label_windows = label.unfold(0, window_w, stride_w)
                    
                    # Reshape to have all windows as the first dimension
                    h, window_h = data_windows.shape
                    
                    data_windows = data_windows.reshape(-1, window_h)
                    label_windows = label_windows.reshape(-1, window_h)
                    
                    
                    data_windows_list.append(data_windows)
                    label_windows_list.append(label_windows)
                
                data_windows = torch.cat(data_windows_list, dim=0)
                label_windows = torch.cat(label_windows_list, dim=0)
                
                if norm_flag:
                    # normalize each item in the batch (a single window)
                    data_windows = normalize_windows(data_windows)
                    
                return data_windows, label_windows
            
            return copy.deepcopy(traces_collate)
        
        
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
    
def normalize_trace(trace):
    # Normalize a single trace
    mean = trace.mean()
    std = trace.std()
    trace = (trace - mean) / (std + 1e-8)
    return trace
    
def normalize_windows(windows): 
    if windows.ndim == 3:
        # shape (batch, z, x)
        # Given a batch of windows, normalize each window individually
        mean = windows.mean(dim=(1, 2), keepdim=True)
        std = windows.std(dim=(1, 2), keepdim=True)
        # print ("Mean:", mean, "Std:", std)
        windows = (windows - mean) / (std + 1e-8)
    else: 
        # shape: (batch, z)
        mean = windows.mean(dim=1, keepdim=True)
        std = windows.std(dim=1, keepdim=True)
        windows = (windows - mean) / (std + 1e-8)
        
    return windows

def normalize_item(item):
    # print ("----------- Normalizing Slice ------------------", slice.mean(), slice.std())
    # Normalize a single slice
    mean = item.mean()
    std = item.std()
    item =(item - mean) / (std + 1e-8)
    # print ("Normalized:", slice.mean(), slice.std())
    return item