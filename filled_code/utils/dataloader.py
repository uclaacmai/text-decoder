from pandas import read_csv
from torch.utils.data import Dataset

class CSVdataset(Dataset):
    # filepath is the path to the csv, nrows is the number of rows to read
    def __init__(self, filepath: str, nrows = None): 
        assert type(nrows) == int or nrows is None, "nrows must be an integer or None"
        self.df = read_csv(filepath, header = None, nrows = nrows)
        self._validate_init()

    def _validate_init(self):
        assert (len(self.df.shape) == 1 or (len(self.df.shape) == 2 and 
        self.df.shape[1] == 1)), f"Each row of the dataframe should only contain one entry, found {self.df.shape[1]}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx, 0]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = CSVdataset("../../test.csv")
    print(str(dataset[1]))
    
    loader = DataLoader(dataset, batch_size = 1)
    for element in loader:
        print(element)