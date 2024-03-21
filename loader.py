import torch
from torch.utils.data import Dataset

# primer lastnega nalagalnika, ki implementira Dataset
# smiselno kadar zelimo neko kompleksno nalaganje in pripravljanje podatkov
# ce lahko podatko samo nalozimo v pomnilnik kot tenzorje potem bi enakovreden 
# 
# from torch.utils.data import TensorDataset
# dataset = TensorDataset(input_data, target_data)

class My3DDataset(Dataset):
    def __init__(self):
        # nakljucni primeri
        self.input_data = torch.randn(10, 1, 32, 32, 32) # povprecje 0, standardni odklon 1
        self.target_data = torch.rand(10, 1, 32, 32, 32) # med 0 in 1
    
    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = My3DDataset()
    
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4) # 
    
    data, target = next(data_loader.__iter__())

    # pricakujemo obliko 2, 1, 32, 32, 32
    # 2 - velikost paketa
    # 1 - stevilo kanalov
    # 32, 32, 32 - velikost 3D volumna
    print(f'{data.shape=}')
    print(f'{target.shape=}')

