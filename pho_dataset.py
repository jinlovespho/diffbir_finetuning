import torch 
from torch.utils.data import Dataset, DataLoader 

class PhoDataset(Dataset):
    def __init__(self):
        self.tmp1 = []
        self.tmp1.append({
            'img': 'img1',
            'box': [ [1,2,3,4], [5,6,7,8] ],
        })

        self.tmp1.append({
            'img': 'img2',
            'box': [ [1,2,3,4] ],
        })


    def __len__(self):
        return len(self.tmp1)

    def __getitem__(self, idx):
        tmp_img = self.tmp1[idx]['img']
        tmp_box = self.tmp1[idx]['box']

        return tmp_img, tmp_box

def collate_fn(batch):
    breakpoint()
    imgs, boxes = zip(*batch)  # Unzip batch
    return imgs, list(boxes)  # Keep boxes as a list of lists

pho_data = PhoDataset()
pho_loader = DataLoader(pho_data, batch_size=2, collate_fn=collate_fn)

for batch in pho_loader:
    breakpoint()