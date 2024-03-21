import torch

from torch.utils.data import DataLoader
from loader import My3DDataset
from model import Unet3D

if __name__ == '__main__':

    # imamo na voljo CUDA graficno? drugace uporabimo CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet3D()
    model = model.to(device) # dajmo ga na pravo napravo

    # za testni primer bi najprej tukaj nalozil samo en primer
    # in model naučil (overfit-al) na tem primeru
    # tako bi preveril, da je model pravilno povezan
    dataset = My3DDataset()
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4) 
    
    # metoda optimizacije, z učno hitrostjo 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # običajna izbira izgube za učenje izhoda sigmoidne funkcije
    criterion = torch.nn.functional.binary_cross_entropy 
    
    epochs=10
    
    for e in range(epochs):
        print(f'epoha {e+1}/{epochs}')
        for data, target in data_loader:
            optimizer.zero_grad() # resetiramo gradiente, alternativno bi jih lahko akumulirali cez vec korakov
            # tukaj naredimo predikcijo z modelom
            # enako lahko naredimo pri predikciji
            
            # to z napravami je lahko vcasih hinavsko
            # ko jih nalagamo podatki pristanejo na CPU pomnilniku, model smo prestavili na GPU
            # in sedaj bomo morda morali podatke ročno premakniti na GPU
            data = data.to(device)
            target = target.to(device)

            output = model(data) 
            # izracunamo izgubo
            loss = criterion(output, target)
            # izracunamo gradient
            loss.backward()
            # opravimo korak optimizacije
            optimizer.step()
            
            # ce smo morda na GPU, moramo izgubo najprej lociti iz racunskega grafa, premakniti na CPU in nato dobiti vrednost
            loss_val = loss.detach().cpu().item()
            print(f'loss: {loss_val}')
        # tu lahko sedaj opravimo se validacijo modela
        # tukaj bom to samo ponovil na učnih podatkih
        
        # model nastavimo v način predikcije, pomembno za nekatere sloje, ki se obnašajo drugače pri predikciji kot pri učenju
        model.eval()
        # z torch.no_grad() onemogočimo računanje gradientov, kar pohitri računanje
        with torch.no_grad():
            # tu je potrebno uporabiti podatke validacije
            for data, target in data_loader:
                # enako lahko naredimo pri predikciji
                output = model(data) 
                # izračunamo izgubo ali kakšno drugo oceno,
                # jo povzamemo preko vseh vzorcev 
                loss = criterion(output, target)
        # nazaj v način učenja
        model.train()
            
        # shranimo zadnji model, najverjetneje bi želeli shraniti najboljšega
        # ali pa kar vsakega posebej
        torch.save(model.state_dict(), f'./model_last.pth')

