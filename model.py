import torch

# preprost block, ki ga veckrat uporabim v Unet3D
# tega bi kasneje lahko zamenjali z kompleksnejšim blokom, recimo Resnet
def conv_block_3d(in_channels : int, out_channels : int) -> torch.nn.Module:
    # s Sequential zapakiram operacije, ki se vedno izvedejo v zaporedju, kot cevovod
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
        # tu bi lahko dodal še BatchNorm, ki bi pohitril in olajšal učenje. vendar pa hitreje prekomerno prilagaja
        torch.nn.ReLU(),
        torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
        torch.nn.ReLU()
    )


'''
precej klasična arhitektura za segmentacijo
ki sliko obdeluje na več ločljivostnih nivojih
na visokih ločljivostih se uči natančne detajle, na nizkih pa zajame večje strukture
'''
class Unet3D(torch.nn.Module):
    # inicializacija modela, priprava slojev
    def __init__(self):
        super(Unet3D, self).__init__()
        
        '''
        tu si pripravim vse sloje, ki jih bom uporabljal
        ce bi jih zelel pripraviti v zanki, ker se ponavljalo
        bi jih dodal v torch.nn.ModuleList ali pa bi vsakega
        posebej moral registrirati z self.register_module
        '''
        self.conv_block_l1_down = conv_block_3d(1, 32)
        self.pool_l1 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_block_l2_down = conv_block_3d(32, 64)
        self.pool_l2 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv_block_l3_out = conv_block_3d(64, 128)
        self.unpool_l3 = torch.nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.conv_block_l2_out = conv_block_3d(192, 64) # 128 + 64 = 192
        self.unpool_l2 = torch.nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.conv_block_l1_out = conv_block_3d(96, 32)  # 64 + 32 = 96
        
        self.predictor = torch.nn.Conv3d(32, 1, kernel_size=1)
        # sigmoidna prenosna funkcija preslika vrednosti [-inf, inf] v [0, 1], se lahko interpretira kot verjetnost
        self.sigmoid_tf = torch.nn.Sigmoid()
        

    # izračuna modela
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        # izvedba modela/mreže
        x_l1 = x

        # 1. NIVO (originalni)
        # vhod najprej obdelamo z blokom operacij, povečamo število kanalov 
        x_l1_down = self.conv_block_l1_down(x_l1)
        # nato zmanjšamo prostorsko ločljivost
        x_l2 = self.pool_l1(x_l1_down)

        # 2. NIVO
        # ponovimo prejšnja koraka drugič
        # če imamo več nivojev, bi to lahko delali v zanki
        x_l2_down = self.conv_block_l2_down(x_l2)
        x_l3 = self.pool_l2(x_l2_down)


        # 3. NIVO (v tem primeru zadnji)
        x_l3_out = self.conv_block_l3_out(x_l3)
        # rezultat povečamo nazaj v drugi nivo
        x_l2_unpool = self.unpool_l3(x_l3_out)

        # 2. NIVO
        # podatke ki so prišli navzgor iz 3. nivoja združimo s podatki na poti navzdol, preden smo jih zmanjšali na 3. nivo
        x_l2_cat = torch.concat((x_l2_unpool, x_l2_down), dim=1)
        # združene podatke obdelamo z blokom operacij
        x_l2_out = self.conv_block_l2_out(x_l2_cat)
        # rezultat povečamo nazaj v 1. nivo
        x_l1_unpool = self.unpool_l2(x_l2_out)

        # 1. NIVO
        # podatke ki so prišli navzgor iz 2. nivoja združimo s podatki na poti navzdol, preden smo jih zmanjšali na 2. nivo
        x_l1_cat = torch.concat((x_l1_unpool, x_l1_down), dim=1)
        x_l1_out = self.conv_block_l1_out(x_l1_cat)
        
        # iz izhoda sedaj naredimo napoved za 1 razred/1 kanal
        # z sigmoidno funkcijo preslikamo vrednosti v [0, 1]
        y = self.sigmoid_tf(self.predictor(x_l1_out))

        # izpis za pregled
        # print('tensor shapes:')
        # for x in [x_l1, x_l1_down,
        #           x_l2, x_l2_down,
        #           x_l3, x_l3_out,
        #           x_l2_unpool, x_l2_cat, x_l2_out,
        #           x_l1_unpool, x_l1_cat, x_l1_out,
        #           y]:
        #     print(x.shape)

        return y
    
if __name__ == '__main__':
    # samo hiter preizkus
    model = Unet3D()
    x = torch.randn(2, 1, 32, 32, 32)
    y = model(x)
    print(y.shape)