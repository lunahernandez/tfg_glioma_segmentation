import torch
import torch.nn as nn

class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)

class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer3D(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AttentionBlock3D(nn.Module):
    """
    Mecanismo de atencion propuesto en DenseUNet+ para suprimir fondo sano.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DenseUNetPlus3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5, init_features=32, growth_rate=16, block_layers=3):
        super().__init__()

        self.init_conv = nn.Conv3d(in_channels, init_features, kernel_size=3, padding=1)

        # Encoder (Dense Blocks)
        self.enc1 = DenseBlock3D(init_features, growth_rate, block_layers)
        channels_enc1 = init_features + block_layers * growth_rate
        self.pool1 = nn.Conv3d(channels_enc1, channels_enc1, kernel_size=3, stride=2, padding=1)

        self.enc2 = DenseBlock3D(channels_enc1, growth_rate, block_layers)
        channels_enc2 = channels_enc1 + block_layers * growth_rate
        self.pool2 = nn.Conv3d(channels_enc2, channels_enc2, kernel_size=3, stride=2, padding=1)

        self.enc3 = DenseBlock3D(channels_enc2, growth_rate, block_layers)
        channels_enc3 = channels_enc2 + block_layers * growth_rate
        self.pool3 = nn.Conv3d(channels_enc3, channels_enc3, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = DenseBlock3D(channels_enc3, growth_rate, block_layers)
        channels_bot = channels_enc3 + block_layers * growth_rate

        # Decoder con Attention Blocks integrados (La magia del '+')
        self.up3 = nn.ConvTranspose3d(channels_bot, channels_enc3, kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(F_g=channels_enc3, F_l=channels_enc3, F_int=channels_enc3 // 2)
        self.dec3 = DenseBlock3D(channels_enc3 * 2, growth_rate, block_layers)
        channels_dec3 = (channels_enc3 * 2) + block_layers * growth_rate
        # Capa de reduccion para evitar colapso de VRAM en la RTX 5090
        self.reduce3 = nn.Conv3d(channels_dec3, channels_enc2, kernel_size=1)

        self.up2 = nn.ConvTranspose3d(channels_enc2, channels_enc2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(F_g=channels_enc2, F_l=channels_enc2, F_int=channels_enc2 // 2)
        self.dec2 = DenseBlock3D(channels_enc2 * 2, growth_rate, block_layers)
        channels_dec2 = (channels_enc2 * 2) + block_layers * growth_rate
        self.reduce2 = nn.Conv3d(channels_dec2, channels_enc1, kernel_size=1)

        self.up1 = nn.ConvTranspose3d(channels_enc1, channels_enc1, kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(F_g=channels_enc1, F_l=channels_enc1, F_int=channels_enc1 // 2)
        self.dec1 = DenseBlock3D(channels_enc1 * 2, growth_rate, block_layers)
        channels_dec1 = (channels_enc1 * 2) + block_layers * growth_rate

        # Salida
        self.final_conv = nn.Conv3d(channels_dec1, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        bot = self.bottleneck(p3)

        u3 = self.up3(bot)
        a3 = self.att3(g=u3, x=e3)
        cat3 = torch.cat([u3, a3], dim=1)
        d3 = self.dec3(cat3)
        r3 = self.reduce3(d3)

        u2 = self.up2(r3)
        a2 = self.att2(g=u2, x=e2)
        cat2 = torch.cat([u2, a2], dim=1)
        d2 = self.dec2(cat2)
        r2 = self.reduce2(d2)

        u1 = self.up1(r2)
        a1 = self.att1(g=u1, x=e1)
        cat1 = torch.cat([u1, a1], dim=1)
        d1 = self.dec1(cat1)

        out = self.final_conv(d1)
        return out