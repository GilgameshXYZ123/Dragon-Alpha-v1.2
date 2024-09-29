/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.net;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * The backone of ConvNext [3, 3, 9, 3] without stem and classifier.
 * @author Gilgamesh
 */
public class ConvNeXt extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit layer1, layer2, layer3, layer4;
    public Unit downsample1, downsample2, downsample3;
    
    public ConvNeXt(int in_channel, int expand) {
        final int C1 = in_channel;
        final int C2 = in_channel << 1;
        final int C4 = in_channel << 2;
        final int C8 = in_channel << 3;
        
        layer1 = nn.sequence(//3
                nn.ConvNeXtBlock(C1, C1, expand),
                nn.ConvNeXtBlock(C1, C1, expand),
                nn.ConvNeXtBlock(C1, C1, expand)
        );
        downsample1 = nn.sequence(//patchfy downsample: 2
                nn.layerNorm(C1),
                nn.conv3D(false, C1, C2, 2, 2, 0)
        );

        layer2 = nn.sequence(//4
                nn.ConvNeXtBlock(C2, C2, expand),
                nn.ConvNeXtBlock(C2, C2, expand),
                nn.ConvNeXtBlock(C2, C2, expand)
        );
        downsample2 = nn.sequence(//patchfy downsample
                nn.layerNorm(C2),
                nn.conv3D(false, C2, C4, 2, 2, 0)
        );

        layer3 = nn.sequence(//9
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand),
                nn.ConvNeXtBlock(C4, C4, expand)
        );
        downsample3 = nn.sequence(//patchfy downsample
                nn.layerNorm(C4),
                nn.conv3D(false, C4, C8, 2, 2, 0)
        );

        layer4 = nn.sequence(//3
                nn.ConvNeXtBlock(C8, C8, expand),
                nn.ConvNeXtBlock(C8, C8, expand),
                nn.ConvNeXtBlock(C8, C8, expand)
        );
    }
  
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = downsample1.forward(layer1.forward(X));
        X = downsample2.forward(layer2.forward(X));
        X = downsample3.forward(layer3.forward(X));
        X = layer4.forward(X);
        return X;
    }
}
