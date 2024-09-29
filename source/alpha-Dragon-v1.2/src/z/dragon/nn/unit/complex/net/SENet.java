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
 * The backbone of Squeeze-Extract Net without stem and classifier.
 * @author Gilgamesh
 */
public class SENet extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit act, layer1, layer2, layer3, layer4;
    public SENet(Unit activation) {
        act = activation;
        layer1 = nn.sequence(//div = 2
                nn.SEBlock(act, 64, 64, 64, 256, 2),
                nn.SEBlock(act, 256, 64, 64, 256, 1),
                nn.SEBlock(act, 256, 64, 64, 256, 1)
        );
        layer2 = nn.sequence(//div = 4
                nn.SEBlock(act, 256, 128, 128, 512, 2),
                nn.SEBlock(act, 512, 128, 128, 512, 1),
                nn.SEBlock(act, 512, 128, 128, 512, 1),
                nn.SEBlock(act, 512, 128, 128, 512, 1)
        );
        layer3 = nn.sequence(//div = 8
                nn.SEBlock(act, 512, 256, 256, 1024, 2),
                nn.SEBlock(act, 1024, 256, 256, 1024, 1),
                nn.SEBlock(act, 1024, 256, 256, 1024, 1),
                nn.SEBlock(act, 1024, 256, 256, 1024, 1),
                nn.SEBlock(act, 1024, 256, 256, 1024, 1),
                nn.SEBlock(act, 1024, 256, 256, 1024, 1)
        );
        layer4 = nn.sequence(//div = 16
                nn.SEBlock(act, 1024, 512, 512, 2048, 2),
                nn.SEBlock(act, 2048, 512, 512, 2048, 1),
                nn.SEBlock(act, 2048, 512, 512, 2048, 1)
        );
    }

    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = layer1.forward(X);
        X = layer2.forward(X);
        X = layer3.forward(X);
        X = layer4.forward(X);
        return X;
    }
}
    
