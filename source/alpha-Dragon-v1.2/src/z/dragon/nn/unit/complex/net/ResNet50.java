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
 * The backbone of ResNet50 without stem and classifier.
 * @author Gilgamesh
 */
public class ResNet50 extends Module {
    private static final long serialVersionUID = 1L;

    public Unit act, layer1, layer2, layer3, layer4;
    public ResNet50(Unit activation) {
        act = activation;
        layer1 = nn.sequence(//div = 2
                nn.BottleNeck(act, 64, 64, 2, 4),
                nn.BottleNeck(act, 256, 64, 1, 4),
                nn.BottleNeck(act, 256, 64, 1, 4)
        );
        layer2 = nn.sequence(//div = 4
                nn.BottleNeck(act, 256, 128, 2, 4),
                nn.BottleNeck(act, 512, 128, 2, 4),
                nn.BottleNeck(act, 512, 128, 2, 4),
                nn.BottleNeck(act, 512, 128, 2, 4)
        );
        layer3 = nn.sequence(//div16
                nn.BottleNeck(act, 512, 256, 2, 4),
                nn.BottleNeck(act, 1024, 256, 2, 4),
                nn.BottleNeck(act, 1024, 256, 2, 4),
                nn.BottleNeck(act, 1024, 256, 2, 4),
                nn.BottleNeck(act, 1024, 256, 2, 4),
                nn.BottleNeck(act, 1024, 256, 2, 4)
        );
        layer4 = nn.sequence(//div32
                nn.BottleNeck(act, 1024, 512, 2, 4),
                nn.BottleNeck(act, 2048, 512, 2, 4),
                nn.BottleNeck(act, 2048, 512, 2, 4)
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
