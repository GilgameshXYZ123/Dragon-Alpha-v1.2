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
 * The backbone of ResNet18 without stem and classifier.
 * @author Gilgamesh
 */
public class ResNet18 extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit act, layer1, layer2, layer3, layer4;
    public ResNet18(Unit activation) {
        act = activation;
        layer1 = nn.sequence(
                nn.BasicBlock(act, 64, 64, 1),
                nn.BasicBlock(act, 64, 64, 1)
        );
        layer2 = nn.sequence(//div = 2
                nn.BasicBlock(act, 64, 128, 2),
                nn.BasicBlock(act, 128, 128, 1)
        );
        layer3 = nn.sequence(//div = 4
                nn.BasicBlock(act, 128, 256, 2),
                nn.BasicBlock(act, 256, 256, 1)
        );
        layer4 = nn.sequence(//div = 8
                nn.BasicBlock(act, 256, 512, 2),
                nn.BasicBlock(act, 512, 512, 1)
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
