/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.block;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * Large Kernel Attention Block
 * @author Gilgamesh
 */
public class LargeKernelAttn extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit conv1, conv2, conv3, bn, mul;
    public Unit act, mask;
    
    public LargeKernelAttn(Unit activation, Unit mask, 
            int channel, int hidden, int kernel) 
    {
        this.act = activation;
        this.mask = mask;
        if (hidden == -1) hidden = channel;
        
        bn = nn.batchNorm(channel);
        conv1 = nn.fullconnect(false, channel, hidden);
        conv2 = nn.depthwise_conv3D(false, hidden, kernel, 1, kernel / 2);
        conv3 = nn.fullconnect(false, hidden, channel);
        mul = nn.mul();
    }
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        Tensor res = X[0];
        
        X = bn.forward(X);
        X = conv1.forward(X); if (act != null) X = act.forward(X);
        X = conv2.forward(X);
        X = conv3.forward(X); if (mask != null) X = mask.forward(X);
        
        return mul.forward(X[0], res);
    }
}
