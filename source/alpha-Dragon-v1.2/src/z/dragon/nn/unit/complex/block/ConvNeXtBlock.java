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
 *
 * @author Gilgamesh
 */
public class ConvNeXtBlock extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit dwconv, pwconv1, pwconv2, add;
    public Unit norm, act;
    
    public ConvNeXtBlock(Unit activation, Unit normalize,
            int in_channel, int expand, int out_channel,
            int kernel)
    {
        act = activation;//Gelu
        norm = normalize;//LayerNorm
        
        int hidden = in_channel * expand;
        dwconv = nn.depthwise_conv3D(false, in_channel, kernel, 1, kernel / 2);
        pwconv1 = nn.fullconnect(false, in_channel, hidden);
        pwconv2 = nn.fullconnect(false, hidden, out_channel);
        add = nn.add();
    }
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        Tensor[] res = X;

        X = norm.forward(dwconv.forward(X));
        X = act.forward(pwconv1.forward(X));
        X = pwconv2.forward(X);
        
        return add.forward(X[0], res[0]);
    }
}
