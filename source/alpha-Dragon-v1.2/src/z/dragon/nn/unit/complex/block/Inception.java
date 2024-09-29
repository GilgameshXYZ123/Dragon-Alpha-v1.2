/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.block;

import z.dragon.alpha.Alpha.Line;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class Inception extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit branch1, branch2, branch3, branch4, act, concat;
    public Inception(Unit activation, int in_channels,
            int channel_1x1, 
            int channel_3x3_reduce, int channel_3x3,
            int channel_5x5_reduce, int channel_5x5,
            int pool_proj)
    {
        if (act == null) throw new NullPointerException("activation can't be null");
        act = activation;
        
        branch1 = nn.sequence(nn.conv3D(true, in_channels, channel_1x1, 1, 1, 0), act);
        branch2 = nn.sequence(
                nn.conv3D(true, in_channels, channel_3x3_reduce, 1, 1, 0), act,
                nn.conv3D(true, channel_3x3_reduce, channel_3x3, 3, 1, 1), act);
        branch3 = nn.sequence(
                nn.conv3D(true, in_channels, channel_5x5_reduce, 1, 1, 0), act,
                nn.conv3D(true, channel_5x5_reduce, channel_5x5, 5, 1, 2), act);
        branch4 = nn.sequence(
                nn.maxPool2D(3, 1, 1),
                nn.conv3D(true, in_channels, pool_proj, 1, 1, 0), act);
        concat = nn.concat(-1);
    }
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        Line<Tensor> X1 = alpha.line(()-> branch1.forward(X)[0]);
        Line<Tensor> X2 = alpha.line(()-> branch2.forward(X)[0]);
        Line<Tensor> X3 = alpha.line(()-> branch3.forward(X)[0]);
        Line<Tensor> X4 = alpha.line(()-> branch4.forward(X)[0]);
        return concat.forward(X1.c(), X2.c(), X3.c(), X4.c());
    }
}
