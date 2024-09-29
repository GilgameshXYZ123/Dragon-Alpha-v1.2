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
public class BasicBlock extends Module {
    private static final long serialVersionUID = 1L;
    
    public Unit conv1, bn1, conv2, bn2, downsample;
    public Unit out, act;
    
    public BasicBlock(Unit activation, int in_channel, int out_channel, int stride) {
        act = activation; 
        
        conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
        bn1 = nn.fuse(nn.batchNorm(out_channel), act);

        conv2 = nn.conv3D(false, out_channel, out_channel, 3, 1, 1);
        bn2 = nn.batchNorm(out_channel);

        downsample = null;
        if (stride != 1 || out_channel != in_channel) 
            downsample = nn.sequence(
                    nn.conv3D(false, in_channel, out_channel, 3, stride, 1),
                    nn.batchNorm(out_channel)
            );
        
        out = nn.fuse(nn.add(), act);
    }
    
     @Override
    public Tensor[] __forward__(Tensor... X) {
        Tensor[] res = X;
        X = bn1.forward(conv1.forward(X));
        X = bn2.forward(conv2.forward(X));
        if (downsample != null) res = downsample.forward(res);
        return out.forward(X[0], res[0]);
    }
}
