/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.block;

import java.util.ArrayList;
import java.util.List;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class BottleNeck extends Module {
    private static final long serialVersionUID = 1L;

    public Unit conv1, bn1, conv2, bn2, conv3, bn3, downsample;
    public Unit act, out;
        
    public BottleNeck(Unit activation, int in_channel, int hidden, int stride, int expand) {
        int out_channel = hidden * expand; 
        act = activation; 
            
        conv1 = nn.conv3D(false, in_channel, hidden, 1, 1, 0);
        bn1 = nn.fuse(nn.batchNorm(hidden), act);
            
        conv2 = nn.conv3D(false, hidden, hidden, 3, stride, 1);
        bn2 = nn.fuse(nn.batchNorm(hidden), act);
            
        conv3 = nn.conv3D(false, hidden, out_channel, 1, 1, 0);
        bn3 =  nn.batchNorm(out_channel);
            
        downsample = null;
        if(stride != 1 || in_channel != hidden) {
            List<Unit> seq = new ArrayList<>();
            if(stride != 1) {
                seq.add(nn.conv3D(false, in_channel, in_channel, 3, stride, 1));
                seq.add(act);
            }
            seq.add(nn.conv3D(false, in_channel, out_channel, 1, 1, 0));
            seq.add(nn.batchNorm(out_channel));
            downsample = nn.sequence(seq);
        }
        
        out = nn.fuse(nn.add(), act);
    }
        
    @Override
    public Tensor[] __forward__(Tensor... X) {
        Tensor[] res = X;
        X = bn1.forward(conv1.forward(X));
        X = bn2.forward(conv2.forward(X));
        X = bn3.forward(conv3.forward(X));
        if(downsample != null) res = downsample.forward(res);
        return out.forward(X[0], res[0]);
    }
}
