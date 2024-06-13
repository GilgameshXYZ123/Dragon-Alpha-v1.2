/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package paper.net_present;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;
import z.dragon.nn.unit.simple.affine.BatchNorm;

/**
 *
 * @author Gilgamesh
 */
public class Net
{
    public static class Block extends Module {
        Unit conv1, bn1, downsample;
        public Block(int in_channel, int out_channel, int stride) {
            conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
            bn1 = nn.batchNorm(false, out_channel);
            if(stride !=1  || out_channel != in_channel) 
                downsample = nn.sequence(
                        nn.conv3D(false, in_channel, out_channel, 3, stride, 1),
                        nn.batchNorm(out_channel));
        }

        @Override
        public void __init__(Engine eg) {
            super.__init__(eg); 
            for(BatchNorm bn : this.find(BatchNorm.class)) 
                bn.affine(true).beta1(0.1f).beta2(0.1f).eps(1e-5f);
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor[] res = X;
            X = F.leakyRelu(bn1.forward(conv1.forward(X)));
            if(downsample != null) res = downsample.forward(res);
            return F.leakyRelu(F.add(X[0], res[0]));
        }
    }
    
    public static class Network extends Module {
        Unit conv1 = nn.conv3D(false, 3, 64, 3, 1, 1);
        Unit bn1 = nn.batchNorm(false, 64);
        Block block1 = new Block(64, 128, 2);
        Block block2 = new Block(128, 256, 2);
        Unit fc = nn.fullconnect(true, 256, 10);
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = F.leakyRelu(bn1.forward(conv1.forward(X)));
            X = block1.forward(X);
            X = block2.forward(X);
            X = F.adaptive_avgPool2D(1, X);
            return fc.forward(F.flatten(X));
        }
    }
    
    static { alpha.home("alpha-home"); }
    static Mempool memp = alpha.engine.memp1(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
}
