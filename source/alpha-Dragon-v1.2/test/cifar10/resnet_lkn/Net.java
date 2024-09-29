/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet_lkn;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.optim.Optimizer;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * @author Gilgamesh
 */
public class Net {
    public static class ResLKN18 extends Module {
        Unit stem = nn.sequence(//div2, channel = 64
                nn.conv3D(false, 3, 64, 3, 1, 1),
                nn.fuse(nn.batchNorm(false, 64), nn.gelu())
        );
        
        Unit layer1 = nn.sequence(//32 * 32
                nn.BasicBlock(nn.leakyRelu(), 64, 64, 1),
                nn.BasicBlock(nn.gelu(), 64, 64, 1)
        );
        
        Unit layer2 = nn.sequence(//16 * 16
                nn.BasicBlock(nn.leakyRelu(), 64, 128, 2),
                nn.BasicBlock(nn.gelu(), 128, 128, 1)
        );
        
        Unit layer3 = nn.sequence(//8*8
                nn.BasicBlock(nn.leakyRelu(), 128, 256, 2),
                nn.BasicBlock(nn.gelu(), 256, 256, 1),
                nn.LargeKernelAttn(256, 256, 7, nn.dropout(0.7f))
        );
        
        Unit layer4 = nn.sequence(//4*4
                nn.BasicBlock(nn.leakyRelu(), 256, 512, 2),
                nn.BasicBlock(nn.gelu(), 512, 512, 1)
//                nn.LargeKernelAttn(512, 512, 5, nn.dropout(0.7f))
        );
        
        Unit fc = nn.sequence(
                nn.adaptive_avgPool2D(1, 1),
                nn.flaten(),
                nn.dropout(0.9f),
                nn.fullconnect(true, 512, 128),
                nn.fuse(nn.leakyRelu(), nn.dropout(0.9f)),
                nn.fullconnect(true, 128, 10)
        );
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = stem.forward(X);
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = layer4.forward(X);
            return fc.forward(X);
        } 
        
        static String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-reslkn18.zip";
        static String opt_weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-reslkn18-opt.zip";
        
        public void load(Optimizer opt) { 
            alpha.stat.load_zip(this, weight); 
            if (opt != null) alpha.stat.load_zip(opt, opt_weight);
        };
        
        public void save(Optimizer opt) { 
            alpha.stat.save_zip(this, weight); 
            if (opt != null) alpha.stat.save_zip(opt, opt_weight);
        }
    }
}
