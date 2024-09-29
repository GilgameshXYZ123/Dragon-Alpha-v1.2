/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.gated_cnn;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.optim.Optimizer;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class Net {
    
    public static class GatedCNN extends Module {
        Unit stem = nn.sequence(
                nn.conv3D(false,  3, 64, 3, 1, 1),
                nn.leakyRelu(),
                nn.conv3D(false, 64, 64, 3, 1, 1),
                nn.fuse(nn.batchNorm(64), nn.gelu())
        );
        
        Unit layer1 = nn.sequence(//32
                nn.GatedBlock(nn.gelu(), nn.fuse(nn.batchNorm(64), nn.leakyRelu()), 64, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(64), nn.leakyRelu()), 64, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(64), nn.leakyRelu()), 64, 7),
                nn.conv3D(false, 64, 128, 3, 2, 1)
        );
        
        Unit layer2 = nn.sequence(//16
                nn.GatedBlock(nn.gelu(), nn.fuse(nn.batchNorm(128), nn.leakyRelu()), 128, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(128), nn.leakyRelu()), 128, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(128), nn.leakyRelu()), 128, 7),
                nn.conv3D(false, 128, 256, 3, 2, 1)
        );
        
        Unit layer3 = nn.sequence(//8
                nn.GatedBlock(nn.gelu(), nn.fuse(nn.batchNorm(256), nn.leakyRelu()), 256, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(256), nn.leakyRelu()), 256, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(256), nn.leakyRelu()), 256, 7),
                nn.conv3D(false, 256, 512, 3, 2, 1)
        );
        
        Unit layer4 = nn.sequence(//4
                nn.GatedBlock(nn.gelu(), nn.fuse(nn.batchNorm(512), nn.leakyRelu()), 512, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(512), nn.leakyRelu()), 512, 7),
                nn.GatedBlock(nn.leakyRelu(), nn.fuse(nn.batchNorm(512), nn.leakyRelu()), 512, 7),
                nn.conv3D(false, 512, 512, 3, 2, 1)
        );
        
        Unit classifier = nn.sequence(
                nn.adaptive_avgPool2D(1),
                nn.fuse(nn.gelu(), nn.dropout(0.8f)),
                nn.fullconnect(true, 512, 512),
                nn.fuse(nn.gelu(), nn.dropout(0.8f)),
                nn.fullconnect(true, 512, 10)
        );

        @Override
        public Tensor[] __forward__(Tensor... X) {
            X =  stem.forward(X);
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = layer4.forward(X);
            return classifier.forward(X);
        }
        
        static String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-GatedCNN.zip";
        static String opt_weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-GatedCNN-opt.zip";
        
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
