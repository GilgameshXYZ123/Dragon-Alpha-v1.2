/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.TNet;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class Net 
{
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.1"); }
    static Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 2048);
    
    public static class BasicBlock extends Module {
        Unit conv1, bn1, conv2, bn2, downsample;
        public BasicBlock(int in_channel, int out_channel, int stride) {
            conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
            bn1 = nn.batchNorm_leakyRelu(nn.batchNorm(out_channel), nn.leakyRelu());
            
            conv2 = nn.conv3D(false, out_channel, out_channel, 3, 1, 1);
            bn2 = nn.batchNorm(out_channel);
            
            if(stride != 1 || out_channel != in_channel)
                downsample = nn.sequence(
                        nn.conv3D(false, in_channel, out_channel, 3, stride, 1),
                        nn.batchNorm(out_channel)
                );
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor[] res = X;
            
            X = bn1.forward(conv1.forward(X));
            X = bn2.forward(conv2.forward(X));
            
            if(downsample != null) res = downsample.forward(res);
            return F.add_leakyRelu(X[0], res[0]);
        }
    }
    
    public static class TBlock extends Module {
        int H, W, C;
        Unit conv1, conv2, conv3, conv4;
        Unit bn1, bn2, bn3, bn4;
        public TBlock(int C) {
            conv1 = nn.conv3D(false, C, C, 3, 1, 1);
            conv2 = nn.conv3D(false, C, C, 3, 1, 1);
            conv3 = nn.conv3D(false, C, C, 3, 1, 1);
            
            bn1 = nn.batchNorm_leakyRelu(nn.batchNorm(C), nn.leakyRelu());
            bn2 = nn.batchNorm_leakyRelu(nn.batchNorm(C), nn.leakyRelu());
            bn3 = nn.batchNorm_leakyRelu(nn.batchNorm(C), nn.leakyRelu());
            
            conv4 = nn.conv3D(false, C, C, 1, 1, 0);
            bn4 = nn.batchNorm_leakyRelu(nn.batchNorm(C), nn.leakyRelu());
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {//[n: 0, h: 1, w: 2, c: 3]
            int[] dim = X[0].dim();
            Tensor[] X1 = F.reshape(true, F.transpose(false, 2, 3, X), dim);
            Tensor[] X2 = F.reshape(true, F.transpose(false, 1, 3, X), dim);
            
            X1 = bn1.forward(conv1.forward(X1));
            X2 = bn2.forward(conv2.forward(X2));
            Tensor[] W = F.mul(X1[0], X2[0]);
            
            X = bn3.forward(conv3.forward(X));
            X = bn4.forward(conv4.forward(X));
            return F.add_leakyRelu(X[0], W[0]);
        }
    }
    
    public static class TNet extends Module {
        Unit prepare = nn.sequence(//32 - 16
                nn.conv3D(false, 3, 64, 7, 2, 3),
                nn.batchNorm_leakyRelu(nn.batchNorm(64), nn.leakyRelu())
        );
        
        Unit layer1 = nn.sequence(//16 - 8
                new BasicBlock(64, 128, 2),
                new TBlock(128)
        );
        
        Unit layer2 = nn.sequence(//8 - 4
                new BasicBlock(128, 256, 2),
                new TBlock(256)
        );
        
        Unit layer3 = nn.sequence(//4 - 2
                new BasicBlock(256, 512, 2),
                new TBlock(512)
        );
        
        Unit layer4 = nn.sequence(//2
                new BasicBlock(512, 512, 2),
                new TBlock(512)
        );
        
        Unit pool = nn.adaptive_avgPool2D(1);
        
        Unit classifier = nn.sequence(
                nn.fullconnect(true, 512, 512),
                nn.leakyRelu_dropout(nn.leakyRelu(), nn.dropout(0.9f)),
                nn.fullconnect(true, 512, 10)
        );
        
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = prepare.forward(X);
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = pool.forward(X);
            X = classifier.forward(X);
            return X;
        }
        
          
        static String weight = "C:\\Users\\Gilgamesh\\Desktop\\Alpha-v1.1-cifar10-TNet.zip";
        public void load() { alpha.stat.load_zip(this, weight); }
        public void save() { alpha.stat.save_zip(this, weight);}
    }
}
