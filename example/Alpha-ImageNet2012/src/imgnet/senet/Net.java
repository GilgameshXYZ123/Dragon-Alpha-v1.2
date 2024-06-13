/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.senet;

import imgnet.Config;
import java.util.ArrayList;
import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class Net 
{
     public static class SEBlock extends Module{
        Unit conv1, bn1, conv2, bn2, conv3, bn3, downsample, se;
        
        public SEBlock(int in_channel, int filter1, int filter2, int filter3, int stride) {
            conv1 = nn.conv3D(false, in_channel, filter1, 1, 1, 0);
            bn1 = nn.batchNorm_leakyRelu(nn.batchNorm(filter1), nn.leakyRelu());
            
            conv2 = nn.conv3D(false, filter1, filter2, 3, stride, 1);
            bn2 = nn.batchNorm_leakyRelu(nn.batchNorm(filter2), nn.leakyRelu());
            
            conv3 = nn.conv3D(false, filter2, filter3, 1, 1, 0);
            bn3 = nn.batchNorm(filter3);
            
            downsample = null;
            if(stride != 1 || in_channel != filter3) {
                ArrayList<Unit> seq = new ArrayList<>();
                if(stride != 1) {
                    seq.add(nn.conv3D(false, in_channel, in_channel, 3, stride, 1));
                    seq.add(nn.leakyRelu());
                }
                seq.add(nn.conv3D(false, in_channel, filter3, 1, 1, 0));
                seq.add(nn.batchNorm(filter3));
                downsample = nn.sequence(seq);
            }
            
            se = nn.sequence(
                    nn.adaptive_avgPool2D(1),//[N, 1, 1, filter3] -> [N, filter3]
                    nn.conv3D(false, filter3, filter3 / 16, 1, 1, 0),
                    nn.leakyRelu(),
                    nn.conv3D(false, filter3 / 16, filter3, 1, 1, 0),
                    nn.sigmoid());
        }

        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor[] res = X;

            X = bn1.forward(conv1.forward(X));
            X = bn2.forward(conv2.forward(X));
            X = bn3.forward(conv3.forward(X));
            X = F.mul_center(X[0], se.forward(X)[0]);//Sequeeze-Extract
            
            if(downsample != null) res = downsample.forward(res);
            return F.add_leakyRelu(X[0], res[0]);
        }
    }
    
    public static class SENet extends Module {
        Unit prepare = nn.sequence(//div = 4, 3 -> 64
                nn.conv3D(false, 3, 64, 7, 2, 3),
                nn.batchNorm_leakyRelu(nn.batchNorm(64), nn.leakyRelu())
        );
        
        Unit layer1 = nn.sequence(//div = 2, 64 -> 64
                new SEBlock(64, 64, 64, 256, 2),
                new SEBlock(256, 64, 64, 256, 1),
                new SEBlock(256, 64, 64, 256, 1)
        );
        
        Unit layer2 = nn.sequence(//div = 4, 64 -> 128
                new SEBlock(256, 128, 128, 512, 2), 
                new SEBlock(512, 128, 128, 512, 1),
                new SEBlock(512, 128, 128, 512, 1),
                new SEBlock(512, 128, 128, 512, 1)
        );
        
        Unit layer3 = nn.sequence(//div = 8, 128-> 1024
                new SEBlock(512, 256, 256, 1024, 2),
                new SEBlock(1024, 256, 256, 1024, 1),
                new SEBlock(1024, 256, 256, 1024, 1),
                new SEBlock(1024, 256, 256, 1024, 1),
                new SEBlock(1024, 256, 256, 1024, 1),    
                new SEBlock(1024, 256, 256, 1024, 1)         
        );
        
        Unit pool = nn.adaptive_avgPool2D(1);
        
        Unit layer4 = nn.sequence(//div = 16, 1024 -> 2048
                new SEBlock(1024, 512, 512, 2048, 2),
                new SEBlock(2048, 512, 512, 2048, 1),
                new SEBlock(2048, 512, 512, 2048, 1)
        );
        
        Unit fc = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 2048, 512),
                nn.leakyRelu_dropout(nn.leakyRelu(), nn.dropout(0.9f)),
                nn.fullconnect(true, 512, 1000));
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = prepare.forward(X);
            
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = layer4.forward(X);
            X = pool.forward(X);
            
            X = fc.forward(F.flatten(X));
            return X;
        }
        
        static String weight     =  Config.weight_home + "Alpha-v1.1-imgnet2012-SENet.zip";
        static String opt_weight =  Config.weight_home + "Alpha-v1.1-imgnet2012-SENet-opt.zip";
    }
}
