/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.resnet50;

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
    public static class BottleNeck extends Module {
        Unit conv1, bn1, conv2, bn2, conv3, bn3, downsample;
        
        public BottleNeck(int in_channel, int out_channel, int stride, int expand) {
            int expand_channel = out_channel * expand;
            
            conv1 = nn.conv3D(false, in_channel, out_channel, 1, 1, 0);
            bn1 = nn.batchNorm_leakyRelu(nn.batchNorm(out_channel), nn.leakyRelu());
            
            conv2 = nn.conv3D(false, out_channel, out_channel, 3, stride, 1);
            bn2 = nn.batchNorm_leakyRelu(nn.batchNorm(out_channel), nn.leakyRelu());
            
            conv3 = nn.conv3D(false, out_channel, expand_channel, 1, 1, 0);
            bn3 =  nn.batchNorm(expand_channel);
            
            downsample = null;
            if(stride != 1 || in_channel != out_channel) {
                ArrayList<Unit> seq = new ArrayList<>();
                if(stride != 1) {
                    seq.add(nn.conv3D(false, in_channel, in_channel, 3, stride, 1));
                    seq.add(nn.leakyRelu());
                }
                seq.add(nn.conv3D(false, in_channel, expand_channel, 1, 1, 0));
                seq.add(nn.batchNorm(expand_channel));
                downsample = nn.sequence(seq);
            }
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor[] res = X;
            
            X = bn1.forward(conv1.forward(X));
            X = bn2.forward(conv2.forward(X));
            X = bn3.forward(conv3.forward(X));
            
            if(downsample != null) res = downsample.forward(res);
            
            return F.add_leakyRelu(X[0], res[0]);
        }
    }
    
    public static class ResNet50 extends Module {
        Unit pre = nn.sequence(//div2
                nn.conv3D(false, 3, 64, 7, 2, 3),
                nn.batchNorm_leakyRelu(nn.batchNorm(64), nn.leakyRelu())
        );
        
        Unit layer1 = nn.sequence(//div4
                new BottleNeck( 64, 64, 2, 4),//64 -> 256
                new BottleNeck(256, 64, 1, 4),//256 -> 256
                new BottleNeck(256, 64, 1, 4) //256 -> 256
        );
        
        Unit layer2 = nn.sequence(//div8
                new BottleNeck(256, 128, 2, 4),//256 -> 512
                new BottleNeck(512, 128, 1, 4),//512 -> 512
                new BottleNeck(512, 128, 1, 4),//512 -> 512
                new BottleNeck(512, 128, 1, 4) //512 -> 512
        );
        
        Unit layer3 = nn.sequence(//div16
                new BottleNeck( 512, 256, 2, 4),//512 -> 1024
                new BottleNeck(1024, 256, 1, 4),//1024 -> 1024
                new BottleNeck(1024, 256, 1, 4),//1024 -> 1024
                new BottleNeck(1024, 256, 1, 4),//1024 -> 1024
                new BottleNeck(1024, 256, 1, 4),//1024 -> 1024
                new BottleNeck(1024, 256, 1, 4) //1024 -> 1024
        );
        
        Unit layer4 = nn.sequence(//div32
                new BottleNeck(1024, 512, 2, 4),//512 -> 2048
                new BottleNeck(2048, 512, 1, 4),//2048 -> 2048
                new BottleNeck(2048, 512, 1, 4) //2048 -> 2048
        );
        
        Unit pool = nn.adaptive_avgPool2D(1);
        
        Unit fc = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 2048, 512),
                nn.leakyRelu_dropout(nn.leakyRelu(), nn.dropout(0.9f)),
                nn.fullconnect(true, 512, 1000)
        );

        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = pre.forward(X);
            
            X = layer1.forward(X);
            X = layer2.forward(X);
            X = layer3.forward(X);
            X = layer4.forward(X);
            X = pool.forward(X);
            
            X = fc.forward(F.flatten(X));
            return X;
        }
        
        static String weight     =  Config.weight_home + "Alpha-v1.1-imgnet2012-Resnet50.zip";
        static String opt_weight =  Config.weight_home + "Alpha-v1.1-imgnet2012-Resnet50-opt.zip";
    }
}
