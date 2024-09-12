/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg19;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 *
 * @author Gilgamesh
 */
public class VGG19 extends Module {
    public static class AttentionBlock extends Module {
        Unit Wq, Wk, Wv; int channel; float div;
        public AttentionBlock(int channel, int D) {
            this.channel = channel;
            this.div = (float) Math.sqrt(channel);
            Wq = nn.fullconnect(false, channel, channel);
            Wk = nn.fullconnect(false, channel, D);
            Wv = nn.fullconnect(false, channel, D);
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            int dim[] = X[0].dim();
            X = F.view(X, dim[0], dim[1]*dim[2], dim[3]);//[N, H, W, C] -> [N, HW, C]

            Tensor Q = Wq.forward(X)[0];//[N, HW, C / 2]
            Tensor K = Wk.forward(X)[0];//[N, HW, C]
            Tensor V = Wv.forward(X)[0];//[N, HW, C / 2]
            
            Tensor score = F.batchMatMulT1(Q, K)[0];//[N, C / 2, HW] * [N, HW, C] -> [N, C / 2, C]
            score = F.softmax(-1, F.sdiv(div, score))[0];//[N, C / 2, C / 2]
            return F.view(F.batchMatMulT2(V, score), dim);//[N, HW, C / 2] * [N, C / 2, C] -> N[N, HW, C]
        }
    }
    
    Unit conv1 = nn.conv3D(false,  3, 64, 3, 1, 1);
    Unit conv2 = nn.conv3D(false, 64, 64, 3, 1, 1);
    Unit bn1 = nn.batchNorm_gelu(nn.batchNorm(64), nn.gelu());
    
    Unit conv3 = nn.conv3D(false,  64, 128, 3, 1, 1);
    Unit conv4 = nn.conv3D(false, 128, 128, 3, 1, 1);
    Unit bn2 = nn.batchNorm_gelu(nn.batchNorm(128), nn.gelu());
    
    Unit conv5 = nn.conv3D(false, 128, 256, 3, 1, 1);
    Unit conv6 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit conv7 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit conv8 = nn.conv3D(false, 256, 256, 3, 1, 1);
    Unit bn3 = nn.batchNorm_gelu(nn.batchNorm(256), nn.gelu());
    
    Unit conv9  = nn.conv3D(false, 256, 512, 3, 1, 1);
    Unit conv10 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv11 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv12 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit bn4 = nn.batchNorm_gelu(nn.batchNorm(512),nn.gelu());
   
    Unit conv13 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv14 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv15 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit conv16 = nn.conv3D(false, 512, 512, 3, 1, 1);
    Unit bn5 = nn.batchNorm_gelu(nn.batchNorm(512), nn.gelu());
    Unit attn = new AttentionBlock(512, 512);
    
    Unit classifier = nn.sequence(
            nn.fullconnect(false, 512, 512),
            nn.gelu_dropout(nn.gelu(), nn.dropout(false, 0.9f)),
            nn.fullconnect(false, 512, 128),
            nn.gelu_dropout(nn.gelu(), nn.dropout(false, 0.9f)),
            nn.fullconnect(true, 128, 10));
    
    @Override
    public Tensor[] __forward__(Tensor... X) {
        X = F.gelu(conv1.forward(X));//div2: channel = 3 -> 64
        X = bn1.forward(conv2.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv3.forward(X));//div4: channel 64 -> 128
        X = bn2.forward(conv4.forward(X));
        X = F.maxPool2D(2, X);
        
        X = F.leakyRelu(conv5.forward(X));//div8: channel 128 -> 256
        X = F.leakyRelu(conv6.forward(X));
        X = F.leakyRelu(conv7.forward(X));
        X = bn3.forward(conv8.forward(X));
        X = F.maxPool2D(2, X);

        X = F.leakyRelu(conv9.forward(X));//div16: 256 -> 512
        X = F.leakyRelu(conv10.forward(X));
        X = F.leakyRelu(conv11.forward(X));
        X = bn4.forward(conv12.forward(X));
        X = F.maxPool2D(2, X);
      
        X = F.leakyRelu(conv13.forward(X));//div32: 512 -> 512
        X = F.leakyRelu(conv14.forward(X));
        X = F.leakyRelu(conv15.forward(X));
        X = bn5.forward(conv16.forward(X));
        X = F.add_gelu(attn.forward(X)[0], X[0]);
        X = F.maxPool2D(2, X);
        
        X = classifier.forward(F.flatten(X));
        return X;
    }
    
    String weight = "C:\\Users\\Gilgamesh\\Desktop\\Alpha-v1.1-VGG19-cifar10.zip";
    public void load() { alpha.stat.load_zip(this, weight); };
    public void save() { alpha.stat.save_zip(this, weight); }
}
