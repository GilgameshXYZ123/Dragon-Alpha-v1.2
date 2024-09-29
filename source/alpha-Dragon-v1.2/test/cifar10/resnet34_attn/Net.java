/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet34_attn;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
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
    public static class ImageAttn extends Module {
        Unit Wq, Wk, Wv, Wy, dropout, attn;
        public ImageAttn(int channel, int hidden, int head) {
            Wq = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wk = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wv = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wy = nn.conv3D(false, hidden, channel, 3, 1, 1);
            dropout = nn.dropout(0.8f);
            attn = nn.ImageMHA(head, dropout);
        }

        @Override
        public Tensor[] __forward__(Tensor... X) {
            Tensor Q = Wq.forward(X)[0];//[N, H, W, hidden]
            Tensor K = Wk.forward(X)[0];//[N, H, W, hidden]
            Tensor V = Wv.forward(X)[0];//[N, H, W, hidden]
            Tensor Y = attn.forward(Q, K, V)[0];
            return Wy.forward(Y);//[N, H, W, channel]
        }
    }
    
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
    
    public static class BasicBlock extends Module {
        Unit conv1, bn1, conv2, bn2, downsample;
        
        public BasicBlock(int in_channel, int out_channel, int stride) {
            conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
            bn1 = nn.batchNorm_gelu(nn.batchNorm(out_channel), nn.gelu());
            
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
            return F.add_gelu(X[0], res[0]);
        }
    }
    
    public static class ResNet34 extends Module {
        Unit prepare = nn.sequence(
                nn.conv3D(false, 3, 64, 7, 2, 3),
                nn.batchNorm_gelu(nn.batchNorm(64), nn.gelu())
        );
        
        Unit layer1 = nn.sequence(//div1, 64 -> 64
                nn.BasicBlock(nn.gelu(), 64, 64, 1),
                nn.BasicBlock(nn.gelu(), 64, 64, 1),
                nn.BasicBlock(nn.gelu(), 64, 64, 1),
                nn.BasicBlock(nn.gelu(), 64, 64, 1)
        );
        
        Unit attn2 = new ImageAttn(64, 64, 8);
        Unit layer2 = nn.sequence(//div2, 64 -> 128
                new BasicBlock(64, 128, 2),
                new BasicBlock(128, 128, 1),
                new BasicBlock(128, 128, 1),
                new BasicBlock(128, 128, 1)
        );
        
        Unit attn3 = new ImageAttn(128, 64, 8);
        Unit layer3 = nn.sequence(//div4, 128 -> 256
                new BasicBlock(128, 256, 2),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1),
                new BasicBlock(256, 256, 1)
        );
        
        Unit attn4 = new ImageAttn(256, 256, 8);
        Unit layer4 = nn.sequence(//div2, 64 -> 64
                new BasicBlock(256, 512, 2),
                new BasicBlock(512, 512, 1),
                new BasicBlock(512, 512, 1)
        );
        
        Unit pool = nn.adaptive_avgPool2D(1);// -> 512
        
        Unit fc = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 512, 256),
                nn.gelu_dropout(nn.gelu(), nn.dropout(false, 0.9f)),
                nn.fullconnect(true, 256, 10)
        );

        @Override
        public Tensor[] __forward__(Tensor... X) {
            X = prepare.forward(X);
            
            X = layer1.forward(X);
            
            X = F.add_gelu(attn2.forward(X)[0], X[0]);
            X = layer2.forward(X);
          
            X = F.add_gelu(attn3.forward(X)[0], X[0]);
            X = layer3.forward(X);
            
            X = F.add_gelu(attn4.forward(X)[0], X[0]);
            X = layer4.forward(X);
            
            X = pool.forward(X);
            return fc.forward(F.flatten(X));
        }
        
        String net_weight = "C:\\Users\\Gilgamesh\\Desktop\\Alpha-v1.1-cifar10-Resnet34.zip";
        String opt_weight = "C:\\Users\\Gilgamesh\\Desktop\\Alpha-v1.1-cifar10-Resnet34_opt.zip";
        
        public void load(Optimizer opt) {
            alpha.stat.load_zip(this, net_weight, false); 
            if (opt != null) alpha.stat.load_zip(opt, opt_weight);
        }
        public void save(Optimizer opt) { 
            alpha.stat.save_zip(this, net_weight); 
            if (opt != null) alpha.stat.save_zip(opt, opt_weight);
        }
    }
}
