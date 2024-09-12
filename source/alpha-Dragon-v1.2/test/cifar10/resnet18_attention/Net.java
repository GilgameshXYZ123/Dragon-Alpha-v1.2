/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.resnet18_attention;

import static z.dragon.alpha.Alpha.UnitBuilder.nn;
import static z.dragon.alpha.Alpha.UnitFunctional.F;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * @author Gilgamesh
 */
public class Net {
    public static class BasicBlock extends Module {
        Unit conv1, bn1, conv2, bn2, downsample;
        public BasicBlock(int in_channel, int out_channel, int stride) {
            conv1 = nn.conv3D(false, in_channel, out_channel, 3, stride, 1);
            bn1 = nn.fuse(nn.batchNorm(out_channel), nn.gelu());
            
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
    
    //sequence_length = channel, dim = H * W
    public static class ImageAttn extends Module {
        Unit Wq, Wk, Wv, Wy, dropout, attn; 
        private int head, hidden;
        
        public ImageAttn(int channel, int hidden, int head) {
            this.hidden = hidden;
            this.head = head;
            
            Wq = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wk = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wv = nn.conv3D(false, channel, hidden, 3, 1, 1);
            Wy = nn.conv3D(false, hidden, channel, 3, 1, 1);
            dropout = nn.dropout(0.9f);
            attn = nn.ImageAttn(head, dropout);
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
    
    //sequence_length = H * W, dim = channel
    public static class ChannelAttn extends Module {
        Unit Wq, Wk, Wv, Wy, dropout, attn;
        
        private final int channel, head, hidden;
        public ChannelAttn(int channel, int hidden, int head) {
            this.channel = channel;
            this.hidden = hidden;
            this.head = head;
            
            Wq = nn.fullconnect(false, channel, hidden);
            Wk = nn.fullconnect(false, channel, hidden);
            Wv = nn.fullconnect(false, channel, hidden);
            Wy = nn.fullconnect(false, hidden, channel);
            dropout = nn.dropout(0.9f);
            attn = nn.ChannelAttn(head, dropout);
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            int dim[] = X[0].dim(), N = dim[0], L = dim[1]*dim[2];
            X = F.view(X, N, L, channel);//[N, H, W, C] -> [N, L, C]

            Tensor Q = Wq.forward(X)[0];//[N, L, hidden]
            Tensor K = Wk.forward(X)[0];//[N, L, hidden]
            Tensor V = Wv.forward(X)[0];//[N, L, channel]
            Tensor Y = attn.forward(Q, K, V)[0];
            
            Y = Wy.forward(Y)[0];//[N, L, channel]
            return F.view(Y, dim);
        }
    }
    
    public static class ConvAttentionBlock extends Module {
        Unit Wq, Wk, Wv, dropout; int channel; float div;
        public ConvAttentionBlock(int channel) {
            this.channel = channel;
            this.div = (float) Math.sqrt(channel);
            Wq = nn.conv3D(false, channel, channel, 3, 1, 1);
            Wk = nn.conv3D(false, channel, channel, 3, 1, 1);
            Wv = nn.conv3D(false, channel, channel, 3, 1, 1);
        }
        
        @Override
        public Tensor[] __forward__(Tensor... X) {
            int[] dim = X[0].dim();
            int[] new_dim = new int[]{dim[0], dim[1]*dim[2], dim[3]};
            Tensor Q = F.view(Wq.forward(X), new_dim)[0];//[N, H, W, C] -> [N, HW, C]
            Tensor K = F.view(Wk.forward(X), new_dim)[0];//[N, H, W, C]
            Tensor V = F.view(Wv.forward(X), new_dim)[0];//[N, H, W, C]
            
            Tensor score = F.batchMatMulT1(Q, K)[0];//[N, C, HW] * [N, HW, C] -> [N, C, C]
            score = F.softmax(-1, F.sdiv(div, score))[0];//[N, C, C]
            return F.view(F.batchMatMulT2(V, score), dim);//[N, HW, C] * [N, C, C] -> N[N, HW, C]
        }
    }
    
    public static class ResNet18 extends Module {
        Unit prepare = nn.sequence(//div2, channel = 64, 32
                nn.conv3D(false, 3, 64, 7, 2, 3),
                nn.fuse(nn.batchNorm(64), nn.gelu())
        );
        
        Unit layer1 = nn.sequence(//div2, channel = 64, 16
                new BasicBlock(64, 64, 1),
                new BasicBlock(64, 64, 1)
        );
        
        Unit attn2 = new ImageAttn(64, 64, 4);
        Unit layer2 = nn.sequence(//div4, channel = 128, 16
                new BasicBlock(64, 128, 2),
                new BasicBlock(128, 128, 1)
        );
        
        //Unit attn3 = new ChannelAttn(128, 64, 8);
        Unit attn3 = new ImageAttn(128, 64, 4);
        Unit layer3 = nn.sequence(//div8, channel = 256, 8
                new BasicBlock(128, 256, 2),
                new BasicBlock(256, 256, 1)
        );
        
        //Unit attn4 = new ChannelAttn(256, 128, 8);
        Unit attn4 = new ImageAttn(256, 128, 4);
        Unit layer4 = nn.sequence(//div16, channel = 256, 4
                new BasicBlock(256, 512, 2),
                new BasicBlock(512, 512, 1)
        );
        
        Unit pool = nn.adaptive_avgPool2D(1, 1);
        
        Unit fc = nn.sequence(
                nn.dropout(0.9f),
                nn.fullconnect(true, 512, 128),
                nn.fuse(nn.gelu(), nn.dropout(0.9f)),
                nn.fullconnect(true, 128, 10)
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
        
        static String weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-resnet18-attention.zip";
        static String opt_weight = "C:\\Users\\Gilgamesh\\Desktop\\cifar10-resnet18-attention-opt.zip";
        public void load() { alpha.stat.load_zip(this, weight); };
        public void save() { alpha.stat.save_zip(this, weight); }
    }
}
