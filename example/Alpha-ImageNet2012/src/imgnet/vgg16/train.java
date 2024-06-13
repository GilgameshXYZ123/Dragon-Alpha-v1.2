/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package imgnet.vgg16;

import static imgnet.Config.ImgNet2012_trainset;
import static imgnet.Config.eg;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.Pair;
import z.dragon.engine.Tensor;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train 
{
    static int batch_size = 256;
    static float lr = 0.001f;
    
    public static void train(int epoch) {
        VGG16 net = new VGG16().train().init(eg).println();
        
//        Optimizer opt = alpha.optim.SGDMN(net.param_map(), lr).momentum(0.9f).nestorv(1.0f).println(); 
        Optimizer opt = alpha.optim.Adam(net.param_map(), lr).println();
        
        alpha.stat.load_zip(net, VGG16.weight);
        alpha.stat.load_zip(opt, VGG16.opt_weight); 
        
        BufferedTensorIter iter = ImgNet2012_trainset.init().shuffle_sort().buffered_iter(eg, batch_size);
        LossFunction loss = alpha.loss.softmax_crossEntropy();
        
        int batchIdx = 0; 
        eg.sync(false).check(false);
        SimpleTimer timer = new SimpleTimer().record();
        
        for(int i=0; i<epoch; i++) {
            System.out.println("\n\nepoch = " + i);
            for(iter.reset().shuffle_sort(); iter.hasNext(); batchIdx++){
                Pair<Tensor, Tensor> pair = iter.next();
                Tensor X = pair.input.c().linear(true, 2.0f, -1.0f);
                Tensor Y = pair.label;
                
                Tensor Yh = net.forward(X)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(Yh, Y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(Yh, Y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f * div / 1000));
        System.out.println("for each sample:" + time / batch_size);
        
        alpha.stat.save_zip(net, VGG16.weight); 
        alpha.stat.save_zip(opt, VGG16.opt_weight);
    }
    
    public static void main(String[] args) {
        try { train(10); }
        catch(Exception e) { e.printStackTrace(); }
    }
}
