/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cifar10.vgg19;

import static z.dragon.alpha.Alpha.alpha;
import z.dragon.data.BufferedTensorIter;
import z.dragon.data.DataSet;
import z.dragon.data.Pair;
import z.dragon.dataset.Cifar10;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.memp.Mempool;
import z.dragon.nn.loss.LossFunction;
import z.dragon.nn.optim.Optimizer;
import z.util.lang.SimpleTimer;

/**
 *
 * @author Gilgamesh
 */
public class train {
    static { alpha.home("C:\\Users\\Gilgamesh\\Desktop\\Dragon-alpha-v1.2"); }
    static final Mempool memp = alpha.engine.memp2(alpha.MEM_1GB * 8);
    static Engine eg = alpha.engine.cuda_float32(0, memp, alpha.MEM_1MB * 1024);
    
    static float lr = 0.0005f;
    static int batch_size = 512;
    
    public static void train(int epoch) {
        VGG19 net = new VGG19().train().init(eg).println();
    
        Optimizer opt = alpha.optim.AdamW(net.params(), lr, 1e-3f).amsgrad(true).println();//45 epoch
//        Optimizer opt = alpha.optim.SGDMN(net.params(), lr)//40 epoch
//                .momentum(0.9f).nestorv(1).println();
        
        LossFunction loss = alpha.loss.softmax_crossEntropy();

        DataSet<byte[], Integer> train_set = Cifar10.train();
        BufferedTensorIter iter = train_set.buffered_iter(eg, batch_size);
        eg.sync(false).check(false);
        
        int batchIdx = 0;
        SimpleTimer timer = new SimpleTimer().record();
        for(int i=0; i<epoch; i++) {
            System.out.println("\nepoch = " + i);
            for(iter.reset().shuffle_swap(); iter.hasNext(); batchIdx++) {
                Pair<Tensor, Tensor> pair = iter.next();
                Tensor x = pair.input.c().linear(true, 2.0f, -1.0f);
                Tensor y = pair.label;
                
                Tensor yh = net.forward(x)[0];
                
                if(batchIdx % 10 == 0) {
                    float ls = loss.loss(yh, y).get();
                    System.out.println(batchIdx + ": loss = " + ls + ", lr = " + opt.learning_rate());
                }
                
                net.backward(loss.gradient(yh, y));
                opt.update().clear_grads();
                net.gc();
            }
        }
        
        long div = timer.record().timeStamp_dif_millis();
        float time = 1.0f * div / batchIdx;
        System.out.println("total = " + (1.0f*div/1000));
        System.out.println("for each sample:" + time/batch_size);
        net.save();
    }
    
    public static void main(String[] args)
    {
        try
        {
            train(44);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}
